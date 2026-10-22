from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn

from dcvpp.models.workload.cfc_w import CfCW


OUTPUT_KEYS = (
    "count_mean",
    "count_dispersion",
    "volume_log_mu",
    "volume_log_sigma",
    "volume_gamma_shape",
    "volume_gamma_rate",
)


class _DefaultCfCCell(nn.Module):
    """The installed ncps default CfC branch without unreachable modes."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.backbone = source.backbone
        self.ff1 = source.ff1
        self.ff2 = source.ff2
        self.time_a = source.time_a
        self.time_b = source.time_b

    def forward(self, inputs: Tensor, hidden: Tensor, elapsed: Tensor) -> Tensor:
        joined = self.backbone(torch.cat([inputs, hidden], dim=1))
        first = torch.tanh(self.ff1(joined))
        second = torch.tanh(self.ff2(joined))
        interpolation = torch.sigmoid(
            self.time_a(joined) * elapsed + self.time_b(joined)
        )
        return first * (1.0 - interpolation) + interpolation * second


class _DefaultCfCEncoder(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.rnn_cell = _DefaultCfCCell(source.rnn_cell)
        self.state_size = int(source.state_size)


class _TupleCfCWCore(nn.Module):
    """CfCW with tuple outputs, required by make_graphed_callables."""

    def __init__(self, source: CfCW) -> None:
        super().__init__()
        self.encoder = _DefaultCfCEncoder(source.encoder)
        self.count_head = source.count_head
        self.lognormal_head = source.lognormal_head
        self.gamma_head = source.gamma_head
        self.horizon = int(source.horizon)
        self.cohorts = int(source.cohorts)

    def forward(self, inputs: Tensor, elapsed: Tensor) -> Tuple[Tensor, ...]:
        hidden = torch.zeros(
            (inputs.size(0), self.encoder.state_size),
            dtype=inputs.dtype,
            device=inputs.device,
        )
        for step in range(inputs.size(1)):
            hidden = self.encoder.rnn_cell(
                inputs[:, step], hidden, elapsed[:, step]
            )
        shape = (inputs.size(0), self.cohorts, self.horizon, 2)
        count = self.count_head(hidden).reshape(shape)
        lognormal = self.lognormal_head(hidden).reshape(shape)
        gamma = self.gamma_head(hidden).reshape(shape)
        return (
            torch.nn.functional.softplus(count[..., 0]) + 1e-4,
            torch.nn.functional.softplus(count[..., 1]) + 1e-4,
            lognormal[..., 0],
            torch.nn.functional.softplus(lognormal[..., 1]) + 1e-4,
            torch.nn.functional.softplus(gamma[..., 0]) + 1e-4,
            torch.nn.functional.softplus(gamma[..., 1]) + 1e-4,
        )


class CUDAGraphedCfCWRecovery(nn.Module):
    """Checkpoint-compatible CfCW using a CUDA graph for full training batches.

    Evaluation and partial batches use the same eager core. The graph records
    the original operator sequence; it does not replace the model, loss,
    optimizer, batch size, seed order, or checkpoint state.
    """

    def __init__(self, source: CfCW, *, capture_batch_size: int = 32) -> None:
        super().__init__()
        self.core = _TupleCfCWCore(source)
        self._eager_forward = self.core.forward
        self.capture_batch_size = int(capture_batch_size)
        self._architecture_contract = dict(source.architecture_contract)
        self._graphed_training: nn.Module | None = None
        self._first_input_gate_passed = False

    @property
    def architecture_contract(self) -> dict[str, int | str]:
        return dict(self._architecture_contract)

    @property
    def acceleration_contract(self) -> dict[str, Any]:
        return {
            "backend": "torch.cuda.make_graphed_callables",
            "model": "CfCW",
            "capture_batch_size": self.capture_batch_size,
            "captured_scope": "forward_and_backward_for_full_training_batches",
            "partial_batch_and_evaluation_fallback": "same_eager_default_cfc_core",
            "model_math_changed": False,
            "loss_changed": False,
            "optimizer_changed": False,
            "checkpoint_state_key_order_changed": False,
        }

    def _validate_first_inputs(self, inputs: Tensor, elapsed: Tensor) -> None:
        if inputs.ndim != 3 or elapsed.ndim != 3:
            raise ValueError("CfCW expects batched event sequences")
        if inputs.shape[:2] != elapsed.shape[:2] or elapsed.shape[-1] != 1:
            raise ValueError("elapsed time must align to event sequences")
        if (elapsed <= 0).any() or not torch.isfinite(elapsed).all():
            raise ValueError("elapsed time must be finite and strictly positive")
        self._first_input_gate_passed = True

    def _capture(self, inputs: Tensor, elapsed: Tensor) -> nn.Module:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.get_autocast_dtype("cuda"),
            enabled=torch.is_autocast_enabled("cuda"),
            cache_enabled=False,
        ):
            graphed = torch.cuda.make_graphed_callables(
                self.core, (inputs, elapsed), num_warmup_iters=2
            )
        self.zero_grad(set_to_none=True)
        return graphed

    def forward(self, inputs: Tensor, elapsed: Tensor) -> Dict[str, Tensor]:
        if not self._first_input_gate_passed:
            self._validate_first_inputs(inputs, elapsed)
        elif inputs.ndim != 3 or elapsed.shape[:2] != inputs.shape[:2]:
            raise ValueError("CfCW input shape changed after the verified first batch")

        use_graph = (
            self.training
            and torch.is_grad_enabled()
            and inputs.is_cuda
            and inputs.shape[0] == self.capture_batch_size
        )
        if use_graph:
            if self._graphed_training is None:
                self._graphed_training = self._capture(inputs, elapsed)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.get_autocast_dtype("cuda"),
                enabled=torch.is_autocast_enabled("cuda"),
                cache_enabled=False,
            ):
                values = self._graphed_training(inputs, elapsed)
        else:
            # make_graphed_callables patches core.forward in-place. Retain the
            # original bound method so the final partial batch is not replayed
            # against the static full-batch graph.
            values = self._eager_forward(inputs, elapsed)
        return dict(zip(OUTPUT_KEYS, values, strict=True))

    def state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Tensor]:
        return self.core.state_dict(*args, **kwargs)

    def load_state_dict(
        self, state_dict: Mapping[str, Tensor], strict: bool = True, assign: bool = False
    ):
        # Loading a best/checkpoint state copies into the same Parameter
        # objects already referenced by the graph. Keep the captured graph
        # alive; destroying it mid-process can invalidate CUDA stream state.
        return self.core.load_state_dict(state_dict, strict=strict, assign=assign)


class _TupleMappingCore(nn.Module):
    """Expose a mapping-output workload model as a stable tensor tuple."""

    def __init__(self, source: nn.Module, output_keys: Sequence[str]) -> None:
        super().__init__()
        self.source = source
        self.output_keys = tuple(str(key) for key in output_keys)
        if not self.output_keys:
            raise ValueError("CUDA-graph workload wrapper requires output keys")

    def forward(self, inputs: Tensor, elapsed: Tensor) -> Tuple[Tensor, ...]:
        unchecked = getattr(self.source, "_cuda_graph_forward_unchecked", None)
        output = unchecked(inputs, elapsed) if unchecked is not None else self.source(inputs, elapsed)
        return tuple(output[key] for key in self.output_keys)


class CUDAGraphedMappingWorkloadModel(nn.Module):
    """CUDA-graph runtime wrapper for mapping-output CfC ablations.

    The wrapped module and its Parameter objects are retained verbatim.  Only
    the repeated full-batch forward/backward launch sequence is captured;
    partial batches and all evaluation calls use the original eager method.
    """

    def __init__(
        self,
        source: nn.Module,
        *,
        output_keys: Sequence[str],
        capture_batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.core = _TupleMappingCore(source, output_keys)
        self._eager_forward = self.core.forward
        self.output_keys = tuple(str(key) for key in output_keys)
        self.capture_batch_size = int(capture_batch_size)
        self._architecture_contract = dict(
            getattr(source, "architecture_contract", {})
        )
        self._graphed_training: nn.Module | None = None
        self._first_input_gate_passed = False

    @property
    def architecture_contract(self) -> dict[str, Any]:
        return dict(self._architecture_contract)

    @property
    def acceleration_contract(self) -> dict[str, Any]:
        return {
            "backend": "torch.cuda.make_graphed_callables",
            "capture_batch_size": self.capture_batch_size,
            "captured_scope": "forward_and_backward_for_full_training_batches",
            "partial_batch_and_evaluation_fallback": "same_eager_model",
            "model_math_changed": False,
            "loss_changed": False,
            "optimizer_changed": False,
            "checkpoint_state_key_order_changed": False,
        }

    def _validate_first_inputs(self, inputs: Tensor, elapsed: Tensor) -> None:
        if inputs.ndim != 3 or elapsed.ndim != 3:
            raise ValueError("workload model expects batched event sequences")
        if inputs.shape[:2] != elapsed.shape[:2] or elapsed.shape[-1] != 1:
            raise ValueError("elapsed time must align to event sequences")
        if (elapsed <= 0).any() or not torch.isfinite(elapsed).all():
            raise ValueError("elapsed time must be finite and strictly positive")
        self._first_input_gate_passed = True

    def _capture(self, inputs: Tensor, elapsed: Tensor) -> nn.Module:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.get_autocast_dtype("cuda"),
            enabled=torch.is_autocast_enabled("cuda"),
            cache_enabled=False,
        ):
            graphed = torch.cuda.make_graphed_callables(
                self.core, (inputs, elapsed), num_warmup_iters=2
            )
        self.zero_grad(set_to_none=True)
        return graphed

    def forward(self, inputs: Tensor, elapsed: Tensor) -> Dict[str, Tensor]:
        if not self._first_input_gate_passed:
            self._validate_first_inputs(inputs, elapsed)
        elif inputs.ndim != 3 or elapsed.shape[:2] != inputs.shape[:2]:
            raise ValueError("workload input shape changed after the verified first batch")
        use_graph = (
            self.training
            and torch.is_grad_enabled()
            and inputs.is_cuda
            and inputs.shape[0] == self.capture_batch_size
        )
        if use_graph:
            if self._graphed_training is None:
                self._graphed_training = self._capture(inputs, elapsed)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.get_autocast_dtype("cuda"),
                enabled=torch.is_autocast_enabled("cuda"),
                cache_enabled=False,
            ):
                values = self._graphed_training(inputs, elapsed)
        else:
            values = self._eager_forward(inputs, elapsed)
        return dict(zip(self.output_keys, values, strict=True))

    def state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Tensor]:
        return self.core.source.state_dict(*args, **kwargs)

    def load_state_dict(
        self, state_dict: Mapping[str, Tensor], strict: bool = True, assign: bool = False
    ):
        return self.core.source.load_state_dict(
            state_dict, strict=strict, assign=assign
        )


def wrap_cuda_graphed_workload_model(
    source: nn.Module,
    *,
    output_keys: Sequence[str],
    capture_batch_size: int = 32,
) -> nn.Module:
    """Wrap an existing CfC workload model without replacing its parameters."""

    source_parameters = list(source.parameters())
    source_state_keys = list(source.state_dict())
    if isinstance(source, CfCW):
        wrapper: nn.Module = CUDAGraphedCfCWRecovery(
            source, capture_batch_size=capture_batch_size
        )
    else:
        wrapper = CUDAGraphedMappingWorkloadModel(
            source,
            output_keys=output_keys,
            capture_batch_size=capture_batch_size,
        )
    if source_state_keys != list(wrapper.state_dict()):
        raise RuntimeError("CUDA graph changed workload checkpoint state keys")
    if source_parameters != list(wrapper.parameters()):
        raise RuntimeError("CUDA graph changed workload optimizer parameter order")
    return wrapper


def build_cuda_graphed_cfc_w_recovery(
    *,
    input_size: int,
    hidden_size: int,
    horizon: int,
    cohorts: int = 4,
    capture_batch_size: int = 32,
) -> CUDAGraphedCfCWRecovery:
    source = CfCW(
        input_size=input_size,
        hidden_size=hidden_size,
        horizon=horizon,
        cohorts=cohorts,
    )
    wrapper = CUDAGraphedCfCWRecovery(
        source, capture_batch_size=capture_batch_size
    )
    if list(source.state_dict()) != list(wrapper.state_dict()):
        raise RuntimeError("CUDA-graph CfC-W changed checkpoint state-dict key ordering")
    if list(source.parameters()) != list(wrapper.parameters()):
        raise RuntimeError("CUDA-graph CfC-W changed optimizer parameter ordering")
    return wrapper
