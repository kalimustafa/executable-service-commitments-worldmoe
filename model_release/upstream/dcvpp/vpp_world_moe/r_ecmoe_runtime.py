"""Strict, solver-free deployment runtime for one R-ECMoE forward pass."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import wraps
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor

from .config import WorldMoEConfig
from .model import VPPWorldMoE, load_plannerless_released_w1_state
from .packets import WorldActionPacket, WorldInputPacket
from .r_ecmoe import RECMoE, RECMoEConfig, RECMoEOutput, RiskConditionPacket


CHECKPOINT_SCHEMA = "dcvpp.vpp-world-moe.checkpoint.v1"
RUNTIME_SCHEMA = "dcvpp.r-ecmoe.inference-runtime.v1"
BENCHMARK_SCHEMA = "dcvpp.r-ecmoe.inference-benchmark.v1"


class RECMoEInferenceRuntimeError(RuntimeError):
    """The runtime cannot preserve its strict released-W1 contract."""


@dataclass(frozen=True)
class RECMoEInferenceResult:
    """Compact deployable result with internal W1 expert state removed."""

    actions: WorldActionPacket
    day_ahead_commitment: Tensor
    target_soc: Tensor
    cost_quantiles: Tensor
    expected_cost: Tensor
    upper_tail_cost: Tensor
    sla_risk: Tensor
    constraint_risk: Tensor
    grid_risk: Tensor
    deadline_expired_total: Tensor
    constraint_residual_max: Tensor
    constraint_residual_max_by_name: Mapping[str, Tensor]
    source_state_retained_internal_only: bool = True


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device=device)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                item.name: _to_device(getattr(value, item.name), device)
                for item in fields(value)
            },
        )
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("latency sample cannot be empty")
    ordered = sorted(float(value) for value in values)
    position = float(probability) * float(len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - float(lower)
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("latency sample cannot be empty")
    return {
        "mean": float(sum(values) / len(values)),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _validate_checkpoint_mechanism(
    payload: Mapping[str, Any], config: RECMoEConfig
) -> None:
    """Prevent a trained ablation from silently loading as the full model.

    Historical checkpoints predate this field and represent ``full``. New
    callers must supply the saved model configuration, rather than selecting a
    different inference-only intervention while loading the same weights.
    """

    variants = []
    if "mechanism_variant" in payload:
        variants.append(payload["mechanism_variant"])
    model_config = payload.get("model_config")
    if isinstance(model_config, Mapping):
        variants.append(model_config.get("mechanism_variant", "full"))
    resolved = payload.get("resolved_config")
    if isinstance(resolved, Mapping) and isinstance(resolved.get("model"), Mapping):
        variants.append(resolved["model"].get("mechanism_variant", "full"))
    if not variants:
        variants.append("full")
    if any(value != config.mechanism_variant for value in variants):
        raise RECMoEInferenceRuntimeError(
            "R checkpoint mechanism_variant differs from the requested model config"
        )


class _ComponentProfiler:
    """Record nested CPU/CUDA sections inside the same measured forward.

    CUDA events are deliberately resolved only after the enclosing inference
    event has synchronized.  Consequently the profiler adds no per-component
    host synchronization and does not add diagnostic model forwards.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.records: dict[str, list[tuple[Any, Any]]] = {}

    def begin(self) -> Any:
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def end(self, name: str, started: Any) -> None:
        if self.device.type == "cuda":
            finished = torch.cuda.Event(enable_timing=True)
            finished.record()
        else:
            finished = time.perf_counter()
        self.records.setdefault(name, []).append((started, finished))

    def values(self, name: str) -> list[float]:
        pairs = self.records.get(name, ())
        if self.device.type == "cuda":
            return [float(start.elapsed_time(finish)) for start, finish in pairs]
        return [float((finish - start) * 1_000.0) for start, finish in pairs]


class _ExecutedParameterTracker:
    """Trace parameters reached by one eager diagnostic forward.

    Forward hooks mark only parameters owned directly by modules whose
    ``forward`` method actually executes.  The W1 dynamics response is invoked
    through ``parallel_action_response`` rather than ``Module.__call__``; its
    three directly consumed parameters are therefore marked by a temporary
    method wrapper.  Child modules used by that method remain covered by the
    ordinary forward hooks.

    The tracker is intentionally installed only around a separate diagnostic
    forward after latency and peak-memory measurement have completed.
    """

    _DYNAMICS_DIRECT_PARAMETERS = (
        "regime_embedding",
        "regime_gate",
        "residual_gate",
    )

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self._parameter_name_by_id = {
            id(parameter): name for name, parameter in model.named_parameters()
        }
        self._executed_parameter_ids: set[int] = set()
        self._handles: list[Any] = []
        self._method_originals: list[tuple[torch.nn.Module, str, bool, Any]] = []

    def _mark_parameter(self, parameter: Tensor | None) -> None:
        if parameter is not None and id(parameter) in self._parameter_name_by_id:
            self._executed_parameter_ids.add(id(parameter))

    def _mark_module_parameters(self, module: torch.nn.Module) -> None:
        for parameter in module.parameters(recurse=False):
            self._mark_parameter(parameter)

    def _module_pre_hook(self, module: torch.nn.Module, _args: Any) -> None:
        self._mark_module_parameters(module)

    def _wrap_direct_method(
        self,
        module: torch.nn.Module,
        method_name: str,
        parameter_names: tuple[str, ...],
    ) -> None:
        original = getattr(module, method_name, None)
        if not callable(original):
            return
        had_instance_value = method_name in module.__dict__
        prior_instance_value = module.__dict__.get(method_name)

        @wraps(original)
        def tracked(*args: Any, **kwargs: Any) -> Any:
            for name in parameter_names:
                parameter = getattr(module, name, None)
                if isinstance(parameter, Tensor):
                    self._mark_parameter(parameter)
            return original(*args, **kwargs)

        setattr(module, method_name, tracked)
        self._method_originals.append(
            (module, method_name, had_instance_value, prior_instance_value)
        )

    def install(self) -> None:
        if self._handles or self._method_originals:
            raise RuntimeError("executed-parameter tracker is already installed")
        self._handles = [
            module.register_forward_pre_hook(self._module_pre_hook)
            for module in self.model.modules()
        ]
        world_model = getattr(self.model, "world_model", None)
        dynamics = getattr(world_model, "dynamics", None)
        if isinstance(dynamics, torch.nn.Module):
            self._wrap_direct_method(
                dynamics,
                "parallel_action_response",
                self._DYNAMICS_DIRECT_PARAMETERS,
            )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for module, method_name, had_instance_value, prior_instance_value in reversed(
            self._method_originals
        ):
            if had_instance_value:
                setattr(module, method_name, prior_instance_value)
            else:
                delattr(module, method_name)
        self._method_originals.clear()

    @property
    def executed_parameter_names(self) -> frozenset[str]:
        return frozenset(
            self._parameter_name_by_id[parameter_id]
            for parameter_id in self._executed_parameter_ids
        )


class RECMoEInferenceRuntime:
    """Load one released W1 and execute R-ECMoE without search or a solver."""

    def __init__(
        self,
        model: RECMoE,
        *,
        world_config: WorldMoEConfig,
        rec_config: RECMoEConfig,
        checkpoint_path: Path,
        checkpoint_load: Mapping[str, Any],
        device: torch.device,
    ) -> None:
        if not bool(getattr(model, "external_solver_free", False)) or not bool(
            getattr(model, "candidate_search_free", False)
        ):
            raise RECMoEInferenceRuntimeError(
                "R-ECMoE runtime prohibits external solvers and candidate search"
            )
        self.model = model
        self.world_config = world_config
        self.rec_config = rec_config
        self.checkpoint_path = checkpoint_path
        self.checkpoint_load = dict(checkpoint_load)
        self.device = device
        self.autocast_dtype = (
            torch.bfloat16
            if device.type in {"cuda", "cpu"}
            else torch.float16
        )
        self._compiled_forward: Any | None = None
        self._compile_fallback_on_error = False
        self.compile_status: dict[str, Any] = {
            "requested": "off",
            "active": False,
            "backend": None,
            "mode": None,
            "fallback": False,
            "reason": "disabled_by_config",
        }

    def configure_torch_compile(
        self,
        requested: str | bool | None,
        *,
        backend: str = "inductor",
        mode: str = "reduce-overhead",
    ) -> Mapping[str, Any]:
        """Enable an opt-in fixed-shape compiled forward with eager fallback.

        The default remains eager because packed expert dispatch and
        conditional tail execution can change the executed graph.  ``auto``
        is therefore accepted only on CUDA and always falls back to the exact
        eager forward if compilation or its first execution fails.
        """

        normalized = str(requested).strip().lower() if requested is not None else "off"
        if requested is True:
            normalized = "auto"
        elif requested is False:
            normalized = "off"
        if normalized in {"", "0", "false", "none", "off", "disabled"}:
            self._compiled_forward = None
            self._compile_fallback_on_error = False
            self.compile_status = {
                "requested": normalized or "off",
                "active": False,
                "backend": None,
                "mode": None,
                "fallback": False,
                "reason": "disabled_by_config",
            }
            return dict(self.compile_status)
        if normalized not in {"auto", "on", "true"}:
            raise ValueError("torch_compile must be off or auto")
        if self.device.type != "cuda":
            self._compiled_forward = None
            self._compile_fallback_on_error = False
            self.compile_status = {
                "requested": normalized,
                "active": False,
                "backend": backend,
                "mode": mode,
                "fallback": True,
                "reason": "compile_auto_is_cuda_only",
            }
            return dict(self.compile_status)
        compiler = getattr(torch, "compile", None)
        if compiler is None:
            self._compiled_forward = None
            self._compile_fallback_on_error = False
            self.compile_status = {
                "requested": normalized,
                "active": False,
                "backend": backend,
                "mode": mode,
                "fallback": True,
                "reason": "torch_compile_unavailable",
            }
            return dict(self.compile_status)
        try:
            self._compiled_forward = compiler(
                self.model.forward,
                backend=str(backend),
                mode=str(mode),
                dynamic=False,
                fullgraph=False,
            )
        except Exception as exc:  # pragma: no cover - backend/platform dependent
            self._compiled_forward = None
            self._compile_fallback_on_error = False
            self.compile_status = {
                "requested": normalized,
                "active": False,
                "backend": backend,
                "mode": mode,
                "fallback": True,
                "reason": f"compile_setup_failed:{type(exc).__name__}",
            }
            return dict(self.compile_status)
        self._compile_fallback_on_error = True
        self.compile_status = {
            "requested": normalized,
            "active": True,
            "backend": str(backend),
            "mode": str(mode),
            "fallback": False,
            "reason": "pending_first_execution",
        }
        return dict(self.compile_status)

    @staticmethod
    def _world_config(value: WorldMoEConfig | Mapping[str, Any]) -> WorldMoEConfig:
        if isinstance(value, WorldMoEConfig):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("world_config must be WorldMoEConfig or a mapping")
        model = value.get("model", value)
        if not isinstance(model, Mapping):
            raise TypeError("world_config.model must be a mapping")
        return WorldMoEConfig.from_mapping(model)

    @staticmethod
    def _rec_config(value: RECMoEConfig | Mapping[str, Any]) -> RECMoEConfig:
        if isinstance(value, RECMoEConfig):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("rec_config must be RECMoEConfig or a mapping")
        model = value.get("model", value)
        if not isinstance(model, Mapping):
            raise TypeError("rec_config.model must be a mapping")
        return RECMoEConfig.from_mapping(dict(model))

    @classmethod
    def from_w1_checkpoint(
        cls,
        world_config: WorldMoEConfig | Mapping[str, Any],
        rec_config: RECMoEConfig | Mapping[str, Any],
        checkpoint_path: str | Path,
        device: str | torch.device,
        *,
        r_checkpoint_path: str | Path | None = None,
    ) -> "RECMoEInferenceRuntime":
        """Construct directly from one exact released-W1 ``model_state``.

        No optimizer/RNG state is loaded and no file digest is computed or
        persisted.  A mismatched schema, phase, locked-test flag, or model key
        fails before inference is exposed.
        """

        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RECMoEInferenceRuntimeError("CUDA runtime requested but unavailable")
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise RECMoEInferenceRuntimeError("W1 checkpoint payload must be a mapping")
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise RECMoEInferenceRuntimeError("W1 checkpoint schema mismatch")
        if str(payload.get("phase", "")).lower() != "w1":
            raise RECMoEInferenceRuntimeError("runtime accepts only phase=w1")
        if payload.get("locked_2025_opened") is not False:
            raise RECMoEInferenceRuntimeError(
                "runtime requires locked_2025_opened=false"
            )
        model_state = payload.get("model_state")
        if not isinstance(model_state, Mapping):
            raise RECMoEInferenceRuntimeError("W1 checkpoint omitted model_state")
        resume_state = payload.get("resume_state", {})
        if not isinstance(resume_state, Mapping):
            resume_state = {}

        resolved_world = cls._world_config(world_config)
        resolved_rec = cls._rec_config(rec_config)
        world_model = VPPWorldMoE(
            resolved_world,
            include_legacy_planner=False,
        )
        try:
            load_plannerless_released_w1_state(world_model, model_state)
        except RuntimeError as exc:
            raise RECMoEInferenceRuntimeError(
                "strict W1 model_state load failed"
            ) from exc
        model = RECMoE(world_model, resolved_rec).to(resolved_device)
        resolved_r_checkpoint: Path | None = None
        if r_checkpoint_path is not None:
            resolved_r_checkpoint = Path(r_checkpoint_path).expanduser().resolve()
            if not resolved_r_checkpoint.is_file():
                raise FileNotFoundError(resolved_r_checkpoint)
            r_payload = torch.load(
                resolved_r_checkpoint, map_location=resolved_device, weights_only=False
            )
            if (
                not isinstance(r_payload, Mapping)
                or r_payload.get("schema") != "dcvpp.r_ecmoe.runner.v1"
            ):
                raise RECMoEInferenceRuntimeError("R checkpoint schema mismatch")
            _validate_checkpoint_mechanism(r_payload, resolved_rec)
            r_state = r_payload.get("r_state")
            if not isinstance(r_state, Mapping) or any(
                str(name).startswith("world_model.") for name in r_state
            ):
                raise RECMoEInferenceRuntimeError("R checkpoint state is malformed")
            incompatible_r = model.load_state_dict(r_state, strict=False)
            if incompatible_r.unexpected_keys or any(
                not name.startswith("world_model.")
                for name in incompatible_r.missing_keys
            ):
                raise RECMoEInferenceRuntimeError("R checkpoint does not match model")
            if r_payload.get("tuned_dynamics_state") not in (None, {}):
                raise RECMoEInferenceRuntimeError(
                    "formal R runtime requires the shared W1 dynamics to remain frozen"
                )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        checkpoint_load = {
            "path": str(path),
            "schema": str(payload["schema"]),
            "phase": "w1",
            "epoch": int(resume_state.get("epoch", payload.get("epoch", -1))),
            "update": int(
                resume_state.get(
                    "global_update",
                    resume_state.get(
                        "update",
                        payload.get("global_update", payload.get("update", -1)),
                    ),
                )
            ),
            "locked_2025_opened": False,
            "strict": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "optimizer_loaded": False,
            "rng_loaded": False,
            "hash_computed": False,
            "hash_saved": False,
            "r_checkpoint_path": (
                None if resolved_r_checkpoint is None else str(resolved_r_checkpoint)
            ),
        }
        if resolved_r_checkpoint is not None:
            checkpoint_load["mechanism_variant"] = resolved_rec.mechanism_variant
        return cls(
            model,
            world_config=resolved_world,
            rec_config=resolved_rec,
            checkpoint_path=path,
            checkpoint_load=checkpoint_load,
            device=resolved_device,
        )

    def _prepare_packet(self, packet: WorldInputPacket) -> WorldInputPacket:
        if not isinstance(packet, WorldInputPacket):
            raise TypeError("infer requires one WorldInputPacket")
        moved = _to_device(packet, self.device)
        if not isinstance(moved, WorldInputPacket):
            raise TypeError("device transfer changed the WorldInputPacket type")
        return moved.validate(self.world_config)

    def _prepare_risk(
        self, risk: RiskConditionPacket, *, batch: int, horizon: int
    ) -> RiskConditionPacket:
        if not isinstance(risk, RiskConditionPacket):
            raise TypeError("infer requires one RiskConditionPacket")
        moved = _to_device(risk, self.device)
        if not isinstance(moved, RiskConditionPacket):
            raise TypeError("device transfer changed the RiskConditionPacket type")
        return moved.validate(batch, horizon)

    def _prepare_anchors(
        self, anchors: Mapping[str, Any]
    ) -> tuple[Tensor, Tensor | None]:
        if not isinstance(anchors, Mapping):
            raise TypeError("anchors must be a mapping")
        nested_anchor = anchors.get("forecast_anchors")
        nested_cap = anchors.get("forecast_residual_caps")
        direct_da = anchors.get("da_anchor", anchors.get("da"))
        if isinstance(nested_anchor, Mapping):
            if direct_da is not None:
                raise ValueError("DA anchor was supplied twice")
            direct_da = nested_anchor.get("da")
        direct_cap = anchors.get("da_residual_cap")
        if isinstance(nested_cap, Mapping):
            if direct_cap is not None:
                raise ValueError("DA residual cap was supplied twice")
            direct_cap = nested_cap.get("da")
        if not isinstance(direct_da, Tensor):
            raise ValueError("anchors must contain a Tensor DA anchor")
        if direct_cap is not None and not isinstance(direct_cap, Tensor):
            raise ValueError("DA residual cap must be a Tensor when supplied")
        da = direct_da.to(device=self.device, dtype=torch.float32)
        cap = (
            None
            if direct_cap is None
            else direct_cap.to(device=self.device, dtype=torch.float32)
        )
        if not bool(torch.isfinite(da).all()) or (
            cap is not None and not bool(torch.isfinite(cap).all())
        ):
            raise ValueError("anchors must be finite")
        return da, cap

    def _autocast(self) -> torch.autocast:
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.device.type in {"cuda", "cpu"},
        )

    def _infer_prepared(
        self,
        packet: WorldInputPacket,
        risk: RiskConditionPacket,
        da_anchor: Tensor,
        da_residual_cap: Tensor | None,
        *,
        force_eager: bool = False,
    ) -> RECMoEOutput:
        self.model.eval()
        with torch.inference_mode(), self._autocast():
            call = self.model if force_eager else (self._compiled_forward or self.model)
            calls_before = int(self.model.w1_backbone_calls)
            try:
                output = call(
                    packet,
                    risk,
                    da_anchor=da_anchor,
                    da_residual_cap=da_residual_cap,
                    return_diagnostics=False,
                )
                if self._compiled_forward is not None and not force_eager:
                    self.compile_status["reason"] = "compiled_execution_succeeded"
            except Exception as exc:
                if (
                    force_eager
                    or self._compiled_forward is None
                    or not self._compile_fallback_on_error
                ):
                    raise
                # Compilation can fail after Python-side counters have run but
                # before a result exists.  Restore that diagnostic counter so
                # the eager retry still satisfies exactly one W1 pass.
                self.model.w1_backbone_calls = calls_before
                self._compiled_forward = None
                self._compile_fallback_on_error = False
                self.compile_status.update(
                    {
                        "active": False,
                        "fallback": True,
                        "reason": f"compiled_execution_failed:{type(exc).__name__}",
                    }
                )
                output = self.model(
                    packet,
                    risk,
                    da_anchor=da_anchor,
                    da_residual_cap=da_residual_cap,
                    return_diagnostics=False,
                )
        if not isinstance(output, RECMoEOutput):
            raise RECMoEInferenceRuntimeError("R-ECMoE returned an invalid output")
        if output.diagnostics:
            raise RECMoEInferenceRuntimeError(
                "default inference unexpectedly materialized diagnostics"
            )
        return output

    def infer(
        self,
        packet: WorldInputPacket,
        risk: RiskConditionPacket,
        anchors: Mapping[str, Any],
    ) -> RECMoEInferenceResult:
        """Execute exactly one direct W1-backbone plus R-ECMoE decision pass."""

        prepared_packet = self._prepare_packet(packet)
        batch, horizon = prepared_packet.fast_values.shape[:2]
        prepared_risk = self._prepare_risk(
            risk, batch=int(batch), horizon=int(horizon)
        )
        da_anchor, da_residual_cap = self._prepare_anchors(anchors)
        before = int(self.model.w1_backbone_calls)
        output = self._infer_prepared(
            prepared_packet,
            prepared_risk,
            da_anchor,
            da_residual_cap,
        )
        if int(self.model.w1_backbone_calls) - before != 1:
            raise RECMoEInferenceRuntimeError(
                "one inference must execute the W1 backbone exactly once"
            )
        return self._compact_result(output)

    @staticmethod
    def _batch_maximum(value: Tensor, batch: int) -> Tensor:
        numeric = value.detach().float().abs()
        if numeric.ndim == 0:
            return numeric.expand(batch)
        if numeric.shape[0] != batch:
            return numeric.amax().expand(batch)
        return numeric.reshape(batch, -1).amax(dim=-1)

    @classmethod
    def _compact_result(cls, output: RECMoEOutput) -> RECMoEInferenceResult:
        batch = int(output.expected_cost.shape[0])
        by_name = {
            str(name): cls._batch_maximum(value, batch)
            for name, value in output.constraint_residuals.items()
            if isinstance(value, Tensor)
        }
        residual_max = (
            torch.stack(tuple(by_name.values()), dim=-1).amax(dim=-1)
            if by_name
            else output.expected_cost.detach().new_zeros((batch,), dtype=torch.float32)
        )
        deadline = output.deadline_expired_work.detach().float()
        deadline_total = (
            deadline.reshape(batch, -1).sum(dim=-1)
            if deadline.ndim > 0 and deadline.shape[0] == batch
            else deadline.sum().expand(batch)
        )
        return RECMoEInferenceResult(
            actions=output.actions,
            day_ahead_commitment=output.day_ahead_commitment,
            target_soc=output.target_soc,
            cost_quantiles=output.cost_quantiles,
            expected_cost=output.expected_cost,
            upper_tail_cost=output.upper_tail_cost,
            sla_risk=output.sla_risk,
            constraint_risk=output.constraint_risk,
            grid_risk=output.grid_risk,
            deadline_expired_total=deadline_total,
            constraint_residual_max=residual_max,
            constraint_residual_max_by_name=by_name,
        )

    def _install_component_profiler(
        self, profiler: _ComponentProfiler
    ) -> tuple[list[tuple[str, bool, Any]], list[Any]]:
        """Instrument the actual forward without adding another model call."""

        originals: list[tuple[str, bool, Any]] = []
        handles: list[Any] = []

        def wrap_method(method_name: str, section: str) -> None:
            original = getattr(self.model, method_name, None)
            if not callable(original):
                return
            had_instance_value = method_name in self.model.__dict__
            prior_instance_value = self.model.__dict__.get(method_name)

            @wraps(original)
            def timed(*args: Any, **kwargs: Any) -> Any:
                started = profiler.begin()
                try:
                    return original(*args, **kwargs)
                finally:
                    profiler.end(section, started)

            setattr(self.model, method_name, timed)
            originals.append(
                (method_name, had_instance_value, prior_instance_value)
            )

        wrap_method("encode_expert_context", "w1_packet")
        wrap_method("decide", "r_decision_total")
        wrap_method("apply_action_response", "action_response")

        assembler = getattr(self.model, "assembler", None)
        if isinstance(assembler, torch.nn.Module):
            active_tokens: list[Any] = []

            def assembler_pre(_module: torch.nn.Module, _args: Any) -> None:
                active_tokens.append(profiler.begin())

            def assembler_post(
                _module: torch.nn.Module, _args: Any, _output: Any
            ) -> None:
                profiler.end("constraint_assembler", active_tokens.pop())

            handles.extend(
                (
                    assembler.register_forward_pre_hook(assembler_pre),
                    assembler.register_forward_hook(assembler_post),
                )
            )
        return originals, handles

    def _remove_component_profiler(
        self,
        originals: list[tuple[str, bool, Any]],
        handles: list[Any],
    ) -> None:
        for handle in handles:
            handle.remove()
        for method_name, had_instance_value, prior_instance_value in originals:
            if had_instance_value:
                setattr(self.model, method_name, prior_instance_value)
            else:
                delattr(self.model, method_name)

    @staticmethod
    def _component_latency(
        profiler: _ComponentProfiler, full_latencies: list[float]
    ) -> dict[str, Any]:
        w1 = profiler.values("w1_packet")
        decision = profiler.values("r_decision_total")
        response = profiler.values("action_response")
        assembler = profiler.values("constraint_assembler")
        complete = all(
            len(values) == len(full_latencies)
            for values in (w1, decision, response, assembler)
        )
        if not complete:
            return {
                "complete": False,
                "measured_in_same_forward": True,
                "reason": "model_did_not_expose_all_profiled_stage_boundaries",
                "measured_counts": {
                    "full": len(full_latencies),
                    "w1_packet": len(w1),
                    "r_decision_total": len(decision),
                    "action_response": len(response),
                    "constraint_assembler": len(assembler),
                },
            }
        route_adapter_heads = [
            max(0.0, total - constraint)
            for total, constraint in zip(decision, assembler, strict=True)
        ]
        post_decision = [
            max(0.0, full - packet - total - action)
            for full, packet, total, action in zip(
                full_latencies, w1, decision, response, strict=True
            )
        ]
        r_overhead = [
            max(0.0, full - packet)
            for full, packet in zip(full_latencies, w1, strict=True)
        ]
        w1_mean = sum(w1) / len(w1)
        overhead_mean = sum(r_overhead) / len(r_overhead)
        return {
            "complete": True,
            "measured_in_same_forward": True,
            "host_synchronization_per_component": False,
            "w1_packet": _latency_summary(w1),
            "r_route_adapter_coordinator_and_heads": _latency_summary(
                route_adapter_heads
            ),
            "action_response": _latency_summary(response),
            "constraint_assembler": _latency_summary(assembler),
            "post_decision_risk_value_and_cost": _latency_summary(post_decision),
            "r_overhead_relative_to_w1": {
                **_latency_summary(r_overhead),
                "mean_ratio": float(overhead_mean / max(w1_mean, 1.0e-12)),
                "mean_percent": float(
                    100.0 * overhead_mean / max(w1_mean, 1.0e-12)
                ),
            },
        }

    def _execution_counts(
        self,
        output: RECMoEOutput,
        *,
        executed_parameter_names: frozenset[str],
    ) -> dict[str, Any]:
        named_parameters = tuple(self.model.named_parameters())
        parameter_by_name = dict(named_parameters)
        executed_names = {
            name for name in executed_parameter_names if name in parameter_by_name
        }
        if not executed_names:
            raise RECMoEInferenceRuntimeError(
                "diagnostic forward traced no executed model parameters"
            )
        total = sum(parameter.numel() for _, parameter in named_parameters)
        world_total = sum(
            parameter.numel()
            for name, parameter in named_parameters
            if name.startswith("world_model.")
        )
        r_total = total - world_total

        def prefixes_for(sample_index: int | None) -> tuple[list[str], dict[str, list[int]]]:
            inactive_prefixes: list[str] = []
            w1_active: dict[str, list[int]] = {}
            for diagnostic in output.expert_packet.source_state.router_diagnostics:
                block = int(diagnostic.block_index)
                indices = diagnostic.topk_indices.detach()
                if sample_index is not None and indices.ndim > 0:
                    indices = indices[sample_index]
                active = sorted(set(int(item) for item in indices.reshape(-1).cpu()))
                w1_active[str(block)] = active
                for expert in range(self.world_config.general_experts):
                    if expert not in active:
                        inactive_prefixes.append(
                            f"world_model.backbone.blocks.{block}.moe.experts.{expert}."
                        )
                tail_mask = diagnostic.tail_active_mask
                if sample_index is not None and tail_mask.ndim > 0:
                    tail_mask = tail_mask[sample_index]
                if not bool(tail_mask.any()):
                    inactive_prefixes.append(
                        f"world_model.backbone.blocks.{block}.moe.tail_expert."
                    )
            continued_ids = output.selected_expert_ids.detach()
            if sample_index is not None and continued_ids.ndim > 0:
                continued_ids = continued_ids[sample_index]
            continued = set(int(item) for item in continued_ids.reshape(-1).cpu())
            for expert in range(self.rec_config.general_experts):
                if expert not in continued:
                    inactive_prefixes.append(f"adapter_bank.adapters.{expert}.")
            tail_decision = output.tail_decision_active
            if sample_index is not None and tail_decision.ndim > 0:
                tail_decision = tail_decision[sample_index]
            if not bool(tail_decision.any()):
                inactive_prefixes.append("tail_adapter.adapter.")
            return inactive_prefixes, w1_active

        def active_breakdown(prefixes: list[str]) -> tuple[int, int, int]:
            active_names = {
                name
                for name in executed_names
                if not any(name.startswith(prefix) for prefix in prefixes)
            }
            active_world = sum(
                parameter_by_name[name].numel()
                for name in active_names
                if name.startswith("world_model.")
            )
            active_r = sum(
                parameter_by_name[name].numel()
                for name in active_names
                if not name.startswith("world_model.")
            )
            return active_world + active_r, active_world, active_r

        inactive_prefixes, w1_active = prefixes_for(None)
        active, active_world, active_r = active_breakdown(inactive_prefixes)
        batch = int(output.expected_cost.shape[0])
        per_sample = [
            active_breakdown(prefixes_for(index)[0]) for index in range(batch)
        ]
        active_per_sample = [item[0] for item in per_sample]
        world_active_per_sample = [item[1] for item in per_sample]
        r_active_per_sample = [item[2] for item in per_sample]

        continued = sorted(
            set(int(item) for item in output.selected_expert_ids.detach().reshape(-1).cpu())
        )
        tail_decision_active = bool(output.tail_decision_active.any())
        return {
            "active_parameters": int(active),
            "total_parameters": int(total),
            "active_parameter_fraction": float(active / max(1, total)),
            "active_parameter_count_method": (
                "one_untimed_eager_diagnostic_forward_module_trace_with_"
                "per_sample_sparse_route_refinement"
            ),
            "diagnostic_executed_parameter_tensors": int(len(executed_names)),
            "w1_total_parameters": int(world_total),
            "r_total_parameters": int(r_total),
            "w1_active_parameters_batch_union": int(active_world),
            "r_active_parameters_batch_union": int(active_r),
            "active_parameters_per_sample": [int(value) for value in active_per_sample],
            "active_parameters_per_sample_mean": float(
                sum(active_per_sample) / max(1, len(active_per_sample))
            ),
            "active_parameters_per_sample_min": int(min(active_per_sample)),
            "active_parameters_per_sample_max": int(max(active_per_sample)),
            "w1_active_parameters_per_sample": [
                int(value) for value in world_active_per_sample
            ],
            "r_active_parameters_per_sample": [
                int(value) for value in r_active_per_sample
            ],
            "w1_active_general_experts_by_block": w1_active,
            "continued_general_experts": continued,
            "continued_general_expert_count": len(continued),
            "tail_decision_active": tail_decision_active,
            "tail_decision_activation_rate": float(
                output.tail_decision_active.float().mean().detach().cpu()
            ),
            "w1_tail_activation_rate": float(
                torch.stack(
                    tuple(
                        diagnostic.tail_active_mask.float().mean()
                        for diagnostic in output.expert_packet.source_state.router_diagnostics
                    )
                )
                .mean()
                .detach()
                .cpu()
            ),
        }

    def benchmark(
        self,
        packet: WorldInputPacket,
        risk: RiskConditionPacket,
        anchors: Mapping[str, Any],
        *,
        warmup: int = 5,
        repeats: int = 30,
    ) -> dict[str, Any]:
        """Benchmark timed forwards, then add one untimed parameter diagnostic."""

        if int(warmup) < 0 or int(repeats) < 1:
            raise ValueError("benchmark requires warmup >= 0 and repeats >= 1")
        prepared_packet = self._prepare_packet(packet)
        batch, horizon = prepared_packet.fast_values.shape[:2]
        prepared_risk = self._prepare_risk(
            risk, batch=int(batch), horizon=int(horizon)
        )
        da_anchor, da_residual_cap = self._prepare_anchors(anchors)
        calls_before = int(self.model.w1_backbone_calls)
        for _ in range(int(warmup)):
            output = self._infer_prepared(
                prepared_packet,
                prepared_risk,
                da_anchor,
                da_residual_cap,
            )
            del output
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            base_allocated = torch.cuda.memory_allocated(self.device)
            base_reserved = torch.cuda.memory_reserved(self.device)
        else:
            base_allocated = 0
            base_reserved = 0
        latencies: list[float] = []
        last_output: RECMoEOutput | None = None
        profiler = _ComponentProfiler(self.device)
        originals: list[tuple[str, bool, Any]] = []
        handles: list[Any] = []
        component_profile_reason: str | None = None
        if self._compiled_forward is None:
            originals, handles = self._install_component_profiler(profiler)
        else:
            component_profile_reason = (
                "compiled_graph_bypasses_python_stage_boundaries; "
                "rerun with torch_compile=off for component attribution"
            )
        try:
            for _ in range(int(repeats)):
                if self.device.type == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    finish = torch.cuda.Event(enable_timing=True)
                    start.record()
                    last_output = self._infer_prepared(
                        prepared_packet,
                        prepared_risk,
                        da_anchor,
                        da_residual_cap,
                    )
                    finish.record()
                    finish.synchronize()
                    latencies.append(float(start.elapsed_time(finish)))
                else:
                    started = time.perf_counter()
                    last_output = self._infer_prepared(
                        prepared_packet,
                        prepared_risk,
                        da_anchor,
                        da_residual_cap,
                    )
                    latencies.append((time.perf_counter() - started) * 1000.0)
        finally:
            self._remove_component_profiler(originals, handles)
        if last_output is None:
            raise RECMoEInferenceRuntimeError("benchmark produced no inference output")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_allocated = torch.cuda.max_memory_allocated(self.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.device)
            timing_method = "cuda_event"
        else:
            peak_allocated = 0
            peak_reserved = 0
            timing_method = "cpu_perf_counter"
        timed_calls_after = int(self.model.w1_backbone_calls)
        expected_calls = int(warmup) + int(repeats)
        if timed_calls_after - calls_before != expected_calls:
            raise RECMoEInferenceRuntimeError(
                "benchmark repeated or omitted a W1 backbone execution"
            )
        del last_output
        # Trace actual parameter execution in one separate eager forward.  It is
        # deliberately run after latency and peak-memory values are finalized,
        # so parameter accounting cannot contaminate either benchmark.
        tracker = _ExecutedParameterTracker(self.model)
        diagnostic_before = int(self.model.w1_backbone_calls)
        try:
            tracker.install()
            diagnostic_output = self._infer_prepared(
                prepared_packet,
                prepared_risk,
                da_anchor,
                da_residual_cap,
                force_eager=True,
            )
        finally:
            tracker.remove()
        calls_after = int(self.model.w1_backbone_calls)
        if calls_after - diagnostic_before != 1:
            raise RECMoEInferenceRuntimeError(
                "parameter diagnostic must execute the W1 backbone exactly once"
            )
        execution = self._execution_counts(
            diagnostic_output,
            executed_parameter_names=tracker.executed_parameter_names,
        )
        del diagnostic_output
        mean_latency = sum(latencies) / float(len(latencies))
        component_latency = self._component_latency(profiler, latencies)
        if component_profile_reason is not None:
            component_latency = {
                "complete": False,
                "measured_in_same_forward": False,
                "reason": component_profile_reason,
            }
        return {
            "schema": BENCHMARK_SCHEMA,
            "runtime_schema": RUNTIME_SCHEMA,
            "device": str(self.device),
            "device_name": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else "cpu"
            ),
            "autocast_dtype": str(self.autocast_dtype).replace("torch.", ""),
            "batch_size": int(batch),
            "horizon": int(horizon),
            "warmup": int(warmup),
            "repeats": int(repeats),
            "timing_method": timing_method,
            "full_inference_latency_ms": _latency_summary(latencies),
            "latency_scope": (
                "deployment_batch1" if int(batch) == 1 else "whole_batch"
            ),
            "throughput_windows_per_second": float(
                int(batch) * 1_000.0 / max(mean_latency, 1.0e-12)
            ),
            "throughput_scope": "complete_rolling_windows",
            "component_latency_ms": component_latency,
            "peak_memory_mib": {
                "base_allocated": float(base_allocated / 1024**2),
                "peak_allocated": float(peak_allocated / 1024**2),
                "incremental_peak_allocated": float(
                    (peak_allocated - base_allocated) / 1024**2
                ),
                "base_reserved": float(base_reserved / 1024**2),
                "peak_reserved": float(peak_reserved / 1024**2),
            },
            "parameters": {
                key: execution[key]
                for key in (
                    "active_parameters",
                    "total_parameters",
                    "active_parameter_fraction",
                    "active_parameter_count_method",
                    "diagnostic_executed_parameter_tensors",
                    "w1_total_parameters",
                    "r_total_parameters",
                    "w1_active_parameters_batch_union",
                    "r_active_parameters_batch_union",
                    "active_parameters_per_sample",
                    "active_parameters_per_sample_mean",
                    "active_parameters_per_sample_min",
                    "active_parameters_per_sample_max",
                    "w1_active_parameters_per_sample",
                    "r_active_parameters_per_sample",
                )
            },
            "experts": {
                key: execution[key]
                for key in (
                    "w1_active_general_experts_by_block",
                    "continued_general_experts",
                    "continued_general_expert_count",
                    "tail_decision_active",
                )
            },
            "sparsity_contract": {
                "w1_general_experts": int(self.world_config.general_experts),
                "w1_active_experts_per_token": int(
                    self.world_config.regular_top_k
                ),
                "r_active_experts_per_token": int(self.rec_config.active_top_k),
                "r_unique_experts_used_somewhere_in_window": int(
                    execution["continued_general_expert_count"]
                ),
                "active_parameter_count_semantics": (
                    "traced_union_executed_anywhere_in_the_batch_window_with_"
                    "per_sample_sparse_route_refinement"
                ),
            },
            "tail_activation_rate": {
                "w1": execution["w1_tail_activation_rate"],
                "decision": execution["tail_decision_activation_rate"],
            },
            "w1_backbone_calls": {
                "before": calls_before,
                "after": calls_after,
                "warmup": int(warmup),
                "measured": int(repeats),
                "diagnostic": 1,
                "timed_delta": timed_calls_after - calls_before,
                "delta": calls_after - calls_before,
                "per_full_inference": 1,
            },
            "checkpoint_load": dict(self.checkpoint_load),
            "torch_compile": dict(self.compile_status),
            "diagnostic_forwards_added": 1,
            "diagnostic_forward_included_in_latency": False,
            "diagnostic_forward_included_in_peak_memory": False,
            "candidate_search": False,
            "external_solver": False,
        }


__all__ = [
    "BENCHMARK_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "RECMoEInferenceResult",
    "RECMoEInferenceRuntime",
    "RECMoEInferenceRuntimeError",
    "RUNTIME_SCHEMA",
]
