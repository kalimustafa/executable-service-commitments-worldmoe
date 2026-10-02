from typing import Any, Dict, Mapping

import torch
from torch import Tensor, nn

from dcvpp.models.workload.cfc_w import CfCW


class _DefaultCfCCell(nn.Module):
    """TorchScript-compatible default-mode ncps CfC cell.

    This is algebraically identical to the installed ncps default branch.  It
    deliberately omits the unreachable pure/no-gate branches that prevent the
    third-party cell from scripting.
    """

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
    state_size: int

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.rnn_cell = _DefaultCfCCell(source.rnn_cell)
        self.state_size = int(source.state_size)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs


class _ScriptableCfCWCore(nn.Module):
    horizon: int
    cohorts: int

    def __init__(self, source: CfCW) -> None:
        super().__init__()
        self.encoder = _DefaultCfCEncoder(source.encoder)
        self.count_head = source.count_head
        self.lognormal_head = source.lognormal_head
        self.gamma_head = source.gamma_head
        self.horizon = int(source.horizon)
        self.cohorts = int(source.cohorts)

    def forward(self, inputs: Tensor, elapsed: Tensor) -> Dict[str, Tensor]:
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
        return {
            "count_mean": torch.nn.functional.softplus(count[..., 0]) + 1e-4,
            "count_dispersion": torch.nn.functional.softplus(count[..., 1]) + 1e-4,
            "volume_log_mu": lognormal[..., 0],
            "volume_log_sigma": torch.nn.functional.softplus(lognormal[..., 1]) + 1e-4,
            "volume_gamma_shape": torch.nn.functional.softplus(gamma[..., 0]) + 1e-4,
            "volume_gamma_rate": torch.nn.functional.softplus(gamma[..., 1]) + 1e-4,
        }


class ScriptedCfCWRecovery(nn.Module):
    """Checkpoint-compatible wrapper around a scripted CfC recurrence."""

    def __init__(self, source: CfCW) -> None:
        super().__init__()
        self.scripted = torch.jit.script(_ScriptableCfCWCore(source))
        self._architecture_contract = dict(source.architecture_contract)

    @property
    def architecture_contract(self) -> dict[str, int | str]:
        return dict(self._architecture_contract)

    def forward(self, inputs: Tensor, elapsed: Tensor) -> Dict[str, Tensor]:
        return self.scripted(inputs, elapsed)

    def state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Tensor]:
        return self.scripted.state_dict(*args, **kwargs)

    def load_state_dict(
        self, state_dict: Mapping[str, Tensor], strict: bool = True, assign: bool = False
    ):
        return self.scripted.load_state_dict(state_dict, strict=strict, assign=assign)


def build_scripted_cfc_w_recovery(
    *, input_size: int, hidden_size: int, horizon: int, cohorts: int = 4
) -> ScriptedCfCWRecovery:
    source = CfCW(
        input_size=input_size,
        hidden_size=hidden_size,
        horizon=horizon,
        cohorts=cohorts,
    )
    wrapper = ScriptedCfCWRecovery(source)
    if list(source.state_dict()) != list(wrapper.state_dict()):
        raise RuntimeError("scripted CfC-W changed checkpoint state-dict key ordering")
    return wrapper
