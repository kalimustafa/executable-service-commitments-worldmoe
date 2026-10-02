import time
from typing import Dict

import torch
from torch import Tensor, nn

from dcvpp.models.workload.cfc_w import CfCW


class DefaultCfCCell(nn.Module):
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


class DefaultCfCEncoder(nn.Module):
    state_size: int

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.rnn_cell = DefaultCfCCell(source.rnn_cell)
        self.state_size = int(source.state_size)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs


class ScriptableCfCW(nn.Module):
    horizon: int
    cohorts: int

    def __init__(self, source: CfCW) -> None:
        super().__init__()
        self.encoder = DefaultCfCEncoder(source.encoder)
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


def main() -> None:
    eager = CfCW(input_size=14, hidden_size=64, horizon=96, cohorts=4).cuda().eval()
    scriptable = ScriptableCfCW(eager).cuda().eval()
    if list(eager.state_dict()) != list(scriptable.state_dict()):
        raise RuntimeError("state-dict key ordering changed")
    scripted = torch.jit.script(scriptable)
    inputs = torch.randn(4, 672, 14, device="cuda")
    elapsed = torch.ones(4, 672, 1, device="cuda")
    with torch.no_grad():
        eager_output = eager(inputs, elapsed)
        scripted_output = scripted(inputs, elapsed)
    difference = max(
        (eager_output[key] - scripted_output[key]).abs().max().item()
        for key in eager_output
    )
    print(f"max_abs_difference={difference:.12g}")
    for model, label in ((eager, "eager"), (scripted, "scripted")):
        with torch.no_grad():
            model(inputs, elapsed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            for _ in range(5):
                model(inputs, elapsed)
        torch.cuda.synchronize()
        print(f"{label}_seconds={(time.perf_counter() - started) / 5:.6f}")


if __name__ == "__main__":
    main()
