from __future__ import annotations

import torch
from torch import nn


class TinyCfCCell(nn.Module):
    """Closed-form elapsed-time gated recurrent cell for the demo."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)
        self.tau = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        candidate = torch.tanh(self.candidate(torch.cat([x, hidden], dim=-1)))
        decay = torch.exp(-torch.nn.functional.softplus(self.tau) * dt)
        return decay * hidden + (1.0 - decay) * candidate


class TinyCfCW(nn.Module):
    def __init__(self, input_size: int = 2, hidden_size: int = 32, horizon: int = 96) -> None:
        super().__init__()
        self.cell = TinyCfCCell(input_size, hidden_size)
        self.horizon = horizon
        self.count_head = nn.Linear(hidden_size, horizon)
        self.volume_head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(x.shape[0], self.cell.tau.numel(), device=x.device, dtype=x.dtype)
        for step in range(x.shape[1]):
            hidden = self.cell(x[:, step], hidden, dt[:, step])
        counts = torch.nn.functional.softplus(self.count_head(hidden))
        volume = torch.nn.functional.softplus(self.volume_head(hidden))
        return counts, volume


class CfCW(nn.Module):
    """Paper event-time CfC encoder with cohort-specific distribution heads."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int = 64,
        horizon: int = 96,
        cohorts: int = 4,
    ) -> None:
        super().__init__()
        if hidden_size not in {64, 96}:
            raise ValueError("paper CfC-W hidden size must be 64 or 96")
        if cohorts < 1:
            raise ValueError("CfC-W requires at least one cohort")
        try:
            from ncps.torch import CfC
        except ImportError as exc:
            raise RuntimeError("paper CfC-W requires ncps") from exc
        self.horizon = int(horizon)
        self.hidden_size = int(hidden_size)
        self.cohorts = int(cohorts)
        self.encoder = CfC(
            input_size=input_size,
            units=hidden_size,
            return_sequences=False,
            batch_first=True,
            mixed_memory=False,
            mode="default",
            backbone_units=hidden_size,
            backbone_layers=1,
            backbone_dropout=0.0,
        )
        output_size = cohorts * horizon * 2
        self.count_head = nn.Linear(hidden_size, output_size)
        self.lognormal_head = nn.Linear(hidden_size, output_size)
        self.gamma_head = nn.Linear(hidden_size, output_size)

    @property
    def architecture_contract(self) -> dict[str, int | str]:
        return {
            "hidden_size": self.hidden_size,
            "horizon": self.horizon,
            "cohorts": self.cohorts,
            "encoder": "ncps.torch.CfC",
            "elapsed_time_input": "explicit_hours_since_previous_event",
            "count_distribution": "negative_binomial",
            "volume_candidates": "lognormal_and_gamma",
        }

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or dt.ndim != 3 or x.shape[:2] != dt.shape[:2]:
            raise ValueError("CfC-W expects x=(batch,event,features), dt=(batch,event,1)")
        if dt.shape[-1] != 1 or (dt <= 0).any() or not torch.isfinite(dt).all():
            raise ValueError("CfC-W elapsed times must be finite positive values")
        # ncps 1.0.1 squeezes batched timespans to (batch,), which does not
        # broadcast with (batch, hidden) when batch > 1.  Calling its public
        # cell keeps the elapsed-time dimension and uses the same recurrence.
        hidden = torch.zeros(
            (x.shape[0], self.encoder.state_size), dtype=x.dtype, device=x.device
        )
        encoded = hidden
        for step in range(x.shape[1]):
            encoded, hidden = self.encoder.rnn_cell.forward(
                x[:, step], hidden, dt[:, step]
            )
        if self.encoder.fc is not None:
            encoded = self.encoder.fc(encoded)

        shape = (len(x), self.cohorts, self.horizon, 2)
        count_raw = self.count_head(encoded).reshape(shape)
        lognormal_raw = self.lognormal_head(encoded).reshape(shape)
        gamma_raw = self.gamma_head(encoded).reshape(shape)
        return {
            "count_mean": torch.nn.functional.softplus(count_raw[..., 0]) + 1e-4,
            "count_dispersion": torch.nn.functional.softplus(count_raw[..., 1]) + 1e-4,
            "volume_log_mu": lognormal_raw[..., 0],
            "volume_log_sigma": torch.nn.functional.softplus(lognormal_raw[..., 1]) + 1e-4,
            "volume_gamma_shape": torch.nn.functional.softplus(gamma_raw[..., 0]) + 1e-4,
            "volume_gamma_rate": torch.nn.functional.softplus(gamma_raw[..., 1]) + 1e-4,
        }


def negative_binomial_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    dispersion: torch.Tensor,
) -> torch.Tensor:
    if target.shape != mean.shape or target.shape != dispersion.shape:
        raise ValueError("negative-binomial target and parameters must share a shape")
    probs = mean / (dispersion + mean)
    distribution = torch.distributions.NegativeBinomial(
        total_count=dispersion, probs=probs.clamp(1e-7, 1.0 - 1e-7)
    )
    return -distribution.log_prob(target.clamp_min(0)).mean()


def lognormal_volume_nll(
    target: torch.Tensor,
    log_mu: torch.Tensor,
    log_sigma: torch.Tensor,
) -> torch.Tensor:
    positive = target > 0
    if not positive.any():
        return (log_mu.sum() + log_sigma.sum()) * 0.0
    distribution = torch.distributions.LogNormal(log_mu[positive], log_sigma[positive])
    return -distribution.log_prob(target[positive].clamp_min(1e-6)).mean()


def gamma_volume_nll(
    target: torch.Tensor,
    shape: torch.Tensor,
    rate: torch.Tensor,
) -> torch.Tensor:
    positive = target > 0
    if not positive.any():
        return (shape.sum() + rate.sum()) * 0.0
    distribution = torch.distributions.Gamma(shape[positive], rate[positive])
    return -distribution.log_prob(target[positive].clamp_min(1e-6)).mean()
