"""Solver-free closed-form Liquid sequence primitives for R4.

The recurrence is an analytic first-order relaxation.  It deliberately does
not depend on an ODE package: one call to :meth:`ClosedFormLiquidCell.step`
corresponds to one observed 15-minute transition and can therefore be replayed
exactly by the rolling runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LiquidState:
    """Explicit recurrent state used by incremental rolling inference."""

    hidden: Tensor

    def detach(self) -> "LiquidState":
        return LiquidState(self.hidden.detach())


class ClosedFormLiquidCell(nn.Module):
    """Stable elapsed-time-conditioned continuous-state recurrent cell.

    For input ``u`` and previous state ``h`` the update is

    ``tau = tau_min + (tau_max - tau_min) * sigmoid(W_tau[u,h])``
    ``g = tanh(W_g[u,h])``
    ``a = exp(-dt / tau)``
    ``h_next = a*h + (1-a)*g``.

    The exponential is evaluated in fp32 under autocast because it is the only
    numerically sensitive operation.  The returned state is converted back to
    the input dtype, preserving mixed-precision use by the surrounding model.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        tau_min_hours: float = 0.25,
        tau_max_hours: float = 24.0,
        default_dt_hours: float = 0.25,
    ) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1:
            raise ValueError("Liquid dimensions must be positive")
        if not 0.0 < tau_min_hours < tau_max_hours:
            raise ValueError("tau bounds must satisfy 0 < tau_min < tau_max")
        if default_dt_hours <= 0.0:
            raise ValueError("default_dt_hours must be positive")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.tau_min_hours = float(tau_min_hours)
        self.tau_max_hours = float(tau_max_hours)
        self.default_dt_hours = float(default_dt_hours)
        joined = self.input_size + self.hidden_size
        self.tau_projection = nn.Linear(joined, self.hidden_size)
        self.candidate_projection = nn.Linear(joined, self.hidden_size)

    def initial_state(self, reference: Tensor, batch_size: int | None = None) -> Tensor:
        batch = int(reference.shape[0] if batch_size is None else batch_size)
        return reference.new_zeros((batch, self.hidden_size))

    def _normalize_dt(self, dt: Tensor | float | None, reference: Tensor) -> Tensor:
        if dt is None:
            value = reference.new_full((reference.shape[0], 1), self.default_dt_hours)
        else:
            value = torch.as_tensor(dt, device=reference.device, dtype=reference.dtype)
            if value.ndim == 0:
                value = value.expand(reference.shape[0]).unsqueeze(-1)
            elif value.ndim == 1:
                value = value.unsqueeze(-1)
            if value.shape != (reference.shape[0], 1):
                raise ValueError(
                    "elapsed time must be scalar, [batch], or [batch, 1], got "
                    f"{tuple(value.shape)}"
                )
        if not torch.isfinite(value).all() or (value <= 0).any():
            raise ValueError("elapsed time must be finite and strictly positive")
        return value

    def step(
        self,
        inputs: Tensor,
        hidden: Tensor | LiquidState | None = None,
        *,
        dt: Tensor | float | None = None,
    ) -> Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_size:
            raise ValueError(
                f"Liquid input must be [batch, {self.input_size}], got "
                f"{tuple(inputs.shape)}"
            )
        if isinstance(hidden, LiquidState):
            hidden = hidden.hidden
        if hidden is None:
            hidden = self.initial_state(inputs)
        if hidden.shape != (inputs.shape[0], self.hidden_size):
            raise ValueError(
                f"Liquid hidden state must be {(inputs.shape[0], self.hidden_size)}, "
                f"got {tuple(hidden.shape)}"
            )
        elapsed = self._normalize_dt(dt, inputs)
        joined = torch.cat((inputs, hidden), dim=-1)
        tau = self.tau_min_hours + (
            self.tau_max_hours - self.tau_min_hours
        ) * torch.sigmoid(self.tau_projection(joined))
        candidate = torch.tanh(self.candidate_projection(joined))
        # Explicit fp32 evaluation makes the decay stable under fp16/bf16 AMP.
        decay = torch.exp(-elapsed.float() / tau.float()).to(dtype=inputs.dtype)
        return decay * hidden + (1.0 - decay) * candidate

    def forward(
        self,
        inputs: Tensor,
        hidden: Tensor | LiquidState | None = None,
        dt: Tensor | float | None = None,
    ) -> Tensor:
        return self.step(inputs, hidden, dt=dt)


class ClosedFormLiquidSequence(nn.Module):
    """Token-preserving Liquid encoder with identical full/replay semantics."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        *,
        tau_min_hours: float = 0.25,
        tau_max_hours: float = 24.0,
        default_dt_hours: float = 0.25,
        input_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.input_norm = nn.LayerNorm(input_size) if input_layer_norm else nn.Identity()
        self.cell = ClosedFormLiquidCell(
            input_size,
            hidden_size,
            tau_min_hours=tau_min_hours,
            tau_max_hours=tau_max_hours,
            default_dt_hours=default_dt_hours,
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def _dt_at(self, dt: Tensor | float | None, step: int, inputs: Tensor) -> Tensor | float | None:
        if dt is None or not isinstance(dt, Tensor) or dt.ndim <= 1:
            return dt
        if dt.ndim == 2 and dt.shape == inputs.shape[:2]:
            return dt[:, step]
        if dt.ndim == 3 and dt.shape == (*inputs.shape[:2], 1):
            return dt[:, step]
        raise ValueError(
            "sequence elapsed time must be scalar, [batch], [batch,time], or "
            f"[batch,time,1], got {tuple(dt.shape)}"
        )

    def step(
        self,
        inputs: Tensor,
        hidden: Tensor | LiquidState | None = None,
        *,
        dt: Tensor | float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(normalized_output, raw_recurrent_state)`` for one step."""

        state = self.cell.step(self.input_norm(inputs), hidden, dt=dt)
        return self.output_norm(state), state

    def forward(
        self,
        inputs: Tensor,
        hidden: Tensor | LiquidState | None = None,
        *,
        dt: Tensor | float | None = None,
    ) -> tuple[Tensor, Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError(
                f"Liquid sequence must be [batch,time,{self.input_size}], got "
                f"{tuple(inputs.shape)}"
            )
        if isinstance(hidden, LiquidState):
            hidden = hidden.hidden
        normalized = self.input_norm(inputs)
        state = hidden
        outputs: list[Tensor] = []
        for index in range(inputs.shape[1]):
            state = self.cell.step(
                normalized[:, index], state, dt=self._dt_at(dt, index, inputs)
            )
            outputs.append(self.output_norm(state))
        sequence = torch.stack(outputs, dim=1)
        # Return the raw recurrent state.  Applying output LayerNorm to the
        # recurrent feedback would make incremental and full-prefix paths differ.
        assert state is not None
        return sequence, state

    def replay(
        self,
        inputs: Tensor,
        *,
        dt: Tensor | float | None = None,
        hidden: Tensor | LiquidState | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Named alias used by the formal prefix-replay equivalence check."""

        return self.forward(inputs, hidden, dt=dt)


__all__ = ["ClosedFormLiquidCell", "ClosedFormLiquidSequence", "LiquidState"]
