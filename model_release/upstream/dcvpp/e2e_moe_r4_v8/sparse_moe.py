"""Genuinely sparse token-level Top-2 mixture of experts for R4-V8.

The dispatcher groups all assignments for an active expert and calls that
expert exactly once per forward pass.  Experts with no routed tokens are not
executed.  The module is deliberately self-contained and has no optimization
solver dependency.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_TOP_K: Final = 2


class _SwiGLUExpert(nn.Module):
    """A residual delta expert whose width is independent of the token width."""

    def __init__(self, hidden_dim: int, expert_hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.input_projection = nn.Linear(hidden_dim, 2 * expert_hidden_dim)
        self.output_projection = nn.Linear(expert_hidden_dim, hidden_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        gate, value = self.input_projection(self.norm(tokens)).chunk(2, dim=-1)
        return self.output_projection(F.silu(gate) * value)


class SparseTop2MoE(nn.Module):
    """Dropless Top-2 routing with sparse, once-per-active-expert dispatch.

    ``tokens`` may have any non-empty prefix shape, for example ``[B, T, H]``
    or ``[N, H]``.  If ``router_context`` is supplied, routing is conditioned
    on ``cat(tokens, router_context)`` while the experts themselves still see
    only ``tokens``.  ``router_input_dim`` denotes that concatenated width.
    """

    def __init__(
        self,
        hidden_dim: int,
        expert_hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        router_input_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if expert_hidden_dim <= 0:
            raise ValueError("expert_hidden_dim must be positive")
        if num_experts < _TOP_K:
            raise ValueError("num_experts must be at least two")
        if top_k != _TOP_K:
            raise ValueError("SparseTop2MoE requires top_k=2")

        resolved_router_dim = (
            hidden_dim if router_input_dim is None else int(router_input_dim)
        )
        if resolved_router_dim < hidden_dim:
            raise ValueError("router_input_dim cannot be smaller than hidden_dim")

        self.hidden_dim = int(hidden_dim)
        self.expert_hidden_dim = int(expert_hidden_dim)
        self.num_experts = int(num_experts)
        self.top_k = _TOP_K
        self.router_input_dim = resolved_router_dim
        self.router_context_dim = resolved_router_dim - self.hidden_dim

        self.router = nn.Linear(self.router_input_dim, self.num_experts)
        self.experts = nn.ModuleList(
            _SwiGLUExpert(self.hidden_dim, self.expert_hidden_dim)
            for _ in range(self.num_experts)
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def _router_inputs(self, tokens: Tensor, router_context: Tensor | None) -> Tensor:
        if tokens.ndim < 2 or tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                "tokens must have shape [..., hidden_dim]; "
                f"expected hidden_dim={self.hidden_dim}, got {tuple(tokens.shape)}"
            )
        if not tokens.is_floating_point():
            raise TypeError("tokens must be floating point")
        if tokens.numel() == 0:
            raise ValueError("tokens must contain at least one routed token")

        if self.router_context_dim == 0:
            if router_context is not None:
                raise ValueError(
                    "router_context was supplied but router_input_dim equals hidden_dim"
                )
            return tokens

        if router_context is None:
            raise ValueError(
                "router_context is required because router_input_dim exceeds hidden_dim"
            )
        expected_shape = (*tokens.shape[:-1], self.router_context_dim)
        if tuple(router_context.shape) != expected_shape:
            raise ValueError(
                "router_context must share the token prefix and fill the configured "
                f"router width; expected {expected_shape}, got {tuple(router_context.shape)}"
            )
        if router_context.device != tokens.device:
            raise ValueError("tokens and router_context must be on the same device")
        if router_context.dtype != tokens.dtype:
            # Autocast can leave a residual stream in FP32 while an eligible
            # projection is BF16. Routing is defined in the token dtype.
            router_context = router_context.to(dtype=tokens.dtype)
        return torch.cat((tokens, router_context), dim=-1)

    def forward(
        self, tokens: Tensor, router_context: Tensor | None = None
    ) -> dict[str, Tensor]:
        router_inputs = self._router_inputs(tokens, router_context)
        router_logits = self.router(router_inputs)
        router_probabilities = torch.softmax(router_logits, dim=-1)
        top_values, topk_indices = router_probabilities.topk(self.top_k, dim=-1)
        topk_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(top_values.dtype).tiny
        )

        flat_tokens = tokens.reshape(-1, self.hidden_dim)
        flat_indices = topk_indices.reshape(-1, self.top_k)
        flat_weights = topk_weights.reshape(-1, self.top_k)
        routed_experts = flat_indices.reshape(-1)
        routed_weights = flat_weights.reshape(-1)
        token_indices = torch.arange(
            flat_tokens.shape[0], device=tokens.device
        ).repeat_interleave(self.top_k)

        expert_token_counts = torch.bincount(routed_experts, minlength=self.num_experts)
        expert_call_counts = (expert_token_counts > 0).to(torch.long)

        # The only expert calls in this module occur in this loop.  Each active
        # expert receives all of its routed tokens in one aggregate call. A
        # fixed Python range avoids copying active indices from CUDA to the
        # host; empty experts are still never executed.
        flat_delta = torch.zeros_like(flat_tokens)
        for expert_index in range(self.num_experts):
            assignment_indices = torch.nonzero(
                routed_experts == expert_index, as_tuple=False
            ).flatten()
            if assignment_indices.numel() == 0:
                continue
            selected_token_indices = token_indices.index_select(0, assignment_indices)
            selected_tokens = flat_tokens.index_select(0, selected_token_indices)
            expert_delta = self.experts[expert_index](selected_tokens)
            selected_weights = routed_weights.index_select(
                0, assignment_indices
            ).unsqueeze(-1)
            flat_delta = flat_delta.index_add(
                0, selected_token_indices, expert_delta * selected_weights
            )

        output = self.output_norm(tokens + flat_delta.reshape_as(tokens))

        flat_probabilities = router_probabilities.reshape(-1, self.num_experts)
        expert_load = expert_token_counts.to(router_probabilities.dtype) / float(
            flat_tokens.shape[0] * self.top_k
        )
        expert_importance = flat_probabilities.mean(dim=0)
        expert_weight_mass = torch.zeros(
            self.num_experts,
            dtype=topk_weights.dtype,
            device=topk_weights.device,
        ).scatter_add(0, routed_experts, routed_weights) / float(flat_tokens.shape[0])
        router_balance_loss = self.num_experts * (expert_load * expert_importance).sum()
        router_z_loss = torch.logsumexp(router_logits.float(), dim=-1).square().mean()
        entropy_probabilities = flat_probabilities.float()
        router_entropy = (
            -(
                entropy_probabilities
                * entropy_probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
            )
            .sum(dim=-1)
            .mean()
        )

        return {
            "tokens": output,
            "router_logits": router_logits,
            "router_probabilities": router_probabilities,
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
            "router_balance_loss": router_balance_loss,
            "router_z_loss": router_z_loss,
            "expert_token_counts": expert_token_counts,
            "expert_call_counts": expert_call_counts,
            "active_expert_mask": expert_token_counts > 0,
            "expert_load": expert_load,
            "expert_importance": expert_importance,
            "expert_weight_mass": expert_weight_mass,
            "router_entropy": router_entropy,
        }


__all__ = ["SparseTop2MoE"]
