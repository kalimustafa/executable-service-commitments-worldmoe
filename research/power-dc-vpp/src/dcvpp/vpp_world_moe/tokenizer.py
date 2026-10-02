"""Unified causal tokenizer for typed VPP events at three time scales."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import WorldMoEConfig
from .packets import (
    ENTITIES_PER_NODE,
    GLOBAL_SLOTS,
    NODES,
    TOTAL_SLOTS,
    WorldInputPacket,
)


# These identifiers are part of the registered data provenance vocabulary.
# Completed observations (6), missing values (0), and prior actions (8) are
# deliberately absent.  A caller-provided mask can remove entries but cannot
# elevate a source outside this fail-closed list.
KNOWN_FUTURE_SOURCE_IDS = (1, 2, 3, 4, 5, 7, 9)


def resolve_known_future_mask(packet: WorldInputPacket) -> Tensor:
    """Return the hard source- and origin-bounded fast-horizon mask.

    Bidirectional attention is scientifically causal only over information
    already visible at the issue/rolling origin.  The provenance whitelist is
    therefore applied after any optional caller mask, so a mislabeled boolean
    mask cannot expose completed observations or future actions.
    """

    source_allowed = torch.zeros_like(packet.source_ids, dtype=torch.bool)
    for source_id in KNOWN_FUTURE_SOURCE_IDS:
        source_allowed |= packet.source_ids.eq(source_id)
    time = torch.arange(
        packet.fast_values.shape[1], device=packet.fast_values.device
    )[None, :, None]
    at_or_after_origin = time >= packet.origin_index[:, None, None]
    result = packet.fast_mask & source_allowed & at_or_after_origin
    if packet.known_future_mask is not None:
        result &= packet.known_future_mask
    return result


def _causal_hold_last(value: Tensor, target_length: int) -> Tensor:
    """Causally align a coarse stream without interpolation from the future."""

    source_length = value.shape[1]
    if source_length == target_length:
        return value
    index = torch.div(
        torch.arange(target_length, device=value.device) * source_length,
        target_length,
        rounding_mode="floor",
    ).clamp_max(source_length - 1)
    return value.index_select(1, index)


class KnownFutureTransformerFusion(nn.Module):
    """Encode the admitted 96-step future without opening realized channels.

    The Transformer sees exactly one token per 15-minute lead.  Each token is
    a learned compression of four node summaries plus one global summary; this
    keeps attention quadratic in 96 rather than in ``96 * 40`` while retaining
    fixed-node distinctions.  Its output is expanded back to the unchanged
    40-slot workspace through bounded, gated node-specific contexts.
    """

    _GROUPS = NODES + 1
    _LAYERS = 2

    def __init__(self, config: WorldMoEConfig) -> None:
        super().__init__()
        width = config.d_model
        self.config = config
        self.group_norm = nn.LayerNorm(width)
        self.group_projection = nn.Linear(self._GROUPS * width, width)
        self.availability_projection = nn.Linear(self._GROUPS, width, bias=False)
        self.position_embedding = nn.Embedding(config.horizon, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.num_heads,
            dim_feedforward=2 * width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=self._LAYERS,
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )
        self.group_embedding = nn.Parameter(
            torch.randn(self._GROUPS, width) / width**0.5
        )
        self.group_output = nn.Linear(width, width)
        self.output_norm = nn.LayerNorm(width)
        self.global_to_local_gate = nn.Parameter(torch.tensor(-1.5))
        self.local_to_global_gate = nn.Parameter(torch.tensor(-1.5))
        self.context_gate = nn.Parameter(torch.tensor(-1.0))

    @staticmethod
    def _masked_mean(value: Tensor, mask: Tensor, *, dim: int) -> Tensor:
        weight = mask[..., None].to(value.dtype)
        count = weight.sum(dim=dim).clamp_min(1.0)
        return (value * weight).sum(dim=dim) / count

    def _group_summaries(
        self, fast_tokens: Tensor, known_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch, time, _, width = fast_tokens.shape
        local = fast_tokens[:, :, : NODES * ENTITIES_PER_NODE].reshape(
            batch, time, NODES, ENTITIES_PER_NODE, width
        )
        local_mask = known_mask[:, :, : NODES * ENTITIES_PER_NODE].reshape(
            batch, time, NODES, ENTITIES_PER_NODE
        )
        local_summary = self._masked_mean(local, local_mask, dim=3)
        global_summary = self._masked_mean(
            fast_tokens[:, :, -GLOBAL_SLOTS:],
            known_mask[:, :, -GLOBAL_SLOTS:],
            dim=2,
        )[:, :, None]
        summaries = torch.cat((local_summary, global_summary), dim=2)
        available = torch.cat(
            (
                local_mask.any(dim=3),
                known_mask[:, :, -GLOBAL_SLOTS:].any(dim=2)[..., None],
            ),
            dim=2,
        )
        # Re-mask after LayerNorm so learned affine offsets can never turn an
        # unavailable source group into apparent future evidence.
        summaries = self.group_norm(summaries)
        summaries = summaries * available[..., None].to(summaries.dtype)
        return summaries, available

    def forward(self, fast_tokens: Tensor, packet: WorldInputPacket) -> Tensor:
        known_mask = resolve_known_future_mask(packet)
        summaries, available = self._group_summaries(fast_tokens, known_mask)
        batch, time = summaries.shape[:2]
        sequence = self.group_projection(summaries.flatten(start_dim=2))
        sequence = sequence + self.availability_projection(
            available.to(sequence.dtype)
        )
        sequence = sequence + self.position_embedding(
            torch.arange(time, device=sequence.device)
        )[None]

        time_available = available.any(dim=2)
        has_any = time_available.any(dim=1)
        padding_mask = ~time_available
        # Multi-head attention has no finite softmax when every key is masked.
        # Empty samples receive one inert key and are zeroed after encoding.
        safe_padding_mask = padding_mask.clone()
        safe_padding_mask[~has_any, 0] = False
        safe_sequence = torch.where(
            has_any[:, None, None], sequence, torch.zeros_like(sequence)
        )
        encoded = self.transformer(
            safe_sequence,
            src_key_padding_mask=safe_padding_mask,
        )
        encoded = encoded * has_any[:, None, None].to(encoded.dtype)

        grouped = encoded[:, :, None] + self.group_embedding[None, None].to(
            encoded.dtype
        )
        grouped = self.group_output(torch.tanh(grouped))
        local = grouped[:, :, :NODES]
        global_context = grouped[:, :, NODES]
        local = local + torch.sigmoid(self.global_to_local_gate) * global_context[
            :, :, None
        ]
        global_context = global_context + torch.sigmoid(
            self.local_to_global_gate
        ) * local.mean(dim=2)
        local_slots = local[:, :, :, None].expand(
            -1, -1, -1, ENTITIES_PER_NODE, -1
        ).reshape(batch, time, NODES * ENTITIES_PER_NODE, -1)
        global_slots = global_context[:, :, None].expand(-1, -1, GLOBAL_SLOTS, -1)
        context = torch.cat((local_slots, global_slots), dim=2)
        bounded = torch.tanh(self.output_norm(context))
        return (
            torch.sigmoid(self.context_gate)
            * bounded
            * has_any[:, None, None, None].to(bounded.dtype)
        )


class UnifiedTypedEventTokenizer(nn.Module):
    """Embed values, provenance, quality, entity, node, time and controls jointly."""

    def __init__(self, config: WorldMoEConfig) -> None:
        super().__init__()
        self.config = config
        width = config.d_model
        self.value_projections = nn.ModuleList(
            nn.Linear(config.input_dim, width) for _ in range(3)
        )
        self.entity_embedding = nn.Embedding(16, width)
        self.node_embedding = nn.Embedding(NODES + 1, width)
        self.source_embedding = nn.Embedding(64, width)
        self.quality_embedding = nn.Embedding(16, width)
        self.scale_embedding = nn.Embedding(3, width)
        self.slot_embedding = nn.Embedding(TOTAL_SLOTS, width)
        self.time_embedding = nn.Embedding(config.horizon, width)
        self.missing_embedding = nn.Parameter(torch.zeros(3, width))
        self.scale_score = nn.Linear(width, 1)
        self.known_future_encoder = KnownFutureTransformerFusion(config)
        self.action_projection = nn.Linear(config.action_dim, width)
        self.slack_projection = nn.Linear(config.constraint_dim, width)
        self.output_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(config.dropout)

    def _one_scale(
        self,
        values: Tensor,
        mask: Tensor,
        packet: WorldInputPacket,
        scale: int,
        target_length: int,
        source_ids: Tensor,
        quality_ids: Tensor,
    ) -> Tensor:
        values = _causal_hold_last(values, target_length)
        mask = _causal_hold_last(mask, target_length)
        source_ids = _causal_hold_last(source_ids, target_length)
        quality_ids = _causal_hold_last(quality_ids, target_length)
        batch = values.shape[0]
        device = values.device
        dtype = values.dtype
        source_ids = source_ids.clamp(0, 63)
        quality_ids = quality_ids.clamp(0, 15)
        structural = (
            self.entity_embedding(packet.entity_ids.clamp(0, 15))[None, None]
            + self.node_embedding(packet.node_ids.clamp(0, NODES))[None, None]
            + self.slot_embedding(torch.arange(TOTAL_SLOTS, device=device))[None, None]
            + self.time_embedding(torch.arange(target_length, device=device))[
                None, :, None
            ]
            + self.scale_embedding.weight[scale][None, None, None]
            + self.source_embedding(source_ids)
            + self.quality_embedding(quality_ids)
        )
        projected = self.value_projections[scale](values * mask[..., None].to(dtype))
        missing = (~mask)[..., None].to(dtype) * self.missing_embedding[scale]
        if structural.shape[:3] != (batch, target_length, TOTAL_SLOTS):
            raise RuntimeError("tokenizer structural broadcast failed")
        return projected + structural + missing

    def forward(self, packet: WorldInputPacket) -> tuple[Tensor, Tensor, Tensor]:
        packet.validate(self.config)
        target_length = packet.fast_values.shape[1]
        streams: list[Tensor] = [
            self._one_scale(
                packet.fast_values,
                packet.fast_mask,
                packet,
                0,
                target_length,
                packet.source_ids,
                packet.quality_ids,
            )
        ]
        masks: list[Tensor] = [packet.fast_mask]
        for index, (values, mask, source_ids, quality_ids) in enumerate(
            (
                (
                    packet.medium_values,
                    packet.medium_mask,
                    packet.medium_source_ids,
                    packet.medium_quality_ids,
                ),
                (
                    packet.slow_values,
                    packet.slow_mask,
                    packet.slow_source_ids,
                    packet.slow_quality_ids,
                ),
            ),
            start=1,
        ):
            if values is None or mask is None:
                continue
            if source_ids is None:
                source_ids = torch.full_like(mask, 6, dtype=torch.long)
            if quality_ids is None:
                quality_ids = torch.full_like(mask, 3, dtype=torch.long)
            streams.append(
                self._one_scale(
                    values,
                    mask,
                    packet,
                    index,
                    target_length,
                    source_ids,
                    quality_ids,
                )
            )
            masks.append(_causal_hold_last(mask, target_length))
        stacked = torch.stack(streams, dim=-2)
        available = torch.stack(masks, dim=-1)
        scale_logits = self.scale_score(torch.tanh(stacked)).squeeze(-1)
        scale_logits = scale_logits.masked_fill(~available, -10_000.0)
        all_missing = ~available.any(dim=-1, keepdim=True)
        scale_logits = torch.where(
            all_missing, torch.zeros_like(scale_logits), scale_logits
        )
        weights = torch.softmax(scale_logits, dim=-1)
        tokens = (stacked * weights[..., None]).sum(dim=-2)
        future_context = self.known_future_encoder(streams[0], packet)

        action = packet.previous_actions * packet.action_mask.to(
            packet.previous_actions.dtype
        )
        action_context = self.action_projection(action)
        local_action = action_context.repeat_interleave(
            self.config.entities_per_node, dim=2
        )
        global_action = action_context.mean(dim=2, keepdim=True).expand(
            -1, -1, GLOBAL_SLOTS, -1
        )
        action_slots = torch.cat((local_action, global_action), dim=2)

        slack_context = self.slack_projection(packet.constraint_slacks)
        local_slack = slack_context.repeat_interleave(
            self.config.entities_per_node, dim=2
        )
        global_slack = slack_context.mean(dim=2, keepdim=True).expand(
            -1, -1, GLOBAL_SLOTS, -1
        )
        slack_slots = torch.cat((local_slack, global_slack), dim=2)
        tokens = self.output_norm(
            tokens + action_slots + slack_slots + future_context
        )
        return self.dropout(tokens), action_slots, slack_slots


__all__ = [
    "KNOWN_FUTURE_SOURCE_IDS",
    "KnownFutureTransformerFusion",
    "UnifiedTypedEventTokenizer",
    "resolve_known_future_mask",
]
