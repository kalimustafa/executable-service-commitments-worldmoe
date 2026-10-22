"""Post-decision realized historical replay for R-ECMoE.

The replay boundary is deliberately downstream of :class:`RECMoEOutput`.
Consequently, realized DA/RT, workload, PV and PUE targets cannot influence W1
expert selection, continuation routing, or raw structured action generation.
They are used only to re-run the existing vectorized constraint assembler and
to score the already-chosen action under the realized historical outcome.

This module is solver-free and accepts only the development ``train`` and
``validation`` partitions.  The locked 2025 partition is rejected by date even
if a caller mislabels it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
import math
from typing import Any, Literal, Mapping

import torch
from torch import Tensor

from .packets import NODES, WorldConstraintPacket, WorldObservationPacket
from .r_ecmoe import (
    RECMoE,
    RECMoEOutput,
    RiskConditionPacket,
    StructuredActionHeads,
    W1ExpertPacket,
    NeuralConstraintAssembler,
    differentiable_operating_cost,
)


ReplaySplit = Literal["train", "validation"]


class RECMoEReplayError(ValueError):
    """Raised when a historical replay would violate its causal contract."""


@dataclass(frozen=True)
class RealizedReplayCostWeights:
    """The same scalar cost coefficients used by the live R forward path."""

    battery_degradation_weight: float
    migration_weight: float
    sla_weight: float

    def __post_init__(self) -> None:
        for name in (
            "battery_degradation_weight",
            "migration_weight",
            "sla_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise RECMoEReplayError(
                    f"replay cost coefficient {name} must be finite and nonnegative"
                )


@dataclass(frozen=True)
class RealizedReplayTargets:
    """Detached, raw-unit outcomes admitted to post-decision replay only."""

    public_21: Tensor
    da: Tensor
    rt: Tensor
    dart: Tensor
    arrivals: Tensor
    pv: Tensor
    pue: Tensor
    operating_dates: tuple[str, ...]
    source_split: ReplaySplit

    @classmethod
    def from_corpus_batch(
        cls,
        batch: Mapping[str, Any],
        *,
        constraints: WorldConstraintPacket,
        source_split: ReplaySplit,
        expected_batch: int,
        expected_time: int,
        device: torch.device,
    ) -> "RealizedReplayTargets":
        """Extract only loss-side targets from a :class:`WorldMoECorpus` batch.

        ``world_input`` and every causal/model-side field are intentionally
        ignored.  The aggregate realized workload is distributed across the
        four sites in proportion to their registered compute capacities.  It
        is not capped: overload must remain visible to queue/deadline replay.
        """

        if source_split not in {"train", "validation"}:
            raise RECMoEReplayError(
                "realized replay accepts only train or validation, never test"
            )
        raw_dates = batch.get("operating_dates")
        if not isinstance(raw_dates, (tuple, list)) or len(raw_dates) != expected_batch:
            raise RECMoEReplayError(
                "realized replay requires one operating date per batch item"
            )
        dates = tuple(str(value) for value in raw_dates)
        try:
            parsed_dates = tuple(date.fromisoformat(value) for value in dates)
        except ValueError as error:
            raise RECMoEReplayError(
                "realized replay operating dates must use ISO YYYY-MM-DD"
            ) from error
        if any(value.year >= 2025 for value in parsed_dates):
            raise RECMoEReplayError("locked 2025 outcomes cannot enter R training")

        metadata = batch.get("target_metadata")
        if isinstance(metadata, Mapping) and metadata.get("opened_final_test_2025") is True:
            raise RECMoEReplayError("target metadata reports that locked 2025 was opened")

        raw_public = batch.get("stochastic_target")
        if raw_public is None:
            reconstruction = batch.get("reconstruction_targets")
            if isinstance(reconstruction, Mapping):
                raw_public = reconstruction.get("public_21")
        if not isinstance(raw_public, Tensor):
            raise RECMoEReplayError(
                "WorldMoECorpus batch omitted realized public_21 targets"
            )
        if raw_public.shape != (expected_batch, expected_time, 21):
            raise RECMoEReplayError(
                "realized public target must be [B,T,21] in raw physical units"
            )
        raw_masks = batch.get("quantile_masks")
        if not isinstance(raw_masks, Mapping):
            raise RECMoEReplayError(
                "realized replay requires formal quantile target masks"
            )
        family_shapes = {
            "price": (expected_batch, expected_time, 12),
            "workload": (expected_batch, expected_time, 1),
            "pv": (expected_batch, expected_time, 4),
            "pue": (expected_batch, expected_time, 4),
        }
        masks: list[Tensor] = []
        for family, shape in family_shapes.items():
            value = raw_masks.get(family)
            if not isinstance(value, Tensor) or value.shape != shape:
                raise RECMoEReplayError(
                    f"realized {family} mask must be bool {shape}"
                )
            if value.dtype is not torch.bool:
                raise RECMoEReplayError(f"realized {family} mask must be boolean")
            masks.append(value.detach().to(device=device))
        public_mask = torch.cat(masks, dim=-1)

        # Targets are environment outcomes, never trainable tensors.  Moving a
        # detached tensor here also prevents an accidental target graph from
        # crossing the post-decision boundary.
        public = raw_public.detach().to(device=device, dtype=torch.float32)
        # Formal R replay is deliberately all-target, not a mixture that falls
        # back to model predictions on missing coordinates.  One joint check
        # covers both the registered mask and numeric validity.
        if not bool((public_mask & torch.isfinite(public)).all()):
            raise RECMoEReplayError(
                "realized replay requires every formal target to be valid and finite"
            )

        da = public[..., 0:4]
        rt = public[..., 4:8]
        # DART has no independent replay degree of freedom.
        dart = da - rt
        workload = public[..., 12].clamp_min(0.0)
        pv = public[..., 13:17]
        pue = public[..., 17:21]
        capacity = constraints.compute_capacity.detach().to(
            device=device, dtype=torch.float32
        )
        if capacity.shape != (expected_batch, NODES):
            raise RECMoEReplayError("compute capacity must be [B,4] for replay")
        denominator = capacity.sum(dim=-1, keepdim=True)
        if capacity.device.type == "cpu" and bool((denominator <= 0.0).any()):
            raise RECMoEReplayError("replay compute capacity must be positive")
        shares = capacity / denominator.clamp_min(1.0e-8)
        arrivals = workload[..., None] * shares[:, None]
        if public.device.type == "cpu" and (
            bool((pv < 0.0).any()) or bool((pue < 1.0).any())
        ):
            raise RECMoEReplayError("realized PV/PUE violates physical support")

        closed_public = torch.cat(
            (da, rt, dart, workload[..., None], pv, pue), dim=-1
        )
        return cls(
            public_21=closed_public,
            da=da,
            rt=rt,
            dart=dart,
            arrivals=arrivals,
            pv=pv,
            pue=pue,
            operating_dates=dates,
            source_split=source_split,
        )

    def replace_reference(
        self, template: WorldObservationPacket
    ) -> WorldObservationPacket:
        """Replace only the realized outcome coordinates used by replay."""

        batch, time = self.da.shape[:2]
        if template.da.shape[:2] != (batch, time):
            raise RECMoEReplayError("realized targets and W1 reference timeline differ")
        reference = replace(
            template,
            da=self.da,
            rt=self.rt,
            dart=self.dart,
            arrivals=self.arrivals,
            pv=self.pv,
            pue=self.pue,
            public_21=self.public_21,
        )
        return reference.validate(
            batch=batch, strict_values=reference.da.device.type == "cpu"
        )


def _split_raw_structured_actions(joined: Tensor) -> dict[str, Tensor]:
    if joined.ndim != 3 or joined.shape[-1] != StructuredActionHeads.output_dim:
        raise RECMoEReplayError("raw structured actions must be [B,T,36]")
    return {
        "joined": joined,
        "day_ahead": joined[..., 0:4],
        "battery": joined[..., 4:8],
        "compute": joined[..., 8:12],
        "migration": joined[..., 12:28].reshape(*joined.shape[:2], NODES, NODES),
        "pv": joined[..., 28:32],
        "target_soc": joined[..., 32:36],
    }


@dataclass
class RealizedHistoricalReplay:
    """Re-score one already-made R decision on realized historical outcomes."""

    assembler: NeuralConstraintAssembler
    cost_weights: RealizedReplayCostWeights

    @classmethod
    def from_model(cls, model: RECMoE) -> "RealizedHistoricalReplay":
        config = model.world_model.config
        return cls(
            assembler=model.assembler,
            cost_weights=RealizedReplayCostWeights(
                battery_degradation_weight=float(config.battery_degradation_weight),
                migration_weight=float(config.migration_weight),
                sla_weight=float(config.unserved_power_weight),
            ),
        )

    def __call__(
        self,
        output: RECMoEOutput,
        *,
        batch: Mapping[str, Any],
        risk: RiskConditionPacket,
        source_split: ReplaySplit,
    ) -> RECMoEOutput:
        """Return an output replacement suitable for :func:`r_ecmoe_loss`.

        Routing, selected expert state, value-head predictions, response state,
        and raw action logits are retained from ``output``.  Only the
        deterministic assembler outcome and realized operating cost are
        replaced.
        """

        if not isinstance(output, RECMoEOutput):
            raise RECMoEReplayError("replay input must be one RECMoEOutput")
        packet = output.expert_packet
        batch_size, time = output.raw_structured_actions.shape[:2]
        risk.validate(batch_size, time)
        targets = RealizedReplayTargets.from_corpus_batch(
            batch,
            constraints=packet.constraints,
            source_split=source_split,
            expected_batch=batch_size,
            expected_time=time,
            device=output.raw_structured_actions.device,
        )
        realized_reference = targets.replace_reference(packet.reference)
        replay_packet: W1ExpertPacket = replace(packet, reference=realized_reference)
        raw = _split_raw_structured_actions(output.raw_structured_actions)
        assembled = self.assembler(raw, replay_packet, risk)
        # The assembler is replaying physical execution under realized
        # PV/PUE/workload.  It may therefore produce a different feasible grid,
        # but the already-made DA commitment and target-SOC decision must never
        # be regenerated from those future outcomes.
        committed_day_ahead = output.day_ahead_commitment
        committed_target_soc = output.target_soc
        operating_cost = differentiable_operating_cost(
            reference=realized_reference,
            actions=assembled.actions,
            day_ahead_commitment=committed_day_ahead,
            constraints=replay_packet.constraints,
            decision_active_mask=assembled.decision_active_mask,
            deadline_expired_work=assembled.deadline_expired_work,
            battery_degradation_weight=(
                self.cost_weights.battery_degradation_weight
            ),
            migration_weight=self.cost_weights.migration_weight,
            sla_weight=self.cost_weights.sla_weight,
        )
        diagnostics = dict(output.diagnostics)
        diagnostics.update(
            {
                "outcome_basis": "realized_historical_replay",
                "replay_split": source_split,
                "replay_opened_final_test_2025": 0,
            }
        )
        return replace(
            output,
            expert_packet=replay_packet,
            actions=assembled.actions,
            day_ahead_commitment=committed_day_ahead,
            target_soc=committed_target_soc,
            raw_structured_actions=assembled.raw_structured_actions,
            constraint_residuals=assembled.residuals,
            differentiable_operating_cost=operating_cost,
            decision_active_mask=assembled.decision_active_mask,
            deadline_expired_work=assembled.deadline_expired_work,
            diagnostics=diagnostics,
        )


__all__ = [
    "RECMoEReplayError",
    "RealizedHistoricalReplay",
    "RealizedReplayCostWeights",
    "RealizedReplayTargets",
    "ReplaySplit",
]
