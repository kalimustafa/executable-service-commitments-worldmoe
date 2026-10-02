"""Checkpoint-bound materialization for the formal R4 scenario capability gate.

The expensive phase-end evaluator in this module is deliberately separate from
the differentiable scenario loss.  It executes the immutable candidate on all
179 eligible 2023-H2 dates and compares its exact twenty weighted joint paths
with one frozen, train-only historical residual-sampling baseline.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from . import losses as r4_losses
from .artifact_io import atomic_json, atomic_torch_save, read_json
from .capability_evaluation import (
    CapabilityEvidenceError,
    EXPECTED_VALIDATION_DAYS,
    evaluate_scenario,
)
from .hashing import sha256_file, sha256_json
from .scenario_training_contract import (
    ScenarioTrainingContract,
    attach_scenario_training_contract,
)
from .training_data import PreloadedR4Corpus, move_batch


SCHEMA = "dcvpp.r4.scenario-capability-materialization.v2"
BASELINE_SCHEMA = "dcvpp.r4.frozen-training-residual-sampling.v1"
BASELINE_NAME = "frozen_training_residual_sampling"
SCENARIO_COUNT = 20
HORIZON = 96
CHANNELS = 21


def _operating_date(record: Mapping[str, Any]) -> str:
    value = record.get("operating_date")
    text = value.isoformat() if hasattr(value, "isoformat") else str(value)
    try:
        parsed = date.fromisoformat(text[:10])
    except ValueError as exc:
        raise CapabilityEvidenceError(f"invalid corpus operating date {value!r}") from exc
    return parsed.isoformat()


def _validate_split_dates(
    train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    train_dates = tuple(_operating_date(row) for row in train)
    validation_dates = tuple(_operating_date(row) for row in validation)
    if len(train_dates) != 543 or len(set(train_dates)) != 543:
        raise CapabilityEvidenceError("scenario baseline requires 543 unique training dates")
    if len(validation_dates) != EXPECTED_VALIDATION_DAYS or len(set(validation_dates)) != 179:
        raise CapabilityEvidenceError("scenario gate requires 179 unique validation dates")
    if train_dates != tuple(sorted(train_dates)) or validation_dates != tuple(sorted(validation_dates)):
        raise CapabilityEvidenceError("formal corpus dates must be strictly ordered")
    if not all("2022-01-01" <= item <= "2023-06-30" for item in train_dates):
        raise CapabilityEvidenceError("scenario baseline attempted to use a non-training date")
    if not all("2023-07-01" <= item <= "2023-12-31" for item in validation_dates):
        raise CapabilityEvidenceError("scenario gate attempted to use a date outside 2023-H2")
    return train_dates, validation_dates


def _target(record: Mapping[str, Any]) -> Tensor:
    value = torch.as_tensor(record["scenario_target_zonal"], dtype=torch.float32).cpu()
    if value.shape != (HORIZON, CHANNELS) or not torch.isfinite(value).all():
        raise CapabilityEvidenceError("scenario target must be one finite [96,21] trajectory")
    identity = value[:, 0:4] - value[:, 4:8] - value[:, 8:12]
    if float(identity.abs().max()) > 1.0e-5:
        raise CapabilityEvidenceError("scenario target violates DA-RT-DART identity")
    return value


def _issue_center(record: Mapping[str, Any], pue_center: Tensor) -> Tensor:
    market = torch.as_tensor(record["market_raw"], dtype=torch.float32).cpu()
    workload = torch.as_tensor(record["issue_workload_raw"], dtype=torch.float32).cpu()
    pv = torch.as_tensor(record["issue_pv_raw"], dtype=torch.float32).cpu()
    if market.shape != (HORIZON, 16):
        raise CapabilityEvidenceError("raw issue-time market input must be [96,16]")
    if workload.shape != (HORIZON,) or pv.shape != (HORIZON, 4):
        raise CapabilityEvidenceError("raw issue-time workload/PV input has an invalid shape")
    da = market[:, 0:4]
    rt = market[:, 4:8]
    dart = da - rt
    return torch.cat((da, rt, dart, workload[:, None], pv, pue_center), dim=-1)


def _corpus_signature(
    records: Sequence[Mapping[str, Any]], train_dates: Sequence[str]
) -> str:
    sources: list[dict[str, Any]] = []
    for day, record in zip(train_dates, records, strict=True):
        provenance = record.get("provenance", {})
        if not isinstance(provenance, Mapping):
            provenance = {}
        sources.append(
            {
                "date": day,
                "bundle_sha256": provenance.get("bundle_sha256"),
                "settlement_sha256": provenance.get("settlement_sha256"),
                "physical_sha256": provenance.get("physical_sha256"),
            }
        )
    return sha256_json({"schema": BASELINE_SCHEMA, "training_sources": sources})


def _state_dict_metadata(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name in sorted(state):
        value = state[name]
        if isinstance(value, Tensor):
            result.append(
                {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
            )
        else:
            result.append({"name": name, "extra_state": value})
    return result


def _state_dicts_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    for name in left:
        first, second = left[name], right[name]
        if isinstance(first, Tensor) and isinstance(second, Tensor):
            if first.shape != second.shape or first.dtype != second.dtype:
                return False
            if not torch.equal(first.detach().cpu(), second.detach().cpu()):
                return False
        elif first != second:
            return False
    return True


def _load_checkpoint_binding(
    checkpoint: Path, model: nn.Module, epoch: int | None
) -> tuple[str, str, Mapping[str, Any]]:
    if not checkpoint.is_file():
        raise CapabilityEvidenceError(f"scenario checkpoint does not exist: {checkpoint}")
    checkpoint_sha = sha256_file(checkpoint)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state"), Mapping):
        raise CapabilityEvidenceError("scenario checkpoint is missing model_state")
    if str(payload.get("phase", "")) != "scenario":
        raise CapabilityEvidenceError("scenario capability candidate has the wrong phase")
    completed = int(payload.get("epochs_completed", int(payload.get("epoch", -1)) + 1))
    if epoch is not None and completed != int(epoch):
        raise CapabilityEvidenceError(
            f"scenario evaluator epoch {epoch} does not match checkpoint epoch {completed}"
        )
    live_state = model.state_dict()
    if not _state_dicts_equal(payload["model_state"], live_state):
        raise CapabilityEvidenceError("live scenario model is not the supplied immutable checkpoint")
    checkpoint_state_sha = sha256_json(
        {
            "checkpoint_sha256": checkpoint_sha,
            "state_dict": _state_dict_metadata(payload["model_state"]),
        }
    )
    return checkpoint_sha, checkpoint_state_sha, payload


def _collate(values: Sequence[Any]) -> Any:
    first = values[0]
    if isinstance(first, Tensor):
        return torch.stack([torch.as_tensor(value) for value in values])
    if isinstance(first, Mapping):
        return {key: _collate([value[key] for value in values]) for key in first}
    return list(values)


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return torch.device("cpu") if buffer is None else buffer.device


def _weighted_tail_score(
    particles: Tensor, weights: Tensor, target: Tensor
) -> Tensor:
    """Two-sided 5/95% pinball score, reduced to one scalar per day."""

    values: list[Tensor] = []
    for quantile in (0.05, 0.95):
        prediction = r4_losses.weighted_quantile(particles, weights, quantile)
        residual = target - prediction
        values.append(
            torch.maximum(quantile * residual, (quantile - 1.0) * residual)
            .flatten(1)
            .mean(1)
        )
    return torch.stack(values, dim=1).mean(1)


def _metric_vectors(particles: Tensor, weights: Tensor, target: Tensor) -> dict[str, Tensor]:
    weights = r4_losses.normalize_weights(weights)
    coverage_90, width_90 = r4_losses.empirical_coverage(
        particles, weights, target, 0.90
    )
    coverage_95, width_95 = r4_losses.empirical_coverage(
        particles, weights, target, 0.95
    )
    return {
        "energy_score": r4_losses.weighted_energy_score(particles, weights, target),
        "marginal_crps": r4_losses.weighted_marginal_crps(particles, weights, target),
        "variogram_score": r4_losses.weighted_variogram_score(particles, weights, target),
        "tail_score": _weighted_tail_score(particles, weights, target),
        "coverage_90": coverage_90,
        "coverage_95": coverage_95,
        "interval_width_90": width_90,
        "interval_width_95": width_95,
        "effective_sample_size": r4_losses.effective_sample_size(weights),
        "average_pairwise_particle_distance": r4_losses.mean_pairwise_particle_distance(
            particles
        ),
        "weight_entropy": -(weights * weights.clamp_min(1.0e-12).log()).sum(-1),
    }


def _identity_residual(particles: Tensor) -> Tensor:
    return (
        particles[..., 0:4] - particles[..., 4:8] - particles[..., 8:12]
    ).abs().flatten(1).amax(1)


def _audit_revealed_prefix_and_future_invariance(
    model: nn.Module,
    batch: Mapping[str, Any],
    target: Tensor,
    origins: Tensor,
) -> dict[str, Tensor]:
    """Audit revealed-prefix identity and future-realization invariance.

    The 179 validation dates are assigned rolling origins cyclically, so every
    origin in ``[0, 95]`` is exercised without multiplying the capability-gate
    workload by 96.  The counterfactual changes only the unrevealed suffix and
    preserves the DA/RT/DART identity.  A causal model must therefore emit the
    same current free action and the same revealed particle prefix.
    """

    batch_size = target.shape[0]
    if origins.shape != (batch_size,):
        raise CapabilityEvidenceError("scenario audit origins must have shape [B]")
    # The formal policy issues DA once at origin zero and treats it as immutable
    # thereafter.  A nonzero-origin causality audit must reproduce that exact
    # session contract instead of invoking the policy without its issued plan.
    origin_zero_mask = torch.arange(HORIZON, device=target.device)[None].eq(0)
    origin_zero_mask = origin_zero_mask.expand(batch_size, -1)
    origin_zero = dict(batch)
    origin_zero["origin"] = torch.zeros(
        (batch_size,), dtype=torch.long, device=target.device
    )
    origin_zero["revealed"] = {"zonal": target, "mask": origin_zero_mask}
    issued_da = torch.as_tensor(model(origin_zero)["policy"]["day_ahead"]).detach()

    periods = torch.arange(HORIZON, device=target.device)[None]
    mask = periods <= origins[:, None]
    audit = dict(batch)
    audit["origin"] = origins
    audit["revealed"] = {"zonal": target, "mask": mask}
    audit["issued_da_schedule"] = issued_da
    first = model(audit)
    first_particles = first["scenarios"]["particles_zonal"]

    prefix_mask = mask[:, None, :, None]
    prefix_residual = (
        (first_particles - target[:, None])
        .abs()
        .masked_fill(~prefix_mask, 0.0)
        .flatten(1)
        .amax(1)
    )

    # Construct a finite, coherent alternative future in raw units.  Only the
    # suffix changes; RT is then derived from DA-DART and positive channels stay
    # positive.  This catches accidental reads of the masked future while
    # avoiding an invalid counterfactual trajectory.
    alternative = target.clone()
    suffix = ~mask
    zone_shift = target.new_tensor((17.0, 29.0, 43.0, 61.0))[None, None, :]
    alternative[..., 0:4] = torch.where(
        suffix[..., None], alternative[..., 0:4] + zone_shift, alternative[..., 0:4]
    )
    alternative[..., 8:12] = torch.where(
        suffix[..., None], alternative[..., 8:12] - 0.5 * zone_shift, alternative[..., 8:12]
    )
    alternative[..., 4:8] = alternative[..., 0:4] - alternative[..., 8:12]
    positive_shift = target.new_tensor(
        (3.0, 0.7, 1.1, 1.7, 2.3, 0.03, 0.04, 0.05, 0.06)
    )[None, None, :]
    alternative[..., 12:21] = torch.where(
        suffix[..., None],
        (alternative[..., 12:21] + positive_shift).clamp_min(1.0e-6),
        alternative[..., 12:21],
    )

    perturbed = dict(batch)
    perturbed["origin"] = origins
    perturbed["revealed"] = {"zonal": alternative, "mask": mask}
    perturbed["issued_da_schedule"] = issued_da
    second = model(perturbed)
    second_particles = second["scenarios"]["particles_zonal"]
    particle_invariance = (
        (first_particles - second_particles)
        .abs()
        .masked_fill(~prefix_mask, 0.0)
        .flatten(1)
        .amax(1)
    )

    action_residual = target.new_zeros((batch_size,))
    for key in (
        "day_ahead",
        "battery",
        "compute",
        "fallback_confidence",
        "policy_confidence",
    ):
        left = torch.as_tensor(first["policy"][key], device=target.device)
        right = torch.as_tensor(second["policy"][key], device=target.device)
        if left.shape != right.shape or left.shape[0] != batch_size:
            raise CapabilityEvidenceError(
                f"scenario future-invariance action {key!r} has an invalid shape"
            )
        action_residual = torch.maximum(
            action_residual, (left - right).abs().flatten(1).amax(1)
        )

    return {
        "revealed_prefix_identity": prefix_residual,
        "future_action_invariance": action_residual,
        "future_prefix_particle_invariance": particle_invariance,
    }


class ScenarioCapabilityMaterializer:
    """Trainer-compatible callable that produces strict scenario gate evidence."""

    def __init__(
        self,
        config: Mapping[str, Any],
        repository_root: str | Path,
        artifact_root: str | Path,
        *,
        corpus: PreloadedR4Corpus | None = None,
        training_contract: ScenarioTrainingContract,
        batch_size: int = 4,
        audit_origin: int = 47,
        seed: int = 20240813,
    ) -> None:
        self.config = config
        self.repository_root = Path(repository_root).resolve()
        self.artifact_root = Path(artifact_root).resolve()
        self.corpus = corpus or PreloadedR4Corpus(config, self.repository_root)
        self.training_contract = training_contract
        self.batch_size = int(batch_size)
        self.audit_origin = int(audit_origin)
        self.seed = int(seed)
        if self.batch_size < 1:
            raise ValueError("scenario capability batch_size must be positive")
        if not 0 <= self.audit_origin < HORIZON:
            raise ValueError("scenario audit origin must lie in [0,95]")

    @property
    def capability_root(self) -> Path:
        return self.artifact_root / "capability" / "scenario"

    def _baseline(self) -> tuple[dict[str, Any], Path, str]:
        train = tuple(self.corpus.records("train"))
        validation = tuple(self.corpus.records("validation"))
        train_dates, _ = _validate_split_dates(train, validation)
        signature = _corpus_signature(train, train_dates)
        path = self.capability_root / "frozen_training_residual_baseline.pt"
        if path.exists():
            try:
                value = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                value = torch.load(path, map_location="cpu")
            if not isinstance(value, dict):
                raise CapabilityEvidenceError("frozen scenario baseline is malformed")
            if value.get("schema") != BASELINE_SCHEMA or value.get("corpus_sha256") != signature:
                raise CapabilityEvidenceError("frozen scenario baseline does not match the corpus")
            if tuple(value.get("train_dates", ())) != train_dates:
                raise CapabilityEvidenceError("frozen scenario baseline date lineage changed")
        else:
            targets = torch.stack([_target(record) for record in train])
            pue_center = targets[..., 17:21].median(dim=0).values
            centers = torch.stack([_issue_center(record, pue_center) for record in train])
            residuals = targets - centers
            scale = residuals.abs().flatten(0, 1).median(dim=0).values
            scale = torch.where(scale > 1.0e-6, scale, torch.ones_like(scale))
            value = {
                "schema": BASELINE_SCHEMA,
                "name": BASELINE_NAME,
                "fit_split": "2022-01-01_to_2023-06-30_only",
                "train_dates": train_dates,
                "corpus_sha256": signature,
                "sampling_seed": self.seed,
                "trajectory_sampling": "whole_day_joint_residual_without_replacement",
                "normalization": "train_residual_channel_median_absolute_scale",
                "pue_center": pue_center,
                "residuals": residuals,
                "scale": scale,
            }
            atomic_torch_save(path, value)
        residuals = torch.as_tensor(value.get("residuals"))
        scale = torch.as_tensor(value.get("scale"))
        pue_center = torch.as_tensor(value.get("pue_center"))
        if residuals.shape != (543, HORIZON, CHANNELS):
            raise CapabilityEvidenceError("frozen scenario residual pool has an invalid shape")
        if scale.shape != (CHANNELS,) or (scale <= 0).any() or not torch.isfinite(scale).all():
            raise CapabilityEvidenceError("frozen scenario metric scale is invalid")
        if pue_center.shape != (HORIZON, 4):
            raise CapabilityEvidenceError("frozen scenario PUE center is invalid")
        return value, path, sha256_file(path)

    def _sample_indices(self, operating_date: str, baseline_sha: str) -> Tensor:
        ordinal = date.fromisoformat(operating_date).toordinal()
        deterministic_seed = (
            int(baseline_sha[:16], 16) ^ int(self.seed) ^ int(ordinal)
        ) & ((1 << 63) - 1)
        generator = torch.Generator().manual_seed(deterministic_seed)
        return torch.randperm(543, generator=generator)[:SCENARIO_COUNT]

    def _materialize(
        self,
        model: nn.Module,
        checkpoint: Path,
        checkpoint_sha: str,
        state_sha: str,
        epoch: int,
        baseline: Mapping[str, Any],
        baseline_path: Path,
        baseline_sha: str,
    ) -> dict[str, Any]:
        train = tuple(self.corpus.records("train"))
        validation = tuple(self.corpus.records("validation"))
        _, validation_dates = _validate_split_dates(train, validation)
        residuals = torch.as_tensor(baseline["residuals"], dtype=torch.float32)
        scale_cpu = torch.as_tensor(baseline["scale"], dtype=torch.float32)
        pue_center = torch.as_tensor(baseline["pue_center"], dtype=torch.float32)
        if not torch.equal(
            scale_cpu, self.training_contract.score_scales.to(dtype=torch.float32)
        ):
            raise CapabilityEvidenceError(
                "Scenario gate score scale differs from the train-only contract"
            )
        device = _model_device(model)
        was_training = model.training
        model.eval()
        rows: list[dict[str, Any]] = []
        try:
            with torch.inference_mode():
                for start in range(0, len(validation), self.batch_size):
                    records = validation[start : start + self.batch_size]
                    dates = validation_dates[start : start + self.batch_size]
                    batch = move_batch(_collate(records), device)
                    batch["origin"] = torch.zeros(len(records), dtype=torch.long, device=device)
                    batch = attach_scenario_training_contract(
                        batch, self.training_contract, device=device
                    )
                    output = model(batch)
                    particles = torch.as_tensor(output["scenarios"]["particles_zonal"])
                    weights = torch.as_tensor(output["scenarios"]["weights"])
                    reported_caps = output["scenarios"].get(
                        "raw_unit_residual_caps_zonal"
                    )
                    if not isinstance(reported_caps, Tensor):
                        raise CapabilityEvidenceError(
                            "Scenario candidate did not report its raw-unit residual caps"
                        )
                    expected_caps = batch["scenario_residual_caps_zonal"].to(
                        device=device, dtype=reported_caps.dtype
                    )
                    if reported_caps.shape == (HORIZON, CHANNELS):
                        reported_caps = reported_caps[None].expand(
                            len(records), -1, -1
                        )
                    if expected_caps.shape == (HORIZON, CHANNELS):
                        expected_caps = expected_caps[None].expand(
                            len(records), -1, -1
                        )
                    if (
                        reported_caps.shape
                        != (len(records), HORIZON, CHANNELS)
                        or not torch.isfinite(reported_caps).all()
                        or not torch.allclose(
                            reported_caps,
                            expected_caps,
                            rtol=1.0e-6,
                            atol=1.0e-6,
                        )
                    ):
                        raise CapabilityEvidenceError(
                            "Scenario candidate residual caps differ from the train-only contract"
                        )
                    target = torch.as_tensor(
                        batch["scenario_target_zonal"], device=device, dtype=particles.dtype
                    )
                    exact_shape = particles.shape == (
                        len(records), SCENARIO_COUNT, HORIZON, CHANNELS
                    ) and weights.shape == (len(records), SCENARIO_COUNT)
                    finite_weighted = bool(
                        exact_shape
                        and torch.isfinite(particles).all()
                        and torch.isfinite(weights).all()
                        and (weights >= 0).all()
                        and torch.allclose(
                            weights.sum(-1),
                            torch.ones(len(records), device=device, dtype=weights.dtype),
                            atol=1.0e-6,
                        )
                    )
                    if not finite_weighted:
                        raise CapabilityEvidenceError(
                            "candidate did not emit finite weighted joint exact-20 scenarios"
                        )
                    weights = r4_losses.normalize_weights(weights)
                    scale = scale_cpu.to(device=device, dtype=particles.dtype)
                    learned_metrics = _metric_vectors(
                        particles / scale, weights, target / scale
                    )
                    learned_identity = _identity_residual(particles)
                    audit_origins = (
                        torch.arange(len(records), device=device, dtype=torch.long)
                        + len(rows)
                        + self.audit_origin
                    ).remainder(HORIZON)
                    causal_audit = _audit_revealed_prefix_and_future_invariance(
                        model, batch, target, audit_origins
                    )
                    prefix_residual = causal_audit["revealed_prefix_identity"]
                    action_invariance = causal_audit["future_action_invariance"]
                    particle_invariance = causal_audit[
                        "future_prefix_particle_invariance"
                    ]
                    baseline_particles_list: list[Tensor] = []
                    for record, operating_date in zip(records, dates, strict=True):
                        center = _issue_center(record, pue_center)
                        indices = self._sample_indices(operating_date, baseline_sha)
                        sampled = center[None] + residuals.index_select(0, indices)
                        sampled[..., 12:17] = sampled[..., 12:17].clamp_min(0.0)
                        sampled[..., 17:21] = sampled[..., 17:21].clamp_min(1.0e-4)
                        # Preserve exact identity even after defensive transforms.
                        sampled[..., 4:8] = sampled[..., 0:4] - sampled[..., 8:12]
                        baseline_particles_list.append(sampled)
                    baseline_particles = torch.stack(baseline_particles_list).to(
                        device=device, dtype=particles.dtype
                    )
                    baseline_weights = particles.new_full(
                        (len(records), SCENARIO_COUNT), 1.0 / SCENARIO_COUNT
                    )
                    baseline_metrics = _metric_vectors(
                        baseline_particles / scale, baseline_weights, target / scale
                    )
                    for local, operating_date in enumerate(dates):
                        row: dict[str, Any] = {
                            "date": operating_date,
                            "checkpoint_sha256": checkpoint_sha,
                            "model_state_sha256": state_sha,
                            "baseline_name": BASELINE_NAME,
                            "baseline_sha256": baseline_sha,
                            "scenario_training_contract_sha256": (
                                self.training_contract.artifact_sha256
                            ),
                            "finite_weighted_joint_exact20": True,
                            "revealed_prefix_identity_passed": bool(
                                float(prefix_residual[local]) <= 1.0e-6
                            ),
                            "future_leakage_audit_origin": int(audit_origins[local]),
                            "future_leakage_audit_status": (
                                "PASSED"
                                if float(action_invariance[local]) <= 1.0e-6
                                and float(particle_invariance[local]) <= 1.0e-6
                                else "FAILED"
                            ),
                            "future_action_invariance_max_abs": float(
                                action_invariance[local]
                            ),
                            "future_revealed_prefix_particle_invariance_max_abs": float(
                                particle_invariance[local]
                            ),
                            "model_revealed_prefix_identity_max_abs": float(
                                prefix_residual[local]
                            ),
                            "model_da_rt_dart_identity_max_abs": float(
                                learned_identity[local]
                            ),
                        }
                        for name, values in learned_metrics.items():
                            row[f"model_{name}"] = float(values[local])
                        for name in (
                            "energy_score",
                            "marginal_crps",
                            "variogram_score",
                            "tail_score",
                        ):
                            row[f"baseline_{name}"] = float(baseline_metrics[name][local])
                        row["evidence_sha256"] = sha256_json(row)
                        rows.append(row)
        finally:
            model.train(was_training)
        decision = evaluate_scenario(rows, seed=self.seed)
        return {
            "schema": SCHEMA,
            "phase": "scenario",
            "epoch": int(epoch),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "model_state_sha256": state_sha,
            "baseline": str(baseline_path),
            "baseline_sha256": baseline_sha,
            "scenario_training_contract": str(self.training_contract.path),
            "scenario_training_contract_sha256": (
                self.training_contract.artifact_sha256
            ),
            "validation_window": "2023-H2",
            "validation_days": len(rows),
            "scenario_count": SCENARIO_COUNT,
            "metric_space": "train_residual_channel_median_absolute_scaled_joint_space",
            "audit_origin_schedule": "cyclic_all_96_origins_across_179_dates",
            "audit_origin_offset": self.audit_origin,
            "future_leakage_audit_status": (
                "PASSED"
                if len(rows) == EXPECTED_VALIDATION_DAYS
                and {int(row["future_leakage_audit_origin"]) for row in rows}
                == set(range(HORIZON))
                and all(
                    row["future_leakage_audit_status"] == "PASSED" for row in rows
                )
                else "FAILED"
            ),
            "daily_rows": rows,
            "gate": decision.as_evaluator_payload(),
        }

    def __call__(
        self,
        *,
        phase: str,
        horizon: int | None,
        checkpoint: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        epoch: int | None = None,
        model: nn.Module | None = None,
        **_: Any,
    ) -> Mapping[str, Any]:
        if str(phase).lower() != "scenario" or int(horizon or -1) != HORIZON:
            raise CapabilityEvidenceError("scenario materializer requires phase=scenario and K=96")
        if model is None:
            raise CapabilityEvidenceError("scenario materializer requires the reloaded candidate model")
        supplied = checkpoint_path if checkpoint_path is not None else checkpoint
        if supplied is None:
            raise CapabilityEvidenceError("scenario materializer requires a candidate checkpoint")
        source = Path(supplied).resolve()
        checkpoint_sha, state_sha, payload = _load_checkpoint_binding(source, model, epoch)
        completed = int(payload.get("epochs_completed", int(payload.get("epoch", -1)) + 1))
        baseline, baseline_path, baseline_sha = self._baseline()
        if baseline_sha != self.training_contract.frozen_baseline_sha256:
            raise CapabilityEvidenceError(
                "Scenario gate baseline differs from the training contract baseline"
            )
        destination = (
            self.capability_root
            / "checkpoints"
            / checkpoint_sha
            / "daily_evidence_v2.json"
        )
        if destination.exists():
            materialized = read_json(destination)
            if (
                materialized.get("schema") != SCHEMA
                or materialized.get("checkpoint_sha256") != checkpoint_sha
                or materialized.get("model_state_sha256") != state_sha
                or materialized.get("baseline_sha256") != baseline_sha
                or materialized.get("scenario_training_contract_sha256")
                != self.training_contract.artifact_sha256
            ):
                raise CapabilityEvidenceError("persisted scenario evidence binding is invalid")
            rows = materialized.get("daily_rows")
            if not isinstance(rows, list):
                raise CapabilityEvidenceError("persisted scenario daily evidence is malformed")
            decision = evaluate_scenario(rows, seed=self.seed)
            result = decision.as_evaluator_payload()
        else:
            materialized = self._materialize(
                model,
                source,
                checkpoint_sha,
                state_sha,
                completed,
                baseline,
                baseline_path,
                baseline_sha,
            )
            atomic_json(destination, materialized)
            result = dict(materialized["gate"])
        result.update(
            {
                "evidence_path": str(destination),
                "evidence_sha256": sha256_file(destination),
                "checkpoint_sha256": checkpoint_sha,
                "baseline_sha256": baseline_sha,
                "scenario_training_contract_sha256": (
                    self.training_contract.artifact_sha256
                ),
            }
        )
        return result


__all__ = [
    "BASELINE_NAME",
    "BASELINE_SCHEMA",
    "SCHEMA",
    "ScenarioCapabilityMaterializer",
]
