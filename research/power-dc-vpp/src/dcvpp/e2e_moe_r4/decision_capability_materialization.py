"""Checkpoint-bound 179-day hard-gate evidence for R4 decision phases.

This module is deliberately executable rather than a JSON adapter.  A policy,
A1, or A2 K96 checkpoint is replayed through :class:`R4Runtime` at every one of
the 17,184 validation origins.  The learned method and the frozen Price-aware
EDF benchmark share the same analytic shield and the same independent
``validate_record`` audit.  Only the issue context is normalized; official
realized DA/RT settlement remains in raw units inside the runtime.

The implementation opens explicitly named 2023-H2 files only.  It neither
discovers nor accepts a 2024/2025 path and contains no mathematical-solver
dependency.
"""

from __future__ import annotations

from datetime import date
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .artifact_io import atomic_json, atomic_parquet
from .capability_evaluation import build_capability_evaluator
from .capability_materialization import LiveRepresentationCapabilityEvaluator
from .constrained_runtime import R4Runtime, validate_record
from .data import R4DataError, R4SelectionDay, load_r4_day, temporal_split
from .hashing import sha256_file, sha256_json
from .post_shield_particle_replay import (
    DISTRIBUTION_REPLAY_SCHEMA_VERSION,
    FORMAL_EMPIRICAL_RISK_SAMPLES,
    PostShieldDistributionReplay,
    PostShieldParticleReplay,
    PostShieldParticleReplayError,
    REPLAY_SCHEMA_VERSION,
    replay_post_shield_distribution_risk,
)
from .scenario_risk import exact_weighted_cvar
from .statistics import paired_daily_inference
from .training_data import PreloadedR4Corpus, normalize_runtime_batch


DECISION_PHASES = frozenset({"policy", "joint_policy", "a1", "a2"})
_DECISION_EVIDENCE_SCHEMA = "dcvpp.r4.decision_capability_daily.v4"
_PAIRED_PARTICLE_REPLAY_SCHEMA = "dcvpp.r4.paired_post_shield_particle_replay.v1"
_PAIRED_DISTRIBUTION_REPLAY_SCHEMA = (
    "dcvpp.r4.v6.paired_post_shield_distribution_replay.v1"
)
_SINGLE_DISTRIBUTION_REPLAY_SCHEMA = (
    "dcvpp.r4.v6.single_post_shield_distribution_replay.v1"
)
_DISTRIBUTIONAL_ARCHITECTURE_VERSION = "continuous_distributional_joint_moe_r4_v6"
_A1_PREFIXES = (
    "forecast_decoder.dart_head.",
    "forecast_decoder.dart_calibration_",
    "scenario_generator.dart_adapter.",
)
_ACTION_FIELDS = (
    "day_ahead",
    "battery",
    "compute",
    "migration",
)


def _is_distributional_model(model: nn.Module) -> bool:
    return str(getattr(model, "architecture_version", "")) == (
        _DISTRIBUTIONAL_ARCHITECTURE_VERSION
    )


def _sampling_seed(*, checkpoint_sha256: str, operating_date: str, count: int) -> int:
    """Deterministic pre-result seed derived only from frozen identifiers."""

    digest = sha256_json(
        {
            "schema": "dcvpp.r4.v6.decision_sampling_seed.v1",
            "checkpoint_sha256": str(checkpoint_sha256).lower(),
            "operating_date": str(operating_date),
            "sample_count": int(count),
        }
    )
    return int(digest[:16], 16) % (2**63 - 1)


def _a1_trainable_names(model: nn.Module) -> tuple[str, ...]:
    if _is_distributional_model(model) and hasattr(
        model, "phase_trainable_parameter_names"
    ):
        names = tuple(model.phase_trainable_parameter_names("a1"))
        if names:
            return names
    return tuple(
        name
        for name, _ in model.named_parameters()
        if any(name.startswith(prefix) for prefix in _A1_PREFIXES)
    )


def _distribution_support_from_config(
    config: Mapping[str, Any], repository_root: Path
) -> dict[str, Any] | None:
    project = config.get("project")
    architecture = project.get("architecture_version") if isinstance(project, Mapping) else None
    if architecture != _DISTRIBUTIONAL_ARCHITECTURE_VERSION:
        return None
    sampling = config.get("sampling")
    if (
        not isinstance(sampling, Mapping)
        or int(sampling.get("formal_empirical_risk", -1))
        != FORMAL_EMPIRICAL_RISK_SAMPLES
    ):
        raise ValueError("V6 formal empirical-risk sampling budget must be N=512")
    lineage = config.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("V6 config.lineage must declare Distribution support")
    path = _resolve(
        repository_root,
        lineage.get("distribution_support_artifact"),
        "lineage.distribution_support_artifact",
    )
    from .distribution_capability_materialization import (
        load_distribution_support_contract,
    )

    contract = load_distribution_support_contract(path)
    return {
        "lower_zonal": contract.lower_zonal,
        "upper_zonal": contract.upper_zonal,
        "contract_sha256": contract.artifact_sha256,
    }


def _resolve(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise R4DataError(f"decision evaluator requires explicit {label}")
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _tensor(value: Any) -> Tensor:
    return torch.as_tensor(value).detach().cpu()


def _numpy(value: Any) -> np.ndarray:
    return _tensor(value).numpy().copy()


def _all_finite(value: Any) -> bool:
    if isinstance(value, Tensor):
        return bool(torch.isfinite(value.detach()).all())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all()) if value.dtype.kind in "fciu" else True
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _is_sha256(value: Any) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _equal_state(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return False
        return left.shape == right.shape and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        if not isinstance(left, Tensor) or not isinstance(right, Tensor):
            return False
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(
            _equal_state(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            isinstance(left, type(right))
            and len(left) == len(right)
            and all(_equal_state(a, b) for a, b in zip(left, right))
        )
    return left == right


def _checkpoint_audit(
    source: Path,
    *,
    phase: str,
    seed: int,
    epoch: int,
    model: nn.Module,
    config_sha256: str,
) -> tuple[dict[str, Any], dict[str, bool]]:
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older supported Torch
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("decision candidate checkpoint has no model_state")
    checkpoint = dict(payload)
    state = checkpoint["model_state"]
    if not _equal_state(model.state_dict(), state):
        raise ValueError("loaded model does not exactly equal the candidate checkpoint")
    expected_dates = tuple(str(item) for item in checkpoint.get("expected_training_dates", ()))
    lineage = bool(
        checkpoint.get("phase") == phase
        and int(checkpoint.get("seed", -1)) == int(seed)
        and int(checkpoint.get("epochs_completed", -1)) == int(epoch)
        and int(checkpoint.get("horizon", -1)) == 96
        and checkpoint.get("config_sha256") == config_sha256
        and bool(checkpoint.get("protocol_version"))
        and bool(checkpoint.get("lineage_id"))
        and len(expected_dates) == 543
        and len(set(expected_dates)) == 543
        and all(temporal_split(item) == "train" for item in expected_dates)
    )
    resumable = bool(
        lineage
        and isinstance(checkpoint.get("optimizer_state"), Mapping)
        and "torch_rng_state" in checkpoint
        and "cuda_rng_state_all" in checkpoint
        and _all_finite(checkpoint["model_state"])
        and _all_finite(checkpoint["optimizer_state"])
        and _equal_state(model.state_dict(), checkpoint["model_state"])
    )
    return checkpoint, {
        "checkpoint_lineage_valid": lineage,
        "checkpoint_resume_exact": resumable,
    }


class _NormalizedDeploymentModel(nn.Module):
    """Bind one raw runtime day to its frozen train-only normalization."""

    def __init__(
        self,
        model: nn.Module,
        corpus: PreloadedR4Corpus,
        record: Mapping[str, Any],
        *,
        distribution_support: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.corpus = corpus
        self.record = record
        self.is_distributional = _is_distributional_model(model)
        self.distribution_support = (
            dict(distribution_support) if distribution_support is not None else None
        )
        self.latencies_ms: list[float] = []
        self.scenario_contract_passed = True
        self.scenario_count_by_origin: list[int] = []
        self.scenario_weight_sum_max_abs = 0.0
        self.scenario_identity_max_abs = 0.0
        self.scenario_revealed_prefix_max_abs = 0.0
        self.origin_zero_costs: Tensor | None = None
        self.origin_zero_weights: Tensor | None = None
        self.origin_zero_particles: Tensor | None = None
        self.origin_zero_distribution_output: Mapping[str, Any] | None = None
        self.origin_zero_revealed_zonal: Tensor | None = None
        self.origin_zero_revealed_mask: Tensor | None = None
        self.distribution_sampling_latencies_ms: list[float] = []
        self.raw_actions: dict[int, dict[str, np.ndarray]] = {}
        self.causal_input_contract_by_origin: list[bool] = []
        self.forecast_residual_cap_contract = False
        self.distributional_policy_contract_passed = self.is_distributional
        self.policy_particle_input_by_origin: list[bool] = []

    def prepare_issue_context(self, raw_batch: Mapping[str, Any]) -> Mapping[str, Any]:
        batch = normalize_runtime_batch(raw_batch, self.corpus.statistics)
        reference = next(self.model.parameters())

        def day_tensor(name: str) -> Tensor:
            value = self.record[name]
            tensor = torch.as_tensor(value, device=reference.device)
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=reference.dtype)
            return tensor.unsqueeze(0)

        for name in ("historical_market", "historical_market_available"):
            batch[name] = day_tensor(name)
        anchors = self.record.get("forecast_anchors")
        if not isinstance(anchors, Mapping):
            raise ValueError(
                "deployment record lacks the training-identical forecast_anchors contract"
            )
        anchor_tensors: dict[str, Tensor] = {}
        for name in ("da", "pv"):
            value = anchors.get(name)
            if value is None:
                raise ValueError(f"deployment forecast_anchors.{name} is absent")
            tensor = torch.as_tensor(
                value, device=reference.device, dtype=reference.dtype
            )
            if tensor.shape != (96, 4) or not bool(torch.isfinite(tensor).all()):
                raise ValueError(
                    f"deployment forecast_anchors.{name} must be finite [96,4]"
                )
            anchor_tensors[name] = tensor.unsqueeze(0)
        available = self.record.get("causal_da_anchor_available")
        if available is not None and not bool(torch.as_tensor(available).item()):
            raise ValueError("formal deployment requires a completed causal DA anchor")
        batch["forecast_anchors"] = anchor_tensors
        batch["causal_da_anchor_available"] = torch.ones(
            1, device=reference.device, dtype=torch.bool
        )
        caps = batch.get("forecast_residual_caps")
        metadata = batch.get("forecast_residual_cap_metadata")
        record_caps = self.record.get("forecast_residual_caps")
        record_metadata = self.record.get("forecast_residual_cap_metadata")
        if (
            not isinstance(caps, Mapping)
            or not isinstance(metadata, Mapping)
            or not isinstance(record_caps, Mapping)
            or not isinstance(record_metadata, Mapping)
        ):
            raise ValueError("deployment lacks train-frozen forecast residual caps")
        for name in ("da", "pv", "pue"):
            value = torch.as_tensor(
                caps.get(name), device=reference.device, dtype=reference.dtype
            )
            recorded = torch.as_tensor(
                record_caps.get(name), device=reference.device, dtype=reference.dtype
            )
            if value.shape != (4,) or not bool(
                torch.isfinite(value).all() and (value > 0.0).all()
            ):
                raise ValueError(
                    f"deployment forecast_residual_caps.{name} must be positive [4]"
                )
            if recorded.shape != (4,) or not torch.equal(value, recorded):
                raise ValueError(
                    f"deployment forecast_residual_caps.{name} differs from frozen record"
                )
        if (
            metadata.get("fit_split") != "train"
            or dict(record_metadata) != dict(metadata)
            or metadata.get("sha256")
            != self.corpus.statistics.forecast_residual_cap_sha256()
        ):
            raise ValueError("forecast residual caps were not frozen on train")
        self.forecast_residual_cap_contract = True
        for name in ("day_of_week", "month", "holiday", "dst"):
            value = torch.as_tensor(self.record[name], device=reference.device)
            batch[name] = value.reshape(1)
        if self.is_distributional:
            if not isinstance(self.distribution_support, Mapping):
                raise ValueError(
                    "V6 deployment requires the frozen Distribution support contract"
                )
            lower = torch.as_tensor(
                self.distribution_support.get("lower_zonal"),
                device=reference.device,
                dtype=reference.dtype,
            )
            upper = torch.as_tensor(
                self.distribution_support.get("upper_zonal"),
                device=reference.device,
                dtype=reference.dtype,
            )
            contract_sha = str(
                self.distribution_support.get("contract_sha256", "")
            ).lower()
            if (
                lower.shape != (96, 21)
                or upper.shape != (96, 21)
                or not bool(torch.isfinite(lower).all())
                or not bool(torch.isfinite(upper).all())
                or bool((lower > upper).any())
                or not _is_sha256(contract_sha)
            ):
                raise ValueError("V6 deployment Distribution support is malformed")
            batch["distribution_support"] = {
                "schema": "r4_v6_distribution_support.v1",
                "lower_zonal": lower,
                "upper_zonal": upper,
                "contract_sha256": contract_sha,
            }
        return self.model.prepare_issue_context(batch)

    def _runtime_state(self, session_input: Mapping[str, Any]) -> dict[str, Any]:
        prepared = dict(session_input)
        state_value = session_input.get("state")
        state = dict(state_value) if isinstance(state_value, Mapping) else {}
        reference = next(self.model.parameters())
        origin = int(session_input.get("origin", 0))
        if state.get("current_soc") is not None:
            energy = torch.as_tensor(
                np.array(self.corpus.physical.battery_energy_mwh, copy=True),
                device=reference.device,
                dtype=reference.dtype,
            ).clamp_min(1.0e-6)
            state["current_soc"] = torch.as_tensor(
                state["current_soc"], device=reference.device, dtype=reference.dtype
            ) / energy
        revealed_rt = state.get("revealed_rt")
        revealed_workload = state.get("revealed_workload")
        revealed_pv = state.get("revealed_pv")
        market_raw = torch.as_tensor(
            self.record["market_raw"], device=reference.device, dtype=reference.dtype
        )
        workload_raw = torch.as_tensor(
            self.record["issue_workload_raw"], device=reference.device, dtype=reference.dtype
        )
        pv_raw = torch.as_tensor(
            self.record["issue_pv_raw"], device=reference.device, dtype=reference.dtype
        )
        if revealed_rt is not None:
            state["rt_innovation"] = torch.as_tensor(
                revealed_rt, device=reference.device, dtype=reference.dtype
            )[:, origin] - market_raw[origin, 4:8]
        if revealed_workload is not None:
            state["workload_innovation"] = torch.as_tensor(
                revealed_workload, device=reference.device, dtype=reference.dtype
            )[:, origin] - workload_raw[origin]
        if revealed_pv is not None:
            state["pv_innovation"] = torch.as_tensor(
                revealed_pv, device=reference.device, dtype=reference.dtype
            )[:, origin] - pv_raw[origin]
        prepared["state"] = state
        prepared["current_soc"] = state.get("current_soc", prepared.get("current_soc"))
        for name in ("rt_innovation", "workload_innovation", "pv_innovation"):
            if name in state:
                prepared[name] = state[name]
        return prepared

    @staticmethod
    def _synchronize(model: nn.Module) -> None:
        reference = next(model.parameters())
        if reference.is_cuda:
            torch.cuda.synchronize(reference.device)

    @staticmethod
    def _causal_input_contract(session_input: Mapping[str, Any]) -> bool:
        """Prove that every model-visible realization is prefix-masked."""

        origin = int(session_input.get("origin", -1))
        if not 0 <= origin < 96:
            return False
        state = session_input.get("state")
        revealed = session_input.get("revealed")
        if not isinstance(state, Mapping) or not isinstance(revealed, Mapping):
            return False
        mask = _numpy(revealed.get("mask")).astype(bool, copy=False)
        zonal = _numpy(revealed.get("zonal"))
        expected = (np.arange(96) <= origin)[None, :]
        if mask.shape != (1, 96) or not np.array_equal(mask, expected):
            return False
        if zonal.shape != (1, 96, 21) or not np.allclose(
            zonal[:, origin + 1 :], 0.0, atol=0.0, rtol=0.0
        ):
            return False
        for name, width in (("revealed_rt", 4), ("revealed_workload", 1), ("revealed_pv", 4)):
            value = _numpy(state.get(name))
            if value.shape != (1, 96, width) or not np.allclose(
                value[:, origin + 1 :], 0.0, atol=0.0, rtol=0.0
            ):
                return False
        forbidden = {
            "targets",
            "rollout_targets",
            "realized_rt_price",
            "realized_workload",
            "realized_pv",
            "realized_pue",
        }
        return not any(name in session_input for name in forbidden)

    def _capture(self, output: Mapping[str, Any], session_input: Mapping[str, Any]) -> None:
        origin = int(session_input.get("origin", 0))
        policy = output.get("policy", output.get("actions", {}))
        actions = policy
        if self.is_distributional and isinstance(policy, Mapping):
            actions = policy.get("free_actions")
            if not isinstance(actions, Mapping):
                self.distributional_policy_contract_passed = False
                raise ValueError("V6 Policy must expose the explicit free_actions mapping")
        if isinstance(actions, Mapping):
            self.raw_actions[origin] = {
                name: _numpy(actions[name]) for name in _ACTION_FIELDS if actions.get(name) is not None
            }
        if self.is_distributional:
            distribution = output.get("distribution")
            scenario = output.get("scenarios")
            particle_visible = bool(
                isinstance(scenario, Mapping)
                or (
                    isinstance(distribution, Mapping)
                    and any(
                        name in distribution
                        for name in (
                            "samples_zonal",
                            "particles_zonal",
                            "particles",
                        )
                    )
                )
            )
            input_contract = (
                str(policy.get("policy_input_contract", ""))
                if isinstance(policy, Mapping)
                else ""
            )
            latent_ok = bool(
                isinstance(distribution, Mapping)
                and distribution.get("sampling_axis_present") is False
                and distribution.get("fixed_particle_capacity") is False
                and distribution.get("world_latent") is not None
            )
            no_particle_contract = bool(
                not particle_visible
                and input_contract.endswith("no_particles")
                and latent_ok
            )
            self.policy_particle_input_by_origin.append(particle_visible)
            self.distributional_policy_contract_passed = bool(
                self.distributional_policy_contract_passed and no_particle_contract
            )
            if not no_particle_contract:
                raise ValueError(
                    "V6 decision callback must be world-latent based and emit no particles"
                )
            if origin == 0:
                self.origin_zero_distribution_output = output
                revealed = session_input.get("revealed")
                if isinstance(revealed, Mapping):
                    self.origin_zero_revealed_zonal = _tensor(revealed.get("zonal"))
                    self.origin_zero_revealed_mask = _tensor(
                        revealed.get("mask")
                    ).bool()
            return
        scenario = output.get("scenarios")
        if not isinstance(scenario, Mapping):
            self.scenario_contract_passed = False
            return
        particles = _tensor(scenario.get("particles_zonal", scenario.get("particles")))
        weights = _tensor(scenario.get("weights"))
        if particles.shape != (1, 20, 96, 21) or weights.shape != (1, 20):
            raise ValueError("formal decision callback must expose [1,20,96,21] and [1,20]")
        if not torch.isfinite(particles).all() or not torch.isfinite(weights).all():
            raise ValueError("model scenarios or weights are non-finite")
        weight_residual = float((weights.sum(-1) - 1.0).abs().max())
        self.scenario_weight_sum_max_abs = max(
            self.scenario_weight_sum_max_abs, weight_residual
        )
        if bool((weights < 0).any()) or not torch.allclose(
            weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1.0e-6
        ):
            raise ValueError("model scenario weights are not normalized nonnegative weights")
        identity = particles[..., 0:4] - particles[..., 4:8] - particles[..., 8:12]
        identity_residual = float(identity.abs().max())
        self.scenario_identity_max_abs = max(
            self.scenario_identity_max_abs, identity_residual
        )
        if identity_residual > 1.0e-5:
            self.scenario_contract_passed = False
        revealed = session_input.get("revealed")
        if isinstance(revealed, Mapping):
            zonal = _tensor(revealed.get("zonal"))
            mask = _tensor(revealed.get("mask")).bool()
            if zonal.shape == (1, 96, 21) and mask.shape == (1, 96):
                expected = zonal[:, None].expand_as(particles)
                expanded = mask[:, None, :, None].expand_as(particles)
                if expanded.any():
                    prefix_residual = float(
                        (particles[expanded] - expected[expanded]).abs().max()
                    )
                    self.scenario_revealed_prefix_max_abs = max(
                        self.scenario_revealed_prefix_max_abs, prefix_residual
                    )
                    if prefix_residual > 1.0e-5:
                        self.scenario_contract_passed = False
        self.scenario_count_by_origin.append(int(particles.shape[1]))
        if origin == 0:
            risk = output.get("risk")
            costs = risk.get("scenario_costs") if isinstance(risk, Mapping) else None
            if costs is None or _tensor(costs).shape != (1, 20):
                raise ValueError("origin-zero equation-based risk must expose [1,20] costs")
            self.origin_zero_costs = _tensor(costs).float()
            self.origin_zero_weights = weights.float()
            self.origin_zero_particles = particles.float()

    def sample_origin_zero_distribution(
        self, sample_count: int, *, seed: int
    ) -> tuple[Tensor, Tensor]:
        """Independently sample the cached V6 world; Policy is not re-entered."""

        if not self.is_distributional:
            raise PostShieldParticleReplayError(
                "arbitrary-N Distribution sampling is unavailable for V5"
            )
        if self.origin_zero_distribution_output is None:
            raise PostShieldParticleReplayError(
                "origin-zero Distributional World Latent was not captured"
            )
        sampler = getattr(self.model, "sample_distribution", None)
        if not callable(sampler):
            raise PostShieldParticleReplayError(
                "V6 model does not expose sample_distribution(N)"
            )
        self._synchronize(self.model)
        started = time.perf_counter()
        with torch.inference_mode():
            sampled = sampler(
                self.origin_zero_distribution_output,
                int(sample_count),
                seed=int(seed),
                revealed_zonal=self.origin_zero_revealed_zonal,
                revealed_mask=self.origin_zero_revealed_mask,
            )
        self._synchronize(self.model)
        self.distribution_sampling_latencies_ms.append(
            (time.perf_counter() - started) * 1000.0
        )
        if not isinstance(sampled, Mapping):
            raise PostShieldParticleReplayError(
                "sample_distribution did not return a mapping"
            )
        samples = _tensor(
            sampled.get("samples_zonal", sampled.get("particles_zonal"))
        ).float()
        weights = _tensor(sampled.get("weights")).float()
        expected = (1, int(sample_count), 96, 21)
        if samples.shape != expected or weights.shape != (1, int(sample_count)):
            raise PostShieldParticleReplayError(
                f"Distribution sample must be {expected} with [1,N] weights"
            )
        if not bool(torch.isfinite(samples).all() and torch.isfinite(weights).all()):
            raise PostShieldParticleReplayError("Distribution sample is non-finite")
        return samples, weights

    def forward_session(self, session_input: Mapping[str, Any]) -> Mapping[str, Any]:
        self.causal_input_contract_by_origin.append(
            self._causal_input_contract(session_input)
        )
        prepared = self._runtime_state(session_input)
        self._synchronize(self.model)
        started = time.perf_counter()
        output = self.model.forward_session(prepared)
        self._synchronize(self.model)
        self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
        if not isinstance(output, Mapping):
            raise ValueError("R4 model forward_session did not return a mapping")
        self._capture(output, prepared)
        return output


class _PriceAwareEDF:
    """Frozen issue-time price rule; feasibility is supplied by R4Runtime's shield."""

    def __init__(
        self,
        day: R4SelectionDay,
        corpus: PreloadedR4Corpus,
        low_price: np.ndarray,
        high_price: np.ndarray,
    ) -> None:
        self.day = day
        self.physical = corpus.physical
        self.low_price = np.asarray(low_price, dtype=float)
        self.high_price = np.asarray(high_price, dtype=float)
        capacity = np.asarray(self.physical.capacity_work_units, dtype=float)
        issue_total = np.minimum(np.maximum(day.workload[:, 0], 0.0), capacity.sum())
        issue_work = issue_total[:, None] * capacity[None] / capacity.sum()
        plant = R4Runtime(
            self.physical,
            physical_twin_core=getattr(corpus, "physical_core", None),
        ).shield.plant
        load = np.stack(
            [plant.facility_load(day.physical_multiplier, row) for row in issue_work],
            axis=0,
        )
        issue_grid = np.maximum(load - day.physical_dynamic[:, :4], 0.0)
        self.day_ahead = np.minimum(
            issue_grid, self.physical.grid_import_limit_mw[None]
        ).T

    def __call__(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        origin = int(context["origin"])
        price = self.day.market[origin, 4:8]
        cheap = price <= self.low_price
        expensive = price >= self.high_price
        compute = np.where(cheap, self.physical.capacity_work_units, 0.0)
        battery = np.where(
            cheap,
            self.physical.battery_power_mw,
            np.where(expensive, -self.physical.battery_power_mw, 0.0),
        )
        action: dict[str, Any] = {"compute": compute, "battery": battery}
        if origin == 0:
            action["day_ahead"] = self.day_ahead
        return {"actions": action}


def _benchmark_definition(
    corpus: PreloadedR4Corpus,
    *,
    config_sha256: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    records = [*corpus.records("train"), *corpus.records("validation")]
    price = torch.cat([_tensor(row["market_raw"])[..., 4:8] for row in records], dim=0)
    low = torch.quantile(price, 0.25, dim=0).numpy()
    high = torch.quantile(price, 0.75, dim=0).numpy()
    dates = tuple(str(row["operating_date"]) for row in records)
    payload = {
        "schema_version": "dcvpp.r4.price_aware_edf_shield.v2",
        "name": "price_aware_edf_shield",
        "calibration_split": ["train", "validation"],
        "calibration_dates_sha256": sha256_json(dates),
        "calibration_date_count": len(dates),
        "low_rt_issue_price_quantile": 0.25,
        "high_rt_issue_price_quantile": 0.75,
        "low_rt_issue_price_by_zone": low.tolist(),
        "high_rt_issue_price_by_zone": high.tolist(),
        "compute_rule": "capacity_on_low_price_else_edf_minimum_from_identical_shield",
        "battery_rule": "charge_low_discharge_high_else_idle",
        "day_ahead_rule": "issue_mean_workload_capacity_proportional_net_issue_mean_pv",
        "shield": "dcvpp.e2e_moe_r4.constrained_runtime.AnalyticConstraintShield",
        "physical_sha256": corpus.physical_sha256,
        "config_sha256": config_sha256,
        "final_test_frozen": True,
    }
    return payload, low, high


def _valid_replay_evidence(value: Mapping[str, Any]) -> bool:
    unsigned = {key: item for key, item in value.items() if key != "evidence_sha256"}
    return value.get("evidence_sha256") == sha256_json(unsigned)


def _distribution_replay_passes(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("schema_version") == DISTRIBUTION_REPLAY_SCHEMA_VERSION
        and value.get("contract_passed") is True
        and value.get("formal_empirical_risk_sample_count")
        == FORMAL_EMPIRICAL_RISK_SAMPLES
        and value.get("particle_count") == FORMAL_EMPIRICAL_RISK_SAMPLES
        and value.get("origin_count_per_particle") == 96
        and value.get("policy_uses_distribution_samples") is False
        and value.get("all_particles_completed_96_origins") is True
        and value.get("all_particles_independently_validated") is True
        and _valid_replay_evidence(value)
    )


def _single_post_shield_distribution_replay(
    *,
    day: R4SelectionDay,
    actual_result: Mapping[str, Any],
    adapter: _NormalizedDeploymentModel,
    model: nn.Module,
    corpus: PreloadedR4Corpus,
    deployment_record: Mapping[str, Any],
    runtime_config: Mapping[str, Any] | None,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Materialize formal V6 empirical risk from one independent N=512 draw."""

    payload: dict[str, Any] = {
        "schema_version": _SINGLE_DISTRIBUTION_REPLAY_SCHEMA,
        "operating_date": day.operating_date,
        "distribution_contract_sha256": checkpoint_sha256,
        "learned_policy_contract_sha256": checkpoint_sha256,
        "replay_schema_version": DISTRIBUTION_REPLAY_SCHEMA_VERSION,
        "formal_empirical_risk_sample_count": FORMAL_EMPIRICAL_RISK_SAMPLES,
        "policy_uses_distribution_samples": False,
        "contract_passed": False,
    }
    try:
        records = tuple(actual_result.get("records", ()))
        if len(records) != 96:
            raise PostShieldParticleReplayError(
                "learned actual-day execution did not complete 96 origins"
            )
        seed = _sampling_seed(
            checkpoint_sha256=checkpoint_sha256,
            operating_date=day.operating_date,
            count=FORMAL_EMPIRICAL_RISK_SAMPLES,
        )
        samples, weights = adapter.sample_origin_zero_distribution(
            FORMAL_EMPIRICAL_RISK_SAMPLES, seed=seed
        )
        frozen_day_ahead = np.asarray(
            records[0].get("day_ahead_schedule_mw"), dtype=float
        )
        learned = replay_post_shield_distribution_risk(
            day=day,
            samples_zonal=_numpy(samples[0]),
            weights=_numpy(weights[0]),
            frozen_day_ahead=frozen_day_ahead,
            policy_callback_factory=lambda _index, _sample_day: (
                _NormalizedDeploymentModel(
                    model,
                    corpus,
                    deployment_record,
                    distribution_support=adapter.distribution_support,
                ).eval()
            ),
            physical=corpus.physical,
            policy_contract_sha256=checkpoint_sha256,
            distribution_contract_sha256=checkpoint_sha256,
            runtime_config=runtime_config,
            physical_twin_core=getattr(corpus, "physical_core", None),
        )
        passed = _distribution_replay_passes(learned)
        payload.update(
            {
                "sampling_seed": seed,
                "sampling_latency_ms": adapter.distribution_sampling_latencies_ms[-1],
                "samples_sha256": learned.get("particles_sha256", ""),
                "weights_sha256": learned.get("weights_sha256", ""),
                "scenario_sha256": learned.get("scenario_sha256", ""),
                "learned_replay": learned,
                "contract_passed": passed,
            }
        )
        if not passed:
            payload["failure_reason"] = (
                "N=512 Distribution replay did not complete the full post-shield contract"
            )
    except (PostShieldParticleReplayError, R4DataError, RuntimeError, ValueError) as exc:
        payload["status"] = "FAILED"
        payload["failure_reason"] = f"{type(exc).__name__}: {exc}"
    payload.setdefault("status", "COMPLETE" if payload["contract_passed"] else "FAILED")
    payload["evidence_sha256"] = sha256_json(payload)
    return payload


def _paired_post_shield_distribution_replay(
    *,
    day: R4SelectionDay,
    actual_result: Mapping[str, Any],
    adapter: _NormalizedDeploymentModel,
    model: nn.Module,
    corpus: PreloadedR4Corpus,
    deployment_record: Mapping[str, Any],
    runtime_config: Mapping[str, Any] | None,
    low_price: np.ndarray,
    high_price: np.ndarray,
    checkpoint_sha256: str,
    benchmark_sha256: str,
) -> dict[str, Any]:
    """Replay learned and benchmark policies on one shared V6 N=512 draw."""

    payload: dict[str, Any] = {
        "schema_version": _PAIRED_DISTRIBUTION_REPLAY_SCHEMA,
        "operating_date": day.operating_date,
        "distribution_contract_sha256": checkpoint_sha256,
        "learned_policy_contract_sha256": checkpoint_sha256,
        "benchmark_policy_contract_sha256": benchmark_sha256,
        "replay_schema_version": DISTRIBUTION_REPLAY_SCHEMA_VERSION,
        "formal_empirical_risk_sample_count": FORMAL_EMPIRICAL_RISK_SAMPLES,
        "policy_uses_distribution_samples": False,
        "contract_passed": False,
    }
    try:
        records = tuple(actual_result.get("records", ()))
        if len(records) != 96:
            raise PostShieldParticleReplayError(
                "learned actual-day execution did not complete 96 origins"
            )
        seed = _sampling_seed(
            checkpoint_sha256=checkpoint_sha256,
            operating_date=day.operating_date,
            count=FORMAL_EMPIRICAL_RISK_SAMPLES,
        )
        samples, weights = adapter.sample_origin_zero_distribution(
            FORMAL_EMPIRICAL_RISK_SAMPLES, seed=seed
        )
        samples_value = _numpy(samples[0])
        weights_value = _numpy(weights[0])
        learned_day_ahead = np.asarray(
            records[0].get("day_ahead_schedule_mw"), dtype=float
        )
        benchmark_template = _PriceAwareEDF(
            day, corpus, low_price=low_price, high_price=high_price
        )

        learned = replay_post_shield_distribution_risk(
            day=day,
            samples_zonal=samples_value,
            weights=weights_value,
            frozen_day_ahead=learned_day_ahead,
            policy_callback_factory=lambda _index, _sample_day: (
                _NormalizedDeploymentModel(
                    model,
                    corpus,
                    deployment_record,
                    distribution_support=adapter.distribution_support,
                ).eval()
            ),
            physical=corpus.physical,
            policy_contract_sha256=checkpoint_sha256,
            distribution_contract_sha256=checkpoint_sha256,
            runtime_config=runtime_config,
            physical_twin_core=getattr(corpus, "physical_core", None),
        )
        benchmark = replay_post_shield_distribution_risk(
            day=day,
            samples_zonal=samples_value,
            weights=weights_value,
            frozen_day_ahead=np.asarray(benchmark_template.day_ahead, dtype=float),
            policy_callback_factory=lambda _index, sample_day: _PriceAwareEDF(
                sample_day,
                corpus,
                low_price=low_price,
                high_price=high_price,
            ),
            physical=corpus.physical,
            policy_contract_sha256=benchmark_sha256,
            distribution_contract_sha256=checkpoint_sha256,
            runtime_config=runtime_config,
            physical_twin_core=getattr(corpus, "physical_core", None),
        )
        shared_fields = (
            "particles_sha256",
            "weights_sha256",
            "scenario_sha256",
            "scenario_contract_sha256",
            "day_contract_sha256",
            "physical_contract_sha256",
            "runtime_config_sha256",
        )
        common_source = all(
            learned.get(name) == benchmark.get(name) for name in shared_fields
        )
        passed = bool(
            common_source
            and _distribution_replay_passes(learned)
            and _distribution_replay_passes(benchmark)
        )
        payload.update(
            {
                "sampling_seed": seed,
                "sampling_latency_ms": adapter.distribution_sampling_latencies_ms[-1],
                "common_distribution_source": common_source,
                "common_particle_source": common_source,
                "samples_sha256": learned.get("particles_sha256", ""),
                "particles_sha256": learned.get("particles_sha256", ""),
                "weights_sha256": learned.get("weights_sha256", ""),
                "scenario_sha256": learned.get("scenario_sha256", ""),
                "learned_replay": learned,
                "benchmark_replay": benchmark,
                "contract_passed": passed,
            }
        )
        if not passed:
            payload["failure_reason"] = (
                "learned/baseline N=512 replay did not satisfy the shared-source contract"
            )
    except (PostShieldParticleReplayError, R4DataError, RuntimeError, ValueError) as exc:
        payload["status"] = "FAILED"
        payload["failure_reason"] = f"{type(exc).__name__}: {exc}"
    payload.setdefault("status", "COMPLETE" if payload["contract_passed"] else "FAILED")
    payload["evidence_sha256"] = sha256_json(payload)
    return payload


def _paired_post_shield_particle_replay(
    *,
    day: R4SelectionDay,
    actual_result: Mapping[str, Any],
    adapter: _NormalizedDeploymentModel,
    model: nn.Module,
    corpus: PreloadedR4Corpus,
    deployment_record: Mapping[str, Any],
    runtime_config: Mapping[str, Any] | None,
    low_price: np.ndarray,
    high_price: np.ndarray,
    checkpoint_sha256: str,
    benchmark_sha256: str,
) -> dict[str, Any]:
    """Replay learned and benchmark policies on one hash-identical particle set.

    The two policies retain their own origin-zero day-ahead schedules.  Their
    scenario and weight hashes, runtime, physical twin, and synthetic particle
    days must be identical.  Any absent or incomplete component is represented
    by a signed failure payload so the decision gate fails closed.
    """

    if adapter.is_distributional:
        return _paired_post_shield_distribution_replay(
            day=day,
            actual_result=actual_result,
            adapter=adapter,
            model=model,
            corpus=corpus,
            deployment_record=deployment_record,
            runtime_config=runtime_config,
            low_price=low_price,
            high_price=high_price,
            checkpoint_sha256=checkpoint_sha256,
            benchmark_sha256=benchmark_sha256,
        )

    payload: dict[str, Any] = {
        "schema_version": _PAIRED_PARTICLE_REPLAY_SCHEMA,
        "operating_date": day.operating_date,
        "scenario_contract_sha256": checkpoint_sha256,
        "learned_policy_contract_sha256": checkpoint_sha256,
        "benchmark_policy_contract_sha256": benchmark_sha256,
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "contract_passed": False,
    }
    try:
        particles_value = adapter.origin_zero_particles
        weights_value = adapter.origin_zero_weights
        if particles_value is None or weights_value is None:
            raise PostShieldParticleReplayError(
                "origin-zero learned particles or weights were not captured"
            )
        particles = _numpy(particles_value)
        weights = _numpy(weights_value)
        if particles.shape != (1, 20, 96, 21) or weights.shape != (1, 20):
            raise PostShieldParticleReplayError(
                "origin-zero replay source must be [1,20,96,21] and [1,20]"
            )
        records = tuple(actual_result.get("records", ()))
        if len(records) != 96:
            raise PostShieldParticleReplayError(
                "learned actual-day execution did not complete 96 origins"
            )
        learned_day_ahead = np.asarray(
            records[0].get("day_ahead_schedule_mw"), dtype=float
        )
        benchmark_template = _PriceAwareEDF(
            day, corpus, low_price=low_price, high_price=high_price
        )
        benchmark_day_ahead = np.asarray(benchmark_template.day_ahead, dtype=float)
        replay = PostShieldParticleReplay(
            corpus.physical,
            runtime_config,
            physical_twin_core=getattr(corpus, "physical_core", None),
        )

        learned_binding = replay.bind(
            day=day,
            particles_zonal=particles[0],
            weights=weights[0],
            frozen_day_ahead=learned_day_ahead,
            policy_contract_sha256=checkpoint_sha256,
            scenario_contract_sha256=checkpoint_sha256,
        )
        learned = replay.run(
            day=day,
            particles_zonal=particles[0],
            weights=weights[0],
            frozen_day_ahead=learned_day_ahead,
            policy_callback_factory=lambda _index, _particle_day: (
                _NormalizedDeploymentModel(model, corpus, deployment_record).eval()
            ),
            binding=learned_binding,
        )

        benchmark_binding = replay.bind(
            day=day,
            particles_zonal=particles[0],
            weights=weights[0],
            frozen_day_ahead=benchmark_day_ahead,
            policy_contract_sha256=benchmark_sha256,
            scenario_contract_sha256=checkpoint_sha256,
        )
        benchmark = replay.run(
            day=day,
            particles_zonal=particles[0],
            weights=weights[0],
            frozen_day_ahead=benchmark_day_ahead,
            policy_callback_factory=lambda _index, particle_day: _PriceAwareEDF(
                particle_day,
                corpus,
                low_price=low_price,
                high_price=high_price,
            ),
            binding=benchmark_binding,
        )

        def evidence_hash_valid(value: Mapping[str, Any]) -> bool:
            unsigned = {
                key: item for key, item in value.items() if key != "evidence_sha256"
            }
            return value.get("evidence_sha256") == sha256_json(unsigned)

        shared_fields = (
            "particles_sha256",
            "weights_sha256",
            "scenario_sha256",
            "scenario_contract_sha256",
            "day_contract_sha256",
            "physical_contract_sha256",
            "runtime_config_sha256",
        )
        common_source = all(
            learned.get(name) == benchmark.get(name) for name in shared_fields
        )
        passed = bool(
            learned.get("contract_passed") is True
            and benchmark.get("contract_passed") is True
            and evidence_hash_valid(learned)
            and evidence_hash_valid(benchmark)
            and common_source
            and learned.get("particle_count") == 20
            and benchmark.get("particle_count") == 20
            and learned.get("origin_count_per_particle") == 96
            and benchmark.get("origin_count_per_particle") == 96
        )
        payload.update(
            {
                "common_particle_source": common_source,
                "particles_sha256": learned.get("particles_sha256", ""),
                "weights_sha256": learned.get("weights_sha256", ""),
                "scenario_sha256": learned.get("scenario_sha256", ""),
                "day_contract_sha256": learned.get("day_contract_sha256", ""),
                "physical_contract_sha256": learned.get(
                    "physical_contract_sha256", ""
                ),
                "runtime_config_sha256": learned.get("runtime_config_sha256", ""),
                "learned_replay": learned,
                "benchmark_replay": benchmark,
                "contract_passed": passed,
            }
        )
        if not passed:
            payload["failure_reason"] = (
                "learned/baseline particle replay did not satisfy the shared-source contract"
            )
    except (PostShieldParticleReplayError, R4DataError, ValueError) as exc:
        payload["status"] = "FAILED"
        payload["failure_reason"] = f"{type(exc).__name__}: {exc}"
    payload.setdefault("status", "COMPLETE" if payload["contract_passed"] else "FAILED")
    payload["evidence_sha256"] = sha256_json(payload)
    return payload


def _run_summary(
    result: Mapping[str, Any],
    *,
    day: R4SelectionDay,
    corpus: PreloadedR4Corpus,
    adapter: _NormalizedDeploymentModel | None = None,
    particle_replay: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    records = tuple(dict(row) for row in result.get("records", ()))
    physical_core = getattr(corpus, "physical_core", None)
    independent_plant = R4Runtime(
        corpus.physical, physical_twin_core=physical_core
    ).shield.plant
    independent = [
        validate_record(
            row,
            physical=corpus.physical,
            day=day,
            physical_twin_core=physical_core,
            _physical_plant=independent_plant,
        )
        for row in records
    ]
    completed = len(records)
    feasible = sum(bool(item["passed"]) for item in independent)
    workload = float(np.asarray(day.realized_workload).sum())
    sla = float(sum(float(row["sla_shortfall_work_units"]) for row in records))
    costs = np.asarray([float(row["total_cost_usd"]) for row in records], dtype=float)
    corrections = np.asarray(
        [float(row["corrective_intervention_magnitude"]) for row in records], dtype=float
    )
    corrected = np.asarray(
        [bool(row["corrective_intervention"]) for row in records], dtype=bool
    )
    raw_violation_categories: dict[str, int] = {}
    for row in records:
        categories = row.get("corrective_intervention_categories", ())
        if isinstance(categories, str):
            categories = (categories,)
        for category in categories:
            name = str(category)
            raw_violation_categories[name] = raw_violation_categories.get(name, 0) + 1
    cvar = 0.0
    origin_zero_plan_cvar = 0.0
    latency = 0.0
    is_distributional = bool(adapter is not None and adapter.is_distributional)
    scenario_contract = adapter is None
    decision_uncertainty_contract = adapter is None
    if adapter is not None:
        if is_distributional:
            scenario_contract = False
            decision_uncertainty_contract = bool(
                adapter.distributional_policy_contract_passed
                and len(adapter.policy_particle_input_by_origin) == 96
                and not any(adapter.policy_particle_input_by_origin)
            )
        else:
            if adapter.origin_zero_costs is None or adapter.origin_zero_weights is None:
                scenario_contract = False
            else:
                origin_zero_plan_cvar = float(
                    exact_weighted_cvar(
                        adapter.origin_zero_costs,
                        adapter.origin_zero_weights,
                        alpha=0.95,
                    )[0]
                )
            scenario_contract = bool(
                adapter.scenario_contract_passed
                and len(adapter.scenario_count_by_origin) == 96
                and set(adapter.scenario_count_by_origin) == {20}
            )
            decision_uncertainty_contract = scenario_contract
        latency = float(np.mean(adapter.latencies_ms)) if adapter.latencies_ms else math.inf
    replay_value = dict(particle_replay or {})
    replay_unsigned = {
        key: value for key, value in replay_value.items() if key != "evidence_sha256"
    }
    learned_replay = replay_value.get("learned_replay")
    benchmark_replay = replay_value.get("benchmark_replay")
    replay_hash_valid = bool(
        replay_value
        and replay_value.get("evidence_sha256") == sha256_json(replay_unsigned)
    )
    legacy_post_shield_contract = bool(
        replay_hash_valid
        and replay_value.get("schema_version") == _PAIRED_PARTICLE_REPLAY_SCHEMA
        and replay_value.get("contract_passed") is True
        and replay_value.get("common_particle_source") is True
        and isinstance(learned_replay, Mapping)
        and isinstance(benchmark_replay, Mapping)
        and learned_replay.get("contract_passed") is True
        and benchmark_replay.get("contract_passed") is True
        and learned_replay.get("scenario_sha256")
        == benchmark_replay.get("scenario_sha256")
        and learned_replay.get("weights_sha256")
        == benchmark_replay.get("weights_sha256")
        and learned_replay.get("particle_count") == 20
        and benchmark_replay.get("particle_count") == 20
        and learned_replay.get("origin_count_per_particle") == 96
        and benchmark_replay.get("origin_count_per_particle") == 96
    )
    paired_distribution = bool(
        replay_hash_valid
        and replay_value.get("schema_version") == _PAIRED_DISTRIBUTION_REPLAY_SCHEMA
        and replay_value.get("contract_passed") is True
        and replay_value.get("common_distribution_source") is True
        and isinstance(learned_replay, Mapping)
        and isinstance(benchmark_replay, Mapping)
        and _distribution_replay_passes(learned_replay)
        and _distribution_replay_passes(benchmark_replay)
        and learned_replay.get("scenario_sha256")
        == benchmark_replay.get("scenario_sha256")
        and learned_replay.get("weights_sha256")
        == benchmark_replay.get("weights_sha256")
    )
    single_distribution = bool(
        replay_hash_valid
        and replay_value.get("schema_version") == _SINGLE_DISTRIBUTION_REPLAY_SCHEMA
        and replay_value.get("contract_passed") is True
        and isinstance(learned_replay, Mapping)
        and _distribution_replay_passes(learned_replay)
    )
    distribution_post_shield_contract = bool(
        paired_distribution or single_distribution
    )
    post_shield_cvar_contract = bool(
        legacy_post_shield_contract or distribution_post_shield_contract
    )
    benchmark_cvar = 0.0
    if post_shield_cvar_contract:
        cvar = float(learned_replay["weighted_cvar95_usd"])
        if isinstance(benchmark_replay, Mapping):
            benchmark_cvar = float(benchmark_replay["weighted_cvar95_usd"])
    else:
        # Retain the pre-shield plan diagnostic for failure analysis, but never
        # allow it to satisfy the operational post-shield CVaR contract.
        cvar = origin_zero_plan_cvar
    finite_values = np.asarray(
        [
            costs.sum(initial=0.0),
            sla,
            cvar,
            benchmark_cvar,
            origin_zero_plan_cvar,
            latency,
            corrections.sum(initial=0.0),
        ],
        dtype=float,
    )
    return {
        "completed_origins": completed,
        "independently_validated_origins": feasible,
        "post_shield_feasible_origins": feasible,
        "non_exogenous_rejected_origins": 96 - completed,
        "operating_cost_usd": float(costs.sum(initial=0.0)),
        "sla_shortfall_raw_work_units": sla,
        "normalized_sla_shortfall": sla / max(workload, 1.0e-8),
        "weighted_cvar95_usd": cvar,
        "benchmark_weighted_cvar95_usd": benchmark_cvar,
        "origin_zero_pre_shield_plan_cvar95_usd": origin_zero_plan_cvar,
        "inference_latency_ms": latency,
        "corrective_intervention_rate": float(corrected.mean()) if completed else 0.0,
        "corrective_intervention_mean": float(corrections.mean()) if completed else 0.0,
        "corrective_intervention_p95": (
            float(np.quantile(corrections, 0.95)) if completed else 0.0
        ),
        "raw_action_violation_categories": raw_violation_categories,
        "day_ahead_frozen": bool(result.get("day_ahead_frozen_after_origin_zero", False)),
        "all_actions_independently_validated": bool(
            result.get("all_actions_independently_validated", False)
            and completed == 96
            and feasible == 96
        ),
        "all_values_finite": bool(np.isfinite(finite_values).all()),
        "exact_20_model_scenario_contract": scenario_contract,
        "decision_uncertainty_contract": decision_uncertainty_contract,
        "distributional_policy_no_particle_contract": bool(
            is_distributional and decision_uncertainty_contract
        ),
        "policy_requires_fixed_particles": bool(
            adapter is not None and not is_distributional
        ),
        "post_shield_particle_cvar_contract": post_shield_cvar_contract,
        "post_shield_distribution_cvar_contract": distribution_post_shield_contract,
        "formal_empirical_risk_sample_count": (
            int(learned_replay.get("formal_empirical_risk_sample_count", 0))
            if isinstance(learned_replay, Mapping)
            else 0
        ),
        "distribution_sampling_latency_ms": (
            float(np.mean(adapter.distribution_sampling_latencies_ms))
            if adapter is not None and adapter.distribution_sampling_latencies_ms
            else 0.0
        ),
        "post_shield_particle_replay_schema_version": replay_value.get(
            "replay_schema_version", ""
        ),
        "post_shield_particle_replay_status": replay_value.get("status", "NOT_RUN"),
        "post_shield_particle_replay_evidence_sha256": replay_value.get(
            "evidence_sha256", ""
        ),
        "post_shield_particle_replay_failure_reason": replay_value.get(
            "failure_reason", ""
        ),
        "post_shield_particle_common_source": bool(
            replay_value.get(
                "common_distribution_source",
                replay_value.get("common_particle_source", False),
            )
        ),
        "post_shield_particle_particles_sha256": replay_value.get(
            "particles_sha256", ""
        ),
        "post_shield_particle_weights_sha256": replay_value.get(
            "weights_sha256", ""
        ),
        "post_shield_particle_scenario_sha256": replay_value.get(
            "scenario_sha256", ""
        ),
        "post_shield_particle_learned_replay_sha256": (
            learned_replay.get("evidence_sha256", "")
            if isinstance(learned_replay, Mapping)
            else ""
        ),
        "post_shield_particle_benchmark_replay_sha256": (
            benchmark_replay.get("evidence_sha256", "")
            if isinstance(benchmark_replay, Mapping)
            else ""
        ),
        "post_shield_particle_learned_costs_usd": (
            list(learned_replay.get("particle_day_costs_usd", ()))
            if isinstance(learned_replay, Mapping)
            else []
        ),
        "post_shield_particle_benchmark_costs_usd": (
            list(benchmark_replay.get("particle_day_costs_usd", ()))
            if isinstance(benchmark_replay, Mapping)
            else []
        ),
        "post_shield_particle_learned_sla_raw_work_units": (
            list(learned_replay.get("particle_sla_shortfall_raw_work_units", ()))
            if isinstance(learned_replay, Mapping)
            else []
        ),
        "post_shield_particle_benchmark_sla_raw_work_units": (
            list(benchmark_replay.get("particle_sla_shortfall_raw_work_units", ()))
            if isinstance(benchmark_replay, Mapping)
            else []
        ),
        "model_scenario_origins": (
            len(adapter.scenario_count_by_origin) if adapter is not None else 0
        ),
        "model_scenario_count": (
            min(adapter.scenario_count_by_origin, default=0) if adapter is not None else 0
        ),
        "model_scenario_weight_sum_max_abs": (
            adapter.scenario_weight_sum_max_abs if adapter is not None else 0.0
        ),
        "model_scenario_da_rt_dart_identity_max_abs": (
            adapter.scenario_identity_max_abs if adapter is not None else 0.0
        ),
        "model_scenario_revealed_prefix_max_abs": (
            adapter.scenario_revealed_prefix_max_abs if adapter is not None else 0.0
        ),
        "all_origins_causal_input_contract": bool(
            adapter is not None
            and len(adapter.causal_input_contract_by_origin) == 96
            and all(adapter.causal_input_contract_by_origin)
        ),
        "physical_plant_contract": dict(result.get("physical_plant_contract", {})),
        "artifact_piecewise_core_exact": bool(
            result.get("physical_plant_contract", {}).get(
                "artifact_piecewise_core_exact", False
            )
        ),
        "train_frozen_forecast_residual_cap_contract": bool(
            adapter is not None and adapter.forecast_residual_cap_contract
        ),
    }


def _a1_optimizer_gradient_evidence(
    checkpoint: Mapping[str, Any], model: nn.Module
) -> bool:
    optimizer = checkpoint.get("optimizer_state")
    if not isinstance(optimizer, Mapping):
        return False
    groups = optimizer.get("param_groups")
    states = optimizer.get("state")
    if not isinstance(groups, list) or not isinstance(states, Mapping):
        return False
    ids = [identifier for group in groups for identifier in group.get("params", ())]
    names = list(_a1_trainable_names(model))
    if len(ids) != len(names) or not names:
        return False
    strict_exact_allowlist = _is_distributional_model(model)
    observed = (
        {name: False for name in names}
        if strict_exact_allowlist
        else {prefix: False for prefix in _A1_PREFIXES}
    )
    for identifier, name in zip(ids, names):
        state = states.get(identifier, states.get(str(identifier), {}))
        moment = state.get("exp_avg") if isinstance(state, Mapping) else None
        if not isinstance(moment, Tensor) or not torch.isfinite(moment).all():
            return False
        if bool(moment.abs().sum() > 0):
            if strict_exact_allowlist:
                observed[name] = True
            else:
                for prefix in observed:
                    if name.startswith(prefix):
                        observed[prefix] = True
    return all(observed.values())


def _a1_frozen_unchanged(
    checkpoint: Mapping[str, Any],
    predecessor: Mapping[str, Any] | None,
    model: nn.Module | None = None,
) -> bool:
    if predecessor is None or not isinstance(predecessor.get("model_state"), Mapping):
        return False
    current = checkpoint.get("model_state")
    previous = predecessor["model_state"]
    if not isinstance(current, Mapping) or set(current) != set(previous):
        return False
    allowed = set(_a1_trainable_names(model)) if model is not None else {
        name
        for name in current
        if any(name.startswith(prefix) for prefix in _A1_PREFIXES)
    }
    return bool(allowed) and all(
        _equal_state(value, previous[name])
        for name, value in current.items()
        if name not in allowed
    )


class LiveDecisionCapabilityEvaluator:
    """Materialize Policy/A1/A2 candidate evidence on the complete validation split."""

    def __init__(
        self,
        corpus: PreloadedR4Corpus,
        phase_directory: str | Path,
        *,
        config: Mapping[str, Any],
        repository_root: str | Path,
        seed: int,
    ) -> None:
        self.corpus = corpus
        self.phase_directory = Path(phase_directory).resolve()
        self.config = config
        self.repository_root = Path(repository_root).resolve()
        self.seed = int(seed)
        self.config_sha256 = str(config.get("_config_sha256", ""))
        if len(self.config_sha256) != 64:
            raise ValueError("live decision evaluator requires the hashed formal config")
        self.normalization_sha256 = sha256_json(corpus.statistics.as_json())
        self.forecast_residual_caps_sha256 = (
            corpus.statistics.forecast_residual_cap_sha256()
        )
        self.distribution_support = _distribution_support_from_config(
            config, self.repository_root
        )
        data = config.get("data")
        if not isinstance(data, Mapping):
            raise ValueError("formal config.data must be a mapping")
        self.bundle_root = _resolve(
            self.repository_root, data.get("selection_bundle_root"), "selection_bundle_root"
        )
        self.settlement_root = _resolve(
            self.repository_root, data.get("official_settlement_root"), "official_settlement_root"
        )
        self.records = {
            str(record["operating_date"]): record
            for record in corpus.records("validation")
        }
        if len(self.records) != 179 or any(
            temporal_split(day_text) != "validation" for day_text in self.records
        ):
            raise ValueError("decision evaluator requires exactly 179 validation records")
        definition, self.low_price, self.high_price = _benchmark_definition(
            corpus, config_sha256=self.config_sha256
        )
        benchmark_path = self.phase_directory.parent.parent / "data" / "price_aware_edf_shield.json"
        atomic_json(benchmark_path, definition)
        self.benchmark_path = benchmark_path
        self.benchmark_sha256 = sha256_file(benchmark_path)

    def _day(self, day_text: str) -> R4SelectionDay:
        parsed = date.fromisoformat(day_text)
        if temporal_split(parsed) != "validation" or parsed.year != 2023:
            raise R4DataError("decision materializer may open only 2023-H2 validation days")
        bundle = self.bundle_root / f"{day_text}.npz"
        settlement = self.settlement_root / f"{day_text}.npz"
        return load_r4_day(
            bundle,
            settlement,
            allow_sealed_selection=False,
            capacity_work_units=self.corpus.physical.capacity_work_units,
            physical_twin_core=self.corpus.physical_core,
        )

    def _benchmark_rows(self) -> dict[str, dict[str, Any]]:
        path = self.phase_directory.parent.parent / "data" / "price_aware_edf_validation_daily.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            cached_rows = payload.get("rows") if isinstance(payload, Mapping) else None
            if (
                isinstance(payload, Mapping)
                and payload.get("schema_version")
                == "dcvpp.r4.price_aware_edf_validation_daily.v2"
                and payload.get("benchmark_sha256") == self.benchmark_sha256
                and isinstance(cached_rows, list)
                and len(cached_rows) == 179
                and {str(row.get("date")) for row in cached_rows} == set(self.records)
                and all(
                    int(row.get("completed_origins", -1)) == 96
                    and int(row.get("independently_validated_origins", -1)) == 96
                    and bool(row.get("all_values_finite", False))
                    and bool(row.get("artifact_piecewise_core_exact", False))
                    for row in cached_rows
                )
            ):
                return {str(row["date"]): dict(row) for row in cached_rows}
        rows: list[dict[str, Any]] = []
        runtime = R4Runtime(
            self.corpus.physical,
            self.config.get("runtime"),
            physical_twin_core=self.corpus.physical_core,
        )
        for day_text in sorted(self.records):
            day = self._day(day_text)
            result = runtime.run_day(
                day,
                _PriceAwareEDF(day, self.corpus, self.low_price, self.high_price),
            )
            summary = _run_summary(result, day=day, corpus=self.corpus)
            if (
                summary["completed_origins"] != 96
                or summary["independently_validated_origins"] != 96
                or not summary["all_values_finite"]
            ):
                raise ValueError(f"frozen Price-aware EDF benchmark failed on {day_text}")
            rows.append({"date": day_text, **summary})
            atomic_json(
                path,
                {
                    "schema_version": "dcvpp.r4.price_aware_edf_validation_daily.v2",
                    "benchmark_sha256": self.benchmark_sha256,
                    "complete": len(rows) == 179,
                    "rows": rows,
                },
            )
        return {str(row["date"]): row for row in rows}

    def _policy_predecessor(self) -> tuple[Mapping[str, Any] | None, list[dict[str, Any]] | None]:
        predecessor_phase = (
            "joint_policy" if self.distribution_support is not None else "policy"
        )
        selected = (
            self.phase_directory.parent / f"phase={predecessor_phase}" / "checkpoints"
            / f"{predecessor_phase}_seed{self.seed}_selected_best.pt"
        )
        if not selected.is_file():
            return None, None
        try:
            payload = torch.load(selected, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover
            payload = torch.load(selected, map_location="cpu")
        if not isinstance(payload, Mapping):
            return None, None
        source_sha = str(payload.get("selected_from_sha256", ""))
        evidence = (
            self.phase_directory.parent / f"phase={predecessor_phase}"
            / "capability" / source_sha / f"{predecessor_phase}_daily.json"
        )
        if not evidence.is_file():
            return payload, None
        content = json.loads(evidence.read_text(encoding="utf-8"))
        rows = content.get("rows") if isinstance(content, Mapping) else None
        if (
            not isinstance(content, Mapping)
            or
            content.get("checkpoint_sha256") != source_sha
            or not isinstance(rows, list)
            or len(rows) != 179
        ):
            return payload, None
        return payload, [dict(row) for row in rows]

    def _a1_audits(
        self,
        checkpoint: Mapping[str, Any],
        *,
        source: Path,
        epoch: int,
        model: nn.Module,
    ) -> dict[str, bool]:
        predecessor, policy_rows = self._policy_predecessor()
        gradient = _a1_optimizer_gradient_evidence(checkpoint, model)
        frozen = _a1_frozen_unchanged(checkpoint, predecessor, model)
        dart_evaluator = LiveRepresentationCapabilityEvaluator(
            self.corpus,
            self.phase_directory / "a1_dart_retention",
            seed=self.seed,
        )
        try:
            dart_payload = dart_evaluator(
                phase="representation",
                horizon=96,
                checkpoint=str(source),
                epoch=int(epoch),
                model=model,
            )
            dart_retained = dart_payload.get("passed") is True
        except (ValueError, RuntimeError):
            dart_retained = False
        behavioral = False
        if policy_rows is not None:
            # The actual A1 daily rows are compared after materialization.
            behavioral = True
        return {
            "all_allowed_gradients_finite_nonzero": gradient,
            "frozen_parameters_unchanged": frozen,
            "dart_capability_gate_retained": dart_retained,
            "pre_a1_policy_daily_evidence_available": behavioral,
        }

    def _physical_twin_frozen(self, model: nn.Module) -> bool:
        stream = getattr(model, "physical_stream", None)
        core = getattr(stream, "physical_twin_core", None)
        if core is None or core.training or any(parameter.requires_grad for parameter in core.parameters()):
            return False
        return _equal_state(core.state_dict(), self.corpus.physical_core.state_dict())

    @staticmethod
    def _a1_behavioral_noninferiority(
        rows: Sequence[Mapping[str, Any]],
        policy_rows: Sequence[Mapping[str, Any]] | None,
        seed: int,
    ) -> bool:
        if policy_rows is None or len(policy_rows) != 179:
            return False
        current = {str(row["date"]): row for row in rows}
        frozen = {str(row["date"]): row for row in policy_rows}
        if set(current) != set(frozen):
            return False
        ordered = sorted(current)
        cost = np.asarray(
            [current[day]["operating_cost_usd"] - frozen[day]["operating_cost_usd"] for day in ordered]
        )
        baseline = np.asarray([abs(float(frozen[day]["operating_cost_usd"])) for day in ordered])
        sla = np.asarray(
            [current[day]["normalized_sla_shortfall"] - frozen[day]["normalized_sla_shortfall"] for day in ordered]
        )
        cost_ci = paired_daily_inference(
            cost - 0.01 * baseline,
            block_days=7,
            replicates=10_000,
            confidence=0.95,
            seed=seed,
        )
        return bool(cost_ci.upper <= 0.0 and float(sla.mean()) <= 0.001)

    def __call__(
        self,
        *,
        phase: str,
        horizon: int | None,
        checkpoint: str | None = None,
        checkpoint_path: str | None = None,
        epoch: int | None = None,
        model: nn.Module,
        **_: Any,
    ) -> Mapping[str, Any]:
        phase_name = str(phase).lower()
        if phase_name not in DECISION_PHASES or horizon != 96:
            raise ValueError("decision evaluator requires Policy/A1/A2 at K=96")
        source = Path(checkpoint_path or checkpoint or "").resolve()
        if not source.is_file() or epoch is None:
            raise ValueError("decision gate requires an immutable checkpoint and epoch")
        checkpoint_sha256 = sha256_file(source)
        checkpoint_payload, checkpoint_flags = _checkpoint_audit(
            source,
            phase=phase_name,
            seed=self.seed,
            epoch=int(epoch),
            model=model,
            config_sha256=self.config_sha256,
        )
        model.eval()
        phase_audits: dict[str, bool] = {}
        if phase_name == "a1":
            phase_audits = self._a1_audits(
                checkpoint_payload, source=source, epoch=int(epoch), model=model
            )
        elif phase_name == "a2":
            phase_audits = {"physical_twin_core_frozen": self._physical_twin_frozen(model)}

        evidence_dir = self.phase_directory / "capability" / checkpoint_sha256
        evidence_path = evidence_dir / f"{phase_name}_daily.json"
        existing_rows: dict[str, dict[str, Any]] = {}
        if evidence_path.is_file():
            previous = json.loads(evidence_path.read_text(encoding="utf-8"))
            if (
                previous.get("schema_version") != _DECISION_EVIDENCE_SCHEMA
                or
                previous.get("checkpoint_sha256") != checkpoint_sha256
                or previous.get("benchmark_sha256") != self.benchmark_sha256
                or previous.get("phase") != phase_name
            ):
                raise ValueError("checkpoint-keyed decision evidence metadata mismatch")
            existing_rows = {
                str(row["date"]): dict(row) for row in previous.get("rows", ())
            }
            for day_text, row in existing_rows.items():
                unsigned = {
                    key: value for key, value in row.items() if key != "evidence_sha256"
                }
                replay_path = (
                    evidence_dir / "particle_replays" / f"{day_text}.json"
                ).resolve()
                replay_payload: Mapping[str, Any] = {}
                if replay_path.is_file():
                    loaded_replay = json.loads(replay_path.read_text(encoding="utf-8"))
                    if isinstance(loaded_replay, Mapping):
                        replay_payload = loaded_replay
                replay_unsigned = {
                    key: value
                    for key, value in replay_payload.items()
                    if key != "evidence_sha256"
                }
                replay_attested = bool(
                    replay_payload.get("schema_version")
                    in {
                        _PAIRED_PARTICLE_REPLAY_SCHEMA,
                        _PAIRED_DISTRIBUTION_REPLAY_SCHEMA,
                    }
                    and replay_payload.get("evidence_sha256")
                    == sha256_json(replay_unsigned)
                    and row.get("post_shield_particle_replay_evidence_sha256")
                    == replay_payload.get("evidence_sha256")
                    and bool(row.get("post_shield_particle_cvar_contract", False))
                    == bool(replay_payload.get("contract_passed", False))
                )
                if (
                    row.get("evidence_sha256") != sha256_json(unsigned)
                    or row.get("checkpoint_sha256") != checkpoint_sha256
                    or int(row.get("checkpoint_epoch", -1)) != int(epoch)
                    or row.get("benchmark_sha256") != self.benchmark_sha256
                    or row.get("train_only_normalization_sha256")
                    != self.normalization_sha256
                    or row.get("forecast_residual_caps_sha256")
                    != self.forecast_residual_caps_sha256
                    or Path(
                        str(row.get("post_shield_particle_replay_evidence_path", ""))
                    ).resolve()
                    != replay_path
                    or not replay_path.is_file()
                    or row.get("post_shield_particle_replay_file_sha256")
                    != sha256_file(replay_path)
                    or not replay_attested
                ):
                    raise ValueError(
                        f"cached decision evidence attestation failed for {day_text}"
                    )
        benchmark_rows = self._benchmark_rows()
        rows = dict(existing_rows)
        runtime = R4Runtime(
            self.corpus.physical,
            self.config.get("runtime"),
            physical_twin_core=self.corpus.physical_core,
        )
        for day_text in sorted(self.records):
            if day_text in rows:
                continue
            if not _equal_state(model.state_dict(), checkpoint_payload["model_state"]):
                raise ValueError("candidate checkpoint/model changed during materialization")
            day = self._day(day_text)
            adapter = _NormalizedDeploymentModel(
                model,
                self.corpus,
                self.records[day_text],
                distribution_support=self.distribution_support,
            ).eval()
            result = runtime.run_day(day, adapter)
            particle_replay = _paired_post_shield_particle_replay(
                day=day,
                actual_result=result,
                adapter=adapter,
                model=model,
                corpus=self.corpus,
                deployment_record=self.records[day_text],
                runtime_config=self.config.get("runtime"),
                low_price=self.low_price,
                high_price=self.high_price,
                checkpoint_sha256=checkpoint_sha256,
                benchmark_sha256=self.benchmark_sha256,
            )
            particle_replay_path = (
                evidence_dir / "particle_replays" / f"{day_text}.json"
            ).resolve()
            atomic_json(particle_replay_path, particle_replay)
            summary = _run_summary(
                result,
                day=day,
                corpus=self.corpus,
                adapter=adapter,
                particle_replay=particle_replay,
            )
            summary.update(
                {
                    "post_shield_particle_replay_evidence_path": str(
                        particle_replay_path
                    ),
                    "post_shield_particle_replay_file_sha256": sha256_file(
                        particle_replay_path
                    ),
                }
            )
            benchmark = benchmark_rows[day_text]
            provenance = self.records[day_text].get("provenance", {})
            settlement_sha256 = str(provenance.get("settlement_sha256", ""))
            if not _is_sha256(settlement_sha256):
                raise ValueError(f"official settlement SHA-256 is absent for {day_text}")
            row: dict[str, Any] = {
                "date": day_text,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_epoch": int(epoch),
                "benchmark_sha256": self.benchmark_sha256,
                "benchmark_name": "price_aware_edf_shield",
                "official_settlement_sha256": settlement_sha256,
                "deployment_issue_context_normalized": True,
                "train_only_normalization_sha256": self.normalization_sha256,
                "forecast_residual_caps_sha256": self.forecast_residual_caps_sha256,
                **summary,
                "benchmark_operating_cost_usd": float(benchmark["operating_cost_usd"]),
                "benchmark_normalized_sla_shortfall": float(
                    benchmark["normalized_sla_shortfall"]
                ),
                "future_leakage_audit_status": "NOT_RUN",
                "checkpoint_immutable_during_materialization": True,
                **checkpoint_flags,
                **phase_audits,
            }
            if phase_name == "a1":
                row.setdefault("all_allowed_gradients_finite_nonzero", False)
                row.setdefault("frozen_parameters_unchanged", False)
                row.setdefault("dart_capability_gate_retained", False)
            if phase_name == "a2":
                row.setdefault("physical_twin_core_frozen", False)
            row["evidence_sha256"] = sha256_json(
                {key: value for key, value in row.items() if key != "evidence_sha256"}
            )
            rows[day_text] = row
            atomic_json(
                evidence_path,
                {
                    "schema_version": _DECISION_EVIDENCE_SCHEMA,
                    "phase": phase_name,
                    "epoch": int(epoch),
                    "checkpoint": str(source),
                    "checkpoint_sha256": checkpoint_sha256,
                    "benchmark_sha256": self.benchmark_sha256,
                    "future_leakage_audit_status": "NOT_RUN",
                    "complete": len(rows) == 179,
                    "rows": [rows[key] for key in sorted(rows)],
                },
            )

        if sha256_file(source) != checkpoint_sha256 or not _equal_state(
            model.state_dict(), checkpoint_payload["model_state"]
        ):
            raise ValueError("candidate checkpoint/model changed after materialization")

        daily = [rows[key] for key in sorted(rows)]
        if phase_name == "a1":
            _, policy_rows = self._policy_predecessor()
            behavioral = self._a1_behavioral_noninferiority(
                daily, policy_rows, self.seed
            )
            phase_audits["policy_cost_and_sla_noninferiority"] = behavioral
            for row in daily:
                row["policy_cost_and_sla_noninferiority"] = behavioral
                row["evidence_sha256"] = sha256_json(
                    {key: value for key, value in row.items() if key != "evidence_sha256"}
                )
            atomic_json(
                evidence_path,
                {
                    "schema_version": _DECISION_EVIDENCE_SCHEMA,
                    "phase": phase_name,
                    "epoch": int(epoch),
                    "checkpoint": str(source),
                    "checkpoint_sha256": checkpoint_sha256,
                    "benchmark_sha256": self.benchmark_sha256,
                    "future_leakage_audit_status": "NOT_RUN",
                    "complete": True,
                    "rows": daily,
                },
            )
        atomic_parquet(evidence_dir / f"{phase_name}_daily.parquet", daily)

        evaluator_phase = "policy" if phase_name == "joint_policy" else phase_name
        evaluator = build_capability_evaluator(evaluator_phase, daily, seed=self.seed)
        result = dict(evaluator(phase=evaluator_phase, horizon=96))
        metrics = dict(result.get("metrics", {}))
        uncertainty_contract = all(
            bool(row.get("decision_uncertainty_contract", False)) for row in daily
        )
        independent = all(
            bool(row["all_actions_independently_validated"]) for row in daily
        )
        finite = all(bool(row["all_values_finite"]) for row in daily)
        resumable = all(
            bool(row["checkpoint_lineage_valid"] and row["checkpoint_resume_exact"])
            for row in daily
        )
        causal_contract = all(
            bool(row.get("all_origins_causal_input_contract", False)) for row in daily
        )
        physical_contract = all(
            bool(row.get("artifact_piecewise_core_exact", False)) for row in daily
        )
        residual_cap_contract = all(
            bool(row.get("train_frozen_forecast_residual_cap_contract", False))
            and row.get("forecast_residual_caps_sha256")
            == self.forecast_residual_caps_sha256
            for row in daily
        )
        post_shield_cvar = all(
            bool(row.get("post_shield_particle_cvar_contract", False)) for row in daily
        )
        checkpoint_attested = all(
            bool(row.get("checkpoint_immutable_during_materialization", False))
            for row in daily
        ) and sha256_file(source) == checkpoint_sha256
        day_ahead_frozen = all(bool(row["day_ahead_frozen"]) for row in daily)
        passed = bool(
            result.get("passed")
            and uncertainty_contract
            and independent
            and finite
            and resumable
            and causal_contract
            and physical_contract
            and residual_cap_contract
            and post_shield_cvar
            and checkpoint_attested
            and day_ahead_frozen
        )
        result.update(
            {
                "passed": passed,
                "hard_shield": True,
                "validation_origins": int(metrics.get("validation_origins", 0)),
                "all_actions_independently_validated": independent,
                "finite": finite,
                "feasibility_fraction": float(metrics.get("post_shield_feasibility", 0.0)),
                "non_exogenous_rejected_origins": int(
                    metrics.get("non_exogenous_rejected_origins", -1)
                ),
                "normalized_sla_increase": float(
                    metrics.get("aggregate_normalized_sla_increase", math.inf)
                ),
                "paired_daily_cost_upper_ci": float(
                    metrics.get("cost_delta_upper", math.inf)
                ),
                "day_ahead_frozen_after_origin_zero": day_ahead_frozen,
                "future_leakage_audit_status": "NOT_RUN",
                "all_origins_causal_input_contract": causal_contract,
                "artifact_piecewise_core_exact": physical_contract,
                "train_frozen_forecast_residual_cap_contract": residual_cap_contract,
                "forecast_residual_caps_sha256": self.forecast_residual_caps_sha256,
                "post_shield_particle_cvar_contract": post_shield_cvar,
                "checkpoint_immutable_during_materialization": checkpoint_attested,
                "finite_and_resumable": bool(finite and resumable),
                "operating_cost": float(
                    np.mean([row["operating_cost_usd"] for row in daily])
                ),
                "cvar": float(np.mean([row["weighted_cvar95_usd"] for row in daily])),
                "corrective_magnitude": float(
                    np.mean([row["corrective_intervention_mean"] for row in daily])
                ),
                "decision_uncertainty_contract": uncertainty_contract,
                "exact_20_model_scenario_contract": all(
                    bool(row.get("exact_20_model_scenario_contract", False))
                    for row in daily
                ),
                "evidence_path": str(evidence_path),
                "evidence_sha256": sha256_file(evidence_path),
            }
        )
        if phase_name == "a1":
            result.update(
                {
                    "representation_dart_gate_retained": bool(
                        phase_audits.get("dart_capability_gate_retained", False)
                    ),
                    "allowed_gradients_finite_nonzero": bool(
                        phase_audits.get("all_allowed_gradients_finite_nonzero", False)
                    ),
                    "frozen_parameters_unchanged": bool(
                        phase_audits.get("frozen_parameters_unchanged", False)
                    ),
                    "policy_cost_and_sla_noninferiority": bool(
                        phase_audits.get("policy_cost_and_sla_noninferiority", False)
                    ),
                }
            )
            result["passed"] = bool(
                result["passed"]
                and result["representation_dart_gate_retained"]
                and result["allowed_gradients_finite_nonzero"]
                and result["frozen_parameters_unchanged"]
                and result["policy_cost_and_sla_noninferiority"]
            )
        if phase_name == "a2":
            inherited = bool(
                result["passed"] and phase_audits.get("physical_twin_core_frozen", False)
            )
            result["inherit_full_policy_gate_without_relaxation"] = inherited
            result["passed"] = inherited
        extra_reasons: list[str] = []
        if not uncertainty_contract:
            extra_reasons.append("decision uncertainty input contract failed")
        if not causal_contract:
            extra_reasons.append("causal prefix input contract failed at one or more origins")
        if not physical_contract:
            extra_reasons.append("formal runtime did not use the frozen piecewise physical core")
        if not residual_cap_contract:
            extra_reasons.append("train-frozen forecast residual cap contract failed")
        if not post_shield_cvar:
            extra_reasons.append(
                "20-particle post-shield CVaR replay is absent; risk gate failed closed"
            )
        if not checkpoint_attested:
            extra_reasons.append("checkpoint immutability attestation failed")
        if not independent:
            extra_reasons.append("one or more executed actions failed independent validation")
        if not finite:
            extra_reasons.append("one or more daily evidence rows contain non-finite values")
        if not resumable:
            extra_reasons.append("checkpoint exact-resume lineage contract failed")
        if not day_ahead_frozen:
            extra_reasons.append("day-ahead schedule changed after origin zero")
        if phase_name == "a1":
            for name in (
                "representation_dart_gate_retained",
                "allowed_gradients_finite_nonzero",
                "frozen_parameters_unchanged",
                "policy_cost_and_sla_noninferiority",
            ):
                if not bool(result.get(name, False)):
                    extra_reasons.append(f"A1 evidence failed closed: {name}")
        if phase_name == "a2" and not bool(
            result.get("inherit_full_policy_gate_without_relaxation", False)
        ):
            extra_reasons.append("A2 did not inherit the full unrelaxed policy gate")
        if result.get("passed") is not True:
            result["status"] = "FAIL"
        if extra_reasons:
            result["reasons"] = tuple(result.get("reasons", ())) + tuple(extra_reasons)
        return result


__all__ = ["LiveDecisionCapabilityEvaluator"]
