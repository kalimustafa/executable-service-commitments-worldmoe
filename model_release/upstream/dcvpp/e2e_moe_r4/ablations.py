"""Preregistered, solver-free architecture and operational ablations for R4.

This module deliberately separates three concerns:

* a deterministic preregistration that fixes the comparisons before results;
* executable model factories for the five architecture variants and the
  operational variants that actually change the neural computation;
* a fail-closed receiver for externally measured, artifact-backed results.

It never opens formal data and it never invents a metric.  Training/evaluation
runners consume :class:`AblationRunSpec` objects and must submit their measured
artifacts through :class:`AblationResultReceiver`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import json
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .artifact_io import atomic_json
from .config import (
    ARCHITECTURE_VERSION,
    EXPERIMENT_ID,
    FORMAL_SEEDS,
    TRAINING_SEMANTICS_REVISION,
)
from .hashing import sha256_file, sha256_json
from .liquid import LiquidState
from .model import HybridCfCTransformerMoER4, R4Config
from .scenario_risk import ZONAL_VARIABLE_COUNT, zonal_to_legacy


ABLATION_SCHEMA_VERSION = "dcvpp.e2e_moe_hybrid_r4.ablation_preregistration.v1"
RESULT_SCHEMA_VERSION = "dcvpp.e2e_moe_hybrid_r4.ablation_result.v1"
ARCHITECTURE_METHODS = ("A0", "A1", "A2", "A3", "A4")
OPERATIONAL_METHODS = (
    "dense_no_moe",
    "one_scenario",
    "no_cvar",
    "no_a1",
    "raw_vs_shield",
)
ALL_ABLATION_METHODS = ARCHITECTURE_METHODS + OPERATIONAL_METHODS
FULL_PHASES = ("representation", "scenario", "policy", "a1", "a2")
NO_A1_PHASES = ("representation", "scenario", "policy", "a2")
VALIDATION_DATES = 179
ORIGINS_PER_DAY = 96


class AblationContractError(ValueError):
    """Raised before execution when a preregistered invariant is violated."""


class MeasuredResultError(ValueError):
    """Raised when submitted evidence cannot support an ablation result."""


@dataclass(frozen=True)
class ArchitectureAblation:
    method_id: str
    market_encoder: str
    workload_encoder: str
    physical_dynamic_encoder: str
    rolling_state_encoder: str
    token_width: int
    last_hidden_bottleneck: bool
    post_fusion: str
    experts: int
    top_k: int
    hypothesis: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "market_encoder": self.market_encoder,
            "workload_encoder": self.workload_encoder,
            "physical_dynamic_encoder": self.physical_dynamic_encoder,
            "rolling_state_encoder": self.rolling_state_encoder,
            "token_width": self.token_width,
            "last_hidden_bottleneck": self.last_hidden_bottleneck,
            "post_fusion": self.post_fusion,
            "experts": self.experts,
            "top_k": self.top_k,
            "hypothesis": self.hypothesis,
        }


@dataclass(frozen=True)
class OperationalAblation:
    method_id: str
    architecture_id: str
    scenario_count: int
    cvar_training_weight_enabled: bool
    phase_sequence: tuple[str, ...]
    execution_views: tuple[str, ...]
    routing_enabled: bool
    checkpoint_source: str
    hypothesis: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "architecture_id": self.architecture_id,
            "scenario_count": self.scenario_count,
            "cvar_training_weight_enabled": self.cvar_training_weight_enabled,
            "phase_sequence": list(self.phase_sequence),
            "execution_views": list(self.execution_views),
            "routing_enabled": self.routing_enabled,
            "checkpoint_source": self.checkpoint_source,
            "hypothesis": self.hypothesis,
        }


@dataclass(frozen=True)
class AblationMethod:
    method_id: str
    family: str
    architecture_id: str
    scenario_count: int
    cvar_training_weight_enabled: bool
    phase_sequence: tuple[str, ...]
    execution_views: tuple[str, ...]
    routing_enabled: bool
    checkpoint_source: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "family": self.family,
            "architecture_id": self.architecture_id,
            "scenario_count": self.scenario_count,
            "cvar_training_weight_enabled": self.cvar_training_weight_enabled,
            "phase_sequence": list(self.phase_sequence),
            "execution_views": list(self.execution_views),
            "routing_enabled": self.routing_enabled,
            "checkpoint_source": self.checkpoint_source,
        }


@dataclass(frozen=True)
class AblationPreregistration:
    schema_version: str
    experiment_id: str
    architecture_version: str
    training_semantics_revision: str
    shared_contract: Mapping[str, Any]
    architecture_ablations: tuple[ArchitectureAblation, ...]
    operational_ablations: tuple[OperationalAblation, ...]

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "architecture_version": self.architecture_version,
            "training_semantics_revision": self.training_semantics_revision,
            "status": "PREREGISTERED_NO_RESULTS",
            "shared_contract": deepcopy(dict(self.shared_contract)),
            "architecture_ablations": [
                item.to_payload() for item in self.architecture_ablations
            ],
            "operational_ablations": [
                item.to_payload() for item in self.operational_ablations
            ],
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.unsigned_payload())

    @property
    def shared_contract_digest(self) -> str:
        return sha256_json(dict(self.shared_contract))

    def to_payload(self) -> dict[str, Any]:
        payload = self.unsigned_payload()
        payload["preregistration_sha256"] = self.digest
        payload["shared_contract_sha256"] = self.shared_contract_digest
        return payload

    def write_once(self, path: str | Path) -> Path:
        destination = Path(path).resolve()
        expected = self.to_payload()
        if destination.exists():
            try:
                current = json.loads(destination.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise AblationContractError(
                    f"existing preregistration is unreadable: {destination}"
                ) from exc
            if current != expected:
                raise AblationContractError(
                    "immutable ablation preregistration already exists with different "
                    f"content: {destination}"
                )
            return destination
        atomic_json(destination, expected)
        return destination


def _shared_contract() -> dict[str, Any]:
    """Return the canonical comparability contract used by every method."""

    return {
        "data_contract": {
            "train": ["2022-01-01", "2023-06-30"],
            "validation": ["2023-07-01", "2023-12-31"],
            "validation_dates": VALIDATION_DATES,
            "final_temporal_test": ["2024-01-01", "2024-12-31"],
            "final_test_sealed_until_freeze": True,
            "locked_year": 2025,
            "reject_2025_path_before_open": True,
            "normalization_fit_split": "train",
            "availability_rule": "feature_available_at <= decision_time",
        },
        "decoder_contract": {
            "class": "dcvpp.e2e_moe_r4.model.ZonalForecastDecoder",
            "per_token": True,
            "zonal_width": 21,
            "identity": "rt = da - dart",
        },
        "shield_contract": {
            "class": "dcvpp.e2e_moe_r4.constrained_runtime.AnalyticConstraintShield",
            "same_instance_semantics_for_all_methods": True,
            "origins_per_day": ORIGINS_PER_DAY,
            "day_ahead_frozen_after_origin_zero": True,
            "migration": "fixed_zero",
            "independent_post_shield_validator": True,
        },
        "training_budget": {
            "effective_batch_days": 512,
            "microbatch_fallback": [512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
            "representation": {"max_epochs": 800, "patience": 30},
            "scenario": {"max_epochs": 200, "patience": 15},
            "policy": {
                "max_epochs": 320,
                "curriculum": [[8, 40], [24, 60], [48, 80], [96, 140]],
                "k96_patience": 20,
            },
            "a1": {"max_epochs": 120, "patience": 15},
            "a2": {"max_epochs": 180, "patience": 20},
            "optimizer": "adamw",
        },
        "seeds": list(FORMAL_SEEDS),
        "seed_order": list(FORMAL_SEEDS),
        "seed1_must_pass_mainline_before_later_seeds": True,
        "scenario_contract": {
            "mainline_count": 20,
            "joint_weighted": True,
            "persistent_particle_identity": True,
        },
        "risk_contract": {
            "formal_reporting_cvar_alpha": 0.95,
            "formal_cvar_always_recomputed_from_final_particle_costs": True,
        },
        "external_solver": False,
        "forbidden_active_imports": [
            "pyomo",
            "gurobipy",
            "cvxpy",
            "cvxpylayers",
            "optnet",
        ],
        "result_policy": {
            "artifact_backed_measured_results_only": True,
            "synthetic_results_forbidden": True,
            "daily_paired_records_required": True,
            "day_is_primary_statistical_unit": True,
        },
    }


def default_preregistration() -> AblationPreregistration:
    """Build the complete deterministic R4 ablation preregistration."""

    architectures = (
        ArchitectureAblation(
            "A0",
            "gru64_last_hidden",
            "gru64_last_hidden",
            "gru64_last_hidden",
            "gru64_last_hidden",
            64,
            True,
            "dense_ffn",
            0,
            0,
            "Test whether the legacy 64-wide day bottleneck loses decision signal.",
        ),
        ArchitectureAblation(
            "A1",
            "token_gru128",
            "token_gru128",
            "token_gru128",
            "token_gru128",
            128,
            False,
            "dense_ffn",
            0,
            0,
            "Isolate token preservation and width from Transformer/Liquid effects.",
        ),
        ArchitectureAblation(
            "A2",
            "market_transformer_2l4h",
            "token_gru128",
            "token_gru128",
            "token_gru128",
            128,
            False,
            "dense_ffn",
            0,
            0,
            "Measure the market Transformer with non-Liquid token branches.",
        ),
        ArchitectureAblation(
            "A3",
            "market_transformer_2l4h",
            "closed_form_liquid128",
            "closed_form_liquid128",
            "closed_form_liquid128",
            128,
            False,
            "dense_ffn",
            0,
            0,
            "Measure closed-form Liquid streams without sparse routing.",
        ),
        ArchitectureAblation(
            "A4",
            "market_transformer_2l4h",
            "closed_form_liquid128",
            "closed_form_liquid128",
            "closed_form_liquid128",
            128,
            False,
            "top2_moe",
            8,
            2,
            "Measure the incremental value of token-level eight-expert Top-2 MoE.",
        ),
    )
    operational = (
        OperationalAblation(
            "dense_no_moe",
            "A3",
            20,
            True,
            (),
            ("shielded_execution",),
            False,
            "A3_same_seed",
            "Operational no-MoE comparison under the full R4 training program.",
        ),
        OperationalAblation(
            "one_scenario",
            "A4",
            1,
            True,
            FULL_PHASES,
            ("shielded_execution",),
            True,
            "fresh",
            "Test whether learned joint uncertainty improves risk-aware control.",
        ),
        OperationalAblation(
            "no_cvar",
            "A4",
            20,
            False,
            FULL_PHASES,
            ("shielded_execution",),
            True,
            "fresh",
            "Remove CVaR from training while retaining ex-post CVaR reporting.",
        ),
        OperationalAblation(
            "no_a1",
            "A4",
            20,
            True,
            NO_A1_PHASES,
            ("shielded_execution",),
            True,
            "fresh",
            "Skip DART-focused A1 and continue from selected policy to A2.",
        ),
        OperationalAblation(
            "raw_vs_shield",
            "A4",
            20,
            True,
            (),
            ("raw_proposal", "shielded_execution"),
            True,
            "A4_same_seed",
            "Paired audit of learned proposals and the exact shield-executed actions.",
        ),
    )
    registration = AblationPreregistration(
        schema_version=ABLATION_SCHEMA_VERSION,
        experiment_id=f"{EXPERIMENT_ID}_ablations_v1",
        architecture_version=ARCHITECTURE_VERSION,
        training_semantics_revision=TRAINING_SEMANTICS_REVISION,
        shared_contract=_shared_contract(),
        architecture_ablations=architectures,
        operational_ablations=operational,
    )
    validate_preregistration(registration)
    return registration


def validate_preregistration(registration: AblationPreregistration) -> None:
    if registration.schema_version != ABLATION_SCHEMA_VERSION:
        raise AblationContractError("unknown R4 ablation preregistration schema")
    architecture_ids = tuple(item.method_id for item in registration.architecture_ablations)
    operational_ids = tuple(item.method_id for item in registration.operational_ablations)
    if architecture_ids != ARCHITECTURE_METHODS:
        raise AblationContractError(f"architecture ablations must be {ARCHITECTURE_METHODS}")
    if operational_ids != OPERATIONAL_METHODS:
        raise AblationContractError(f"operational ablations must be {OPERATIONAL_METHODS}")
    a4 = registration.architecture_ablations[-1]
    if (a4.experts, a4.top_k, a4.post_fusion) != (8, 2, "top2_moe"):
        raise AblationContractError("A4 must remain the complete eight-expert Top-2 R4")
    for item in registration.architecture_ablations[:-1]:
        if item.experts != 0 or item.top_k != 0 or item.post_fusion != "dense_ffn":
            raise AblationContractError(f"{item.method_id} must use the dense comparator")
    if registration.architecture_ablations[0].token_width != 64:
        raise AblationContractError("A0 must use GRU64")
    if any(item.token_width != 128 for item in registration.architecture_ablations[1:]):
        raise AblationContractError("A1-A4 must preserve width-128 tokens")
    contract = registration.shared_contract
    if tuple(contract.get("seeds", ())) != FORMAL_SEEDS:
        raise AblationContractError("all ablations must use the three formal seeds")
    if contract.get("external_solver") is not False:
        raise AblationContractError("R4 ablations must remain solver-free")
    if contract.get("shield_contract", {}).get("origins_per_day") != 96:
        raise AblationContractError("all ablations require 96 rolling origins")
    if contract.get("training_budget", {}).get("effective_batch_days") != 512:
        raise AblationContractError("all ablations require effective batch 512")


def method_registry(
    registration: AblationPreregistration | None = None,
) -> dict[str, AblationMethod]:
    registration = registration or default_preregistration()
    validate_preregistration(registration)
    result: dict[str, AblationMethod] = {}
    for architecture in registration.architecture_ablations:
        result[architecture.method_id] = AblationMethod(
            method_id=architecture.method_id,
            family="architecture",
            architecture_id=architecture.method_id,
            scenario_count=20,
            cvar_training_weight_enabled=True,
            phase_sequence=FULL_PHASES,
            execution_views=("shielded_execution",),
            routing_enabled=architecture.method_id == "A4",
            checkpoint_source="fresh",
        )
    for operational in registration.operational_ablations:
        result[operational.method_id] = AblationMethod(
            method_id=operational.method_id,
            family="operational",
            architecture_id=operational.architecture_id,
            scenario_count=operational.scenario_count,
            cvar_training_weight_enabled=operational.cvar_training_weight_enabled,
            phase_sequence=operational.phase_sequence,
            execution_views=operational.execution_views,
            routing_enabled=operational.routing_enabled,
            checkpoint_source=operational.checkpoint_source,
        )
    if tuple(result) != ALL_ABLATION_METHODS:
        raise AblationContractError("ablation registry order or membership drifted")
    return result


class _GRUStepCell(nn.Module):
    """Small GRUCell shim with the Liquid ``step(..., dt=...)`` signature."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.cell = nn.GRUCell(input_size, hidden_size)

    def step(
        self,
        inputs: Tensor,
        hidden: Tensor | LiquidState | None = None,
        *,
        dt: Tensor | float | None = None,
    ) -> Tensor:
        del dt
        if isinstance(hidden, LiquidState):
            hidden = hidden.hidden
        if hidden is None:
            hidden = inputs.new_zeros((inputs.shape[0], self.hidden_size))
        return self.cell(inputs, hidden)

    def forward(
        self,
        inputs: Tensor,
        hidden: Tensor | LiquidState | None = None,
        dt: Tensor | float | None = None,
    ) -> Tensor:
        return self.step(inputs, hidden, dt=dt)


class TokenGRUSequence(nn.Module):
    """GRU comparator with an explicit token-preserving/last-hidden switch."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 128,
        *,
        recurrent_width: int = 128,
        repeat_last_hidden: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(recurrent_width)
        self.output_size = int(output_size)
        self.repeat_last_hidden = bool(repeat_last_hidden)
        self.input_norm = nn.LayerNorm(input_size)
        self.cell = _GRUStepCell(input_size, recurrent_width)
        projection: nn.Module
        if recurrent_width == output_size:
            projection = nn.Identity()
        else:
            projection = nn.Linear(recurrent_width, output_size)
        self.output_norm = nn.Sequential(projection, nn.LayerNorm(output_size))

    def forward(
        self,
        inputs: Tensor,
        hidden: Tensor | LiquidState | None = None,
        *,
        dt: Tensor | float | None = None,
    ) -> tuple[Tensor, Tensor]:
        del dt
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError(
                f"GRU sequence must be [batch,time,{self.input_size}], got {tuple(inputs.shape)}"
            )
        if isinstance(hidden, LiquidState):
            hidden = hidden.hidden
        normalized = self.input_norm(inputs)
        state = hidden
        raw_tokens: list[Tensor] = []
        for step in range(inputs.shape[1]):
            state = self.cell.step(normalized[:, step], state)
            raw_tokens.append(state)
        assert state is not None
        if self.repeat_last_hidden:
            tokens = state[:, None].expand(-1, inputs.shape[1], -1)
        else:
            tokens = torch.stack(raw_tokens, dim=1)
        return self.output_norm(tokens), state


class GRUMarketEncoder(nn.Module):
    """Market adapter matching the formal Transformer's dictionary interface."""

    def __init__(self, config: R4Config, *, recurrent_width: int, repeat_last_hidden: bool):
        super().__init__()
        self.config = config
        self.context_length = config.history_days + config.horizon
        self.encoder = TokenGRUSequence(
            config.market_dim,
            config.hidden_dim,
            recurrent_width=recurrent_width,
            repeat_last_hidden=repeat_last_hidden,
        )
        self.history_missing = nn.Parameter(
            torch.zeros(1, config.history_days, config.market_dim)
        )

    def forward(
        self,
        market: Tensor,
        *,
        historical_market: Tensor | None = None,
        historical_market_available: Tensor | None = None,
        **calendar: Any,
    ) -> dict[str, Tensor]:
        del calendar
        cfg = self.config
        if market.ndim != 3 or market.shape[1:] != (cfg.horizon, cfg.market_dim):
            raise ValueError(f"market must be [batch,96,{cfg.market_dim}]")
        batch = market.shape[0]
        if historical_market is None:
            available = torch.zeros(
                (batch, cfg.history_days), dtype=torch.bool, device=market.device
            )
            history = self.history_missing.expand(batch, -1, -1)
        else:
            if historical_market.shape != (batch, cfg.history_days, cfg.market_dim):
                raise ValueError(f"historical_market must be [batch,7,{cfg.market_dim}]")
            available = (
                torch.ones((batch, cfg.history_days), dtype=torch.bool, device=market.device)
                if historical_market_available is None
                else torch.as_tensor(
                    historical_market_available, dtype=torch.bool, device=market.device
                )
            )
            if available.shape != (batch, cfg.history_days):
                raise ValueError("historical_market_available must be [batch,7]")
            history = torch.where(
                available[..., None],
                historical_market,
                self.history_missing.expand(batch, -1, -1),
            )
        context, _ = self.encoder(torch.cat((history, market), dim=1))
        return {
            "context_tokens": context,
            "tokens": context[:, cfg.history_days :],
            "history_tokens": context[:, : cfg.history_days],
            "history_available": available,
        }


class DenseTokenFFN(nn.Module):
    """Dense post-fusion comparator; intentionally exposes no fake router."""

    routing_enabled = False

    def __init__(self, hidden_dim: int = 128, ff_dim: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 2 * ff_dim),
            nn.SiLU(),
            nn.Linear(2 * ff_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        tokens: Tensor,
        market_tokens: Tensor,
        state_tokens: Tensor,
        *,
        origin: int | Tensor = 0,
    ) -> dict[str, Tensor]:
        del market_tokens, state_tokens, origin
        output = self.output_norm(tokens + self.ff(self.norm(tokens)))
        return {
            "tokens": output,
            "latent": output,
            "routing_enabled": torch.tensor(False, device=tokens.device),
        }


class LearnedOneScenarioGenerator(nn.Module):
    """One unique learned path carried through the fixed width-20 runtime seam.

    R4 policy/loss/runtime code deliberately fixes the scenario tensor axis at
    twenty.  This ablation learns exactly one code and one trajectory, then
    repeats that same trajectory with uniform mass solely at the interface.
    Repetition adds neither parameters nor uncertainty and is audited by the
    one-scenario live gate as zero pairwise diversity.
    """

    def __init__(
        self,
        token_dim: int = 128,
        *,
        horizon: int = 96,
        scenario_code_dim: int = 32,
        hidden_dim: int = 64,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if horizon != 96:
            raise ValueError("one-scenario R4 ablation requires horizon 96")
        from .liquid import ClosedFormLiquidCell

        self.horizon = 96
        self.num_scenarios = 1
        self.scenario_codes = nn.Embedding(1, scenario_code_dim)
        self.decoder = ClosedFormLiquidCell(
            token_dim + scenario_code_dim,
            hidden_dim,
            tau_min_hours=0.25,
            tau_max_hours=24.0,
            default_dt_hours=0.25,
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 14),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 14),
            nn.Softplus(),
        )
        self.dart_adapter = nn.Linear(token_dim, 4)
        self.residual_scale = float(residual_scale)

    @staticmethod
    def _inverse_softplus(value: Tensor) -> Tensor:
        value = value.clamp_min(1e-6)
        return torch.where(value > 20.0, value, value + torch.log(-torch.expm1(-value)))

    def _decode(self, raw: Tensor, baseline: Tensor, scale: Tensor) -> Tensor:
        da = baseline[..., 0:4] + torch.tanh(raw[..., 0:4]) * scale[..., 0:4]
        dart = baseline[..., 8:12] + torch.tanh(raw[..., 4:8]) * scale[..., 4:8]
        rt = da - dart
        workload = F.softplus(
            self._inverse_softplus(baseline[..., 12:13])
            + torch.tanh(raw[..., 8:9]) * scale[..., 8:9]
        )
        pv = F.softplus(
            self._inverse_softplus(baseline[..., 13:17])
            + torch.tanh(raw[..., 9:13]) * scale[..., 9:13]
        )
        pue = (
            baseline[..., 17:21] + 0.1 * torch.tanh(raw[..., 13:14])
        ).clamp_min(1e-4)
        return torch.cat((da, rt, dart, workload, pv, pue), dim=-1)

    def forward(
        self,
        token_features: Tensor,
        forecast_zonal: Tensor,
        *,
        origin: int | Tensor | None = None,
        revealed_zonal: Tensor | None = None,
        revealed_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if token_features.ndim != 3 or token_features.shape[1] != self.horizon:
            raise ValueError("one-scenario token features must be [batch,96,hidden]")
        if forecast_zonal.shape != (*token_features.shape[:2], ZONAL_VARIABLE_COUNT):
            raise ValueError("forecast_zonal must be [batch,96,21]")
        batch = token_features.shape[0]
        code = self.scenario_codes.weight[None].expand(batch, -1, -1)
        hidden = token_features.new_zeros((batch, 1, self.decoder.hidden_size))
        scale = self.scale_head(token_features).clamp_max(10.0) * self.residual_scale
        outputs: list[Tensor] = []
        for step in range(self.horizon):
            decoder_input = torch.cat((token_features[:, step, None], code), dim=-1)
            hidden = self.decoder.step(
                decoder_input.reshape(batch, -1), hidden.reshape(batch, -1)
            ).reshape(batch, 1, -1)
            outputs.append(
                self._decode(
                    self.residual_head(hidden),
                    forecast_zonal[:, step, None],
                    scale[:, step, None],
                )
            )
        particles = torch.stack(outputs, dim=2)
        if revealed_zonal is not None or revealed_mask is not None:
            if revealed_zonal is None or revealed_mask is None:
                raise ValueError("revealed_zonal and revealed_mask must be supplied together")
            mask = torch.as_tensor(revealed_mask, device=particles.device, dtype=torch.bool)
            if revealed_zonal.shape != (batch, 96, ZONAL_VARIABLE_COUNT):
                raise ValueError("revealed_zonal must be [batch,96,21]")
            if mask.shape != (batch, 96):
                raise ValueError("revealed_mask must be [batch,96]")
            if origin is not None:
                index = torch.as_tensor(origin, device=particles.device)
                if index.ndim == 0:
                    index = index.expand(batch)
                periods = torch.arange(96, device=particles.device)[None]
                if torch.any(mask & (periods > index[:, None])):
                    raise ValueError("revealed_mask exposes a future value")
            identity = (
                revealed_zonal[..., 0:4]
                - revealed_zonal[..., 4:8]
                - revealed_zonal[..., 8:12]
            )
            if torch.any(identity[mask].abs() > 1e-5):
                raise ValueError("revealed zonal values violate DA-RT-DART identity")
            particles = torch.where(mask[:, None, :, None], revealed_zonal[:, None], particles)
        unique_particles = particles
        particles = unique_particles.expand(-1, 20, -1, -1)
        weights = token_features.new_full((batch, 20), 1.0 / 20.0)
        return {
            "particles_zonal": particles,
            "particles": particles,
            "particles_legacy": zonal_to_legacy(particles),
            "weights": weights,
            "weight_logits": token_features.new_zeros((batch, 20)),
            "scale": scale,
            "effective_sample_size": token_features.new_full((batch,), 20.0),
            "weight_entropy": token_features.new_full((batch,), math.log(20.0)),
            "unique_scenario_count": torch.tensor(1, device=token_features.device),
            "transport_scenario_count": torch.tensor(20, device=token_features.device),
            "transport_repeats_single_path": torch.tensor(
                True, device=token_features.device
            ),
        }


def build_ablation_model(
    method_id: str,
    config: R4Config | Mapping[str, Any] | None = None,
    *,
    physical_twin_core: nn.Module | None = None,
) -> HybridCfCTransformerMoER4:
    """Instantiate a real neural comparator without opening any data artifact."""

    methods = method_registry()
    if method_id not in methods:
        raise AblationContractError(f"unknown R4 ablation method {method_id!r}")
    method = methods[method_id]
    model = HybridCfCTransformerMoER4(config, physical_twin_core=physical_twin_core)
    cfg = model.config
    architecture_id = method.architecture_id
    if architecture_id in {"A0", "A1"}:
        recurrent_width = 64 if architecture_id == "A0" else 128
        repeat_last = architecture_id == "A0"
        model.market_encoder = GRUMarketEncoder(
            cfg, recurrent_width=recurrent_width, repeat_last_hidden=repeat_last
        )
        model.workload_encoder = TokenGRUSequence(
            cfg.workload_dim,
            cfg.hidden_dim,
            recurrent_width=recurrent_width,
            repeat_last_hidden=repeat_last,
        )
        model.physical_stream.dynamic_liquid = TokenGRUSequence(
            cfg.physical_dynamic_dim,
            cfg.hidden_dim,
            recurrent_width=recurrent_width,
            repeat_last_hidden=repeat_last,
        )
        model.state_encoder = TokenGRUSequence(
            cfg.state_dim,
            cfg.hidden_dim,
            recurrent_width=recurrent_width,
            repeat_last_hidden=repeat_last,
        )
        model.moe = DenseTokenFFN(cfg.hidden_dim, cfg.fusion_ff_dim)
    elif architecture_id == "A2":
        model.workload_encoder = TokenGRUSequence(cfg.workload_dim, cfg.hidden_dim)
        model.physical_stream.dynamic_liquid = TokenGRUSequence(
            cfg.physical_dynamic_dim, cfg.hidden_dim
        )
        model.state_encoder = TokenGRUSequence(cfg.state_dim, cfg.hidden_dim)
        model.moe = DenseTokenFFN(cfg.hidden_dim, cfg.fusion_ff_dim)
    elif architecture_id == "A3":
        model.moe = DenseTokenFFN(cfg.hidden_dim, cfg.fusion_ff_dim)
    elif architecture_id != "A4":
        raise AblationContractError(f"unimplemented architecture {architecture_id!r}")
    if method.scenario_count == 1:
        model.scenario_generator = LearnedOneScenarioGenerator(
            cfg.hidden_dim,
            horizon=cfg.horizon,
            scenario_code_dim=cfg.scenario_code_dim,
            hidden_dim=cfg.scenario_hidden_dim,
        )
    model.ablation_method_id = method.method_id
    model.ablation_architecture_id = architecture_id
    model.ablation_routing_enabled = method.routing_enabled
    model.ablation_scenario_count = method.scenario_count
    model.ablation_cvar_training_weight_enabled = method.cvar_training_weight_enabled
    model.ablation_phase_sequence = method.phase_sequence
    model.ablation_execution_views = method.execution_views
    return model


def parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": int(total), "trainable": int(trainable), "frozen": int(total - trainable)}


def measure_registered_parameter_counts(
    methods: Iterable[str] = ALL_ABLATION_METHODS,
) -> dict[str, dict[str, int]]:
    """Construct and count registered models; no checkpoint or data is touched."""

    result: dict[str, dict[str, int]] = {}
    for method_id in methods:
        model = build_ablation_model(method_id)
        result[method_id] = parameter_counts(model)
        del model
    return result


def method_training_overrides(method_id: str) -> dict[str, Any]:
    """Explicit runner overrides; absent objectives must never be silently faked."""

    method = method_registry().get(method_id)
    if method is None:
        raise AblationContractError(f"unknown R4 ablation method {method_id!r}")
    overrides: dict[str, Any] = {
        "phase_sequence": list(method.phase_sequence),
        "scenario_count": method.scenario_count,
        "routing_enabled": method.routing_enabled,
        "execution_views": list(method.execution_views),
        "checkpoint_source": method.checkpoint_source,
        "loss_weights": {},
        "remove_loss_terms": [],
        "capability_gate_overrides": {},
    }
    if not method.routing_enabled:
        # PhaseLossComputer checks router-key presence before multiplying by a
        # weight, so dense methods must remove these terms rather than set 0.
        overrides["remove_loss_terms"] = [
            "policy.router_balance",
            "policy.router_stability",
            "a2.router_balance",
            "a2.router_stability",
        ]
    if not method.cvar_training_weight_enabled:
        for phase in ("policy", "a1", "a2"):
            overrides["loss_weights"][f"{phase}.particle_cvar"] = 0.0
    if method.scenario_count == 1:
        overrides["capability_gate_overrides"] = {
            "scenario.expected_particle_count": 1,
            "scenario.minimum_effective_sample_size": 1.0,
            "scope": "preregistered_operational_ablation_only",
            "mainline_twenty_particle_gate_is_not_claimed": True,
        }
    return overrides


@dataclass(frozen=True)
class AblationRunSpec:
    run_id: str
    method_id: str
    family: str
    architecture_id: str
    seed: int
    split_scope: str
    phase_sequence: tuple[str, ...]
    execution_views: tuple[str, ...]
    scenario_count: int
    routing_enabled: bool
    cvar_training_weight_enabled: bool
    checkpoint_source: str
    preregistration_sha256: str
    shared_contract_sha256: str
    parameter_counts: Mapping[str, int]
    training_overrides: Mapping[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "method_id": self.method_id,
            "family": self.family,
            "architecture_id": self.architecture_id,
            "seed": self.seed,
            "split_scope": self.split_scope,
            "phase_sequence": list(self.phase_sequence),
            "execution_views": list(self.execution_views),
            "scenario_count": self.scenario_count,
            "routing_enabled": self.routing_enabled,
            "cvar_training_weight_enabled": self.cvar_training_weight_enabled,
            "checkpoint_source": self.checkpoint_source,
            "preregistration_sha256": self.preregistration_sha256,
            "shared_contract_sha256": self.shared_contract_sha256,
            "parameter_counts": dict(self.parameter_counts),
            "training_overrides": deepcopy(dict(self.training_overrides)),
        }


def build_execution_plan(
    *,
    registration: AblationPreregistration | None = None,
    methods: Sequence[str] = ALL_ABLATION_METHODS,
    seeds: Sequence[int] = FORMAL_SEEDS,
    split_scope: str = "train_validation",
) -> tuple[AblationRunSpec, ...]:
    """Build deterministic run specifications without touching formal data."""

    registration = registration or default_preregistration()
    validate_preregistration(registration)
    if split_scope != "train_validation":
        raise AblationContractError(
            "planning 2024 requires a separate frozen-method attestation; 2025 is closed"
        )
    registry = method_registry(registration)
    if len(set(methods)) != len(methods) or any(item not in registry for item in methods):
        raise AblationContractError("execution methods must be unique registered IDs")
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if any(seed not in FORMAL_SEEDS for seed in normalized_seeds):
        raise AblationContractError("execution plan contains a non-preregistered seed")
    if normalized_seeds != tuple(seed for seed in FORMAL_SEEDS if seed in normalized_seeds):
        raise AblationContractError("formal seeds must retain their preregistered order")
    counts = measure_registered_parameter_counts(methods)
    specs: list[AblationRunSpec] = []
    for method_id in methods:
        method = registry[method_id]
        if counts[method_id]["trainable"] > 5_000_000:
            raise AblationContractError(
                f"{method_id} exceeds the R4 hard parameter limit: {counts[method_id]}"
            )
        for seed in normalized_seeds:
            specs.append(
                AblationRunSpec(
                    run_id=f"r4_ablation_{method_id.lower()}_seed={seed}",
                    method_id=method_id,
                    family=method.family,
                    architecture_id=method.architecture_id,
                    seed=seed,
                    split_scope=split_scope,
                    phase_sequence=method.phase_sequence,
                    execution_views=method.execution_views,
                    scenario_count=method.scenario_count,
                    routing_enabled=method.routing_enabled,
                    cvar_training_weight_enabled=method.cvar_training_weight_enabled,
                    checkpoint_source=method.checkpoint_source,
                    preregistration_sha256=registration.digest,
                    shared_contract_sha256=registration.shared_contract_digest,
                    parameter_counts=counts[method_id],
                    training_overrides=method_training_overrides(method_id),
                )
            )
    return tuple(specs)


def write_execution_plan_once(
    path: str | Path,
    specs: Sequence[AblationRunSpec],
    *,
    registration: AblationPreregistration | None = None,
) -> Path:
    registration = registration or default_preregistration()
    payload = {
        "schema_version": "dcvpp.e2e_moe_hybrid_r4.ablation_execution_plan.v1",
        "status": "PLANNED_NO_RESULTS",
        "preregistration_sha256": registration.digest,
        "shared_contract_sha256": registration.shared_contract_digest,
        "runs": [spec.to_payload() for spec in specs],
    }
    destination = Path(path).resolve()
    if destination.exists():
        current = json.loads(destination.read_text(encoding="utf-8"))
        if current != payload:
            raise AblationContractError(
                f"immutable execution plan differs from existing file: {destination}"
            )
        return destination
    atomic_json(destination, payload)
    return destination


def _finite_tree(value: Any, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise MeasuredResultError(f"{label} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{label}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{label}[{index}]")
        return
    raise MeasuredResultError(f"{label} contains unsupported value {type(value).__name__}")


def _verify_artifact_reference(value: Any, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise MeasuredResultError(f"{label} must be an artifact reference")
    path = Path(str(value.get("path", ""))).resolve()
    digest = str(value.get("sha256", "")).lower()
    if not path.is_file():
        raise MeasuredResultError(f"{label} artifact does not exist: {path}")
    if len(digest) != 64 or sha256_file(path) != digest:
        raise MeasuredResultError(f"{label} SHA-256 mismatch")
    return path


def _verify_daily_records(
    path: Path,
    *,
    expected_dates: int,
    validation_split: bool,
) -> None:
    """Read the submitted paired-day table and validate its primary unit."""

    import pandas as pd

    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            frame = pd.read_parquet(path)
        elif suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix in {".jsonl", ".ndjson"}:
            frame = pd.read_json(path, lines=True)
        elif suffix == ".json":
            frame = pd.read_json(path)
        else:
            raise MeasuredResultError(
                "daily_records must be parquet, csv, json, or jsonl"
            )
    except MeasuredResultError:
        raise
    except Exception as exc:
        raise MeasuredResultError(f"daily_records is unreadable: {path}") from exc
    required = {
        "date",
        "method_operating_cost",
        "benchmark_operating_cost",
        "normalized_sla_shortfall",
        "independent_feasibility_fraction",
        "origins_executed",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise MeasuredResultError(f"daily_records lacks paired columns: {missing}")
    if len(frame) != expected_dates:
        raise MeasuredResultError(
            f"daily_records has {len(frame)} rows, expected {expected_dates}"
        )
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any() or dates.nunique() != expected_dates:
        raise MeasuredResultError("daily_records dates must be finite and unique")
    if validation_split and (
        (dates < pd.Timestamp("2023-07-01")).any()
        or (dates > pd.Timestamp("2023-12-31")).any()
    ):
        raise MeasuredResultError("validation daily_records contains an out-of-split date")
    numeric = frame[list(required - {"date"})].apply(pd.to_numeric, errors="coerce")
    if not torch.isfinite(torch.as_tensor(numeric.to_numpy(dtype="float64"))).all():
        raise MeasuredResultError("daily_records contains a non-finite required value")
    if not (numeric["origins_executed"] == ORIGINS_PER_DAY).all():
        raise MeasuredResultError("every paired day must contain all 96 origins")
    if not (numeric["independent_feasibility_fraction"] == 1.0).all():
        raise MeasuredResultError("daily_records contains an independently infeasible day")


def validate_measured_result(payload: Mapping[str, Any], spec: AblationRunSpec) -> None:
    """Validate one external measured result without inferring missing facts."""

    required = {
        "schema_version",
        "run_id",
        "method_id",
        "architecture_id",
        "seed",
        "split",
        "status",
        "measured",
        "synthetic",
        "preregistration_sha256",
        "shared_contract_sha256",
        "parameter_counts",
        "scenario_count",
        "routing_enabled",
        "cvar_training_weight_enabled",
        "checkpoint_source",
        "completed_phases",
        "execution_views",
        "dates_evaluated",
        "origins_evaluated",
        "metrics",
        "capability_gates",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise MeasuredResultError(f"measured result is missing fields: {missing}")
    expected_pairs = {
        "schema_version": (payload["schema_version"], RESULT_SCHEMA_VERSION),
        "run_id": (payload["run_id"], spec.run_id),
        "method_id": (payload["method_id"], spec.method_id),
        "architecture_id": (payload["architecture_id"], spec.architecture_id),
        "seed": (int(payload["seed"]), spec.seed),
        "preregistration_sha256": (
            payload["preregistration_sha256"],
            spec.preregistration_sha256,
        ),
        "shared_contract_sha256": (
            payload["shared_contract_sha256"],
            spec.shared_contract_sha256,
        ),
        "scenario_count": (int(payload["scenario_count"]), spec.scenario_count),
        "routing_enabled": (payload["routing_enabled"], spec.routing_enabled),
        "cvar_training_weight_enabled": (
            payload["cvar_training_weight_enabled"],
            spec.cvar_training_weight_enabled,
        ),
        "checkpoint_source": (payload["checkpoint_source"], spec.checkpoint_source),
        "execution_views": (tuple(payload["execution_views"]), spec.execution_views),
        "parameter_counts": (dict(payload["parameter_counts"]), dict(spec.parameter_counts)),
    }
    for name, (actual, expected) in expected_pairs.items():
        if actual != expected:
            raise MeasuredResultError(f"{name} differs from preregistration")
    if payload["measured"] is not True or payload["synthetic"] is not False:
        raise MeasuredResultError("only measured, non-synthetic results are admissible")
    if payload["status"] not in {"COMPLETE", "FAILED"}:
        raise MeasuredResultError("result status must be COMPLETE or FAILED")
    completed_phases = tuple(payload["completed_phases"])
    if payload["status"] == "COMPLETE":
        if completed_phases != spec.phase_sequence:
            raise MeasuredResultError("completed_phases differs from preregistration")
    elif completed_phases != spec.phase_sequence[: len(completed_phases)]:
        raise MeasuredResultError("FAILED result phases are not a valid execution prefix")
    if payload["split"] == "2025" or str(payload["split"]).startswith("2025"):
        raise MeasuredResultError("2025 is permanently closed")
    if payload["split"] not in {"validation", "final_temporal_test"}:
        raise MeasuredResultError("result split must be validation or final_temporal_test")
    _finite_tree(payload["metrics"], "metrics")
    _finite_tree(payload["capability_gates"], "capability_gates")
    if payload["status"] == "FAILED":
        if not str(payload.get("failure_reason", "")).strip():
            raise MeasuredResultError("FAILED result requires a concrete failure_reason")
        return
    capability_gates = payload["capability_gates"]
    if not isinstance(capability_gates, Mapping):
        raise MeasuredResultError("capability_gates must be a mapping")
    if set(capability_gates) != set(spec.phase_sequence):
        raise MeasuredResultError(
            "complete result must provide every executed phase capability gate"
        )
    for phase in spec.phase_sequence:
        gate = capability_gates[phase]
        if not isinstance(gate, Mapping) or gate.get("status") != "PASSED":
            raise MeasuredResultError(f"phase {phase} capability gate did not pass")
        _verify_artifact_reference(gate.get("evidence"), f"{phase}_gate_evidence")
    if payload["split"] == "validation":
        if int(payload["dates_evaluated"]) != VALIDATION_DATES:
            raise MeasuredResultError("complete validation requires all 179 dates")
        if int(payload["origins_evaluated"]) != VALIDATION_DATES * ORIGINS_PER_DAY:
            raise MeasuredResultError("complete validation requires all 17,184 origins")
    else:
        _verify_artifact_reference(payload.get("freeze_attestation"), "freeze_attestation")
        if int(payload["dates_evaluated"]) <= 0:
            raise MeasuredResultError("final test must report a positive frozen date count")
        if int(payload["origins_evaluated"]) != int(payload["dates_evaluated"]) * 96:
            raise MeasuredResultError("final test origin count must equal dates times 96")
    _verify_artifact_reference(payload.get("checkpoint"), "checkpoint")
    daily_path = _verify_artifact_reference(payload.get("daily_records"), "daily_records")
    _verify_daily_records(
        daily_path,
        expected_dates=int(payload["dates_evaluated"]),
        validation_split=payload["split"] == "validation",
    )
    metrics = payload["metrics"]
    if not isinstance(metrics, Mapping):
        raise MeasuredResultError("metrics must be a mapping")
    required_metrics = {
        "total_operating_cost",
        "weighted_cvar95",
        "normalized_sla_shortfall",
        "independent_feasibility_fraction",
        "corrective_intervention_rate",
        "inference_latency_ms",
    }
    missing_metrics = sorted(required_metrics - set(metrics))
    if missing_metrics:
        raise MeasuredResultError(f"complete result lacks measured metrics: {missing_metrics}")
    if float(metrics["independent_feasibility_fraction"]) != 1.0:
        raise MeasuredResultError("complete ablation result requires 100% independent feasibility")
    if spec.method_id == "no_cvar" and "weighted_cvar95" not in metrics:
        raise MeasuredResultError("No-CVaR still requires ex-post CVaR95 reporting")
    if spec.method_id == "raw_vs_shield":
        required_raw = {
            "raw_violation_rate",
            "shielded_feasibility_fraction",
            "correction_magnitude_mean",
            "correction_magnitude_p95",
        }
        if required_raw - set(metrics):
            raise MeasuredResultError("raw-vs-shield result lacks paired intervention metrics")
        if payload.get("raw_proposal_sha256") != payload.get(
            "shield_input_proposal_sha256"
        ):
            raise MeasuredResultError(
                "raw and shielded views were not derived from the same proposals"
            )


class AblationResultReceiver:
    """Immutable result sink bound to one preregistered execution plan."""

    def __init__(self, root: str | Path, specs: Sequence[AblationRunSpec]) -> None:
        self.root = Path(root).resolve()
        self.specs = {spec.run_id: spec for spec in specs}
        if len(self.specs) != len(specs):
            raise AblationContractError("execution plan has duplicate run IDs")

    def accept(self, payload: Mapping[str, Any]) -> Path:
        run_id = str(payload.get("run_id", ""))
        spec = self.specs.get(run_id)
        if spec is None:
            raise MeasuredResultError("result run_id is not preregistered")
        validate_measured_result(payload, spec)
        if spec.checkpoint_source != "fresh" and payload.get("status") == "COMPLETE":
            source_method = spec.checkpoint_source.removesuffix("_same_seed")
            source_path = self.root / source_method / f"seed={spec.seed}.json"
            if not source_path.is_file():
                raise MeasuredResultError(
                    f"{spec.method_id} requires the accepted {source_method} same-seed result"
                )
            source = json.loads(source_path.read_text(encoding="utf-8"))
            source_checkpoint = source.get("checkpoint")
            submitted_checkpoint = payload.get("checkpoint")
            if (
                source.get("status") != "COMPLETE"
                or not isinstance(source_checkpoint, Mapping)
                or not isinstance(submitted_checkpoint, Mapping)
                or source_checkpoint.get("sha256") != submitted_checkpoint.get("sha256")
            ):
                raise MeasuredResultError(
                    f"{spec.method_id} did not reuse the accepted {source_method} checkpoint"
                )
        destination = self.root / spec.method_id / f"seed={spec.seed}.json"
        normalized = deepcopy(dict(payload))
        normalized["result_sha256"] = sha256_json(dict(payload))
        if destination.exists():
            current = json.loads(destination.read_text(encoding="utf-8"))
            if current != normalized:
                raise MeasuredResultError(
                    f"immutable measured result already exists: {destination}"
                )
            return destination
        atomic_json(destination, normalized)
        return destination


__all__ = [
    "ABLATION_SCHEMA_VERSION",
    "ALL_ABLATION_METHODS",
    "ARCHITECTURE_METHODS",
    "OPERATIONAL_METHODS",
    "RESULT_SCHEMA_VERSION",
    "AblationContractError",
    "AblationMethod",
    "AblationPreregistration",
    "AblationResultReceiver",
    "AblationRunSpec",
    "ArchitectureAblation",
    "DenseTokenFFN",
    "GRUMarketEncoder",
    "LearnedOneScenarioGenerator",
    "MeasuredResultError",
    "OperationalAblation",
    "TokenGRUSequence",
    "build_ablation_model",
    "build_execution_plan",
    "default_preregistration",
    "measure_registered_parameter_counts",
    "method_registry",
    "method_training_overrides",
    "parameter_counts",
    "validate_measured_result",
    "validate_preregistration",
    "write_execution_plan_once",
]
