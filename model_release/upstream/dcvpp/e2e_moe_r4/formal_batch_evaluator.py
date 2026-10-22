"""Claim-bound real 2024 batch evaluator for frozen R4 methods.

The public :func:`formal_batch_evaluator` is the CLI ``module:function``
adapter.  It refuses direct execution unless
``claim_and_evaluate_final_once`` has already written the atomic ``CLAIMED``
ledger with exactly one evaluator invocation.  Only then may the production
backend open the explicitly registered 2024 dates.

Tests inject a synthetic backend.  The orchestration and cartesian row contract
are therefore testable without creating or opening a 2024/2025 payload.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
import gc
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import numpy as np
import torch

from .ablations import build_ablation_model, method_registry
from .artifact_io import atomic_json, read_json, utc_now
from .config import FORMAL_SEEDS, load_r4_config
from .constrained_runtime import R4Runtime, load_physical_limits
from .data import R4SelectionDay, load_r4_day, temporal_split
from .decision_capability_materialization import (
    _NormalizedDeploymentModel,
    _PriceAwareEDF,
    _distribution_support_from_config,
    _run_summary,
    _single_post_shield_distribution_replay,
)
from .final_evaluation import (
    DAILY_COLUMNS,
    DAILY_SCHEMA_VERSION,
    LEDGER_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    FinalProtocolError,
    verify_final_freeze_manifest,
)
from .hashing import sha256_file, sha256_json
from .physical_twin import build_formal_r4_model, load_frozen_physical_twin_core
from .preflight import verify_locked_path_guard
from .training_data import R4InputStatistics, load_normalization_artifact


EVALUATOR_EVIDENCE_SCHEMA = "dcvpp.e2e_moe_hybrid_r4.formal_batch_evaluator.v1"
_DATE_TOKEN = re.compile(r"(?<!\d)(20\d{2})[-_](\d{2})[-_](\d{2})(?!\d)")
_YEAR_TOKEN = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


class FormalEvaluationBackend(Protocol):
    """Dependency boundary used by the real and synthetic evaluators."""

    def prepare(self, plan: Mapping[str, Any], manifest: Mapping[str, Any]) -> None: ...

    def evaluate_day(
        self,
        method: Mapping[str, Any],
        training_seed: int,
        operating_date: str,
    ) -> Mapping[str, Any]: ...

    def finish(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class _CachedDay:
    day: R4SelectionDay
    deployment_record: Mapping[str, Any]
    bundle_path: Path
    settlement_path: Path
    bundle_sha256_before: str
    settlement_sha256_before: str


class _TimedPolicy:
    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.latencies_ms: list[float] = []

    def __call__(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        import time

        started = time.perf_counter()
        result = self.policy(context)
        self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
        return result


class _PassiveEDF:
    """Frozen work-conserving EDF proposal using the identical R4 shield."""

    def __init__(self, day: R4SelectionDay, physical: Any, physical_core: Any) -> None:
        self.capacity = np.asarray(physical.capacity_work_units, dtype=float)
        issue_total = np.minimum(
            np.maximum(day.workload[:, 0], 0.0), float(self.capacity.sum())
        )
        issue_work = issue_total[:, None] * self.capacity[None] / self.capacity.sum()
        plant = R4Runtime(
            physical, physical_twin_core=physical_core
        ).shield.plant
        load = np.stack(
            [plant.facility_load(day.physical_multiplier, row) for row in issue_work],
            axis=0,
        )
        issue_grid = np.maximum(load - day.physical_dynamic[:, :4], 0.0)
        self.day_ahead = np.minimum(
            issue_grid, np.asarray(physical.grid_import_limit_mw)[None]
        ).T

    def __call__(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        action: dict[str, Any] = {
            "compute": self.capacity,
            "battery": np.zeros(4, dtype=float),
        }
        if int(context["origin"]) == 0:
            action["day_ahead"] = self.day_ahead
        return {"actions": action}


class _CertifiedSafe:
    """By-construction fallback: every free proposal uses the shield default."""

    def __call__(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        del context
        return {"actions": {}}


class ProductionR4EvaluationBackend:
    """Real frozen-checkpoint + R4Runtime + official-settlement backend."""

    def __init__(self) -> None:
        self.plan: Mapping[str, Any] | None = None
        self.manifest: Mapping[str, Any] | None = None
        self.config: Mapping[str, Any] | None = None
        self.repository_root: Path | None = None
        self.statistics: R4InputStatistics | None = None
        self.physical: Any = None
        self.physical_core: Any = None
        self.corpus_view: Any = None
        self.days: dict[str, _CachedDay] = {}
        self._current_key: tuple[str, int] | None = None
        self._current_model: torch.nn.Module | None = None
        self._checkpoint_hashes_after: dict[str, str] = {}
        self._locked_guard_evidence: Mapping[str, Any] = {}
        self.distribution_support: Mapping[str, Any] | None = None

    def prepare(self, plan: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
        self.plan = plan
        self.manifest = manifest
        config_path = Path(str(manifest["config_path"])).resolve()
        self.config = load_r4_config(config_path)
        self.repository_root = config_path.parent.parent.resolve()
        self.distribution_support = _distribution_support_from_config(
            self.config, self.repository_root
        )
        self._locked_guard_evidence = verify_locked_path_guard(self.repository_root)
        self.statistics = self._load_frozen_statistics(manifest)
        self.physical, self.physical_core = self._load_physical_contract()
        self.corpus_view = SimpleNamespace(
            statistics=self.statistics,
            physical=self.physical,
            physical_core=self.physical_core,
        )
        self.days = self._load_formal_days(plan["eligible_dates"])

    def evaluate_day(
        self,
        method: Mapping[str, Any],
        training_seed: int,
        operating_date: str,
    ) -> Mapping[str, Any]:
        if operating_date not in self.days:
            raise FinalProtocolError(f"formal day was not loaded exactly once: {operating_date}")
        cached = self.days[operating_date]
        adapter_name = str(method["runtime_adapter"])
        if method["kind"] == "benchmark":
            return self._evaluate_benchmark(method, cached, adapter_name)
        model = self._activate_model(method, int(training_seed), adapter_name)
        assert self.config is not None and self.corpus_view is not None
        runtime_config = dict(self.config["runtime"])
        runtime_config.update(
            {
                "allow_sealed_selection": True,
                "inference_mode": True,
            }
        )
        runtime = R4Runtime(
            self.physical,
            runtime_config,
            physical_twin_core=self.physical_core,
        )
        deployment = _NormalizedDeploymentModel(
            model,
            self.corpus_view,
            cached.deployment_record,
            distribution_support=self.distribution_support,
        ).eval()
        result = runtime.run_day(cached.day, deployment)
        particle_replay = None
        if deployment.is_distributional:
            checkpoint_sha256 = str(
                method["evidence_by_seed"][str(int(training_seed))]["sha256"]
            )
            particle_replay = _single_post_shield_distribution_replay(
                day=cached.day,
                actual_result=result,
                adapter=deployment,
                model=model,
                corpus=self.corpus_view,
                deployment_record=cached.deployment_record,
                runtime_config=runtime_config,
                checkpoint_sha256=checkpoint_sha256,
            )
        summary = _run_summary(
            result,
            day=cached.day,
            corpus=self.corpus_view,
            adapter=deployment,
            particle_replay=particle_replay,
        )
        required_contracts = {
            "all_actions_independently_validated": summary.get(
                "all_actions_independently_validated"
            ),
            "decision_uncertainty_contract": summary.get(
                "decision_uncertainty_contract"
            ),
            "all_origins_causal_input_contract": summary.get(
                "all_origins_causal_input_contract"
            ),
            "artifact_piecewise_core_exact": summary.get(
                "artifact_piecewise_core_exact"
            ),
            "train_frozen_forecast_residual_cap_contract": summary.get(
                "train_frozen_forecast_residual_cap_contract"
            ),
        }
        failed = [name for name, value in required_contracts.items() if value is not True]
        if failed:
            raise FinalProtocolError(
                "formal neural runtime contract failed: " + ", ".join(failed)
            )
        if summary.get("post_shield_particle_cvar_contract") is not True:
            raise FinalProtocolError(
                "formal neural CVaR lacks a complete post-shield empirical-risk replay"
            )
        if deployment.is_distributional and (
            summary.get("post_shield_distribution_cvar_contract") is not True
            or int(summary.get("formal_empirical_risk_sample_count", 0)) != 512
        ):
            raise FinalProtocolError("formal V6 empirical risk is not N=512")
        return self._metrics_from_summary(summary)

    def finish(self) -> Mapping[str, Any]:
        self._release_model()
        for path_text, expected_sha256 in self._checkpoint_hashes_after.items():
            if sha256_file(Path(path_text)) != expected_sha256:
                raise FinalProtocolError(f"checkpoint changed during evaluation: {path_text}")
        data_records = []
        for operating_date, cached in self.days.items():
            bundle_after = sha256_file(cached.bundle_path)
            settlement_after = sha256_file(cached.settlement_path)
            if bundle_after != cached.bundle_sha256_before:
                raise FinalProtocolError(f"2024 bundle changed during evaluation: {operating_date}")
            if settlement_after != cached.settlement_sha256_before:
                raise FinalProtocolError(
                    f"official settlement changed during evaluation: {operating_date}"
                )
            data_records.append(
                {
                    "operating_date": operating_date,
                    "bundle_path": str(cached.bundle_path),
                    "bundle_sha256_before": cached.bundle_sha256_before,
                    "bundle_sha256_after": bundle_after,
                    "settlement_path": str(cached.settlement_path),
                    "settlement_sha256_before": cached.settlement_sha256_before,
                    "settlement_sha256_after": settlement_after,
                }
            )
        return {
            "backend": "ProductionR4EvaluationBackend",
            "locked_2025_path_before_open": dict(self._locked_guard_evidence),
            "data_sources": data_records,
            "checkpoint_hashes_after": dict(self._checkpoint_hashes_after),
            "formal_dates_loaded_once": len(data_records),
        }

    def _load_frozen_statistics(self, manifest: Mapping[str, Any]) -> R4InputStatistics:
        valid: list[tuple[Path, R4InputStatistics]] = []
        for record in manifest["artifact_groups"]["normalization"]:
            path = Path(str(record["path"]))
            try:
                statistics = load_normalization_artifact(path)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            valid.append((path, statistics))
        if len(valid) != 1:
            raise FinalProtocolError(
                "freeze must contain exactly one train-only R4 normalization artifact"
            )
        return valid[0][1]

    def _resolved_data_path(self, name: str) -> Path:
        assert self.config is not None and self.repository_root is not None
        value = self.config["data"][name]
        path = Path(str(value))
        return (
            (self.repository_root / path).resolve()
            if not path.is_absolute()
            else path.resolve()
        )

    def _load_physical_contract(self) -> tuple[Any, Any]:
        artifact = self._resolved_data_path("frozen_physical_twin")
        surrogate = self._resolved_data_path("frozen_physical_surrogate")
        calibration = self._resolved_data_path("physical_calibration_manifest")
        physical = load_physical_limits(artifact)
        core = load_frozen_physical_twin_core(
            artifact_path=artifact,
            surrogate_path=surrogate,
            calibration_manifest_path=calibration,
        )
        return physical, core

    def _load_formal_days(self, date_values: Sequence[str]) -> dict[str, _CachedDay]:
        bundle_root = self._resolved_data_path("selection_bundle_root")
        settlement_root = self._resolved_data_path("official_settlement_root")
        capacity = np.asarray(self.physical.capacity_work_units, dtype=float)
        histories: list[R4SelectionDay] = []
        first = date.fromisoformat(str(date_values[0])[:10])
        cursor = min(first - timedelta(days=1), date(2023, 12, 31))
        prior: list[R4SelectionDay] = []
        # Only 2023 validation paths may seed the first formal history.  No
        # unregistered 2024 date is ever opened for context construction.
        while len(prior) < 7 and cursor.year == 2023:
            bundle = bundle_root / f"{cursor.isoformat()}.npz"
            settlement = settlement_root / f"{cursor.isoformat()}.npz"
            _guard_payload_path(bundle, expected_date=cursor.isoformat(), allow_2024=False)
            _guard_payload_path(settlement, expected_date=cursor.isoformat(), allow_2024=False)
            if bundle.is_file() and settlement.is_file():
                prior.append(
                    load_r4_day(
                        bundle,
                        settlement,
                        allow_sealed_selection=False,
                        capacity_work_units=capacity,
                        physical_twin_core=self.physical_core,
                    )
                )
            cursor -= timedelta(days=1)
        histories.extend(reversed(prior))
        cached: dict[str, _CachedDay] = {}
        for operating_date in date_values:
            bundle = bundle_root / f"{operating_date}.npz"
            settlement = settlement_root / f"{operating_date}.npz"
            _guard_payload_path(bundle, expected_date=operating_date, allow_2024=True)
            _guard_payload_path(settlement, expected_date=operating_date, allow_2024=True)
            if not bundle.is_file() or not settlement.is_file():
                raise FinalProtocolError(
                    f"eligible 2024 bundle/official settlement is absent: {operating_date}"
                )
            bundle_sha = sha256_file(bundle)
            settlement_sha = sha256_file(settlement)
            day = load_r4_day(
                bundle,
                settlement,
                allow_sealed_selection=True,
                capacity_work_units=capacity,
                physical_twin_core=self.physical_core,
            )
            if day.operating_date != operating_date or temporal_split(operating_date) != "sealed_selection":
                raise FinalProtocolError(f"formal date identity mismatch: {operating_date}")
            record = self._deployment_record(day, histories)
            cached[operating_date] = _CachedDay(
                day=day,
                deployment_record=record,
                bundle_path=bundle,
                settlement_path=settlement,
                bundle_sha256_before=bundle_sha,
                settlement_sha256_before=settlement_sha,
            )
            histories.append(day)
        if list(cached) != list(date_values):
            raise FinalProtocolError("formal dates were not loaded once in frozen order")
        return cached

    def _deployment_record(
        self, day: R4SelectionDay, historical_days: Sequence[R4SelectionDay]
    ) -> dict[str, Any]:
        assert self.statistics is not None
        historical = np.zeros((7, 16), dtype=np.float32)
        available = np.zeros(7, dtype=bool)
        selected = [value.market for value in historical_days[-7:]]
        offset = 7 - len(selected)
        for index, market in enumerate(selected, start=offset):
            historical[index] = np.arcsinh(
                np.asarray(market) / self.statistics.market_scale
            ).mean(axis=0)
            available[index] = True
        operating = date.fromisoformat(day.operating_date)
        local = datetime(
            operating.year,
            operating.month,
            operating.day,
            12,
            tzinfo=ZoneInfo("America/New_York"),
        )
        issue_text = str(day.issue_time_utc).replace("Z", "+00:00")
        issue_time = datetime.fromisoformat(issue_text)
        if issue_time.tzinfo is None:
            raise FinalProtocolError("formal issue_time_utc must be timezone-aware")
        issue_time = issue_time.astimezone(timezone.utc)
        da_anchor: np.ndarray | None = None
        for prior_day in reversed(historical_days):
            completed = datetime.combine(
                date.fromisoformat(prior_day.operating_date) + timedelta(days=1),
                datetime_time.min,
                tzinfo=timezone.utc,
            )
            if completed <= issue_time:
                da_anchor = np.asarray(prior_day.realized_da_price, dtype=np.float32).T
                break
        if da_anchor is None or da_anchor.shape != (96, 4):
            raise FinalProtocolError(
                "formal deployment lacks the latest completed-day causal DA anchor"
            )
        pv_anchor = np.asarray(day.physical_dynamic[:, :4], dtype=np.float32)
        residual_caps = {
            "da": torch.as_tensor(
                self.statistics.forecast_da_residual_cap, dtype=torch.float32
            ),
            "pv": torch.as_tensor(
                self.statistics.forecast_pv_residual_cap, dtype=torch.float32
            ),
            "pue": torch.as_tensor(
                self.statistics.forecast_pue_residual_cap, dtype=torch.float32
            ),
        }
        residual_cap_metadata = {
            "schema": self.statistics.as_json()["forecast_residual_cap_schema"],
            "fit_split": "train",
            "quantile": 0.95,
            "units": "raw_target_units_by_zone",
            "sha256": self.statistics.forecast_residual_cap_sha256(),
        }
        return {
            "market_raw": torch.as_tensor(np.asarray(day.market), dtype=torch.float32),
            "issue_workload_raw": torch.as_tensor(
                np.asarray(day.workload[:, 0]), dtype=torch.float32
            ),
            "issue_pv_raw": torch.as_tensor(
                pv_anchor, dtype=torch.float32
            ),
            "forecast_anchors": {
                "da": torch.as_tensor(da_anchor, dtype=torch.float32),
                "pv": torch.as_tensor(pv_anchor, dtype=torch.float32),
            },
            "causal_da_anchor_available": torch.tensor(True, dtype=torch.bool),
            "forecast_residual_caps": residual_caps,
            "forecast_residual_cap_metadata": residual_cap_metadata,
            "forecast_residual_caps_sha256": (
                self.statistics.forecast_residual_cap_sha256()
            ),
            "historical_market": torch.as_tensor(historical, dtype=torch.float32),
            "historical_market_available": torch.as_tensor(available, dtype=torch.bool),
            "day_of_week": operating.weekday(),
            "month": operating.month - 1,
            "holiday": 0,
            "dst": int(bool(local.dst())),
            "operating_date": day.operating_date,
        }

    def _activate_model(
        self, method: Mapping[str, Any], seed: int, adapter_name: str
    ) -> torch.nn.Module:
        key = (str(method["method_id"]), int(seed))
        if self._current_key == key and self._current_model is not None:
            return self._current_model
        self._release_model()
        assert self.config is not None and self.repository_root is not None
        if adapter_name == "r4_mainline":
            model = build_formal_r4_model(self.config, self.repository_root)
        elif adapter_name.startswith("r4_ablation:"):
            method_id = adapter_name.split(":", 1)[1]
            if method_id not in method_registry():
                raise FinalProtocolError(f"unknown frozen R4 ablation adapter: {method_id}")
            base = build_formal_r4_model(self.config, self.repository_root)
            core = base.physical_stream.physical_twin_core
            model = build_ablation_model(
                method_id,
                base.config,
                physical_twin_core=core,
            )
            del base
        else:
            raise FinalProtocolError(f"unsupported neural runtime adapter: {adapter_name}")
        evidence = method["evidence_by_seed"][str(seed)]
        checkpoint_path = Path(str(evidence["path"]))
        if sha256_file(checkpoint_path) != evidence["sha256"]:
            raise FinalProtocolError(f"checkpoint changed before load: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("model_state"), Mapping):
            raise FinalProtocolError(f"frozen checkpoint has no model_state: {checkpoint_path}")
        if int(checkpoint.get("seed", seed)) != seed:
            raise FinalProtocolError(f"checkpoint seed mismatch: {checkpoint_path}")
        if (
            int(checkpoint.get("horizon", -1)) != 96
            or not checkpoint.get("protocol_version")
            or not checkpoint.get("lineage_id")
            or checkpoint.get("config_sha256") != self.config.get("_config_sha256")
        ):
            raise FinalProtocolError(
                f"checkpoint lineage/config attestation failed: {checkpoint_path}"
            )
        if not _checkpoint_capability_passed(checkpoint):
            raise FinalProtocolError(f"checkpoint lacks a frozen capability PASS: {checkpoint_path}")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval().requires_grad_(False)
        if not torch.cuda.is_available():
            raise FinalProtocolError("formal final evaluator requires the preflight CUDA device")
        model.to(torch.device("cuda:0"))
        self._current_key = key
        self._current_model = model
        self._checkpoint_hashes_after[str(checkpoint_path)] = evidence["sha256"]
        return model

    def _release_model(self) -> None:
        if self._current_model is not None:
            self._current_model.to("cpu")
        self._current_model = None
        self._current_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _evaluate_benchmark(
        self,
        method: Mapping[str, Any],
        cached: _CachedDay,
        adapter_name: str,
    ) -> Mapping[str, Any]:
        evidence = method["evidence_by_seed"]["0"]
        definition_path = Path(str(evidence["path"]))
        definition = read_json(definition_path)
        if definition.get("final_test_frozen") is not True:
            raise FinalProtocolError(
                f"benchmark parameters were not frozen before final test: {definition_path}"
            )
        if str(definition.get("name", "")) != adapter_name:
            raise FinalProtocolError(f"benchmark definition/adapter mismatch: {adapter_name}")
        if adapter_name == "price_aware_edf_shield":
            if (
                definition.get("schema_version")
                != "dcvpp.r4.price_aware_edf_shield.v2"
                or definition.get("calibration_split") != ["train", "validation"]
            ):
                raise FinalProtocolError("price-aware EDF calibration split changed")
            low = np.asarray(definition["low_rt_issue_price_by_zone"], dtype=float)
            high = np.asarray(definition["high_rt_issue_price_by_zone"], dtype=float)
            policy = _PriceAwareEDF(
                cached.day,
                self.corpus_view,
                low,
                high,
            )
        elif adapter_name == "passive_edf_shield":
            policy = _PassiveEDF(cached.day, self.physical, self.physical_core)
        elif adapter_name == "certified_safe_policy":
            policy = _CertifiedSafe()
        else:
            raise FinalProtocolError(f"unsupported frozen benchmark adapter: {adapter_name}")
        timed = _TimedPolicy(policy)
        assert self.config is not None
        runtime_config = dict(self.config["runtime"])
        runtime_config.update({"allow_sealed_selection": True, "inference_mode": True})
        result = R4Runtime(
            self.physical,
            runtime_config,
            physical_twin_core=self.physical_core,
        ).run_day(cached.day, timed)
        summary = _run_summary(
            result,
            day=cached.day,
            corpus=self.corpus_view,
            adapter=None,
        )
        if (
            summary.get("all_actions_independently_validated") is not True
            or summary.get("artifact_piecewise_core_exact") is not True
        ):
            raise FinalProtocolError("formal benchmark runtime/physical contract failed")
        if summary.get("post_shield_particle_cvar_contract") is not True:
            raise FinalProtocolError(
                "benchmark CVaR is unavailable without shared 20-particle post-shield replay"
            )
        metrics = self._metrics_from_summary(summary)
        metrics["inference_latency_ms_b1"] = float(
            np.mean(timed.latencies_ms) if timed.latencies_ms else 0.0
        )
        return metrics

    @staticmethod
    def _metrics_from_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
        completed = int(summary["completed_origins"])
        validated = int(summary["independently_validated_origins"])
        values = {
            "origins_evaluated": 96,
            "operating_cost_usd": float(summary["operating_cost_usd"]),
            "weighted_particle_cvar95_usd": float(summary["weighted_cvar95_usd"]),
            "sla_shortfall_raw": float(summary["sla_shortfall_raw_work_units"]),
            "sla_shortfall_normalized": float(summary["normalized_sla_shortfall"]),
            "independent_feasibility_fraction": validated / 96.0,
            "rejected_origins": 96 - completed,
            "corrective_intervention_rate": float(summary["corrective_intervention_rate"]),
            "corrective_intervention_mean": float(summary["corrective_intervention_mean"]),
            "corrective_intervention_p95": float(summary["corrective_intervention_p95"]),
            "inference_latency_ms_b1": float(summary["inference_latency_ms"]),
        }
        if not all(math.isfinite(float(value)) for key, value in values.items() if key not in {"origins_evaluated", "rejected_origins"}):
            raise FinalProtocolError("formal daily runtime produced a non-finite metric")
        return values


def formal_batch_evaluator(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """CLI adapter: real frozen methods on official eligible 2024 settlements."""

    return evaluate_claimed_final_plan(plan, backend=ProductionR4EvaluationBackend())


def evaluate_claimed_final_plan(
    plan: Mapping[str, Any],
    *,
    backend: FormalEvaluationBackend,
) -> list[dict[str, Any]]:
    """Execute the exact frozen method/seed/date cartesian product once."""

    manifest_path, manifest, ledger_path = _validate_claimed_plan(plan)
    pre_hashes = _method_evidence_hashes(manifest)
    backend.prepare(plan, manifest)
    rows: list[dict[str, Any]] = []
    for method in manifest["methods"]:
        for training_seed in method["training_seeds"]:
            evidence_sha = method["evidence_by_seed"][str(training_seed)]["sha256"]
            for operating_date in manifest["eligible_dates"]:
                measured = dict(
                    backend.evaluate_day(method, int(training_seed), operating_date)
                )
                expected_metrics = set(DAILY_COLUMNS) - {
                    "schema_version",
                    "operating_date",
                    "method_id",
                    "method_kind",
                    "training_seed",
                    "frozen_evidence_sha256",
                    "freeze_sha256",
                }
                if set(measured) != expected_metrics:
                    raise FinalProtocolError(
                        f"backend daily metric schema mismatch for {method['method_id']}/{training_seed}/{operating_date}"
                    )
                rows.append(
                    {
                        "schema_version": DAILY_SCHEMA_VERSION,
                        "operating_date": operating_date,
                        "method_id": method["method_id"],
                        "method_kind": method["kind"],
                        "training_seed": int(training_seed),
                        **measured,
                        "frozen_evidence_sha256": evidence_sha,
                        "freeze_sha256": manifest["freeze_sha256"],
                    }
                )
    backend_evidence = dict(backend.finish())
    post_manifest = verify_final_freeze_manifest(manifest_path)
    post_hashes = _method_evidence_hashes(post_manifest)
    if pre_hashes != post_hashes:
        raise FinalProtocolError("frozen method evidence changed during final evaluation")
    expected_rows = sum(
        len(method["training_seeds"]) * len(manifest["eligible_dates"])
        for method in manifest["methods"]
    )
    if len(rows) != expected_rows:
        raise FinalProtocolError("formal backend did not produce the exact cartesian product")
    evidence_path = ledger_path.parent / "formal_batch_evaluator_evidence.json"
    atomic_json(
        evidence_path,
        {
            "schema_version": EVALUATOR_EVIDENCE_SCHEMA,
            "status": "COMPLETE_PENDING_OUTER_ROW_VALIDATION",
            "created_at_utc": utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "freeze_sha256": manifest["freeze_sha256"],
            "method_evidence_sha256_before": pre_hashes,
            "method_evidence_sha256_after": post_hashes,
            "eligible_dates": list(manifest["eligible_dates"]),
            "eligible_dates_sha256": manifest["eligible_dates_sha256"],
            "rows_returned": len(rows),
            "rows_identity_sha256": sha256_json(
                [
                    [row["method_id"], row["training_seed"], row["operating_date"]]
                    for row in rows
                ]
            ),
            "backend_evidence": backend_evidence,
            "formal_selection_2024_opened": True,
            "locked_test_2025_opened": False,
        },
    )
    return rows


def _validate_claimed_plan(
    plan: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], Path]:
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise FinalProtocolError("formal batch evaluator received an unknown plan schema")
    unsigned = dict(plan)
    claimed_plan_sha = unsigned.pop("plan_sha256", None)
    if claimed_plan_sha != sha256_json(unsigned):
        raise FinalProtocolError("formal evaluation plan digest is invalid")
    manifest_path = Path(str(plan.get("freeze_manifest_path", ""))).resolve()
    manifest = verify_final_freeze_manifest(manifest_path)
    if plan.get("freeze_sha256") != manifest["freeze_sha256"]:
        raise FinalProtocolError("formal evaluation plan belongs to another freeze")
    if plan.get("eligible_dates") != manifest["eligible_dates"]:
        raise FinalProtocolError("formal evaluation dates changed after claim")
    if plan.get("methods") != manifest["methods"]:
        raise FinalProtocolError("formal evaluation methods changed after claim")
    if any(date.fromisoformat(str(value)[:10]).year != 2024 for value in plan["eligible_dates"]):
        raise FinalProtocolError("formal evaluation plan contains a non-2024 date")
    ledger_path = manifest_path.parent.parent / "final_temporal_test" / "execution_ledger.json"
    if not ledger_path.is_file():
        raise FinalProtocolError("formal evaluator has no atomic CLAIMED ledger")
    ledger = read_json(ledger_path)
    if (
        ledger.get("schema_version") != LEDGER_SCHEMA_VERSION
        or ledger.get("status") != "CLAIMED"
        or int(ledger.get("evaluator_invocations", -1)) != 1
        or ledger.get("plan_sha256") != claimed_plan_sha
        or ledger.get("freeze_sha256") != manifest["freeze_sha256"]
        or ledger.get("formal_selection_2024_opened") is not False
        or ledger.get("locked_test_2025_opened") is not False
    ):
        raise FinalProtocolError(
            "formal evaluator is not inside the unique atomic claim authorization"
        )
    return manifest_path, manifest, ledger_path


def _method_evidence_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for method in manifest["methods"]:
        for seed, record in method["evidence_by_seed"].items():
            path = Path(str(record["path"]))
            digest = sha256_file(path)
            if digest != record["sha256"]:
                raise FinalProtocolError(f"frozen method evidence changed: {path}")
            result[f"{method['method_id']}|{seed}"] = digest
    return result


def _checkpoint_capability_passed(checkpoint: Mapping[str, Any]) -> bool:
    if str(checkpoint.get("capability_gate_status", "")).upper() in {"PASS", "PASSED"}:
        return True
    gate = checkpoint.get("selection_gate", checkpoint.get("capability_gate"))
    return isinstance(gate, Mapping) and str(gate.get("status", "")).upper() in {
        "PASS",
        "PASSED",
    }


def _guard_payload_path(
    path: Path,
    *,
    expected_date: str,
    allow_2024: bool,
) -> None:
    text = str(path)
    years = {int(value) for value in _YEAR_TOKEN.findall(text)}
    if 2025 in years:
        raise FinalProtocolError(f"refusing 2025 path before open: {path}")
    tokens = []
    for year, month, day in _DATE_TOKEN.findall(text):
        tokens.append(date(int(year), int(month), int(day)).isoformat())
    if not tokens or tokens[-1] != expected_date:
        raise FinalProtocolError(f"payload path/date mismatch before open: {path}")
    expected_year = date.fromisoformat(expected_date).year
    if expected_year == 2024 and not allow_2024:
        raise FinalProtocolError(f"unclaimed 2024 path refused before open: {path}")
    if expected_year not in ({2024} if allow_2024 else {2022, 2023}):
        raise FinalProtocolError(f"unsupported temporal payload path: {path}")


__all__ = [
    "EVALUATOR_EVIDENCE_SCHEMA",
    "FormalEvaluationBackend",
    "ProductionR4EvaluationBackend",
    "evaluate_claimed_final_plan",
    "formal_batch_evaluator",
]
