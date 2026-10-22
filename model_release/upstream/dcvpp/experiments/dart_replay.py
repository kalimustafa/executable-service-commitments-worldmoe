from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from dcvpp.core import atomic_json, sha256_json


DART_REPLAY_SCHEMA_VERSION = "dart_value_select_replay_panel_v2"
DART_VALUE_SELECT_SEEDS = (20260809, 20260819, 20260829)
EXPERIMENT_IDS = ("A1", "A2")

REQUIRED_EVIDENCE_HASHES = {
    "dart_model_sha256",
    "dart_forecast_sha256",
    "da_forecast_sha256",
    "load_forecast_sha256",
    "workload_forecast_sha256",
    "pv_forecast_sha256",
    "calibration_sha256",
    "dependence_sha256",
    "scenario_builder_sha256",
    "rt_scenario_sha256",
    "nyiso_replay_sha256",
    "workload_replay_sha256",
    "pv_replay_sha256",
    "physical_replay_sha256",
    "user_sha256",
    "twin_sha256",
    "optimizer_sha256",
    "replay_engine_sha256",
    "issue_time_panel_sha256",
    "calibration_panel_sha256",
    "tail_threshold_contract_sha256",
}

A1_ALLOWED_TO_DIFFER = {
    "dart_model_sha256",
    "dart_forecast_sha256",
    "rt_scenario_sha256",
}
A2_ALLOWED_TO_DIFFER = {
    "dart_model_sha256",
    "dart_forecast_sha256",
    "da_forecast_sha256",
    "load_forecast_sha256",
    "workload_forecast_sha256",
    "pv_forecast_sha256",
    "calibration_sha256",
    "dependence_sha256",
    "scenario_builder_sha256",
    "rt_scenario_sha256",
}
CANDIDATE_INDEPENDENT_HASHES = {
    "nyiso_replay_sha256",
    "workload_replay_sha256",
    "pv_replay_sha256",
    "physical_replay_sha256",
    "user_sha256",
    "twin_sha256",
    "optimizer_sha256",
    "replay_engine_sha256",
    "issue_time_panel_sha256",
    "calibration_panel_sha256",
    "tail_threshold_contract_sha256",
}


def _valid_sha256(value: Any) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"not a SHA-256 digest: {value!r}")
    return digest


def _safe_name(value: str) -> str:
    output = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not output:
        raise ValueError("identifier cannot be converted to a safe path component")
    return output


@dataclass(frozen=True)
class CandidateEvidence:
    """Hash-only identity contract for one candidate in A1 or A2."""

    candidate_id: str
    experiment_id: str
    hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        if self.experiment_id not in EXPERIMENT_IDS:
            raise ValueError(f"experiment_id must be one of {EXPERIMENT_IDS}")
        missing = REQUIRED_EVIDENCE_HASHES.difference(self.hashes)
        if missing:
            raise ValueError(f"candidate evidence is missing hashes: {sorted(missing)}")
        normalized = {str(key): _valid_sha256(value) for key, value in self.hashes.items()}
        object.__setattr__(self, "hashes", normalized)

    @property
    def evidence_sha256(self) -> str:
        return sha256_json(
            {
                "candidate_id": self.candidate_id,
                "experiment_id": self.experiment_id,
                "hashes": dict(self.hashes),
            }
        )


def assert_experiment_identity(
    candidates: Sequence[CandidateEvidence],
) -> dict[str, Any]:
    """Enforce the exact A1/A2 attribution boundary before any solve."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    experiment_id = candidates[0].experiment_id
    if any(candidate.experiment_id != experiment_id for candidate in candidates):
        raise ValueError("identity gate accepts one experiment at a time")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate_id values must be unique within an experiment")
    key_sets = {frozenset(candidate.hashes) for candidate in candidates}
    if len(key_sets) != 1:
        raise ValueError("all candidates must declare the same evidence-hash keys")
    allowed = A1_ALLOWED_TO_DIFFER if experiment_id == "A1" else A2_ALLOWED_TO_DIFFER
    differing: list[str] = []
    fixed: dict[str, str] = {}
    for key in sorted(candidates[0].hashes):
        values = {candidate.hashes[key] for candidate in candidates}
        if len(values) == 1:
            fixed[key] = next(iter(values))
        elif key in allowed:
            differing.append(key)
        else:
            raise ValueError(
                f"{experiment_id} identity gate failed: forbidden candidate-varying hash {key}"
            )
    if experiment_id == "A1" and "da_forecast_sha256" not in fixed:
        raise ValueError("A1 requires one fixed DA forecast for every DART candidate")
    missing_fixed = CANDIDATE_INDEPENDENT_HASHES.difference(fixed)
    if missing_fixed:
        raise ValueError(
            f"{experiment_id} lacks candidate-independent replay identities: {sorted(missing_fixed)}"
        )
    return {
        "experiment_id": experiment_id,
        "candidate_ids": sorted(candidate.candidate_id for candidate in candidates),
        "candidate_count": len(candidates),
        "allowed_candidate_varying_hashes": sorted(allowed),
        "observed_candidate_varying_hashes": differing,
        "fixed_hashes": fixed,
        "fixed_identity_sha256": sha256_json(fixed),
        "passed": True,
    }


@dataclass(frozen=True)
class ReplayCell:
    operating_date: str
    candidate_id: str
    experiment_id: str
    seed: int
    evidence_hashes: Mapping[str, str]
    identity_sha256: str
    cell_sha256: str
    schema_version: str = DART_REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.experiment_id not in EXPERIMENT_IDS:
            raise ValueError("ReplayCell has an unknown experiment_id")
        if self.seed not in DART_VALUE_SELECT_SEEDS:
            raise ValueError("ReplayCell seed is outside the amended three-seed panel")
        normalized = {str(key): _valid_sha256(value) for key, value in self.evidence_hashes.items()}
        object.__setattr__(self, "evidence_hashes", normalized)
        if self.cell_sha256 != sha256_json(self._hash_payload()):
            raise ValueError("ReplayCell SHA-256 does not match its payload")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "operating_date": self.operating_date,
            "candidate_id": self.candidate_id,
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "evidence_hashes": dict(self.evidence_hashes),
            "identity_sha256": self.identity_sha256,
        }

    def to_mapping(self) -> dict[str, Any]:
        output = self._hash_payload()
        output["cell_sha256"] = self.cell_sha256
        return output


def build_common_replay_panel(
    operating_dates: Sequence[str],
    candidates_by_experiment: Mapping[str, Sequence[CandidateEvidence]],
    *,
    seeds: Sequence[int] = DART_VALUE_SELECT_SEEDS,
) -> dict[str, Any]:
    """Build the exact date x candidate x A1/A2 x three-seed panel."""

    dates = tuple(sorted(map(str, operating_dates)))
    if not dates or len(set(dates)) != len(dates):
        raise ValueError("operating_dates must be nonempty and unique")
    seed_tuple = tuple(int(seed) for seed in seeds)
    if seed_tuple != DART_VALUE_SELECT_SEEDS:
        raise ValueError(f"seeds must be exactly {DART_VALUE_SELECT_SEEDS}")
    if set(candidates_by_experiment) != set(EXPERIMENT_IDS):
        raise ValueError(f"candidate mapping must contain exactly {EXPERIMENT_IDS}")
    audits: dict[str, dict[str, Any]] = {}
    normalized: dict[str, dict[str, CandidateEvidence]] = {}
    for experiment_id in EXPERIMENT_IDS:
        candidates = tuple(candidates_by_experiment[experiment_id])
        audits[experiment_id] = assert_experiment_identity(candidates)
        normalized[experiment_id] = {candidate.candidate_id: candidate for candidate in candidates}
    candidate_ids = tuple(sorted(normalized["A1"]))
    if candidate_ids != tuple(sorted(normalized["A2"])):
        raise ValueError("A1 and A2 must contain the same candidate IDs")
    for key in sorted(CANDIDATE_INDEPENDENT_HASHES):
        values = {
            normalized[experiment_id][candidate_id].hashes[key]
            for experiment_id in EXPERIMENT_IDS
            for candidate_id in candidate_ids
        }
        if len(values) != 1:
            raise ValueError(f"common A1/A2 panel changed candidate-independent hash {key}")
    cells: list[ReplayCell] = []
    for operating_date in dates:
        for candidate_id in candidate_ids:
            for experiment_id in EXPERIMENT_IDS:
                evidence = normalized[experiment_id][candidate_id]
                for seed in seed_tuple:
                    provisional = {
                        "schema_version": DART_REPLAY_SCHEMA_VERSION,
                        "operating_date": operating_date,
                        "candidate_id": candidate_id,
                        "experiment_id": experiment_id,
                        "seed": seed,
                        "evidence_hashes": dict(evidence.hashes),
                        "identity_sha256": audits[experiment_id]["fixed_identity_sha256"],
                    }
                    cells.append(
                        ReplayCell(
                            operating_date=operating_date,
                            candidate_id=candidate_id,
                            experiment_id=experiment_id,
                            seed=seed,
                            evidence_hashes=evidence.hashes,
                            identity_sha256=audits[experiment_id]["fixed_identity_sha256"],
                            cell_sha256=sha256_json(provisional),
                        )
                    )
    expected_count = len(dates) * len(candidate_ids) * len(EXPERIMENT_IDS) * len(seed_tuple)
    if len(cells) != expected_count:
        raise AssertionError("common DART replay panel is incomplete")
    return {
        "schema_version": DART_REPLAY_SCHEMA_VERSION,
        "operating_dates": list(dates),
        "candidate_ids": list(candidate_ids),
        "experiments": list(EXPERIMENT_IDS),
        "seeds": list(seed_tuple),
        "identity_audits": audits,
        "cell_count": len(cells),
        "cells": cells,
        "panel_sha256": sha256_json([cell.to_mapping() for cell in cells]),
    }


def _cost(value: Mapping[str, Any], *, name: str) -> float:
    if "simulated_total_cost_usd" not in value:
        raise ValueError(f"{name} lacks simulated_total_cost_usd")
    output = float(value["simulated_total_cost_usd"])
    if not np.isfinite(output):
        raise ValueError(f"{name} total cost is not finite")
    return output


def decision_value_metrics(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    perfect_information: Mapping[str, Any],
    *,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Compute cost-based value, regret, and PCR without a revenue claim."""

    candidate_cost = _cost(candidate, name="candidate")
    baseline_cost = _cost(baseline, name="baseline")
    pi_cost = _cost(perfect_information, name="perfect_information")
    if pi_cost > min(candidate_cost, baseline_cost) + tolerance:
        raise ValueError("perfect information is not a cost lower bound")
    incremental_value = baseline_cost - candidate_cost
    opportunity = baseline_cost - pi_cost
    pcr = None if abs(opportunity) <= tolerance else incremental_value / opportunity
    component_values: dict[str, float] = {}
    for key in sorted(set(candidate).intersection(baseline)):
        if key == "simulated_total_cost_usd" or not key.endswith("_cost_usd"):
            continue
        try:
            candidate_component = float(candidate[key])
            baseline_component = float(baseline[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(candidate_component) and np.isfinite(baseline_component):
            component_values[key.replace("_cost_usd", "_value_usd")] = (
                baseline_component - candidate_component
            )
    baseline_planner_lower = baseline.get("planner_objective_lower_bound")
    candidate_planner_upper = candidate.get("planner_objective_upper_bound")
    try:
        baseline_planner_lower = (
            None
            if baseline_planner_lower is None
            else float(baseline_planner_lower)
        )
        candidate_planner_upper = (
            None
            if candidate_planner_upper is None
            else float(candidate_planner_upper)
        )
    except (TypeError, ValueError):
        baseline_planner_lower = None
        candidate_planner_upper = None
    planner_bounds_available = bool(
        baseline_planner_lower is not None
        and candidate_planner_upper is not None
        and np.isfinite(baseline_planner_lower)
        and np.isfinite(candidate_planner_upper)
    )
    planner_certificate = (
        float(baseline_planner_lower - candidate_planner_upper)
        if planner_bounds_available
        else None
    )
    return {
        "candidate_simulated_total_cost_usd": candidate_cost,
        "baseline_simulated_total_cost_usd": baseline_cost,
        "perfect_information_cost_lower_bound_usd": pi_cost,
        "incremental_operational_value_usd": incremental_value,
        "prescriptive_regret_usd": candidate_cost - pi_cost,
        "profit_capture_ratio": pcr,
        "component_operational_value_usd": component_values,
        "planner_objective_conservative_certificate_usd": planner_certificate,
        "planner_objective_conservative_certificate_status": (
            "OK" if planner_bounds_available else "MISSING_EXPLICIT_PLANNER_BOUNDS"
        ),
        "planner_objective_conservative_certificate_formula": (
            "baseline_planner_objective_lower_bound_minus_"
            "candidate_planner_objective_upper_bound"
        ),
        "planner_objective_conservative_certificate_scope": (
            "issue_time_planner_objective_only_not_realized_settlement_value"
        ),
        "planner_certificate_replaces_primary_realized_value": False,
        "counterfactual": True,
        "realized_commercial_profit_claim": False,
        "perfect_information_deployable": False,
    }


CellRunner = Callable[[ReplayCell], Mapping[str, Any]]
BaselineRunner = Callable[[str, str, int], Mapping[str, Any]]
PerfectInformationRunner = Callable[[str], Mapping[str, Any]]


def _load_complete(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("input_sha256") != fingerprint:
        raise ValueError(f"checkpoint fingerprint mismatch: {path}")
    if payload.get("status") != "COMPLETE":
        return None
    if payload.get("output_sha256") != sha256_json(payload.get("result")):
        raise ValueError(f"checkpoint result hash mismatch: {path}")
    return payload


def _run_shared(
    path: Path,
    *,
    fingerprint: str,
    kind: str,
    runner: Callable[[], Mapping[str, Any]],
) -> tuple[dict[str, Any], bool]:
    cached = _load_complete(path, fingerprint)
    if cached is not None:
        return dict(cached["result"]), True
    result = dict(runner())
    _cost(result, name=kind)
    payload = {
        "schema_version": DART_REPLAY_SCHEMA_VERSION,
        "status": "COMPLETE",
        "kind": kind,
        "input_sha256": fingerprint,
        "result": result,
        "output_sha256": sha256_json(result),
    }
    atomic_json(path, payload)
    return result, False


def run_dart_replay_panel(
    panel: Mapping[str, Any],
    checkpoint_root: Path,
    *,
    candidate_runner: CellRunner,
    baseline_runner: BaselineRunner,
    perfect_information_runner: PerfectInformationRunner,
) -> dict[str, Any]:
    """Run or resume cells with shared candidate-independent baseline and PI."""

    raw_cells = panel.get("cells")
    if not isinstance(raw_cells, Sequence) or not raw_cells:
        raise ValueError("panel contains no replay cells")
    cells = list(raw_cells)
    if not all(isinstance(cell, ReplayCell) for cell in cells):
        raise TypeError("panel cells must be ReplayCell instances")
    if panel.get("panel_sha256") != sha256_json([cell.to_mapping() for cell in cells]):
        raise ValueError("panel SHA-256 does not match its cells")
    outputs: list[dict[str, Any]] = []
    resumed_cells = 0
    shared_resume_count = 0
    for cell in cells:
        shared_hashes = {
            key: cell.evidence_hashes[key] for key in sorted(CANDIDATE_INDEPENDENT_HASHES)
        }
        baseline_fingerprint = sha256_json(
            {
                "kind": "candidate_independent_baseline",
                "operating_date": cell.operating_date,
                "experiment_id": cell.experiment_id,
                "seed": cell.seed,
                "shared_hashes": shared_hashes,
            }
        )
        baseline_path = (
            checkpoint_root
            / "shared"
            / "baseline"
            / _safe_name(cell.operating_date)
            / _safe_name(cell.experiment_id)
            / f"seed_{cell.seed}.json"
        )
        baseline, baseline_resumed = _run_shared(
            baseline_path,
            fingerprint=baseline_fingerprint,
            kind="candidate_independent_baseline",
            runner=lambda cell=cell: baseline_runner(
                cell.operating_date, cell.experiment_id, cell.seed
            ),
        )
        pi_fingerprint = sha256_json(
            {
                "kind": "per_day_perfect_information",
                "operating_date": cell.operating_date,
                "shared_hashes": shared_hashes,
            }
        )
        pi_path = (
            checkpoint_root
            / "shared"
            / "perfect_information"
            / f"{_safe_name(cell.operating_date)}.json"
        )
        perfect_information, pi_resumed = _run_shared(
            pi_path,
            fingerprint=pi_fingerprint,
            kind="per_day_perfect_information",
            runner=lambda cell=cell: perfect_information_runner(cell.operating_date),
        )
        shared_resume_count += int(baseline_resumed) + int(pi_resumed)
        cell_path = (
            checkpoint_root
            / "cells"
            / _safe_name(cell.operating_date)
            / _safe_name(cell.candidate_id)
            / _safe_name(cell.experiment_id)
            / f"seed_{cell.seed}.json"
        )
        cell_fingerprint = sha256_json(
            {
                "cell_sha256": cell.cell_sha256,
                "baseline_sha256": sha256_json(baseline),
                "perfect_information_sha256": sha256_json(perfect_information),
            }
        )
        cached = _load_complete(cell_path, cell_fingerprint)
        if cached is not None:
            outputs.append(dict(cached["result"]))
            resumed_cells += 1
            continue
        candidate = dict(candidate_runner(cell))
        _cost(candidate, name="candidate")
        value = decision_value_metrics(candidate, baseline, perfect_information)
        result = {
            "operating_date": cell.operating_date,
            "candidate_id": cell.candidate_id,
            "experiment_id": cell.experiment_id,
            "seed": cell.seed,
            "cell_sha256": cell.cell_sha256,
            "identity_sha256": cell.identity_sha256,
            "evidence_hashes": dict(cell.evidence_hashes),
            "candidate_result": candidate,
            "baseline_result": baseline,
            "perfect_information_result": perfect_information,
            "decision_value": value,
        }
        payload = {
            "schema_version": DART_REPLAY_SCHEMA_VERSION,
            "status": "COMPLETE",
            "kind": "candidate_replay_cell",
            "input_sha256": cell_fingerprint,
            "result": result,
            "output_sha256": sha256_json(result),
        }
        atomic_json(cell_path, payload)
        outputs.append(result)
    return {
        "schema_version": DART_REPLAY_SCHEMA_VERSION,
        "status": "COMPLETE",
        "panel_sha256": panel["panel_sha256"],
        "cell_count": len(outputs),
        "resumed_cell_count": resumed_cells,
        "shared_resume_count": shared_resume_count,
        "results": outputs,
        "results_sha256": sha256_json(outputs),
    }


def derive_a3_metric_contradictions(
    replay_results: Sequence[Mapping[str, Any]],
    *,
    accuracy_metric: str = "mae",
    experiment_id: str = "A1",
    regime_labels_by_date: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Derive aggregate and date-level A3 rows without additional solves."""

    grouped: dict[str, dict[str, list[float]]] = {}
    by_day_candidate: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    source_hashes: list[str] = []
    for record in replay_results:
        if record.get("experiment_id") != experiment_id:
            continue
        operating_date = str(record["operating_date"])
        candidate_id = str(record["candidate_id"])
        candidate_result = record.get("candidate_result", {})
        accuracy = candidate_result.get("accuracy_metrics", {}).get(accuracy_metric)
        value = record.get("decision_value", {}).get("incremental_operational_value_usd")
        if accuracy is None or value is None:
            raise ValueError(
                f"A3 requires candidate_result.accuracy_metrics.{accuracy_metric} and decision value"
            )
        grouped.setdefault(candidate_id, {"accuracy": [], "value": []})
        grouped[candidate_id]["accuracy"].append(float(accuracy))
        grouped[candidate_id]["value"].append(float(value))
        by_day_candidate.setdefault((operating_date, candidate_id), []).append(record)
        source_hashes.append(sha256_json(record))
    if len(grouped) < 2:
        raise ValueError("A3 requires at least two candidates")

    def average_ranks(values: Mapping[str, float], *, higher: bool) -> dict[str, float]:
        ordered = sorted(
            values,
            key=lambda key: ((-values[key]) if higher else values[key], key),
        )
        ranks: dict[str, float] = {}
        index = 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and np.isclose(
                values[ordered[end]], values[ordered[index]], rtol=1e-12, atol=1e-12
            ):
                end += 1
            rank = (index + 1 + end) / 2.0
            for candidate in ordered[index:end]:
                ranks[candidate] = rank
            index = end
        return ranks

    aggregate_accuracy = {
        candidate: float(np.mean(values["accuracy"]))
        for candidate, values in grouped.items()
    }
    aggregate_value = {
        candidate: float(np.mean(values["value"]))
        for candidate, values in grouped.items()
    }
    accuracy_rank = average_ranks(aggregate_accuracy, higher=False)
    value_rank = average_ranks(aggregate_value, higher=True)
    accuracy_order = sorted(accuracy_rank, key=lambda key: (accuracy_rank[key], key))
    value_order = sorted(
        value_rank,
        key=lambda key: (value_rank[key], key),
    )
    rows = [
        {
            "candidate_id": candidate_id,
            f"mean_{accuracy_metric}": aggregate_accuracy[candidate_id],
            "mean_incremental_operational_value_usd": aggregate_value[candidate_id],
            "accuracy_rank": accuracy_rank[candidate_id],
            "decision_value_rank": value_rank[candidate_id],
            "rank_contradiction": accuracy_rank[candidate_id] != value_rank[candidate_id],
        }
        for candidate_id in sorted(grouped)
    ]

    dates = tuple(sorted({key[0] for key in by_day_candidate}))
    candidates = tuple(sorted(grouped))
    labels_by_date = {
        str(day): tuple(map(str, labels))
        for day, labels in (regime_labels_by_date or {}).items()
    }
    date_candidate_rows: list[dict[str, Any]] = []
    date_findings: list[dict[str, Any]] = []
    for operating_date in dates:
        day_accuracy: dict[str, float] = {}
        day_value: dict[str, float] = {}
        for candidate_id in candidates:
            records = by_day_candidate.get((operating_date, candidate_id), [])
            if len(records) != len(DART_VALUE_SELECT_SEEDS) or {
                int(record["seed"]) for record in records
            } != set(DART_VALUE_SELECT_SEEDS):
                raise ValueError(
                    "A3 requires the exact three-seed candidate-day replay panel"
                )
            day_accuracy[candidate_id] = float(
                np.mean(
                    [
                        float(
                            record["candidate_result"]["accuracy_metrics"][
                                accuracy_metric
                            ]
                        )
                        for record in records
                    ]
                )
            )
            day_value[candidate_id] = float(
                np.mean(
                    [
                        float(
                            record["decision_value"][
                                "incremental_operational_value_usd"
                            ]
                        )
                        for record in records
                    ]
                )
            )
        accuracy_ranks = average_ranks(day_accuracy, higher=False)
        value_ranks = average_ranks(day_value, higher=True)
        lowest_accuracy_set = tuple(
            sorted(
                candidate
                for candidate, rank in accuracy_ranks.items()
                if np.isclose(rank, min(accuracy_ranks.values()))
            )
        )
        highest_value_set = tuple(
            sorted(
                candidate
                for candidate, rank in value_ranks.items()
                if np.isclose(rank, min(value_ranks.values()))
            )
        )
        contradiction = set(lowest_accuracy_set).isdisjoint(highest_value_set)
        finding_status = (
            "QUALIFYING_CONTRADICTION"
            if contradiction
            else "NO_QUALIFYING_CONTRADICTION"
        )
        regimes = labels_by_date.get(operating_date, ())
        spike_context = tuple(
            label
            for label in regimes
            if label in {"positive_dart_spike", "negative_dart_spike"}
        )
        if not spike_context:
            spike_context = ("NO_DART_SPIKE_REGIME",)
        date_findings.append(
            {
                "operating_date": operating_date,
                "experiment_id": experiment_id,
                "accuracy_metric": accuracy_metric,
                "status": finding_status,
                "lowest_accuracy_winner_set": list(lowest_accuracy_set),
                "highest_decision_value_winner_set": list(highest_value_set),
                "spike": list(spike_context),
                "regime": list(regimes),
            }
        )
        for candidate_id in candidates:
            records = by_day_candidate[(operating_date, candidate_id)]
            candidate_results = [record["candidate_result"] for record in records]
            flexibility = [
                float(result["gate_time_feasible_flexible_energy_mwh"])
                for result in candidate_results
                if result.get("gate_time_feasible_flexible_energy_mwh") is not None
            ]
            contexts = [
                result.get("a3_context")
                for result in candidate_results
                if isinstance(result.get("a3_context"), Mapping)
            ]
            sign_payloads = [
                context.get("dart_sign")
                for context in contexts
                if isinstance(context.get("dart_sign"), Mapping)
            ]
            sign_labels = sorted(
                {
                    str(payload.get("label"))
                    for payload in sign_payloads
                    if payload.get("label")
                }
            )
            if len(sign_labels) == 1:
                dart_sign: str | None = sign_labels[0]
                dart_sign_status = "AVAILABLE"
                dart_sign_reason = None
            elif sign_labels:
                dart_sign = None
                dart_sign_status = "INCONSISTENT_ACROSS_SEEDS"
                dart_sign_reason = ";".join(sign_labels)
            else:
                dart_sign = None
                dart_sign_status = "UNAVAILABLE"
                dart_sign_reason = "ACTUAL_DART_SIGN_NOT_EXPOSED_BY_REPLAY_EVIDENCE"
            binding_labels: set[str] = set()
            binding_statuses: set[str] = set()
            binding_reasons: set[str] = set()
            for context in contexts:
                raw_binding = context.get("binding_constraint")
                if isinstance(raw_binding, str) and raw_binding:
                    binding_labels.add(raw_binding)
                elif isinstance(raw_binding, Sequence) and not isinstance(
                    raw_binding, (str, bytes)
                ):
                    binding_labels.update(map(str, raw_binding))
                if context.get("binding_constraint_status"):
                    binding_statuses.add(str(context["binding_constraint_status"]))
                if context.get("binding_constraint_undefined_reason"):
                    binding_reasons.add(
                        str(context["binding_constraint_undefined_reason"])
                    )
            date_candidate_rows.append(
                {
                    "operating_date": operating_date,
                    "candidate_id": candidate_id,
                    "experiment_id": experiment_id,
                    "accuracy_metric": accuracy_metric,
                    "mean_accuracy": day_accuracy[candidate_id],
                    "mean_incremental_operational_value_usd": day_value[candidate_id],
                    "accuracy_rank": accuracy_ranks[candidate_id],
                    "decision_value_rank": value_ranks[candidate_id],
                    "is_lowest_accuracy_member": candidate_id in lowest_accuracy_set,
                    "is_highest_decision_value_member": candidate_id in highest_value_set,
                    "contradiction_status": finding_status,
                    "spike": list(spike_context),
                    "dart_sign": dart_sign,
                    "dart_sign_status": dart_sign_status,
                    "dart_sign_undefined_reason": dart_sign_reason,
                    "feasible_flexibility_mwh": (
                        None if not flexibility else float(np.mean(flexibility))
                    ),
                    "feasible_flexibility_status": (
                        "AVAILABLE" if flexibility else "UNAVAILABLE"
                    ),
                    "binding_constraint": (
                        sorted(binding_labels) if binding_labels else None
                    ),
                    "binding_constraint_status": (
                        "REPORTED"
                        if binding_labels
                        else ";".join(sorted(binding_statuses)) or "UNAVAILABLE"
                    ),
                    "binding_constraint_undefined_reason": (
                        None
                        if binding_labels
                        else ";".join(sorted(binding_reasons))
                        or "SOLVER_DID_NOT_EXPOSE_ACTIVE_CONSTRAINT_LABELS"
                    ),
                    "regime": list(regimes),
                    "source_seed_count": len(records),
                }
            )
    any_aggregate = any(row["rank_contradiction"] for row in rows)
    any_date = any(
        row["status"] == "QUALIFYING_CONTRADICTION" for row in date_findings
    )
    return {
        "schema_version": "dart_metric_contradiction_a3_v2",
        "experiment_id": "A3",
        "source_experiment_id": experiment_id,
        "accuracy_metric": accuracy_metric,
        "derived_without_solves": True,
        "candidate_count": len(rows),
        "operating_date_count": len(dates),
        "required_grain": "operating_date_candidate_experiment",
        "required_context": [
            "spike",
            "dart_sign",
            "feasible_flexibility",
            "binding_constraint",
            "regime",
        ],
        "aggregate_accuracy_order": accuracy_order,
        "aggregate_decision_value_order": value_order,
        "any_aggregate_rank_contradiction": any_aggregate,
        "any_date_level_contradiction": any_date,
        "any_rank_contradiction": bool(any_aggregate or any_date),
        "rows": rows,
        "date_findings": date_findings,
        "date_candidate_rows": date_candidate_rows,
        "explicit_null_status": "NO_QUALIFYING_CONTRADICTION",
        "null_findings_retained": True,
        "source_result_sha256": sorted(source_hashes),
        "rows_sha256": sha256_json(rows),
        "date_findings_sha256": sha256_json(date_findings),
        "date_candidate_rows_sha256": sha256_json(date_candidate_rows),
    }


__all__ = [
    "A1_ALLOWED_TO_DIFFER",
    "A2_ALLOWED_TO_DIFFER",
    "CANDIDATE_INDEPENDENT_HASHES",
    "CandidateEvidence",
    "DART_REPLAY_SCHEMA_VERSION",
    "DART_VALUE_SELECT_SEEDS",
    "ReplayCell",
    "assert_experiment_identity",
    "build_common_replay_panel",
    "decision_value_metrics",
    "derive_a3_metric_contradictions",
    "run_dart_replay_panel",
]
