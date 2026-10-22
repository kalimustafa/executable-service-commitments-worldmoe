"""Pre-result protocol for conditional DART value selection.

This module turns a hash-pinned NYISO settlement coverage audit into one
candidate-independent replay mask.  It does not read candidate value results,
materialise counterfactual inputs, or launch a solver.  Incomplete settlement
dates remain in an explicit full-calendar ledger, while the existing real
evidence runner receives only dates whose four-zone 96-interval DA/RT
settlement panel is complete.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import json
from typing import Any

from dcvpp.core import atomic_json, atomic_text, sha256_file, sha256_json, utc_now
from dcvpp.experiments.dart_partial_candidate import (
    ELIGIBLE_CANDIDATES,
    validate_partial_candidate_pool_contract,
)
from dcvpp.experiments.dart_replay import DART_VALUE_SELECT_SEEDS


SCHEMA_VERSION = "dcvpp.dart_conditional_settlement_protocol.v1"
LEDGER_SCHEMA_VERSION = "dcvpp.dart_conditional_settlement_ledger.v1"
CLAIM_SCOPE = "conditional_decision_value_on_fully_observed_settlement_days"
TERMINAL_PAPER_CAP = "PARTIAL_PAPER"
INVALID_STATUS = "INVALID_MISSING_SETTLEMENT"
VALID_STATUS = "ELIGIBLE"
PHASES = ("selection", "locked_test")
EXPECTED_CALENDAR_COUNTS = {"selection": 1096, "locked_test": 365}


class ConditionalSettlementProtocolError(RuntimeError):
    """Raised when the pre-result common-panel contract is not auditable."""


def _resolve(path: str | Path, *, root: Path, sibling: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    from_root = (root / candidate).resolve()
    if from_root.exists() or sibling is None:
        return from_root
    return (sibling / candidate).resolve()


def _relative(path: Path, *, root: Path) -> str:
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ConditionalSettlementProtocolError(f"{label} is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ConditionalSettlementProtocolError(f"{label} is not a JSON mapping")
    return dict(value)


def _manifest_fingerprint(value: Mapping[str, Any]) -> str:
    return sha256_json({key: item for key, item in value.items() if key != "manifest_fingerprint"})


def _source_record(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": _relative(path, root=root),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }


def _normalize_coverage_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    day = str(raw.get("operating_date", ""))
    if not day:
        raise ConditionalSettlementProtocolError("coverage record lacks operating_date")
    valid = raw.get("settlement_evaluable") is True
    status = str(raw.get("decision_replay_cell_status", ""))
    expected_status = VALID_STATUS if valid else INVALID_STATUS
    if status != expected_status:
        raise ConditionalSettlementProtocolError(
            f"coverage status disagrees with settlement_evaluable for {day}"
        )
    missing = int(raw.get("missing_settlement_interval_count", -1))
    if missing < 0 or (valid and missing != 0) or (not valid and missing == 0):
        raise ConditionalSettlementProtocolError(
            f"missing-settlement count is inconsistent for {day}"
        )
    if raw.get("solver_launched_for_missing_settlement") is not False:
        raise ConditionalSettlementProtocolError(
            f"coverage ledger does not forbid a solver for missing settlement on {day}"
        )
    if raw.get("candidate_cost_emitted_for_missing_settlement") is not False:
        raise ConditionalSettlementProtocolError(
            f"coverage ledger emitted a candidate cost for {day}"
        )
    return {
        "operating_date": day,
        "settlement_evaluable": valid,
        "decision_replay_cell_status": expected_status,
        "missing_da_interval_count": int(raw.get("missing_da_interval_count", -1)),
        "missing_rt_interval_count": int(raw.get("missing_rt_interval_count", -1)),
        "missing_settlement_interval_count": missing,
        "missing_zone_value_count": int(raw.get("missing_zone_value_count", -1)),
        "solver_launch_allowed": valid,
        "candidate_cost_allowed": valid,
    }


def write_conditional_settlement_protocol(
    *,
    repository_root: str | Path,
    output_root: str | Path,
    settlement_coverage_index: str | Path,
    official_rt_recovery_manifest: str | Path,
    candidate_pool_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the common observable-settlement panel before value replay.

    Counts and dates are derived from the supplied, post-recovery coverage
    files.  The historical 1001/328 counts are deliberately not constants:
    official raw-data recovery may expand either valid panel before this
    amendment is frozen.
    """

    root = Path(repository_root).resolve()
    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    coverage_index_path = _resolve(settlement_coverage_index, root=root)
    recovery_path = _resolve(official_rt_recovery_manifest, root=root)
    coverage_index = _read_mapping(coverage_index_path, label="settlement coverage index")
    recovery = _read_mapping(recovery_path, label="official RT recovery manifest")
    pool = validate_partial_candidate_pool_contract(candidate_pool_contract)
    candidates = tuple(map(str, pool["eligible_candidates"]))
    if candidates != tuple(ELIGIBLE_CANDIDATES):
        raise ConditionalSettlementProtocolError("eligible candidate pool is not the frozen eight")

    # The recovery artifact is an evidence dependency, never a replacement
    # price source.  Reject any manifest that explicitly admits imputation.
    source_policy = recovery.get("source_policy")
    if not isinstance(source_policy, Mapping):
        raise ConditionalSettlementProtocolError(
            "official RT recovery manifest lacks source_policy"
        )
    forbidden_flags = (
        "forward_fill_used",
        "interpolation_used",
        "neighbor_zone_substitution_used",
        "synthetic_price_used",
        "model_price_fill_used",
    )
    if any(source_policy.get(flag) is not False for flag in forbidden_flags[:3]) or any(
        source_policy.get(flag) is True for flag in forbidden_flags[3:]
    ):
        raise ConditionalSettlementProtocolError(
            "official RT recovery manifest explicitly permits fabricated settlement"
        )

    phase_index = coverage_index.get("phases")
    if not isinstance(phase_index, Mapping):
        raise ConditionalSettlementProtocolError("coverage index lacks phases")
    phase_summaries: dict[str, Any] = {}
    for phase in PHASES:
        indexed = phase_index.get(phase)
        if not isinstance(indexed, Mapping) or not indexed.get("path"):
            raise ConditionalSettlementProtocolError(f"coverage index lacks {phase}")
        coverage_path = _resolve(
            indexed["path"], root=root, sibling=coverage_index_path.parent
        )
        coverage = _read_mapping(coverage_path, label=f"{phase} coverage")
        actual_sha = sha256_file(coverage_path)
        if indexed.get("sha256") != actual_sha:
            raise ConditionalSettlementProtocolError(f"{phase} coverage hash changed")
        raw_records = coverage.get("records")
        if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
            raise ConditionalSettlementProtocolError(f"{phase} coverage records are absent")
        records = [_normalize_coverage_record(dict(record)) for record in raw_records]
        dates = [record["operating_date"] for record in records]
        if dates != sorted(dates) or len(set(dates)) != len(dates):
            raise ConditionalSettlementProtocolError(f"{phase} dates are not sorted and unique")
        expected = EXPECTED_CALENDAR_COUNTS[phase]
        valid_dates = [
            record["operating_date"] for record in records if record["settlement_evaluable"]
        ]
        invalid_dates = [
            record["operating_date"] for record in records if not record["settlement_evaluable"]
        ]
        if (
            len(records) != expected
            or int(coverage.get("expected_operating_date_count", -1)) != expected
            or int(coverage.get("retained_operating_date_count", -1)) != expected
            or int(coverage.get("date_drop_count", -1)) != 0
            or int(coverage.get("settlement_evaluable_date_count", -1)) != len(valid_dates)
            or int(coverage.get("settlement_invalid_date_count", -1)) != len(invalid_dates)
            or coverage.get("missing_settlement_imputed") is not False
            or coverage.get("missing_dates_silently_dropped") is not False
        ):
            raise ConditionalSettlementProtocolError(
                f"{phase} does not retain its exact full-calendar coverage ledger"
            )
        missing_count = sum(record["missing_settlement_interval_count"] for record in records)
        if int(coverage.get("missing_settlement_interval_count", -1)) != missing_count:
            raise ConditionalSettlementProtocolError(
                f"{phase} missing interval total differs from its records"
            )
        ledger = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "phase": phase,
            "status": "FROZEN_COMMON_VALIDITY_MASK",
            "claim_scope": CLAIM_SCOPE,
            "full_calendar_retained": True,
            "date_drop_count": 0,
            "expected_operating_date_count": expected,
            "retained_operating_date_count": len(records),
            "settlement_evaluable_date_count": len(valid_dates),
            "settlement_invalid_date_count": len(invalid_dates),
            "missing_settlement_interval_count": missing_count,
            "valid_operating_dates": valid_dates,
            "invalid_operating_dates": invalid_dates,
            "records": records,
            "candidate_independent": True,
            "eligible_candidates": list(candidates),
            "invalid_date_policy": {
                "status": INVALID_STATUS,
                "solver_launched": False,
                "candidate_cost_emitted": False,
                "imputation_or_substitution": False,
            },
            "source_coverage": _source_record(coverage_path, root=root),
        }
        ledger["ledger_fingerprint"] = sha256_json(ledger)
        ledger_path = destination / f"{phase}_conditional_settlement_ledger.json"
        atomic_json(ledger_path, ledger)
        phase_summaries[phase] = {
            "expected_operating_date_count": expected,
            "retained_operating_date_count": len(records),
            "settlement_evaluable_date_count": len(valid_dates),
            "settlement_invalid_date_count": len(invalid_dates),
            "missing_settlement_interval_count": missing_count,
            "valid_operating_dates_sha256": sha256_json(valid_dates),
            "invalid_operating_dates_sha256": sha256_json(invalid_dates),
            "ledger_path": _relative(ledger_path, root=root),
            "ledger_sha256": sha256_file(ledger_path),
            "ledger_fingerprint": ledger["ledger_fingerprint"],
        }

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_PRE_RESULT_CONDITIONAL_PROTOCOL",
        "created_at_utc": utc_now(),
        "claim_scope": CLAIM_SCOPE,
        "terminal_paper_cap": TERMINAL_PAPER_CAP,
        "formal_complete_calendar_winner_claim_allowed": False,
        "conditional_winner_set_allowed_after_complete_replay": True,
        "protocol_timing": {
            "frozen_before_any_DART_value_result": True,
            "DART_value_result_consumed_by_this_step": False,
            "locked_2025_result_opened_by_this_step": False,
            "solver_launched_by_this_step": False,
        },
        "candidate_contract": {
            "candidate_independent_mask": True,
            "eligible_candidate_count": len(candidates),
            "eligible_candidates": list(candidates),
            "candidate_pool_contract_sha256": pool["contract_sha256"],
            "per_candidate_date_deletion_forbidden": True,
        },
        "settlement_rules": {
            "complete_four_zone_96_interval_DA_RT_required": True,
            "official_raw_recovery_precedes_mask": True,
            "imputation_forward_fill_model_fill_or_neighbor_substitution_forbidden": True,
            "full_calendar_ledger_retained": True,
            "invalid_status": INVALID_STATUS,
            "invalid_date_solver_launch_forbidden": True,
            "invalid_date_cost_regret_PCR_CVaR_forbidden": True,
        },
        "inference_contract": {
            "solver_scenario_replay_seeds": list(DART_VALUE_SELECT_SEEDS),
            "exact_three_replay_cells_required_per_candidate_date_experiment": True,
            "three_training_artifacts_per_candidate_required": False,
            "deterministic_and_statistical_single_forecast_fit_reused_across_replay_seeds": True,
            "daily_value_formula": "mean_seed(C_baseline_d_s_minus_C_candidate_m_d_s)",
            "seed_average_precedes_date_inference": True,
            "model_confidence_set_confidence_level": 0.95,
            "moving_block_bootstrap_days": [7, 14, 28],
            "MCS_and_LCB_dates": "phase_common_valid_operating_dates_only",
            "same_date_panel_for_all_candidates": True,
        },
        "phases": phase_summaries,
        "sources": {
            "post_recovery_settlement_coverage_index": _source_record(
                coverage_index_path, root=root
            ),
            "official_rt_recovery_manifest": _source_record(recovery_path, root=root),
        },
    }
    selection_summary = phase_summaries["selection"]
    locked_summary = phase_summaries["locked_test"]
    markdown = f"""# Frozen conditional DART settlement protocol

Status: `FROZEN_PRE_RESULT_CONDITIONAL_PROTOCOL`

Claim scope: `{CLAIM_SCOPE}`

Terminal paper cap: `{TERMINAL_PAPER_CAP}`

This amendment was materialised before reading any DART value result and did
not launch a solver or open locked-2025 results. Official RT recovery was
attempted first. Remaining missing settlement values are not interpolated,
filled, forecast, or substituted.

| Phase | Full calendar | Common valid dates | Invalid dates | Missing 15-minute intervals |
|---|---:|---:|---:|---:|
| 2022-2024 selection | {selection_summary['retained_operating_date_count']} | {selection_summary['settlement_evaluable_date_count']} | {selection_summary['settlement_invalid_date_count']} | {selection_summary['missing_settlement_interval_count']} |
| 2025 locked test | {locked_summary['retained_operating_date_count']} | {locked_summary['settlement_evaluable_date_count']} | {locked_summary['settlement_invalid_date_count']} | {locked_summary['missing_settlement_interval_count']} |

All full-calendar dates remain in the two JSON ledgers. Invalid dates are
labelled `{INVALID_STATUS}`, do not enter the solver panel, and cannot emit
cost, regret, PCR, or CVaR. The same dynamically derived valid-date mask is
used for all eight eligible candidates. Values are first averaged over common
solver/scenario replay seeds
{list(DART_VALUE_SELECT_SEEDS)} within each date; 95% MCS and 7/14/28-day
moving-block bootstrap operate only on those daily means.

The exact three-seed requirement applies to replay cells. It does not require
three independently trained forecast artifacts for every family:
deterministic/statistical single-fit forecasts are reused across the three
downstream replay seeds under the frozen forecast-training contract.

The result, if later computed, is conditional on fully observed settlement
days and must not be described as a complete-calendar or complete-year winner.

Post-recovery coverage index SHA-256: `{manifest['sources']['post_recovery_settlement_coverage_index']['sha256']}`

Official RT recovery manifest SHA-256: `{manifest['sources']['official_rt_recovery_manifest']['sha256']}`
"""
    markdown_path = destination / "conditional_settlement_protocol.md"
    atomic_text(markdown_path, markdown)
    manifest["authoritative_markdown"] = _source_record(markdown_path, root=root)
    manifest["manifest_fingerprint"] = _manifest_fingerprint(manifest)
    manifest_path = destination / "conditional_settlement_protocol_manifest.json"
    atomic_json(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "manifest": manifest,
        "manifest_path": _relative(manifest_path, root=root),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_fingerprint": manifest["manifest_fingerprint"],
    }


def load_conditional_settlement_protocol(
    manifest_path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    """Load and hash-check a frozen conditional protocol and both ledgers."""

    root = Path(repository_root).resolve()
    path = _resolve(manifest_path, root=root)
    manifest = _read_mapping(path, label="conditional settlement protocol")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "FROZEN_PRE_RESULT_CONDITIONAL_PROTOCOL"
        or manifest.get("claim_scope") != CLAIM_SCOPE
        or manifest.get("terminal_paper_cap") != TERMINAL_PAPER_CAP
        or manifest.get("manifest_fingerprint") != _manifest_fingerprint(manifest)
    ):
        raise ConditionalSettlementProtocolError("conditional protocol identity is invalid")
    for source in manifest.get("sources", {}).values():
        source_path = _resolve(source["path"], root=root)
        if not source_path.is_file() or sha256_file(source_path) != source["sha256"]:
            raise ConditionalSettlementProtocolError("conditional protocol source hash changed")
    for phase in PHASES:
        summary = manifest.get("phases", {}).get(phase, {})
        ledger_path = _resolve(summary.get("ledger_path", ""), root=root)
        ledger = _read_mapping(ledger_path, label=f"{phase} conditional ledger")
        if sha256_file(ledger_path) != summary.get("ledger_sha256"):
            raise ConditionalSettlementProtocolError(f"{phase} conditional ledger hash changed")
        fingerprint = ledger.get("ledger_fingerprint")
        ledger_core = {key: value for key, value in ledger.items() if key != "ledger_fingerprint"}
        if fingerprint != sha256_json(ledger_core):
            raise ConditionalSettlementProtocolError(f"{phase} ledger fingerprint is invalid")
        if (
            ledger.get("full_calendar_retained") is not True
            or int(ledger.get("date_drop_count", -1)) != 0
            or len(ledger.get("records", ())) != EXPECTED_CALENDAR_COUNTS[phase]
        ):
            raise ConditionalSettlementProtocolError(f"{phase} full ledger changed")
    return manifest


def apply_conditional_settlement_overlay(
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Return a replay config restricted to the protocol's common valid dates."""

    root = Path(repository_root).resolve()
    output = deepcopy(dict(config))
    real_contract = output.setdefault("real_evidence_contract", {})
    if not isinstance(real_contract, dict):
        raise ConditionalSettlementProtocolError("real_evidence_contract must be a mapping")
    operating_dates: dict[str, list[str]] = {}
    for phase in PHASES:
        summary = protocol.get("phases", {}).get(phase, {})
        ledger_path = _resolve(summary.get("ledger_path", ""), root=root)
        ledger = _read_mapping(ledger_path, label=f"{phase} conditional ledger")
        valid_dates = list(map(str, ledger.get("valid_operating_dates", ())))
        if sha256_json(valid_dates) != summary.get("valid_operating_dates_sha256"):
            raise ConditionalSettlementProtocolError(f"{phase} valid-date mask changed")
        operating_dates[phase] = valid_dates
    existing = real_contract.get("operating_dates_by_phase")
    if existing not in (None, operating_dates):
        raise ConditionalSettlementProtocolError(
            "config already contains a different operating-date selection"
        )
    real_contract["operating_dates_by_phase"] = operating_dates
    real_contract["conditional_settlement_protocol"] = {
        "schema_version": protocol["schema_version"],
        "manifest_fingerprint": protocol["manifest_fingerprint"],
        "claim_scope": CLAIM_SCOPE,
        "terminal_paper_cap": TERMINAL_PAPER_CAP,
        "invalid_date_status": INVALID_STATUS,
        "full_calendar_ledger_retained_outside_solver_panel": True,
    }
    return output


@contextmanager
def conditional_settlement_runtime(
    *,
    repository_root: str | Path,
    protocol_manifest: str | Path,
    runtime_ledger_path: str | Path | None = None,
):
    """Inject the frozen valid-date overlay at the real-evidence call seam.

    This compatibility context is intentionally external to the frozen DART
    training and replay modules.  It changes only ``config.real_evidence_contract``
    immediately before preflight, and restores the original callable on exit.
    """

    from dcvpp.experiments import dart_real_evidence

    root = Path(repository_root).resolve()
    protocol = load_conditional_settlement_protocol(
        protocol_manifest, repository_root=root
    )
    original = dart_real_evidence.run_real_dart_phase_evidence
    calls: list[dict[str, Any]] = []

    def conditional_run(*args: Any, **kwargs: Any) -> Any:
        if args:
            # The public production seam uses keyword arguments. Fail closed
            # instead of guessing the position of a mutable config mapping.
            raise ConditionalSettlementProtocolError(
                "conditional runtime requires keyword arguments"
            )
        raw_config = kwargs.get("config")
        if not isinstance(raw_config, Mapping):
            raise ConditionalSettlementProtocolError("real-evidence config is absent")
        phase = str(kwargs.get("phase", ""))
        if phase not in PHASES:
            raise ConditionalSettlementProtocolError("unknown conditional replay phase")
        overlaid = apply_conditional_settlement_overlay(
            raw_config, protocol, repository_root=root
        )
        kwargs["config"] = overlaid
        valid_count = len(
            overlaid["real_evidence_contract"]["operating_dates_by_phase"][phase]
        )
        calls.append({"phase": phase, "settlement_evaluable_date_count": valid_count})
        return original(**kwargs)

    dart_real_evidence.run_real_dart_phase_evidence = conditional_run
    try:
        yield {
            "status": "ACTIVE",
            "protocol_manifest_fingerprint": protocol["manifest_fingerprint"],
            "claim_scope": CLAIM_SCOPE,
            "terminal_paper_cap": TERMINAL_PAPER_CAP,
            "solver_invocation_count_by_protocol_prepare": 0,
        }
    finally:
        dart_real_evidence.run_real_dart_phase_evidence = original
        if runtime_ledger_path is not None:
            ledger_path = _resolve(runtime_ledger_path, root=root)
            ledger = {
                "schema_version": "dcvpp.dart_conditional_settlement_runtime.v1",
                "status": "INACTIVE",
                "protocol_manifest_fingerprint": protocol["manifest_fingerprint"],
                "claim_scope": CLAIM_SCOPE,
                "terminal_paper_cap": TERMINAL_PAPER_CAP,
                "calls": calls,
                "runtime_call_count": len(calls),
            }
            atomic_json(ledger_path, ledger)


def aggregate_three_seed_daily_values(
    replay_rows: Sequence[Mapping[str, Any]],
    *,
    valid_operating_dates: Sequence[str],
    candidates: Sequence[str] = ELIGIBLE_CANDIDATES,
    experiments: Sequence[str] = ("A1", "A2"),
) -> list[dict[str, Any]]:
    """Average exactly three shared solver/scenario replay seeds by date.

    Rows are replay cells, not training artifacts. A deterministic or
    statistical forecast may therefore be the same frozen fit across all
    three seeds while downstream scenario/solver randomness differs.
    """

    valid_dates = tuple(map(str, valid_operating_dates))
    if not valid_dates or len(set(valid_dates)) != len(valid_dates):
        raise ConditionalSettlementProtocolError("valid operating dates must be unique")
    candidate_ids = tuple(map(str, candidates))
    experiment_ids = tuple(map(str, experiments))
    grouped: defaultdict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in replay_rows:
        day = str(row.get("operating_date", ""))
        candidate = str(row.get("candidate_id", ""))
        experiment = str(row.get("experiment_id", ""))
        if day not in valid_dates:
            raise ConditionalSettlementProtocolError(
                f"replay result exists outside the common valid panel: {day}"
            )
        if candidate not in candidate_ids or experiment not in experiment_ids:
            raise ConditionalSettlementProtocolError("unexpected candidate or experiment")
        decision = row.get("decision_value")
        if not isinstance(decision, Mapping):
            raise ConditionalSettlementProtocolError("replay row lacks decision_value")
        value = decision.get("incremental_operational_value_usd")
        if value is None:
            raise ConditionalSettlementProtocolError("replay row lacks incremental value")
        grouped[(day, candidate, experiment)].append((int(row["seed"]), float(value)))
    expected_keys = {
        (day, candidate, experiment)
        for day in valid_dates
        for candidate in candidate_ids
        for experiment in experiment_ids
    }
    if set(grouped) != expected_keys:
        raise ConditionalSettlementProtocolError("replay rows do not form the exact common panel")
    output: list[dict[str, Any]] = []
    expected_seeds = set(DART_VALUE_SELECT_SEEDS)
    for day, candidate, experiment in sorted(expected_keys):
        observations = grouped[(day, candidate, experiment)]
        seeds = [seed for seed, _ in observations]
        if len(observations) != 3 or set(seeds) != expected_seeds:
            raise ConditionalSettlementProtocolError(
                f"exact three-seed panel failed for {(day, candidate, experiment)}"
            )
        output.append(
            {
                "operating_date": day,
                "candidate_id": candidate,
                "experiment_id": experiment,
                "seed_count": 3,
                "seeds": list(DART_VALUE_SELECT_SEEDS),
                "mean_daily_incremental_operational_value_usd": sum(
                    value for _, value in observations
                )
                / 3.0,
                "claim_scope": CLAIM_SCOPE,
            }
        )
    return output


__all__ = [
    "CLAIM_SCOPE",
    "ConditionalSettlementProtocolError",
    "INVALID_STATUS",
    "SCHEMA_VERSION",
    "aggregate_three_seed_daily_values",
    "apply_conditional_settlement_overlay",
    "conditional_settlement_runtime",
    "load_conditional_settlement_protocol",
    "write_conditional_settlement_protocol",
]
