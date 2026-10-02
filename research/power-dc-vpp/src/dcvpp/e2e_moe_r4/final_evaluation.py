"""Fail-closed three-seed freeze and exactly-once final evaluation for R4.

This module is intentionally independent of the training protocol and report
adapter.  It does not discover or open a final-test payload.  Instead, it
validates a metadata-only list of eligible 2024 dates, freezes every method and
decision artifact, then gives a single immutable evaluation plan to one batch
evaluator callback.  The callback may be invoked at most once for a lineage.

The durable transaction has three states:

``CLAIMED`` -> ``RESULTS_CAPTURED`` -> ``COMPLETE``.

If the callback raises or returns invalid/incomplete rows, the transaction is
marked failed and cannot be retried.  This conservative rule prevents a crash
after an unrecorded payload access from causing a second formal-test pass.
Post-processing is separately resumable from the captured daily parquet and
never calls the evaluator again.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .artifact_io import atomic_json, atomic_parquet, atomic_text, read_json, utc_now
from .config import (
    ARCHITECTURE_VERSION,
    DISTRIBUTIONAL_ARCHITECTURE_VERSION,
    DISTRIBUTIONAL_EXPERIMENT_ID,
    DISTRIBUTIONAL_PHASE_ORDER,
    DISTRIBUTIONAL_TRAINING_SEMANTICS_REVISION,
    EXPERIMENT_ID,
    FORMAL_SEEDS,
    PHASE_ORDER,
    TRAINING_SEMANTICS_REVISION,
    canonical_config_digest,
    load_r4_config,
)
from .hashing import sha256_file, sha256_json


FREEZE_SCHEMA_VERSION = "dcvpp.e2e_moe_hybrid_r4.final_freeze.v1"
PLAN_SCHEMA_VERSION = "dcvpp.e2e_moe_hybrid_r4.final_evaluation_plan.v1"
DAILY_SCHEMA_VERSION = "dcvpp.e2e_moe_hybrid_r4.final_daily_method.v1"
LEDGER_SCHEMA_VERSION = "dcvpp.e2e_moe_hybrid_r4.final_execution_ledger.v1"

DISTRIBUTIONAL_FREEZE_SCHEMA_VERSION = (
    "dcvpp.e2e_moe_distributional_r4.final_freeze.v1"
)
DISTRIBUTIONAL_PLAN_SCHEMA_VERSION = (
    "dcvpp.e2e_moe_distributional_r4.final_evaluation_plan.v1"
)
DISTRIBUTIONAL_DAILY_SCHEMA_VERSION = (
    "dcvpp.e2e_moe_distributional_r4.final_daily_method.v1"
)
FORMAL_EMPIRICAL_RISK_SAMPLE_COUNT = 512
EMPIRICAL_POST_SHIELD_RISK_MEASUREMENT = (
    "empirical_post_shield_complete_future_replay"
)

REQUIRED_FREEZE_GROUPS = (
    "architecture",
    "inputs",
    "normalization",
    "checkpoints",
    "benchmarks",
    "thresholds",
    "ablations",
    "statistics",
)
ALLOWED_METHOD_KINDS = frozenset({"learned", "derived", "benchmark"})

DAILY_COLUMNS = (
    "schema_version",
    "operating_date",
    "method_id",
    "method_kind",
    "training_seed",
    "origins_evaluated",
    "operating_cost_usd",
    "weighted_particle_cvar95_usd",
    "sla_shortfall_raw",
    "sla_shortfall_normalized",
    "independent_feasibility_fraction",
    "rejected_origins",
    "corrective_intervention_rate",
    "corrective_intervention_mean",
    "corrective_intervention_p95",
    "inference_latency_ms_b1",
    "frozen_evidence_sha256",
    "freeze_sha256",
)
FINITE_DAILY_COLUMNS = (
    "operating_cost_usd",
    "weighted_particle_cvar95_usd",
    "sla_shortfall_raw",
    "sla_shortfall_normalized",
    "independent_feasibility_fraction",
    "corrective_intervention_rate",
    "corrective_intervention_mean",
    "corrective_intervention_p95",
    "inference_latency_ms_b1",
)

DISTRIBUTIONAL_DAILY_COLUMNS = (
    "schema_version",
    "operating_date",
    "method_id",
    "method_kind",
    "training_seed",
    "origins_evaluated",
    "operating_cost_usd",
    "empirical_post_shield_cvar95_usd",
    "formal_empirical_risk_sample_count",
    "risk_measurement",
    "sla_shortfall_raw",
    "sla_shortfall_normalized",
    "independent_feasibility_fraction",
    "rejected_origins",
    "corrective_intervention_rate",
    "corrective_intervention_mean",
    "corrective_intervention_p95",
    "inference_latency_ms_b1",
    "frozen_evidence_sha256",
    "freeze_sha256",
)
DISTRIBUTIONAL_FINITE_DAILY_COLUMNS = (
    "operating_cost_usd",
    "empirical_post_shield_cvar95_usd",
    "sla_shortfall_raw",
    "sla_shortfall_normalized",
    "independent_feasibility_fraction",
    "corrective_intervention_rate",
    "corrective_intervention_mean",
    "corrective_intervention_p95",
    "inference_latency_ms_b1",
)


class FinalProtocolError(RuntimeError):
    """A final-protocol invariant failed before or during formal evaluation."""


class FinalEvaluationAlreadyClaimed(FinalProtocolError):
    """The exactly-once evaluation transaction already exists."""


@dataclass(frozen=True)
class _FinalLineageProfile:
    architecture_version: str
    experiment_id: str
    training_semantics_revision: str
    phase_order: tuple[str, ...]
    terminal_pass_label: str
    freeze_schema_version: str
    plan_schema_version: str
    daily_schema_version: str
    daily_columns: tuple[str, ...]
    finite_daily_columns: tuple[str, ...]
    formal_empirical_risk_sample_count: int | None = None
    risk_measurement: str | None = None


_V5_FINAL_PROFILE = _FinalLineageProfile(
    architecture_version=ARCHITECTURE_VERSION,
    experiment_id=EXPERIMENT_ID,
    training_semantics_revision=TRAINING_SEMANTICS_REVISION,
    phase_order=tuple(PHASE_ORDER),
    terminal_pass_label="R4_MAINLINE_PASS",
    freeze_schema_version=FREEZE_SCHEMA_VERSION,
    plan_schema_version=PLAN_SCHEMA_VERSION,
    daily_schema_version=DAILY_SCHEMA_VERSION,
    daily_columns=DAILY_COLUMNS,
    finite_daily_columns=FINITE_DAILY_COLUMNS,
)
_V6_FINAL_PROFILE = _FinalLineageProfile(
    architecture_version=DISTRIBUTIONAL_ARCHITECTURE_VERSION,
    experiment_id=DISTRIBUTIONAL_EXPERIMENT_ID,
    training_semantics_revision=DISTRIBUTIONAL_TRAINING_SEMANTICS_REVISION,
    phase_order=tuple(DISTRIBUTIONAL_PHASE_ORDER),
    terminal_pass_label="R4_V6_MAINLINE_PASS",
    freeze_schema_version=DISTRIBUTIONAL_FREEZE_SCHEMA_VERSION,
    plan_schema_version=DISTRIBUTIONAL_PLAN_SCHEMA_VERSION,
    daily_schema_version=DISTRIBUTIONAL_DAILY_SCHEMA_VERSION,
    daily_columns=DISTRIBUTIONAL_DAILY_COLUMNS,
    finite_daily_columns=DISTRIBUTIONAL_FINITE_DAILY_COLUMNS,
    formal_empirical_risk_sample_count=FORMAL_EMPIRICAL_RISK_SAMPLE_COUNT,
    risk_measurement=EMPIRICAL_POST_SHIELD_RISK_MEASUREMENT,
)


def _profile_for_architecture(value: Any) -> _FinalLineageProfile:
    architecture = str(value or ARCHITECTURE_VERSION)
    if architecture == ARCHITECTURE_VERSION:
        return _V5_FINAL_PROFILE
    if architecture == DISTRIBUTIONAL_ARCHITECTURE_VERSION:
        return _V6_FINAL_PROFILE
    raise FinalProtocolError(f"unsupported final-evaluation architecture: {architecture}")


def _validate_profile_identity(
    payload: Mapping[str, Any], profile: _FinalLineageProfile
) -> _FinalLineageProfile:
    for field, expected in (
        ("experiment_id", profile.experiment_id),
        ("training_semantics_revision", profile.training_semantics_revision),
    ):
        observed = payload.get(field)
        if observed is not None and observed != expected:
            raise FinalProtocolError(f"final-evaluation {field} mismatch")
    observed_order = payload.get("phase_order")
    if observed_order is not None and tuple(observed_order) != profile.phase_order:
        raise FinalProtocolError("final-evaluation phase_order mismatch")
    return profile


def _profile_from_manifest(manifest: Mapping[str, Any]) -> _FinalLineageProfile:
    profile = _profile_for_architecture(manifest.get("architecture_version"))
    return _validate_profile_identity(manifest, profile)


def _profile_from_config(config: Mapping[str, Any]) -> _FinalLineageProfile:
    project = config.get("project")
    if not isinstance(project, Mapping):
        raise FinalProtocolError("governing config has no project identity")
    profile = _profile_for_architecture(project.get("architecture_version"))
    identity = {
        **project,
        "phase_order": (
            config.get("protocol", {}).get("phase_order")
            if isinstance(config.get("protocol"), Mapping)
            else None
        ),
    }
    return _validate_profile_identity(identity, profile)


def _profile_from_root(root: Path) -> _FinalLineageProfile:
    frozen_preflight = root / "preflight" / "frozen_preflight_manifest.json"
    if not frozen_preflight.is_file():
        return _V5_FINAL_PROFILE
    return _profile_from_manifest(read_json(frozen_preflight))


@dataclass(frozen=True)
class FrozenMethod:
    """One frozen method and the artifact used for each training seed.

    Deterministic benchmarks use the sentinel seed ``0``.  Learned and derived
    methods must use all three registered formal seeds.
    """

    method_id: str
    kind: str
    evidence_by_seed: Mapping[int, str | Path]
    runtime_adapter: str | None = None


BatchEvaluator = Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]]


def assert_seed1_ready_for_later_seeds(output_root: str | Path) -> dict[str, Any]:
    """Prove Seed-1 passed every gate before seeds 2/3 are authorized."""

    root = Path(output_root).resolve()
    profile = _profile_from_root(root)
    phase_evidence = _validate_seed_phases(
        root, FORMAL_SEEDS[0], phase_order=profile.phase_order
    )
    results_path = root / "reports" / "R4_MAINLINE_SEED1_RESULTS.json"
    if not results_path.is_file():
        raise FinalProtocolError("Seed-1 mainline results do not exist")
    results = read_json(results_path)
    gates = results.get("capability_gates")
    if not isinstance(gates, Mapping):
        raise FinalProtocolError("Seed-1 results require capability_gates")
    for phase in profile.phase_order:
        if str(gates.get(phase, "")).upper() not in {"PASS", "PASSED"}:
            raise FinalProtocolError(f"Seed-1 {phase} capability gate did not pass")
    if results.get("terminal_label") != profile.terminal_pass_label:
        raise FinalProtocolError(
            f"later seeds require terminal_label={profile.terminal_pass_label}"
        )
    if int(results.get("seed", -1)) != FORMAL_SEEDS[0]:
        raise FinalProtocolError("Seed-1 result carries the wrong seed")
    if int(results.get("complete_validation_days", -1)) != 179:
        raise FinalProtocolError("Seed-1 must complete all 179 validation dates")
    if int(results.get("complete_validation_origins", -1)) != 17_184:
        raise FinalProtocolError("Seed-1 must complete all 17,184 validation origins")
    if results.get("complete_96_origin_validation_day") is not True:
        raise FinalProtocolError("Seed-1 requires one complete 96-origin validation day")
    if results.get("formal_selection_2024_opened") is not False:
        raise FinalProtocolError("Seed-1 evidence indicates that 2024 was opened")
    if results.get("locked_test_2025_opened") is not False:
        raise FinalProtocolError("Seed-1 evidence indicates that 2025 was opened")
    return {
        "status": "AUTHORIZED",
        "seed": FORMAL_SEEDS[0],
        "results_path": str(results_path),
        "results_sha256": sha256_file(results_path),
        "architecture_version": profile.architecture_version,
        "phase_order": list(profile.phase_order),
        "phases": phase_evidence,
    }


def authorize_later_seed(output_root: str | Path, seed: int) -> dict[str, Any]:
    """Return an auditable authorization for exactly seed 2 or seed 3."""

    selected = int(seed)
    if selected not in FORMAL_SEEDS[1:]:
        raise FinalProtocolError(f"later seed must be one of {FORMAL_SEEDS[1:]}")
    seed1 = assert_seed1_ready_for_later_seeds(output_root)
    return {
        "status": "AUTHORIZED",
        "authorized_seed": selected,
        "prerequisite_seed": FORMAL_SEEDS[0],
        "seed1_results_sha256": seed1["results_sha256"],
    }


def assert_all_seeds_complete(output_root: str | Path) -> dict[str, Any]:
    """Require all 15 train-and-gate phase states before final freezing."""

    root = Path(output_root).resolve()
    profile = _profile_from_root(root)
    seed1 = assert_seed1_ready_for_later_seeds(root)
    seeds = {
        str(seed): _validate_seed_phases(root, seed, phase_order=profile.phase_order)
        for seed in FORMAL_SEEDS
    }
    return {
        "status": "COMPLETE",
        "architecture_version": profile.architecture_version,
        "phase_order": list(profile.phase_order),
        "seed1": seed1,
        "seeds": seeds,
    }


def create_final_freeze_manifest(
    *,
    config_path: str | Path,
    output_root: str | Path,
    artifact_groups: Mapping[str, Sequence[str | Path]],
    methods: Sequence[FrozenMethod],
    eligible_dates: Sequence[str],
    primary_method_id: str,
    primary_benchmark_id: str,
) -> Path:
    """Write the immutable pre-2024 freeze attestation exactly once.

    ``eligible_dates`` must come from metadata-only discovery.  No path or
    payload is accepted here, so this function cannot open the formal outcomes.
    """

    config = load_r4_config(config_path)
    profile = _profile_from_config(config)
    root = Path(output_root).resolve()
    expected_root = (Path(config["_config_path"]).parent.parent / config["project"]["output_root"]).resolve()
    if root != expected_root:
        raise FinalProtocolError(f"output_root must equal governing R4 root: {expected_root}")
    root_profile = _profile_from_root(root)
    if root_profile.architecture_version != profile.architecture_version:
        raise FinalProtocolError("preflight lineage differs from governing config")
    prior_ledger = root / "final_temporal_test" / "execution_ledger.json"
    if prior_ledger.exists() and not (root / "freeze" / "final_freeze_manifest.json").exists():
        raise FinalProtocolError("a final-test ledger exists without a verified pre-open freeze")
    seed_evidence = assert_all_seeds_complete(root)
    dates = _validate_eligible_dates(eligible_dates)
    frozen_groups = _freeze_artifact_groups(artifact_groups)
    method_payload = _freeze_methods(methods)
    checkpoint_hashes = {
        record["sha256"] for record in frozen_groups["checkpoints"]
    }
    benchmark_hashes = {
        record["sha256"] for record in frozen_groups["benchmarks"]
    }
    for method in method_payload:
        registered = benchmark_hashes if method["kind"] == "benchmark" else checkpoint_hashes
        for record in method["evidence_by_seed"].values():
            if record["sha256"] not in registered:
                raise FinalProtocolError(
                    f"method evidence for {method['method_id']} is absent from its freeze group"
                )
    method_ids = {item["method_id"] for item in method_payload}
    by_id = {item["method_id"]: item for item in method_payload}
    if primary_method_id not in method_ids or by_id[primary_method_id]["kind"] == "benchmark":
        raise FinalProtocolError("primary_method_id must name one non-benchmark frozen method")
    if primary_benchmark_id not in method_ids or by_id[primary_benchmark_id]["kind"] != "benchmark":
        raise FinalProtocolError("primary_benchmark_id must name one frozen benchmark")
    procedure: dict[str, Any] = {
        "primary_unit": "day",
        "paired_daily_differences": True,
        "hierarchical_bootstrap": "training_seed_outer_7_day_circular_block_inner",
        "training_seeds": list(FORMAL_SEEDS),
        "block_days": 7,
        "replicates": 10_000,
        "confidence_level": 0.95,
        "holm_secondary_comparisons": True,
        "superiority_boundary": 0.0,
        "noninferiority_margin_fraction_of_benchmark_cost": 0.01,
    }
    if profile.formal_empirical_risk_sample_count is None:
        procedure["weighted_particle_cvar95_is_not_a_confidence_interval"] = True
    else:
        procedure.update(
            {
                "empirical_post_shield_cvar95_is_not_a_confidence_interval": True,
                "formal_empirical_risk_sample_count": (
                    profile.formal_empirical_risk_sample_count
                ),
                "risk_measurement": profile.risk_measurement,
            }
        )
    request = {
        "schema_version": profile.freeze_schema_version,
        "experiment_id": profile.experiment_id,
        "architecture_version": profile.architecture_version,
        "training_semantics_revision": profile.training_semantics_revision,
        "phase_order": list(profile.phase_order),
        "config_path": str(Path(config["_config_path"]).resolve()),
        "config_sha256": config["_config_sha256"],
        "canonical_config_sha256": canonical_config_digest(config),
        "eligible_dates": dates,
        "eligible_dates_sha256": sha256_json(dates),
        "methods": method_payload,
        "primary_method_id": str(primary_method_id),
        "primary_benchmark_id": str(primary_benchmark_id),
        "artifact_groups": frozen_groups,
        "statistical_procedure": procedure,
        "seed_completion_evidence": seed_evidence,
        "formal_selection_2024_opened_before_freeze": False,
        "locked_test_2025_opened": False,
    }
    request_sha = sha256_json(request)
    destination = root / "freeze" / "final_freeze_manifest.json"
    detached = destination.with_suffix(".sha256")
    if destination.exists():
        # A crash between the two exclusive writes is recoverable because no
        # final payload has been authorized yet.  Reconstruct only the detached
        # digest; never replace the manifest itself.
        if not detached.exists():
            provisional = read_json(destination)
            claimed = str(provisional.get("freeze_sha256", ""))
            unsigned = dict(provisional)
            unsigned.pop("freeze_sha256", None)
            if claimed != sha256_json(unsigned):
                raise FinalProtocolError("partial final freeze manifest is invalid")
            _exclusive_text(detached, f"{claimed}  {destination.name}\n")
        current = verify_final_freeze_manifest(destination)
        if current.get("freeze_request_sha256") != request_sha:
            raise FinalProtocolError("immutable final freeze already exists with different content")
        return destination
    payload = {
        **request,
        "frozen_at_utc": utc_now(),
        "freeze_request_sha256": request_sha,
    }
    payload["freeze_sha256"] = sha256_json(payload)
    _exclusive_json(destination, payload)
    _exclusive_text(detached, f"{payload['freeze_sha256']}  {destination.name}\n")
    verify_final_freeze_manifest(destination)
    return destination


def verify_final_freeze_manifest(path: str | Path) -> dict[str, Any]:
    """Verify the self digest, detached digest, and every frozen artifact."""

    manifest_path = Path(path).resolve()
    payload = read_json(manifest_path)
    profile = _profile_from_manifest(payload)
    if payload.get("schema_version") != profile.freeze_schema_version:
        raise FinalProtocolError("unexpected final freeze schema")
    claimed = str(payload.get("freeze_sha256", ""))
    unsigned = dict(payload)
    unsigned.pop("freeze_sha256", None)
    if claimed != sha256_json(unsigned):
        raise FinalProtocolError("final freeze self digest is invalid")
    request = dict(unsigned)
    request.pop("frozen_at_utc", None)
    request_sha = request.pop("freeze_request_sha256", None)
    if request_sha != sha256_json(request):
        raise FinalProtocolError("final freeze request digest is invalid")
    detached = manifest_path.with_suffix(".sha256")
    if not detached.is_file() or detached.read_text(encoding="utf-8").split()[0] != claimed:
        raise FinalProtocolError("final freeze detached digest is absent or invalid")
    _validate_eligible_dates(payload.get("eligible_dates", ()))
    if payload.get("eligible_dates_sha256") != sha256_json(payload["eligible_dates"]):
        raise FinalProtocolError("eligible-date identity changed after freeze")
    config_path = Path(str(payload.get("config_path", "")))
    if not config_path.is_file() or sha256_file(config_path) != payload.get("config_sha256"):
        raise FinalProtocolError("governing config changed after final freeze")
    groups = payload.get("artifact_groups")
    if not isinstance(groups, Mapping) or set(groups) != set(REQUIRED_FREEZE_GROUPS):
        raise FinalProtocolError("frozen artifact groups are missing or unexpected")
    for group_name, records in groups.items():
        if not isinstance(records, list) or not records:
            raise FinalProtocolError(f"frozen group {group_name} is empty")
        for record in records:
            artifact = Path(str(record["path"]))
            if not artifact.is_file():
                raise FinalProtocolError(f"frozen artifact disappeared: {artifact}")
            if int(record["bytes"]) != artifact.stat().st_size:
                raise FinalProtocolError(f"frozen artifact size changed: {artifact}")
            if record["sha256"] != sha256_file(artifact):
                raise FinalProtocolError(f"frozen artifact hash changed: {artifact}")
    _validate_frozen_method_payload(payload.get("methods"))
    seed_evidence = payload.get("seed_completion_evidence")
    if not isinstance(seed_evidence, Mapping):
        raise FinalProtocolError("seed completion evidence is absent")
    for seed_record in seed_evidence.get("seeds", {}).values():
        for record in seed_record.values():
            evidence_path = Path(str(record["path"]))
            if not evidence_path.is_file() or record["sha256"] != sha256_file(evidence_path):
                raise FinalProtocolError(f"post-freeze seed evidence changed: {evidence_path}")
    seed1 = seed_evidence.get("seed1", {})
    results_path = Path(str(seed1.get("results_path", "")))
    if not results_path.is_file() or seed1.get("results_sha256") != sha256_file(results_path):
        raise FinalProtocolError("post-freeze Seed-1 result evidence changed")
    return payload


def claim_and_evaluate_final_once(
    *,
    freeze_manifest_path: str | Path,
    evaluator: BatchEvaluator,
) -> dict[str, Any]:
    """Invoke one batch evaluator once, validate its full cartesian result, and seal it."""

    manifest_path = Path(freeze_manifest_path).resolve()
    manifest = verify_final_freeze_manifest(manifest_path)
    root = manifest_path.parent.parent
    transaction = root / "final_temporal_test"
    ledger_path = transaction / "execution_ledger.json"
    daily_path = transaction / "frozen_method_daily.parquet"
    plan = _evaluation_plan(manifest, manifest_path)
    claimed = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "status": "CLAIMED",
        "claimed_at_utc": utc_now(),
        "freeze_manifest_path": str(manifest_path),
        "freeze_sha256": manifest["freeze_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "evaluator_invocations": 0,
        "formal_selection_2024_opened": False,
        "locked_test_2025_opened": False,
    }
    try:
        _exclusive_json(ledger_path, claimed)
    except FileExistsError as exc:
        raise FinalEvaluationAlreadyClaimed(
            f"formal 2024 transaction already exists: {ledger_path}"
        ) from exc
    claimed["evaluator_invocations"] = 1
    claimed["evaluation_started_at_utc"] = utc_now()
    # At this point the callback owns the one and only authorization to open
    # eligible 2024 payloads.  Any exception is treated as access-ambiguous.
    atomic_json(ledger_path, claimed)
    try:
        raw_rows = evaluator(plan)
    except BaseException as exc:
        claimed.update(
            {
                "status": "FAILED_AMBIGUOUS_NO_RETRY",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "formal_selection_2024_opened": True,
                "failed_at_utc": utc_now(),
            }
        )
        atomic_json(ledger_path, claimed)
        raise
    try:
        rows = _validate_daily_rows(raw_rows, manifest)
        atomic_parquet(daily_path, rows)
    except BaseException as exc:
        claimed.update(
            {
                "status": "FAILED_INVALID_RESULTS_NO_RETRY",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "formal_selection_2024_opened": True,
                "failed_at_utc": utc_now(),
            }
        )
        atomic_json(ledger_path, claimed)
        raise
    claimed.update(
        {
            "status": "RESULTS_CAPTURED",
            "formal_selection_2024_opened": True,
            "locked_test_2025_opened": False,
            "daily_results_path": str(daily_path),
            "daily_results_sha256": sha256_file(daily_path),
            "daily_rows": len(rows),
            "eligible_dates_completed": len(manifest["eligible_dates"]),
            "completed_at_utc": utc_now(),
        }
    )
    atomic_json(ledger_path, claimed)
    return claimed


def load_captured_final_rows(freeze_manifest_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load captured rows for resumable post-processing without re-evaluation."""

    manifest_path = Path(freeze_manifest_path).resolve()
    manifest = verify_final_freeze_manifest(manifest_path)
    ledger_path = manifest_path.parent.parent / "final_temporal_test" / "execution_ledger.json"
    ledger = read_json(ledger_path)
    if ledger.get("status") not in {"RESULTS_CAPTURED", "COMPLETE"}:
        raise FinalProtocolError("final daily results have not been safely captured")
    if ledger.get("freeze_sha256") != manifest["freeze_sha256"]:
        raise FinalProtocolError("execution ledger belongs to another freeze")
    daily_path = Path(str(ledger.get("daily_results_path", "")))
    if not daily_path.is_file() or sha256_file(daily_path) != ledger.get("daily_results_sha256"):
        raise FinalProtocolError("captured daily result artifact is absent or changed")
    frame = pd.read_parquet(daily_path)
    rows = _validate_daily_rows(frame.to_dict(orient="records"), manifest)
    return manifest, rows


def mark_final_outputs_complete(
    freeze_manifest_path: str | Path,
    *,
    output_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Seal derived outputs after their writer completes successfully."""

    manifest_path = Path(freeze_manifest_path).resolve()
    manifest = verify_final_freeze_manifest(manifest_path)
    ledger_path = manifest_path.parent.parent / "final_temporal_test" / "execution_ledger.json"
    ledger = read_json(ledger_path)
    if ledger.get("status") not in {"RESULTS_CAPTURED", "COMPLETE"}:
        raise FinalProtocolError("cannot complete final outputs without captured results")
    records = []
    for value in output_paths:
        path = Path(value).resolve()
        if not path.is_file():
            raise FinalProtocolError(f"required final output is absent: {path}")
        records.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    ledger.update(
        {
            "status": "COMPLETE",
            "outputs": records,
            "finalized_at_utc": utc_now(),
            "formal_selection_2024_opened": True,
            "locked_test_2025_opened": False,
        }
    )
    atomic_json(ledger_path, ledger)
    return ledger


def _validate_seed_phases(
    root: Path,
    seed: int,
    *,
    phase_order: Sequence[str] = PHASE_ORDER,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for phase in phase_order:
        path = root / f"seed={seed}" / f"phase={phase}" / "state.json"
        if not path.is_file():
            raise FinalProtocolError(f"missing phase state: {path}")
        state = read_json(path)
        if int(state.get("seed", -1)) != seed or state.get("phase") != phase:
            raise FinalProtocolError(f"phase identity mismatch: {path}")
        if state.get("status") != "COMPLETE":
            raise FinalProtocolError(f"seed {seed} phase {phase} is not COMPLETE")
        if str(state.get("capability_gate_status", "")).upper() not in {"PASS", "PASSED"}:
            raise FinalProtocolError(f"seed {seed} phase {phase} gate did not pass")
        if state.get("formal_selection_2024_opened") not in (None, False):
            raise FinalProtocolError(f"seed {seed} phase {phase} opened 2024")
        if state.get("locked_test_2025_opened") not in (None, False):
            raise FinalProtocolError(f"seed {seed} phase {phase} opened 2025")
        evidence[phase] = {"path": str(path), "sha256": sha256_file(path)}
    return evidence


def _validate_eligible_dates(values: Sequence[str]) -> list[str]:
    dates = [str(value)[:10] for value in values]
    if not dates or len(dates) != len(set(dates)) or dates != sorted(dates):
        raise FinalProtocolError("eligible 2024 dates must be nonempty, unique, and sorted")
    for value in dates:
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise FinalProtocolError(f"invalid eligible date: {value}") from exc
        if parsed.year != 2024:
            raise FinalProtocolError("final eligible-date metadata must contain only 2024")
    return dates


def _freeze_artifact_groups(groups: Mapping[str, Sequence[str | Path]]) -> dict[str, list[dict[str, Any]]]:
    if set(groups) != set(REQUIRED_FREEZE_GROUPS):
        raise FinalProtocolError(
            f"artifact_groups must contain exactly {REQUIRED_FREEZE_GROUPS}"
        )
    frozen: dict[str, list[dict[str, Any]]] = {}
    for name in REQUIRED_FREEZE_GROUPS:
        values = list(groups[name])
        if not values:
            raise FinalProtocolError(f"freeze artifact group {name} is empty")
        records = []
        for value in values:
            path = Path(value).resolve()
            if not path.is_file():
                raise FinalProtocolError(f"freeze artifact does not exist: {path}")
            records.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
        frozen[name] = records
    return frozen


def _freeze_methods(methods: Sequence[FrozenMethod]) -> list[dict[str, Any]]:
    if not methods:
        raise FinalProtocolError("at least one final method must be frozen")
    payload = []
    seen: set[str] = set()
    for method in methods:
        method_id = str(method.method_id).strip()
        kind = str(method.kind).lower()
        if not method_id or method_id in seen:
            raise FinalProtocolError("frozen method ids must be nonempty and unique")
        if kind not in ALLOWED_METHOD_KINDS:
            raise FinalProtocolError(f"invalid frozen method kind: {kind}")
        expected_seeds = (0,) if kind == "benchmark" else FORMAL_SEEDS
        runtime_adapter = (
            _default_runtime_adapter(method_id, kind)
            if method.runtime_adapter is None
            else str(method.runtime_adapter).strip()
        )
        if not runtime_adapter:
            raise FinalProtocolError(f"method {method_id} requires a runtime adapter")
        evidence = {int(seed): Path(path).resolve() for seed, path in method.evidence_by_seed.items()}
        if tuple(sorted(evidence)) != tuple(sorted(expected_seeds)):
            raise FinalProtocolError(f"method {method_id} requires seeds {expected_seeds}")
        by_seed: dict[str, Any] = {}
        for seed in expected_seeds:
            path = evidence[seed]
            if not path.is_file():
                raise FinalProtocolError(f"method evidence does not exist: {path}")
            by_seed[str(seed)] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        payload.append(
            {
                "method_id": method_id,
                "kind": kind,
                "training_seeds": list(expected_seeds),
                "runtime_adapter": runtime_adapter,
                "evidence_by_seed": by_seed,
            }
        )
        seen.add(method_id)
    _validate_frozen_method_payload(payload)
    return payload


def _validate_frozen_method_payload(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise FinalProtocolError("final freeze has no methods")
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise FinalProtocolError("frozen method record must be a mapping")
        method_id = str(item.get("method_id", ""))
        kind = str(item.get("kind", ""))
        if not method_id or method_id in seen or kind not in ALLOWED_METHOD_KINDS:
            raise FinalProtocolError("invalid or duplicate frozen method")
        expected = [0] if kind == "benchmark" else list(FORMAL_SEEDS)
        if not str(item.get("runtime_adapter", "")).strip():
            raise FinalProtocolError(f"frozen method {method_id} runtime adapter changed")
        if item.get("training_seeds") != expected:
            raise FinalProtocolError(f"frozen method {method_id} seed contract changed")
        evidence = item.get("evidence_by_seed")
        if not isinstance(evidence, Mapping) or set(evidence) != {str(seed) for seed in expected}:
            raise FinalProtocolError(f"frozen method {method_id} evidence mapping changed")
        for seed in expected:
            record = evidence[str(seed)]
            path = Path(str(record["path"]))
            if not path.is_file() or record["sha256"] != sha256_file(path):
                raise FinalProtocolError(f"frozen method evidence changed: {path}")
        seen.add(method_id)


def _default_runtime_adapter(method_id: str, kind: str) -> str:
    if kind == "benchmark":
        return method_id
    registered_ablations = {
        "A0", "A1", "A2", "A3", "A4", "dense_no_moe", "one_scenario",
        "no_cvar", "no_a1", "raw_vs_shield",
    }
    return f"r4_ablation:{method_id}" if method_id in registered_ablations else "r4_mainline"


def _evaluation_plan(manifest: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    profile = _profile_from_manifest(manifest)
    unsigned = {
        "schema_version": profile.plan_schema_version,
        "experiment_id": profile.experiment_id,
        "architecture_version": profile.architecture_version,
        "phase_order": list(profile.phase_order),
        "freeze_manifest_path": str(manifest_path),
        "freeze_sha256": manifest["freeze_sha256"],
        "eligible_dates": list(manifest["eligible_dates"]),
        "methods": list(manifest["methods"]),
        "required_daily_columns": list(profile.daily_columns),
        "origins_per_day": 96,
        "same_dates_for_all_methods": True,
        "exactly_once": True,
        "locked_test_2025_opened": False,
    }
    if profile.formal_empirical_risk_sample_count is not None:
        unsigned.update(
            {
                "formal_empirical_risk_sample_count": (
                    profile.formal_empirical_risk_sample_count
                ),
                "risk_measurement": profile.risk_measurement,
            }
        )
    return {**unsigned, "plan_sha256": sha256_json(unsigned)}


def _validate_daily_rows(
    raw_rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if isinstance(raw_rows, (str, bytes, Mapping)):
        raise FinalProtocolError("batch evaluator must return a sequence of daily rows")
    rows = [dict(row) for row in raw_rows]
    profile = _profile_from_manifest(manifest)
    dates = list(manifest["eligible_dates"])
    methods = {item["method_id"]: item for item in manifest["methods"]}
    expected = {
        (method_id, int(seed), operating_date)
        for method_id, method in methods.items()
        for seed in method["training_seeds"]
        for operating_date in dates
    }
    actual: set[tuple[str, int, str]] = set()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if set(row) != set(profile.daily_columns):
            missing = sorted(set(profile.daily_columns) - set(row))
            extra = sorted(set(row) - set(profile.daily_columns))
            raise FinalProtocolError(f"daily row schema mismatch; missing={missing}, extra={extra}")
        if row["schema_version"] != profile.daily_schema_version:
            raise FinalProtocolError("daily result schema version mismatch")
        method_id = str(row["method_id"])
        seed = int(row["training_seed"])
        operating_date = str(row["operating_date"])[:10]
        key = (method_id, seed, operating_date)
        if key not in expected or key in actual:
            raise FinalProtocolError(f"unexpected or duplicate daily result key: {key}")
        method = methods[method_id]
        if row["method_kind"] != method["kind"]:
            raise FinalProtocolError(f"method kind changed during evaluation: {method_id}")
        if int(row["origins_evaluated"]) != 96:
            raise FinalProtocolError("every final-test day requires exactly 96 origins")
        if int(row["rejected_origins"]) < 0:
            raise FinalProtocolError("rejected_origins cannot be negative")
        for name in profile.finite_daily_columns:
            if not math.isfinite(float(row[name])):
                raise FinalProtocolError(f"daily field {name} is non-finite")
        if profile.formal_empirical_risk_sample_count is not None:
            if (
                int(row["formal_empirical_risk_sample_count"])
                != profile.formal_empirical_risk_sample_count
            ):
                raise FinalProtocolError(
                    "formal empirical risk requires exactly 512 post-shield replays"
                )
            if row["risk_measurement"] != profile.risk_measurement:
                raise FinalProtocolError("formal empirical risk measurement mismatch")
        if not 0.0 <= float(row["independent_feasibility_fraction"]) <= 1.0:
            raise FinalProtocolError("feasibility fraction must lie in [0,1]")
        if not 0.0 <= float(row["corrective_intervention_rate"]) <= 1.0:
            raise FinalProtocolError("intervention rate must lie in [0,1]")
        expected_hash = method["evidence_by_seed"][str(seed)]["sha256"]
        if row["frozen_evidence_sha256"] != expected_hash:
            raise FinalProtocolError(f"method evidence hash mismatch: {key}")
        if row["freeze_sha256"] != manifest["freeze_sha256"]:
            raise FinalProtocolError(f"freeze hash mismatch: {key}")
        normalized.append({name: row[name] for name in profile.daily_columns})
        actual.add(key)
    missing = sorted(expected - actual)
    if missing:
        raise FinalProtocolError(f"final evaluator omitted {len(missing)} method/seed/date rows")
    normalized.sort(key=lambda row: (row["operating_date"], row["method_id"], int(row["training_seed"])))
    return normalized


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    _exclusive_bytes(path, data)


def _exclusive_text(path: Path, value: str) -> None:
    _exclusive_bytes(path, value.encode("utf-8"))


def _exclusive_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # A partial claim must remain in place.  Removing it could authorize a
        # duplicate formal evaluation after an ambiguous crash.
        raise


__all__ = [
    "DAILY_COLUMNS",
    "DAILY_SCHEMA_VERSION",
    "DISTRIBUTIONAL_DAILY_COLUMNS",
    "DISTRIBUTIONAL_DAILY_SCHEMA_VERSION",
    "DISTRIBUTIONAL_PLAN_SCHEMA_VERSION",
    "EMPIRICAL_POST_SHIELD_RISK_MEASUREMENT",
    "FORMAL_EMPIRICAL_RISK_SAMPLE_COUNT",
    "FinalEvaluationAlreadyClaimed",
    "FinalProtocolError",
    "FrozenMethod",
    "assert_all_seeds_complete",
    "assert_seed1_ready_for_later_seeds",
    "authorize_later_seed",
    "claim_and_evaluate_final_once",
    "create_final_freeze_manifest",
    "load_captured_final_rows",
    "mark_final_outputs_complete",
    "verify_final_freeze_manifest",
]
