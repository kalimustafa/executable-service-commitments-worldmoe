"""Fail-isolated completion of an interrupted formal DART hybrid scheduler.

The frozen checkpoint trainer writes the synchronized epoch panel before it
finalizes each group.  If one group has no healthy checkpoint, its finalizer
raises after recording ``checkpoint_group_failure.json`` and the scheduler
does not get a chance to finalize the unaffected groups or its own manifest.

This compatibility layer is deliberately narrow.  It is usable only after
the scheduler state proves that every still-active group has reached the hard
maximum epoch.  It then performs group finalization only, preserves approved
selection failures, continues the remaining groups, and writes one
self-hashed scheduler manifest with separate ``completed_groups`` and
``failed_groups`` inventories.  It never relaxes a health gate and never
trains an additional epoch.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from dcvpp.core import atomic_json, sha256_file, sha256_json, utc_now
from dcvpp.experiments.dart_checkpoint_replay_compat import (
    FROZEN_DART_CHECKPOINT_TRAINING_SHA256,
)


FAILURE_COMPAT_SCHEMA_VERSION = "dcvpp.dart_scheduler_failure_compat.v1"
DEFAULT_RECOVERABLE_STATUSES = ("NO_HEALTHY_EPOCH",)


class DARTSchedulerFailureCompatibilityError(RuntimeError):
    """Raised when scheduler recovery would exceed the frozen contract."""


def _without_digest(value: Mapping[str, Any], digest_key: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != digest_key}


def _read_self_hashed(path: Path, *, digest_key: str) -> dict[str, Any]:
    if not path.is_file():
        raise DARTSchedulerFailureCompatibilityError(f"required file is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get(digest_key) != sha256_json(_without_digest(value, digest_key)):
        raise DARTSchedulerFailureCompatibilityError(
            f"self-hash changed for {path.name}"
        )
    return value


def _scheduler_contract(module: Any, jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    contexts = tuple(module._context_from_job(job) for job in jobs)
    fingerprints = tuple(str(context["group_fingerprint"]) for context in contexts)
    if not fingerprints or len(set(fingerprints)) != len(fingerprints):
        raise DARTSchedulerFailureCompatibilityError(
            "recovery requires a nonempty unique group panel"
        )
    policies = {
        (
            tuple(map(int, context["seeds"])),
            int(job["max_epochs"]),
            int(job.get("minimum_epochs", 40)),
            int(job.get("evaluation_interval_epochs", 5)),
            int(job.get("stop_commit_interval_epochs", 10)),
            int(job.get("stability_evaluations", 3)),
            float(job.get("early_stopping_min_delta", 1.0e-8)),
        )
        for job, context in zip(jobs, contexts, strict=True)
    }
    if len(policies) != 1:
        raise DARTSchedulerFailureCompatibilityError(
            "recovery jobs do not share one frozen scheduler policy"
        )
    (
        seeds,
        max_epochs,
        minimum_epochs,
        interval,
        global_round_interval,
        stability,
        min_delta,
    ) = next(iter(policies))
    scheduler_fingerprint = sha256_json(
        {
            "schema_version": module.SCHEMA_VERSION,
            "order": "warmup_to_40_then_10_epoch_active_group_rounds",
            "group_fingerprints_in_order": list(fingerprints),
            "study_seeds": list(seeds),
            "max_epochs": max_epochs,
            "minimum_epochs": minimum_epochs,
            "evaluation_interval_epochs": interval,
            "global_round_interval_epochs": global_round_interval,
            "stability_evaluations": stability,
            "early_stopping_min_delta": min_delta,
            "module_sha256": sha256_file(Path(module.__file__)),
        }
    )
    return {
        "contexts": contexts,
        "fingerprints": fingerprints,
        "seeds": seeds,
        "max_epochs": max_epochs,
        "minimum_epochs": minimum_epochs,
        "interval": interval,
        "global_round_interval": global_round_interval,
        "scheduler_fingerprint": scheduler_fingerprint,
    }


def _completed_record(
    module: Any,
    *,
    job: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = Path(context["manifest_path"])
    manifest = module._load_group_manifest(path, str(context["group_fingerprint"]))
    if manifest is None:
        return None
    assessments = list(manifest.get("synchronized_early_stopping_assessments", ()))
    return {
        "group_fingerprint": str(context["group_fingerprint"]),
        "task": str(job["task"]),
        "model_name": str(job["model_name"]),
        "output_root": Path(job["output_root"]).as_posix(),
        "completed_common_epoch": int(manifest["completed_common_epoch"]),
        "selected_common_epoch": int(manifest["selected_common_epoch"]),
        "checkpoint_group_manifest_path": path.as_posix(),
        "checkpoint_group_manifest_sha256": sha256_file(path),
        "scheduler_stop_decision": dict(
            manifest.get("synchronized_early_stopping_decision", {})
        ),
        "assessment_history_sha256": sha256_json(assessments),
        "reconstructed_from_hash_verified_group_manifest": True,
    }


def _require_max_epoch_panel(
    *,
    context: Mapping[str, Any],
    seeds: Sequence[int],
    max_epochs: int,
    scheduler_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    coordinator_path = Path(context["group_root"]) / "group_coordinator_state.json"
    if coordinator_path.is_file():
        coordinator = _read_self_hashed(coordinator_path, digest_key="state_sha256")
        if (
            coordinator.get("group_fingerprint") != context["group_fingerprint"]
            or int(coordinator.get("completed_common_epoch", -1)) != int(max_epochs)
            or tuple(map(int, coordinator.get("study_seeds", ()))) != tuple(seeds)
        ):
            raise DARTSchedulerFailureCompatibilityError(
                "active group lacks an exact maximum-epoch three-seed panel"
            )
        latest = coordinator.get("seed_latest_checkpoints", {})
        if set(latest) != {str(seed) for seed in seeds}:
            raise DARTSchedulerFailureCompatibilityError(
                "active group latest-checkpoint seed panel changed"
            )
        for record in latest.values():
            checkpoint = Path(str(record.get("path", "")))
            if (
                not checkpoint.is_file()
                or sha256_file(checkpoint) != record.get("sha256")
            ):
                raise DARTSchedulerFailureCompatibilityError(
                    "active group maximum-epoch checkpoint hash changed"
                )
        return {
            "evidence_source": "group_coordinator_state",
            "path": coordinator_path.as_posix(),
            "sha256": sha256_file(coordinator_path),
        }

    # The frozen scheduler persists the epoch-100 advance state before it
    # enters group finalization.  Thus groups after the first failed finalizer
    # legitimately have no coordinator yet.  Their terminal execution events
    # plus the referenced immutable latest checkpoints prove the same aligned
    # three-seed boundary without invoking training.
    fingerprint = str(context["group_fingerprint"])
    terminal = [
        dict(event)
        for event in scheduler_events
        if str(event.get("group_fingerprint")) == fingerprint
        and int(event.get("common_epoch", -1)) == int(max_epochs)
    ]
    by_seed: dict[int, dict[str, Any]] = {}
    for event in terminal:
        seed = int(event.get("seed", -1))
        if seed in by_seed:
            raise DARTSchedulerFailureCompatibilityError(
                "active group has duplicate terminal scheduler events"
            )
        by_seed[seed] = event
    if set(by_seed) != set(map(int, seeds)):
        raise DARTSchedulerFailureCompatibilityError(
            "active group terminal scheduler event seed panel changed"
        )
    import torch

    verified: dict[str, dict[str, Any]] = {}
    group_root = Path(context["group_root"])
    for seed in seeds:
        event = by_seed[int(seed)]
        if int(event.get("trained_through_epoch", -1)) != int(max_epochs):
            raise DARTSchedulerFailureCompatibilityError(
                "terminal scheduler event did not reach maximum epoch"
            )
        candidates = list(
            (group_root / f"seed_{int(seed)}").glob("batch_*/latest.pt")
        )
        expected_sha = str(event.get("latest_checkpoint_sha256", ""))
        matches = [path for path in candidates if sha256_file(path) == expected_sha]
        if len(matches) != 1:
            raise DARTSchedulerFailureCompatibilityError(
                "terminal scheduler event does not identify one latest checkpoint"
            )
        checkpoint = matches[0]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            int(payload.get("epoch", -1)) != int(max_epochs)
            or int(payload.get("seed", -1)) != int(seed)
            or payload.get("resume_fingerprint") is None
            or str(payload.get("task")) != str(event.get("task"))
            or str(payload.get("model_name")) != str(event.get("model_name"))
        ):
            raise DARTSchedulerFailureCompatibilityError(
                "terminal latest checkpoint identity or epoch changed"
            )
        verified[str(seed)] = {
            "path": checkpoint.as_posix(),
            "sha256": expected_sha,
            "epoch": int(max_epochs),
        }
    return {
        "evidence_source": "hash_verified_terminal_scheduler_events",
        "terminal_event_records_sha256": sha256_json(terminal),
        "latest_checkpoints": verified,
    }


def _failed_record(
    *,
    job: Mapping[str, Any],
    context: Mapping[str, Any],
    recoverable_statuses: frozenset[str],
    max_epochs: int,
) -> dict[str, Any]:
    path = Path(context["group_root"]) / "checkpoint_group_failure.json"
    if not path.is_file():
        raise DARTSchedulerFailureCompatibilityError(
            "selection exception did not leave group failure evidence"
        )
    failure = json.loads(path.read_text(encoding="utf-8"))
    status = str(failure.get("status", ""))
    selection_status = str(failure.get("selection_result", {}).get("status", ""))
    if (
        failure.get("group_fingerprint") != context["group_fingerprint"]
        or status not in recoverable_statuses
        or selection_status != status
        or failure.get("market_replay_or_locked_test_consumed") is not False
    ):
        raise DARTSchedulerFailureCompatibilityError(
            "group failure is not an approved validation-only selection failure"
        )
    return {
        "group_fingerprint": str(context["group_fingerprint"]),
        "task": str(job["task"]),
        "model_name": str(job["model_name"]),
        "output_root": Path(job["output_root"]).as_posix(),
        "completed_common_epoch": int(max_epochs),
        "failure_status": status,
        "checkpoint_group_failure_path": path.as_posix(),
        "checkpoint_group_failure_sha256": sha256_file(path),
        "selection_result": dict(failure["selection_result"]),
        "all_seeds_reached_hard_maximum_before_selection_failure": True,
        "failure_preserved_without_gate_relaxation": True,
        "market_replay_or_locked_test_consumed": False,
    }


def recover_hybrid_scheduler_with_isolated_failures(
    module: Any,
    *,
    jobs: Sequence[Mapping[str, Any]],
    scheduler_root: Path,
    recoverable_statuses: Sequence[str] = DEFAULT_RECOVERABLE_STATUSES,
    enforce_frozen_module_sha: bool = True,
) -> dict[str, Any]:
    """Finalize a maximum-epoch scheduler while isolating approved failures.

    This function refuses to operate before the hard maximum boundary.  Calls
    to the frozen group finalizer therefore only load existing epoch ledgers,
    select/materialize a checkpoint, or preserve ``NO_HEALTHY_EPOCH``; they
    cannot append training epochs.
    """

    if enforce_frozen_module_sha and sha256_file(Path(module.__file__)) != (
        FROZEN_DART_CHECKPOINT_TRAINING_SHA256
    ):
        raise DARTSchedulerFailureCompatibilityError(
            "frozen DART checkpoint trainer SHA-256 changed"
        )
    ordered_jobs = tuple(dict(job) for job in jobs)
    contract = _scheduler_contract(module, ordered_jobs)
    scheduler = Path(scheduler_root)
    manifest_path = scheduler / "hybrid_scheduler_manifest.json"
    if manifest_path.is_file():
        manifest = _read_self_hashed(manifest_path, digest_key="manifest_sha256")
        if (
            manifest.get("scheduler_fingerprint")
            != contract["scheduler_fingerprint"]
            or manifest.get("status") != "COMPLETE"
        ):
            raise DARTSchedulerFailureCompatibilityError(
                "existing hybrid scheduler manifest changed"
            )
        return manifest

    state_path = scheduler / "hybrid_scheduler_state.json"
    state = _read_self_hashed(state_path, digest_key="state_sha256")
    if (
        state.get("scheduler_fingerprint") != contract["scheduler_fingerprint"]
        or tuple(state.get("group_fingerprints_in_order", ()))
        != contract["fingerprints"]
        or int(state.get("common_epoch_boundary", -1)) != contract["max_epochs"]
        or state.get("market_replay_or_locked_test_consumed") is not False
    ):
        raise DARTSchedulerFailureCompatibilityError(
            "scheduler is not at the exact pre-OOS hard-maximum recovery boundary"
        )
    events = list(state.get("execution_events", ()))
    if state.get("execution_events_sha256") != sha256_json(events):
        raise DARTSchedulerFailureCompatibilityError(
            "scheduler execution-event ledger hash changed"
        )
    active = set(map(str, state.get("active_group_fingerprints", ())))
    completed_before = set(
        map(str, state.get("completed_group_fingerprints", ()))
    )
    if (
        active & completed_before
        or active | completed_before != set(contract["fingerprints"])
    ):
        raise DARTSchedulerFailureCompatibilityError(
            "scheduler active/completed partition changed"
        )

    completed_groups: dict[str, dict[str, Any]] = {}
    failed_groups: dict[str, dict[str, Any]] = {}
    approved = frozenset(map(str, recoverable_statuses))
    if not approved:
        raise DARTSchedulerFailureCompatibilityError(
            "at least one explicit recoverable status is required"
        )

    # First validate every predecessor-complete group.  An active group may
    # already have a manifest from a prior interrupted recovery attempt.
    unresolved: list[tuple[dict[str, Any], Mapping[str, Any]]] = []
    for job, context in zip(ordered_jobs, contract["contexts"], strict=True):
        fingerprint = str(context["group_fingerprint"])
        record = _completed_record(module, job=job, context=context)
        if record is not None:
            completed_groups[fingerprint] = record
        elif fingerprint in completed_before:
            raise DARTSchedulerFailureCompatibilityError(
                "scheduler claims a completed group without its manifest"
            )
        else:
            _require_max_epoch_panel(
                context=context,
                seeds=contract["seeds"],
                max_epochs=contract["max_epochs"],
                scheduler_events=events,
            )
            unresolved.append((job, context))

    selection_error = getattr(module, "DARTCheckpointSelectionError")
    for job, context in unresolved:
        fingerprint = str(context["group_fingerprint"])
        try:
            module.fit_dart_energy_seed_group(**job)
        except selection_error:
            failed_groups[fingerprint] = _failed_record(
                job=job,
                context=context,
                recoverable_statuses=approved,
                max_epochs=contract["max_epochs"],
            )
            continue
        record = _completed_record(module, job=job, context=context)
        if record is None:
            raise DARTSchedulerFailureCompatibilityError(
                "group finalizer returned without a complete manifest"
            )
        completed_groups[fingerprint] = record

    if set(completed_groups) | set(failed_groups) != set(contract["fingerprints"]):
        raise DARTSchedulerFailureCompatibilityError(
            "failure-isolated scheduler did not resolve every group"
        )
    rounds = list(
        range(
            contract["minimum_epochs"] + contract["global_round_interval"],
            contract["max_epochs"] + 1,
            contract["global_round_interval"],
        )
    )
    if not rounds or rounds[-1] != contract["max_epochs"]:
        rounds.append(contract["max_epochs"])
    manifest_core = {
        "schema_version": module.SCHEMA_VERSION,
        "status": "COMPLETE",
        "scheduler_scientific_status": (
            "COMPLETE_WITH_FAILED_GROUPS" if failed_groups else "COMPLETE"
        ),
        "failure_compat_schema_version": FAILURE_COMPAT_SCHEMA_VERSION,
        "scheduler_fingerprint": contract["scheduler_fingerprint"],
        "actual_training_order": "warmup_to_40_then_10_epoch_active_group_rounds",
        "efficient_hybrid_group_scheduler": True,
        "seed_major_cross_model_round_robin": False,
        "warmup_common_epoch": contract["minimum_epochs"],
        "active_group_round_epochs": rounds,
        "assessment_interval_epochs": contract["interval"],
        "stop_committed_only_at_global_round_boundary": True,
        "study_seeds": list(contract["seeds"]),
        "group_fingerprints_in_order": list(contract["fingerprints"]),
        "group_count": len(contract["fingerprints"]),
        "completed_group_count": len(completed_groups),
        "failed_group_count": len(failed_groups),
        "resolved_group_count": len(completed_groups) + len(failed_groups),
        "completed_groups": completed_groups,
        "failed_groups": failed_groups,
        "failure_isolation_policy": {
            "recoverable_statuses": sorted(approved),
            "health_or_calibration_gate_relaxed": False,
            "additional_training_epochs_during_recovery": 0,
            "failed_groups_eligible_for_oos_materialization": False,
            "unaffected_groups_continue": True,
        },
        "source_scheduler_state": {
            "path": state_path.as_posix(),
            "sha256_before_recovery": sha256_file(state_path),
            "status": state.get("status"),
            "common_epoch_boundary": contract["max_epochs"],
        },
        "execution_events": events,
        "execution_events_sha256": sha256_json(events),
        "individual_seed_early_stopping": False,
        "market_replay_or_locked_test_consumed": False,
        "written_at_utc": utc_now(),
    }
    manifest = {
        **manifest_core,
        "manifest_sha256": sha256_json(manifest_core),
    }
    atomic_json(manifest_path, manifest)

    final_state_core = {
        **_without_digest(state, "state_sha256"),
        "status": "COMPLETE_WITH_FAILED_GROUPS" if failed_groups else "COMPLETE",
        "active_group_fingerprints": [],
        "completed_group_fingerprints": sorted(completed_groups),
        "failed_group_fingerprints": sorted(failed_groups),
        "failure_compat_schema_version": FAILURE_COMPAT_SCHEMA_VERSION,
        "hybrid_scheduler_manifest_path": manifest_path.as_posix(),
        "hybrid_scheduler_manifest_sha256": sha256_file(manifest_path),
        "written_at_utc": utc_now(),
    }
    atomic_json(
        state_path,
        {**final_state_core, "state_sha256": sha256_json(final_state_core)},
    )
    return manifest


@contextmanager
def isolated_hybrid_scheduler_failures(
    module: Any,
    *,
    recoverable_statuses: Sequence[str] = DEFAULT_RECOVERABLE_STATUSES,
) -> Iterator[None]:
    """Temporarily route the frozen scheduler through fail-isolated recovery."""

    original = module.pretrain_dart_energy_groups_hybrid

    def wrapped(
        jobs: Sequence[Mapping[str, Any]], *, scheduler_root: Path
    ) -> dict[str, Any]:
        state_path = Path(scheduler_root) / "hybrid_scheduler_state.json"
        if state_path.is_file():
            state = _read_self_hashed(state_path, digest_key="state_sha256")
            max_epochs = int(jobs[0]["max_epochs"]) if jobs else -1
            if int(state.get("common_epoch_boundary", -2)) == max_epochs:
                return recover_hybrid_scheduler_with_isolated_failures(
                    module,
                    jobs=jobs,
                    scheduler_root=scheduler_root,
                    recoverable_statuses=recoverable_statuses,
                )
        try:
            return original(jobs, scheduler_root=scheduler_root)
        except getattr(module, "DARTCheckpointSelectionError"):
            return recover_hybrid_scheduler_with_isolated_failures(
                module,
                jobs=jobs,
                scheduler_root=scheduler_root,
                recoverable_statuses=recoverable_statuses,
            )

    module.pretrain_dart_energy_groups_hybrid = wrapped
    try:
        yield
    finally:
        if module.pretrain_dart_energy_groups_hybrid is not wrapped:
            raise DARTSchedulerFailureCompatibilityError(
                "hybrid scheduler function changed inside compatibility context"
            )
        module.pretrain_dart_energy_groups_hybrid = original


__all__ = [
    "DARTSchedulerFailureCompatibilityError",
    "DEFAULT_RECOVERABLE_STATUSES",
    "FAILURE_COMPAT_SCHEMA_VERSION",
    "isolated_hybrid_scheduler_failures",
    "recover_hybrid_scheduler_with_isolated_failures",
]
