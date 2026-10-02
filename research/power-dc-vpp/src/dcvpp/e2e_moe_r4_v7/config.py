"""Configuration contract for the approved R4-V7 joint world-policy lineage."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


ARCHITECTURE_VERSION = "baseline_anchored_joint_world_policy_moe_r4_v7"
PHASE_ORDER = (
    "d0",
    "p0",
    "j1",
    "j2",
    "j3",
    "calibration_restore",
    "a1",
    "a2",
)

SPLIT_CONTRACT: dict[str, tuple[str, str, int, str]] = {
    "train": ("2022-01-01", "2023-12-31", 722, "development_fit"),
    "validation": (
        "2024-01-01",
        "2024-12-31",
        364,
        "development_validation",
    ),
    "trainval_refit": (
        "2022-01-01",
        "2024-12-31",
        1086,
        "frozen_method_refit",
    ),
    "final_test": (
        "2025-01-01",
        "2025-12-31",
        362,
        "unique_final_test",
    ),
}

PROHIBIT_2025_FOR = {
    "training",
    "normalization",
    "architecture_selection",
    "hyperparameter_selection",
    "checkpoint_selection",
    "gate_repair",
    "baseline_calibration",
    "early_stopping",
}


class V7ConfigError(ValueError):
    """Raised when a V7 configuration weakens the approved contract."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V7ConfigError(f"{label} must be a mapping")
    return value


def _resolve(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise V7ConfigError(f"{label} must be a non-empty path")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _validate_split(
    splits: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    entry = _mapping(splits.get(name), f"protocol.splits.{name}")
    start, end, days, role = SPLIT_CONTRACT[name]
    if (
        str(entry.get("start")) != start
        or str(entry.get("end")) != end
        or int(entry.get("days", -1)) != days
        or str(entry.get("role")) != role
    ):
        raise V7ConfigError(
            f"protocol.splits.{name} must be {start}..{end}, "
            f"days={days}, role={role}"
        )
    return entry


def validate_v7_config(config: Mapping[str, Any], repository_root: str | Path) -> dict[str, Path]:
    """Validate the small set of invariants that define the V7 lineage.

    This intentionally avoids a repository-wide compliance layer.  It guards
    only the scientific boundaries that would otherwise change the method.
    """

    root = Path(repository_root).resolve()
    project = _mapping(config.get("project"), "project")
    if project.get("architecture_version") != ARCHITECTURE_VERSION:
        raise V7ConfigError("configuration is not the approved R4-V7 architecture")
    output_root = _resolve(root, project.get("output_root"), "project.output_root")
    if "r4_v6" in output_root.as_posix().lower():
        raise V7ConfigError("V7 may not write into the V6 artifact lineage")

    protocol = _mapping(config.get("protocol"), "protocol")
    if tuple(protocol.get("phase_order", ())) != PHASE_ORDER:
        raise V7ConfigError(f"protocol.phase_order must be {list(PHASE_ORDER)}")
    splits = _mapping(protocol.get("splits"), "protocol.splits")
    train = _validate_split(splits, "train")
    validation = _validate_split(splits, "validation")
    refit = _validate_split(splits, "trainval_refit")
    final_test = _validate_split(splits, "final_test")
    if int(refit["days"]) != int(train["days"]) + int(validation["days"]):
        raise V7ConfigError("trainval_refit days must equal train plus validation")
    if refit.get("allowed_only_after_method_selection") is not True:
        raise V7ConfigError("trainval_refit is allowed only after method selection")
    if (
        int(final_test.get("calendar_days", -1)) != 365
        or int(final_test.get("evaluable_days", -1)) != 362
        or int(final_test.get("missing_or_pending_days", -1)) != 3
        or final_test.get("missing_day_status") != "pending_v7_materialization"
        or final_test.get("sealed_until_all_methods_frozen") is not True
        or final_test.get("execute_once") is not True
    ):
        raise V7ConfigError(
            "2025 final_test must declare 365 calendar days, 362 evaluable, "
            "three pending V7 materializations, pre-freeze sealing, and one execution"
        )
    if int(final_test["evaluable_days"]) + int(final_test["missing_or_pending_days"]) != int(
        final_test["calendar_days"]
    ):
        raise V7ConfigError("2025 final_test evaluable and pending days must close the year")
    if "final_temporal_test" in splits:
        raise V7ConfigError("legacy 2024 final_temporal_test is forbidden in V7")
    locked = _mapping(protocol.get("locked"), "protocol.locked")
    if (
        int(locked.get("year", -1)) != 2025
        or locked.get("reject_path_before_open") is not True
        or locked.get("release_only_for_final_test_after_freeze") is not True
    ):
        raise V7ConfigError("2025 must remain sealed until the unique frozen final test")
    if "prohibit_2024_for" in protocol:
        raise V7ConfigError("2024 is the V7 validation year, not a prohibited year")
    prohibited = protocol.get("prohibit_2025_for")
    if (
        not isinstance(prohibited, (list, tuple))
        or set(str(value) for value in prohibited) != PROHIBIT_2025_FOR
    ):
        raise V7ConfigError(
            "protocol.prohibit_2025_for must contain the complete pre-final prohibition set"
        )

    training = _mapping(config.get("training"), "training")
    coverage_cycle = _mapping(
        training.get("coverage_cycle"), "training.coverage_cycle"
    )
    if (
        int(coverage_cycle.get("days", -1)) != 722
        or int(coverage_cycle.get("optimizer_updates", -1)) != 4
        or list(coverage_cycle.get("update_day_counts", ()))
        != [181, 181, 180, 180]
    ):
        raise V7ConfigError(
            "V7 722-day training coverage must be [181,181,180,180]"
        )
    gates = _mapping(config.get("capability_gates"), "capability_gates")
    distribution_gate = _mapping(
        gates.get("distribution"), "capability_gates.distribution"
    )
    policy_gate = _mapping(gates.get("policy"), "capability_gates.policy")
    if int(distribution_gate.get("validation_days", -1)) != 364:
        raise V7ConfigError("Distribution gate must use all 364 validation days")
    if (
        int(policy_gate.get("validation_days", -1)) != 364
        or int(policy_gate.get("rolling_origins", -1)) != 34_944
    ):
        raise V7ConfigError("Policy gate must use 364 days and 34,944 rolling origins")

    model = _mapping(config.get("model"), "model")
    policy = _mapping(model.get("policy"), "model.policy")
    if policy.get("free_actions") != ["day_ahead", "battery", "compute"]:
        raise V7ConfigError("V7 free actions must be day_ahead, battery, and compute only")
    serialized = repr(config).lower()
    forbidden = (
        "migration_fixed_zero",
        "previous_migration",
        "migration_head",
        "migration_actions_enabled: true",
    )
    if any(token in serialized for token in forbidden):
        raise V7ConfigError("migration implementation fields are forbidden in the V7 mainline")

    runtime = _mapping(config.get("runtime"), "runtime")
    if runtime.get("issue_day_ahead_before_realized_prefix") is not True:
        raise V7ConfigError("DA must be issued before any realized current-day prefix")
    if runtime.get("shared_torch_numpy_shield") is not True:
        raise V7ConfigError("training and deployment must share one shield contract")

    sampling = _mapping(config.get("sampling"), "sampling")
    if int(sampling.get("train", -1)) != 64:
        raise V7ConfigError("V7 joint training uses N=64")
    if int(sampling.get("formal_empirical_risk", -1)) != 512:
        raise V7ConfigError("formal post-shield risk requires N=512")

    lineage = _mapping(config.get("lineage"), "lineage")
    paths = {
        "output_root": output_root,
        "representation_checkpoint": _resolve(
            root, lineage.get("representation_selected_checkpoint"),
            "lineage.representation_selected_checkpoint",
        ),
        "residual_anchor": _resolve(
            root, lineage.get("frozen_residual_anchor"),
            "lineage.frozen_residual_anchor",
        ),
        "source_residual_anchor": _resolve(
            root,
            lineage.get("source_residual_anchor"),
            "lineage.source_residual_anchor",
        ),
        "warm_start_checkpoint": _resolve(
            root,
            lineage.get("warm_start_checkpoint"),
            "lineage.warm_start_checkpoint",
        ),
    }
    for label, path in paths.items():
        if label not in {"output_root", "residual_anchor"} and not path.is_file():
            raise V7ConfigError(f"required {label} is missing: {path}")
    if not paths["residual_anchor"].is_relative_to(output_root):
        raise V7ConfigError("new frozen_residual_anchor must belong to the new V7 output lineage")
    return paths


def load_v7_config(path: str | Path) -> tuple[dict[str, Any], Path, dict[str, Path]]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = dict(_mapping(raw, "configuration"))
    repository_root = config_path.parent.parent.resolve()
    resolved = validate_v7_config(config, repository_root)
    return config, repository_root, resolved


__all__ = [
    "ARCHITECTURE_VERSION",
    "PHASE_ORDER",
    "PROHIBIT_2025_FOR",
    "SPLIT_CONTRACT",
    "V7ConfigError",
    "load_v7_config",
    "validate_v7_config",
]
