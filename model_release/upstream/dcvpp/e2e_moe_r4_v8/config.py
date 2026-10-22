"""Fail-closed configuration contract for the R4-V8 TriFusion lineage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


ARCHITECTURE_VERSION = "causal_trifusion_joint_moe_r4_v8"
BASE_METHOD_REVISION = "v8_base"
REPAIR_METHOD_REVISION = "v8_1_r0repair1"
PHASE_ORDER = ("r0", "d0", "p0", "j1", "j2", "j3", "c0")
PARAMETER_GROUPS = (
    "representation",
    "distribution_truth",
    "decision",
    "shared",
    "router",
    "frozen",
)
SPLITS = {
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


class V8ConfigError(ValueError):
    """A V8 configuration weakens a registered scientific boundary."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V8ConfigError(f"{label} must be a mapping")
    return value


def _resolve(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise V8ConfigError(f"{label} must be a non-empty path")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _split(splits: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = _mapping(splits.get(name), f"protocol.splits.{name}")
    start, end, days, role = SPLITS[name]
    if (
        str(value.get("start")) != start
        or str(value.get("end")) != end
        or int(value.get("days", -1)) != days
        or str(value.get("role")) != role
    ):
        raise V8ConfigError(
            f"{name} must be {start}..{end}, days={days}, role={role}"
        )
    return value


def validate_v8_config(
    config: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Path]:
    root = Path(repository_root).resolve()
    project = _mapping(config.get("project"), "project")
    if project.get("architecture_version") != ARCHITECTURE_VERSION:
        raise V8ConfigError("configuration is not the registered V8 architecture")
    method_revision = str(project.get("method_revision", BASE_METHOD_REVISION))
    if method_revision not in {BASE_METHOD_REVISION, REPAIR_METHOD_REVISION}:
        raise V8ConfigError("project.method_revision is not registered")
    output_root = _resolve(root, project.get("output_root"), "project.output_root")
    if "v8" not in output_root.name.lower():
        raise V8ConfigError("V8 needs an independent artifact lineage")

    protocol = _mapping(config.get("protocol"), "protocol")
    if tuple(protocol.get("phase_order", ())) != PHASE_ORDER:
        raise V8ConfigError(f"protocol.phase_order must be {list(PHASE_ORDER)}")
    splits = _mapping(protocol.get("splits"), "protocol.splits")
    train = _split(splits, "train")
    validation = _split(splits, "validation")
    refit = _split(splits, "trainval_refit")
    final = _split(splits, "final_test")
    if int(refit["days"]) != int(train["days"]) + int(validation["days"]):
        raise V8ConfigError("trainval_refit must close train plus validation")
    if refit.get("allowed_only_after_method_selection") is not True:
        raise V8ConfigError("trainval_refit requires a frozen method selection")
    if (
        int(final.get("calendar_days", -1)) != 365
        or int(final.get("missing_or_pending_days", -1)) != 3
        or final.get("sealed_until_all_methods_frozen") is not True
        or final.get("execute_once") is not True
    ):
        raise V8ConfigError("2025 must remain a sealed, one-shot 362-day final test")
    locked = _mapping(protocol.get("locked"), "protocol.locked")
    if (
        int(locked.get("year", -1)) != 2025
        or locked.get("reject_path_before_open") is not True
        or locked.get("release_only_for_final_test_after_freeze") is not True
    ):
        raise V8ConfigError("2025 must be inaccessible before final-test freeze")

    model = _mapping(config.get("model"), "model")
    representation_repair = model.get("representation_repair", {})
    if not isinstance(representation_repair, Mapping):
        raise V8ConfigError("model.representation_repair must be a mapping")
    repair_enabled = bool(representation_repair.get("enabled", False))
    if repair_enabled != (method_revision == REPAIR_METHOD_REVISION):
        raise V8ConfigError(
            "R0 repair must use project.method_revision=v8_1_r0repair1"
        )
    if repair_enabled:
        if "repair" not in output_root.name.lower():
            raise V8ConfigError("R0 repair requires an independent repair output root")
        if (
            representation_repair.get("baseline") != "causal_persistence"
            or representation_repair.get("distribution_anchor_unchanged") is not True
            or representation_repair.get("detach_backbone") is not True
            or representation_repair.get("exact_price_identity") is not True
            or tuple(representation_repair.get("protected_outputs", ()))
            != ("pv", "pue")
        ):
            raise V8ConfigError("R0 repair causal/protection contract was weakened")
        cap_quantile = float(
            representation_repair.get("residual_cap_quantile", -1.0)
        )
        if abs(cap_quantile - 0.95) > 1.0e-12:
            raise V8ConfigError("R0 repair caps must be train-only q95")
        zero_threshold = float(
            representation_repair.get("workload_zero_threshold", -1.0)
        )
        if not 0.5 <= zero_threshold <= 0.99:
            raise V8ConfigError("workload zero threshold must lie in [0.5,0.99]")
    groups = model.get("parameter_groups")
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        raise V8ConfigError("model.parameter_groups must be a sequence")
    if tuple(str(value) for value in groups) != PARAMETER_GROUPS:
        raise V8ConfigError(f"parameter groups must be {list(PARAMETER_GROUPS)}")
    gradient = _mapping(model.get("gradient_contract"), "model.gradient_contract")
    if (
        gradient.get("policy_to_distribution_truth") != "forbidden"
        or gradient.get("policy_to_shared") != "closed_form_projected"
        or gradient.get("physical_core") != "frozen"
    ):
        raise V8ConfigError("V8 gradient firewall was weakened")

    training = _mapping(config.get("training"), "training")
    microbatch = _mapping(
        training.get("microbatch_days"), "training.microbatch_days"
    )
    if {str(key): int(value) for key, value in microbatch.items()} != {
        "representation": 181,
        "distribution": 4,
        "policy": 1,
        "joint": 1,
    }:
        raise V8ConfigError(
            "V8 10GB profile requires microbatch days R/D/P/J=181/4/1/1"
        )
    if int(training.get("policy_reconditioned_origins_per_rollout", -1)) != 8:
        raise V8ConfigError("V8 training must recondition Policy at eight origins")
    coverage = _mapping(training.get("coverage_cycle"), "training.coverage_cycle")
    if (
        int(coverage.get("days", -1)) != 722
        or int(coverage.get("optimizer_updates", -1)) != 4
        or list(coverage.get("update_day_counts", ())) != [181, 181, 180, 180]
    ):
        raise V8ConfigError("V8 coverage must be 722 -> [181,181,180,180]")
    precision = _mapping(training.get("precision"), "training.precision")
    if (
        precision.get("neural_forward") != "bf16"
        or precision.get("scientific_scores") != "float32"
        or precision.get("shield") != "float32"
    ):
        raise V8ConfigError("BF16 may be used only for neural forward layers")
    phases = _mapping(training.get("phases"), "training.phases")
    if tuple(phases) != PHASE_ORDER:
        raise V8ConfigError(f"training phases must be ordered as {list(PHASE_ORDER)}")
    for phase in PHASE_ORDER:
        entry = _mapping(phases[phase], f"training.phases.{phase}")
        pattern = entry.get("primary_pattern")
        if (
            not isinstance(pattern, Sequence)
            or isinstance(pattern, (str, bytes))
            or not pattern
            or any(value not in {"representation", "distribution", "policy"} for value in pattern)
        ):
            raise V8ConfigError(f"{phase} has an invalid primary task pattern")
    if repair_enabled:
        repair_r0 = _mapping(phases["r0"], "training.phases.r0")
        prefixes = tuple(
            str(value) for value in repair_r0.get("trainable_subscope", ())
        )
        if (
            repair_r0.get("shared_updates") is not False
            or repair_r0.get("router_updates") is not False
            or prefixes != ("representation.repair.",)
        ):
            raise V8ConfigError(
                "R0 repair may train only representation.repair with shared/router frozen"
            )
        repair_loss = _mapping(
            _mapping(config.get("losses"), "losses").get("representation"),
            "losses.representation",
        )
        if repair_loss.get("objective") != "paired_daily_nmae_repair":
            raise V8ConfigError("R0 repair requires the paired daily-NMAE objective")

    gates = _mapping(config.get("capability_gates"), "capability_gates")
    if int(_mapping(gates.get("unified"), "capability_gates.unified").get("validation_days", -1)) != 364:
        raise V8ConfigError("unified gate must use all 364 validation days")
    if _mapping(gates.get("policy"), "capability_gates.policy").get("formal_paths") != 512:
        raise V8ConfigError("Policy formal risk must use N=512 common paths")

    lineage = _mapping(config.get("lineage"), "lineage")
    paths = {
        "output_root": output_root,
        "representation_checkpoint": _resolve(
            root,
            lineage.get("representation_checkpoint"),
            "lineage.representation_checkpoint",
        ),
        "source_residual_anchor": _resolve(
            root,
            lineage.get("source_residual_anchor"),
            "lineage.source_residual_anchor",
        ),
        "residual_anchor": _resolve(
            root,
            lineage.get("frozen_residual_anchor"),
            "lineage.frozen_residual_anchor",
        ),
    }
    if repair_enabled:
        paths["repair_parent_checkpoint"] = _resolve(
            root,
            lineage.get("repair_parent_checkpoint"),
            "lineage.repair_parent_checkpoint",
        )
        paths["repair_parent_gate"] = _resolve(
            root,
            lineage.get("repair_parent_gate"),
            "lineage.repair_parent_gate",
        )
    for name in ("representation_checkpoint", "source_residual_anchor"):
        if not paths[name].is_file():
            raise V8ConfigError(f"required {name} is missing: {paths[name]}")
    for name in ("repair_parent_checkpoint", "repair_parent_gate"):
        if name in paths and not paths[name].is_file():
            raise V8ConfigError(f"required {name} is missing: {paths[name]}")
    if not paths["residual_anchor"].is_relative_to(output_root):
        raise V8ConfigError("new V8 residual anchor must belong to its output lineage")

    data = _mapping(config.get("data"), "data")
    if data.get("verify_daily_hashes") is not False:
        raise V8ConfigError(
            "V8 reuses registered daily provenance without recomputing every file hash"
        )
    required_data = (
        "selection_bundle_root",
        "official_settlement_root",
        "frozen_physical_twin",
        "frozen_physical_surrogate",
        "physical_calibration_manifest",
    )
    for name in required_data:
        _resolve(root, data.get(name), f"data.{name}")
    return paths


def load_v8_config(path: str | Path) -> tuple[dict[str, Any], Path, dict[str, Path]]:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    config = dict(_mapping(raw, "configuration"))
    root = source.parent.parent.resolve()
    return config, root, validate_v8_config(config, root)


__all__ = [
    "ARCHITECTURE_VERSION",
    "BASE_METHOD_REVISION",
    "PARAMETER_GROUPS",
    "PHASE_ORDER",
    "REPAIR_METHOD_REVISION",
    "SPLITS",
    "V8ConfigError",
    "load_v8_config",
    "validate_v8_config",
]
