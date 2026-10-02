from __future__ import annotations

"""CLI for the R4-V7 joint world-policy training lineage.

The command never falls back to the V6 phase runner.  A concrete runtime/data
adapter is injected explicitly for training, while ``prepare`` and ``plan``
remain read-only contract checks.
"""

import importlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

import typer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dcvpp.e2e_moe_r4_v7.config import (  # noqa: E402
    V7ConfigError,
    load_v7_config,
)
from dcvpp.e2e_moe_r4_v7.data import (  # noqa: E402
    V7DataError,
    load_frozen_residual_anchor,
)
from dcvpp.e2e_moe_r4_v7.trainer import (  # noqa: E402
    DistributionGateMetrics,
    JointUpdateScheduler,
    PHASE_ORDER,
    PolicyGateMetrics,
    V7TrainingError,
    evaluate_distribution_gate,
    evaluate_policy_gate,
    training_contract_from_config,
)


DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "e2e_moe_dart_vpp_R4_joint_world_policy_v7.yaml"
)
app = typer.Typer(no_args_is_help=True, add_completion=False)


def _emit(call: Callable[[], Any]) -> None:
    try:
        result = call()
    except (V7ConfigError, V7DataError, V7TrainingError, ValueError) as exc:
        result = {"status": "BLOCKED", "reason": str(exc)}
    if is_dataclass(result) and not isinstance(result, type):
        result = asdict(result)
    typer.echo(json.dumps(result, indent=2, sort_keys=True, default=str))


def prepare_payload(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, repository_root, resolved = load_v7_config(config_path)
    contract = training_contract_from_config(config)
    anchor = load_frozen_residual_anchor(resolved["residual_anchor"])
    return {
        "status": "READY_TO_IMPLEMENT_OR_TRAIN",
        "architecture_version": config["project"]["architecture_version"],
        "repository_root": str(repository_root),
        "output_root": str(resolved["output_root"]),
        "representation_checkpoint": str(resolved["representation_checkpoint"]),
        "residual_anchor": str(anchor.source_path),
        "residual_anchor_days": len(anchor.train_dates),
        "residual_anchor_shape": list(anchor.residuals.shape),
        "phase_order": list(PHASE_ORDER),
        "training_contract": contract.to_dict(),
        "training_started": False,
        "locked_2025_opened": False,
    }


def plan_payload(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, _, _ = load_v7_config(config_path)
    contract = training_contract_from_config(config)
    scheduler = JointUpdateScheduler(contract)
    phases: dict[str, Any] = {}
    for phase in contract.phases:
        first_cycle = scheduler.coverage_cycle(phase.phase_id, 0)
        pattern_preview = [
            scheduler.kind_at(phase.phase_id, index)
            for index in range(max(6, len(phase.update_pattern) * 2))
        ]
        phases[phase.phase_id] = {
            "minimum_cycles": phase.minimum_cycles,
            "maximum_cycles": phase.maximum_cycles,
            "maximum_optimizer_updates": phase.maximum_optimizer_updates,
            "distribution_to_decision": [
                phase.distribution_updates,
                phase.decision_updates,
            ],
            "first_coverage_cycle": [
                {
                    "days": part.stop - part.start,
                    "update_kind": kind,
                }
                for part, kind in first_cycle
            ],
            "update_pattern_preview": pattern_preview,
        }
    return {
        "status": "PLANNED",
        "phases": phases,
        "formal_risk_samples": int(config["sampling"]["formal_empirical_risk"]),
        "pareto_top_k": contract.pareto_top_k,
        "training_started": False,
    }


def _load_adapter(specification: str) -> Callable[[Mapping[str, Any]], Any]:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise V7TrainingError("adapter must use module:function syntax")
    callback = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(callback):
        raise V7TrainingError(f"training adapter is not callable: {specification}")
    return callback


@app.command()
def prepare(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
) -> None:
    """Read-only validation of lineage, frozen anchor and training contract."""

    _emit(lambda: prepare_payload(config))


@app.command()
def plan(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
) -> None:
    """Show phase ratios and the four-way 543-day optimizer partition."""

    _emit(lambda: plan_payload(config))


@app.command()
def phase(
    phase_name: str = typer.Argument(..., metavar="PHASE"),
    adapter: str = typer.Option(..., help="Concrete runner as module:function"),
    seed: int = typer.Option(20240813),
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
    resume: bool = typer.Option(True, "--resume/--fresh"),
) -> None:
    """Run one V7 phase through an explicit, non-V6 adapter."""

    def run() -> Any:
        phase_id = phase_name.strip().lower()
        if phase_id not in PHASE_ORDER:
            raise V7TrainingError(f"phase must be one of {list(PHASE_ORDER)}")
        loaded, repository_root, resolved = load_v7_config(config)
        contract = training_contract_from_config(loaded)
        invocation = {
            "schema": "dcvpp.r4.v7-training-invocation.v1",
            "phase": phase_id,
            "seed": int(seed),
            "resume": bool(resume),
            "config": loaded,
            "config_path": str(config.resolve()),
            "repository_root": str(repository_root),
            "resolved_paths": {key: str(value) for key, value in resolved.items()},
            "training_contract": contract.to_dict(),
        }
        return _load_adapter(adapter)(invocation)

    _emit(run)


@app.command("check-gate")
def check_gate(
    metrics: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    """Evaluate lightweight candidate metrics before a full paired gate."""

    def run() -> dict[str, Any]:
        payload = json.loads(metrics.read_text(encoding="utf-8"))
        distribution = evaluate_distribution_gate(
            DistributionGateMetrics(**payload["distribution"])
        )
        policy = evaluate_policy_gate(PolicyGateMetrics(**payload["policy"]))
        return {
            "status": "PASSED" if distribution.passed and policy.passed else "FAILED",
            "distribution": asdict(distribution),
            "policy": asdict(policy),
            "mae_used": False,
        }

    _emit(run)


if __name__ == "__main__":
    app()
