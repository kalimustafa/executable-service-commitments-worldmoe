"""CLI for V8 prepare, structural smoke, and atomic phase training."""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys
from typing import Any, Callable

import typer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dcvpp.e2e_moe_r4_v8.config import (  # noqa: E402
    PHASE_ORDER,
    V8ConfigError,
)
from dcvpp.e2e_moe_r4_v8.corpus import V8CorpusError  # noqa: E402
from dcvpp.e2e_moe_r4_v8.preflight import (  # noqa: E402
    V8PreflightError,
    prepare_v8,
)
from dcvpp.e2e_moe_r4_v8.runner import (  # noqa: E402
    V8RunnerError,
    run_v8_smoke,
    run_v8_training,
)
from dcvpp.e2e_moe_r4_v8.trainer import V8TrainingError  # noqa: E402


DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "e2e_moe_dart_vpp_R4_trifusion_v8.yaml"
)
app = typer.Typer(no_args_is_help=True, add_completion=False)


def _emit(callback: Callable[[], Any]) -> None:
    try:
        result = callback()
    except (
        V8ConfigError,
        V8CorpusError,
        V8PreflightError,
        V8RunnerError,
        V8TrainingError,
        ValueError,
    ) as exc:
        result = {
            "status": "BLOCKED",
            "reason": str(exc),
            "training_started": False,
            "locked_2025_opened": False,
        }
    if is_dataclass(result) and not isinstance(result, type):
        result = asdict(result)
    typer.echo(json.dumps(result, indent=2, sort_keys=True, default=str))


def _load_callback(specification: str | None) -> Callable[..., Any] | None:
    if specification is None:
        return None
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise V8RunnerError("adapter must use module:function syntax")
    callback = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(callback):
        raise V8RunnerError(f"adapter is not callable: {specification}")
    return callback


def prepare_payload(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    materialize_anchor: bool = True,
) -> dict[str, Any]:
    result = prepare_v8(
        config_path,
        materialize_anchor=materialize_anchor,
        write_summaries=True,
    )
    return {
        "status": "READY",
        "architecture_version": result.config["project"]["architecture_version"],
        "output_root": str(result.resolved_paths["output_root"]),
        "residual_anchor": str(result.residual_anchor.source_path),
        "residual_anchor_shape": list(result.residual_anchor.residuals.shape),
        "anchor_materialized": result.anchor_materialized,
        "phase_order": list(PHASE_ORDER),
        "default_training_phase": "r0",
        "split_summary": str(result.summary_paths["split_summary"]),
        "lineage_summary": str(result.summary_paths["lineage_summary"]),
        "training_started": False,
        "final_test_data_accessed": False,
        "locked_2025_opened": False,
    }


@app.command()
def prepare(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
    materialize_anchor: bool = typer.Option(
        True, "--materialize-anchor/--load-anchor-only"
    ),
) -> None:
    """Validate V8 and prepare its 2022--2023 residual anchor."""

    _emit(lambda: prepare_payload(config, materialize_anchor=materialize_anchor))


@app.command()
def smoke(
    rollout_adapter: str = typer.Option(
        ..., help="Differentiable rollout as module:function"
    ),
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
    seed: int | None = typer.Option(None),
    batch_size: int = typer.Option(2, min=1),
    device: str | None = typer.Option(
        None, help="auto (default), cuda, cuda:0, or cpu"
    ),
) -> None:
    """Run one small R0, D0 and P0 optimizer update; no formal checkpoint."""

    _emit(
        lambda: run_v8_smoke(
            config,
            rollout_fn=_load_callback(rollout_adapter),  # type: ignore[arg-type]
            seed=seed,
            batch_size=batch_size,
            device=device,
        )
    )


@app.command()
def train(
    phase: str = typer.Option("r0", help=f"One of {list(PHASE_ORDER)}"),
    rollout_adapter: str | None = typer.Option(
        None, help="Required for phases containing Policy"
    ),
    validation_adapter: str | None = typer.Option(
        None, help="Optional cheap validation hook as module:function"
    ),
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
    seed: int | None = typer.Option(None),
    resume: bool = typer.Option(True, "--resume/--fresh"),
    max_cycles: int | None = typer.Option(None, min=1),
    device: str | None = typer.Option(None, help="cuda, cuda:0, or cpu"),
) -> None:
    """Train one atomic V8 phase; formal execution defaults to R0."""

    _emit(
        lambda: run_v8_training(
            config,
            phase=phase,
            seed=seed,
            resume=resume,
            rollout_fn=_load_callback(rollout_adapter),
            validation_fn=_load_callback(validation_adapter),
            max_cycles=max_cycles,
            device=device,
        )
    )


if __name__ == "__main__":
    app()
