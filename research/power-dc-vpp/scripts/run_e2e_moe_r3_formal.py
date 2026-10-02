from __future__ import annotations

"""Command-line entry point for the frozen solver-free R3 formal program."""

import json
from pathlib import Path
import sys
from typing import Any, Callable

import typer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dcvpp.e2e_moe_r3.formal_protocol import (  # noqa: E402
    FormalProtocolError,
    prepare_protocol,
    run_full_validation,
    run_initial_validation_day,
    run_mainline_seed1,
    run_preflight,
    run_seed1_lineage,
    write_seed1_report,
)


app = typer.Typer(no_args_is_help=True, add_completion=False)
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "e2e_moe_dart_vpp_R3_solver_free_formal.yaml"
)


def _emit(call: Callable[[], dict[str, Any]]) -> None:
    try:
        result = call()
    except FormalProtocolError as exc:
        result = {"status": "BLOCKED", "reason": str(exc)}
    typer.echo(json.dumps(result, indent=2, sort_keys=True, default=str))


@app.command()
def prepare(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
    calibrate: bool = typer.Option(
        True,
        "--calibrate/--metadata-only",
        help="Freeze train price thresholds and validation-selected battery rules.",
    ),
) -> None:
    """Create manifests and freeze the primary benchmark before preflight."""

    _emit(lambda: prepare_protocol(config, calibrate=calibrate))


@app.command()
def preflight(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
    resume: bool = typer.Option(True, "--resume/--fresh"),
) -> None:
    """Run the single formal preflight, including one audited 96-origin day."""

    _emit(lambda: run_preflight(config, resume=resume))


@app.command("run-seed1")
def run_seed1(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
    resume: bool = typer.Option(True, "--resume/--fresh"),
) -> None:
    """Run representation -> scenario -> policy -> A1 -> A2 for Seed1."""

    _emit(lambda: run_seed1_lineage(config, resume=resume))


@app.command("validation-day")
def validation_day(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
    resume: bool = typer.Option(True, "--resume/--fresh"),
) -> None:
    """Run the earliest eligible 2023-H2 day for all 96 origins."""

    _emit(lambda: run_initial_validation_day(config, resume=resume))


@app.command("evaluate-validation")
def evaluate_validation(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
    resume: bool = typer.Option(True, "--resume/--fresh"),
) -> None:
    """Run paired learned/baseline evaluation on every 2023-H2 date."""

    _emit(lambda: run_full_validation(config, resume=resume))


@app.command()
def report(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
) -> None:
    """Write exact reports only when complete measured evidence exists."""

    _emit(lambda: write_seed1_report(config))


@app.command("mainline-seed1")
def mainline_seed1(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
    resume: bool = typer.Option(True, "--resume/--fresh"),
) -> None:
    """Run the full frozen Seed1 program through validation and reporting."""

    _emit(lambda: run_mainline_seed1(config, resume=resume))


if __name__ == "__main__":
    app()
