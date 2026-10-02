from __future__ import annotations

"""Command-line entry point for the solver-free R4 formal program."""

import json
import importlib
from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys
from typing import Any, Callable

import typer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dcvpp.e2e_moe_r4.formal_protocol import (  # noqa: E402
    R4ProtocolError,
    prepare_protocol,
    protocol_status,
    run_phase,
    run_preflight,
    run_seed1,
    write_seed1_report,
)
from dcvpp.e2e_moe_r4.phase_runner import register_default_phase_runner  # noqa: E402
from dcvpp.e2e_moe_r4.final_evaluation import (  # noqa: E402
    FinalProtocolError,
    FrozenMethod,
    authorize_later_seed,
    claim_and_evaluate_final_once,
    create_final_freeze_manifest,
    verify_final_freeze_manifest,
)
from dcvpp.e2e_moe_r4.final_outputs import write_final_three_seed_outputs  # noqa: E402


app = typer.Typer(no_args_is_help=True, add_completion=False)
LEGACY_V5_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "e2e_moe_dart_vpp_R4_hybrid_solver_free_formal.yaml"
)
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "e2e_moe_dart_vpp_R4_distributional_joint_formal.yaml"
)


def _emit(call: Callable[[], Any]) -> None:
    try:
        result = call()
    except (R4ProtocolError, FinalProtocolError) as exc:
        result = {"status": "BLOCKED", "reason": str(exc)}
    if is_dataclass(result) and not isinstance(result, type):
        result = asdict(result)
    typer.echo(json.dumps(result, indent=2, sort_keys=True, default=str))


@app.command()
def prepare(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
) -> None:
    """Validate and show the immutable R4/V6 protocol contract."""

    _emit(lambda: prepare_protocol(config))


@app.command()
def preflight(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
    resume: bool = typer.Option(True, "--resume/--fresh"),
) -> None:
    """Freeze V6 static/train-only contracts; never open 2024/2025."""

    _emit(lambda: run_preflight(config, resume=resume))


@app.command()
def phase(
    phase_name: str = typer.Argument(..., metavar="PHASE"),
    seed: int = typer.Option(20240813),
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
    resume: bool = typer.Option(True, "--resume/--fresh"),
) -> None:
    """Run one registered phase and its mandatory capability gate."""

    register_default_phase_runner()
    _emit(
        lambda: run_phase(
            config,
            phase=phase_name,
            seed=seed,
            resume=resume,
        )
    )


@app.command("run-seed1")
def run_seed_1(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
    resume: bool = typer.Option(True, "--resume/--fresh"),
) -> None:
    """Run Seed 20240813 in the config's frozen phase order, stopping on FAIL."""

    register_default_phase_runner()
    _emit(lambda: run_seed1(config, resume=resume))


@app.command("seed1-report")
def seed1_report(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
) -> None:
    """Write the Seed-1 report only from complete measured evidence."""

    _emit(lambda: write_seed1_report(config))


@app.command()
def status(
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
) -> None:
    """Show the truthful current R4 protocol and lineage status."""

    _emit(lambda: protocol_status(config))


@app.command("authorize-seed")
def authorize_seed(
    seed: int = typer.Argument(..., metavar="SEED"),
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
) -> None:
    """Authorize Seed 2/3 only after measured Seed-1 R4_MAINLINE_PASS."""

    context = prepare_protocol(config)
    _emit(lambda: authorize_later_seed(context.output_root, seed))


@app.command("freeze-final")
def freeze_final(
    request: Path = typer.Option(..., exists=True, dir_okay=False),
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False),
) -> None:
    """Freeze methods and statistics from a metadata-only JSON request."""

    def run() -> dict[str, Any]:
        payload = json.loads(request.read_text(encoding="utf-8"))
        context = prepare_protocol(config)
        methods = [
            FrozenMethod(
                method_id=str(item["method_id"]),
                kind=str(item["kind"]),
                evidence_by_seed={int(seed): path for seed, path in item["evidence_by_seed"].items()},
                runtime_adapter=item.get("runtime_adapter"),
            )
            for item in payload["methods"]
        ]
        path = create_final_freeze_manifest(
            config_path=config,
            output_root=context.output_root,
            artifact_groups=payload["artifact_groups"],
            methods=methods,
            eligible_dates=payload["eligible_dates"],
            primary_method_id=payload["primary_method_id"],
            primary_benchmark_id=payload["primary_benchmark_id"],
        )
        manifest = verify_final_freeze_manifest(path)
        return {
            "status": "FROZEN",
            "freeze_manifest": str(path),
            "freeze_sha256": manifest["freeze_sha256"],
            "eligible_dates": len(manifest["eligible_dates"]),
            "formal_selection_2024_opened": False,
            "locked_test_2025_opened": False,
        }

    _emit(run)


def _load_evaluator_adapter(specification: str) -> Callable[[dict[str, Any]], Any]:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise FinalProtocolError("adapter must use module:function syntax")
    callback = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(callback):
        raise FinalProtocolError(f"final evaluator adapter is not callable: {specification}")
    return callback


@app.command("evaluate-final")
def evaluate_final(
    freeze_manifest: Path = typer.Option(..., exists=True, dir_okay=False),
    adapter: str = typer.Option(
        "dcvpp.e2e_moe_r4.formal_batch_evaluator:formal_batch_evaluator",
        help="Claim-bound batch evaluator as module:function",
    ),
) -> None:
    """Invoke the registered formal 2024 batch evaluator exactly once."""

    _emit(
        lambda: claim_and_evaluate_final_once(
            freeze_manifest_path=freeze_manifest,
            evaluator=_load_evaluator_adapter(adapter),
        )
    )


@app.command("finalize-final")
def finalize_final(
    freeze_manifest: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    """Generate statistics and exact final outputs from captured daily rows."""

    _emit(lambda: write_final_three_seed_outputs(freeze_manifest))


@app.command("final-status")
def final_status(
    freeze_manifest: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    """Verify the frozen manifest without opening any temporal-test payload."""

    _emit(lambda: verify_final_freeze_manifest(freeze_manifest))


if __name__ == "__main__":
    app()
