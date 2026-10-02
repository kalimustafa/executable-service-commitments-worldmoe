from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from dcvpp.core import environment_snapshot
from dcvpp.data.referenced import build_referenced_data_stage
from dcvpp.pipeline import run_pipeline


app = typer.Typer(no_args_is_help=True, help="Power DC VPP research pipeline")
pipeline_app = typer.Typer(no_args_is_help=True)
referenced_data_app = typer.Typer(no_args_is_help=True)
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(referenced_data_app, name="referenced-data")


@app.command()
def doctor(profile: str = "demo") -> None:
    root = Path.cwd()
    payload = environment_snapshot(root)
    payload["profile"] = profile
    try:
        import highspy  # noqa: F401

        payload["highs"] = "available"
    except ImportError:
        payload["highs"] = "missing"
    typer.echo(json.dumps(payload, indent=2))
    paper_cuda_missing = profile == "paper" and not bool(
        (payload.get("torch") or {}).get("cuda_available")
    )
    if sys.version_info[:2] != (3, 11) or payload["highs"] != "available" or paper_cuda_missing:
        raise typer.Exit(code=1)


@pipeline_app.command("run")
def pipeline_run(
    target: str = typer.Option("paper", "--target"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    config: Path = typer.Option(Path("configs/paper.yaml"), "--config"),
) -> None:
    root = Path.cwd()
    result = run_pipeline(root, root / config, target, resume)
    typer.echo(json.dumps(result, indent=2, default=str))


@referenced_data_app.command("build")
def referenced_data_build(
    sources: str = typer.Option("nasa_power,nyiso,nlr295", "--sources"),
    nyiso_start: str = typer.Option("2021-01-01", "--nyiso-start"),
    nyiso_end: str = typer.Option("2025-12-31", "--nyiso-end"),
    nyiso_max_months: int = typer.Option(
        0,
        "--nyiso-max-months",
        min=0,
        help="Bound NYISO work by month; 0 processes every month in the range.",
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
) -> None:
    """Build referenced third-party data panels with atomic checkpoints."""
    selected = tuple(item.strip() for item in sources.split(",") if item.strip())
    result = build_referenced_data_stage(
        Path.cwd(),
        sources=selected,
        nyiso_start=nyiso_start,
        nyiso_end=nyiso_end,
        nyiso_max_months=nyiso_max_months or None,
        resume=resume,
    )
    typer.echo(json.dumps(result.get("counts", {}), indent=2))


if __name__ == "__main__":
    app()
