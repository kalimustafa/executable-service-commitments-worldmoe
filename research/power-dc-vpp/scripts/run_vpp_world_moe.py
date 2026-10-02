"""Run the independent VPP-WorldMoE smoke, training, or C1 gate."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Windows can access-violate when Arrow's dataset DLLs are loaded lazily after
# PyTorch/CUDA has changed the process DLL search state.  Preload Arrow before
# importing torch in the formal entrypoint, matching the verified probe path.
import pyarrow.dataset  # noqa: F401
import numpy as np
import torch

from dcvpp.vpp_world_moe.config import WorldMoEConfig, load_world_moe_config
from dcvpp.vpp_world_moe.runner import WorldMoERunner


def _construct(factory: Any, *args: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(factory)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    selected = (
        kwargs
        if accepts_kwargs
        else {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }
    )
    return factory(*args, **selected)


def _construct_world_corpus(
    factory: Any,
    config: Any,
    repository_root: Path,
    *,
    smoke: bool,
    **kwargs: Any,
) -> Any:
    """Construct the corpus without allowing the formal weather gate to vanish.

    ``_construct`` intentionally tolerates optional compatibility keywords.  The
    weather coverage gate is different: silently dropping it would allow a
    formal phase to reach W0 with incomplete issue-time weather.  Require that
    the selected corpus factory can receive this keyword, then enable it for
    every non-smoke invocation.
    """

    signature = inspect.signature(factory)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if "weather_coverage_gate" not in signature.parameters and not accepts_kwargs:
        raise RuntimeError(
            "WorldMoECorpus must expose weather_coverage_gate; refusing to run "
            "without the formal pre-W0 weather boundary"
        )
    return _construct(
        factory,
        config,
        repository_root,
        weather_coverage_gate=not smoke,
        **kwargs,
    )


def _default_run_id(seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"worldmoe-{timestamp}-seed{seed}"


def _configure_torch_runtime(config: Any, *, device: str) -> dict[str, Any]:
    """Apply the measured CUDA matmul policy before the model is constructed."""

    training = config.get("training", {}) if isinstance(config, dict) else {}
    runtime = training.get("cuda_runtime", {}) if isinstance(training, dict) else {}
    if not isinstance(runtime, dict):
        raise SystemExit("training.cuda_runtime must be a mapping")
    precision = str(runtime.get("float32_matmul_precision", "highest"))
    if precision not in {"highest", "high", "medium"}:
        raise SystemExit(
            "training.cuda_runtime.float32_matmul_precision must be "
            "highest, high, or medium"
        )
    enabled = str(device).lower().startswith("cuda")
    allow_tf32 = bool(runtime.get("allow_tf32", False)) if enabled else False
    if enabled:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.set_float32_matmul_precision(precision)
    return {
        "device": str(device),
        "allow_tf32": allow_tf32,
        "float32_matmul_precision": precision,
    }


def _seed_before_construction(seed: int, *, device: str) -> None:
    """Seed every constructor-visible RNG before building data or modules.

    The runner also restores RNG state for exact resume.  That is necessarily
    too late for a fresh run, however, because model parameters and corpus
    sampling helpers have already been constructed by then.  Keep this call in
    the entrypoint immediately before those constructors.
    """

    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if str(device).lower().startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_TRAINABLE_PROJECT_VERSIONS = (
    "R5-V2",
    "R5-V2.1",
    "R5-V2.2",
    "R5-V2.3",
    "R5-V2.4",
    "R5-V2.5",
)


def _assert_supported_training_lineage(config: Any, *, resume: bool) -> None:
    """Allow registered R5-V2.x fresh/resume runs and freeze legacy R5-V1."""

    project = config.get("project", {}) if isinstance(config, dict) else {}
    version = str(project.get("version", "")) if isinstance(project, dict) else ""
    if version not in _TRAINABLE_PROJECT_VERSIONS:
        mode = "resume" if resume else "fresh training"
        supported = "/".join(_TRAINABLE_PROJECT_VERSIONS)
        raise SystemExit(
            f"{mode} requires {supported}; R5-V1 is a frozen legacy baseline"
        )


def _guard_run_namespace(
    *,
    output_root: str | Path,
    run_id: str,
    phase: str,
    resume: bool,
    smoke: bool,
    evaluation_resume: bool = False,
) -> Path:
    """Reject ambiguous resumes and accidental same-phase artifact overwrites."""

    if _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise SystemExit("--run-id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    if resume and smoke:
        raise SystemExit("--resume cannot be combined with --smoke")
    if resume and phase == "c1":
        raise SystemExit("C1 is an evaluation gate and cannot be resumed")
    if evaluation_resume and phase != "c1":
        raise SystemExit("evaluation_resume is reserved for C1")

    run_root = Path(output_root).resolve() / f"run={run_id}"
    phase_root = run_root / f"phase={phase}"
    if evaluation_resume:
        if not (phase_root / "schedule.json").is_file():
            raise SystemExit("C1 resume requires an existing frozen schedule")
        return run_root
    if resume:
        latest = phase_root / "checkpoints" / "latest.pt"
        if not latest.is_file():
            raise SystemExit(
                f"--resume requires the exact existing checkpoint: {latest}"
            )
        return run_root

    # A smoke run and a fresh W0 both define a new lineage.  They may not share
    # a populated run root.  Later phases reuse the lineage but may not replace
    # artifacts already written for that same phase.
    collision_root = run_root if smoke or phase == "w0" else phase_root
    if collision_root.exists() and any(collision_root.iterdir()):
        raise SystemExit(
            f"refusing to overwrite existing run artifacts: {collision_root}; "
            "choose a new --run-id or use --resume for the same trainable phase"
        )
    return run_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--phase", choices=("w0", "w1", "u0", "p1", "j1", "c1"), default="w0"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=20240813)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--parent-checkpoint",
        type=Path,
        help=(
            "explicit previous-phase checkpoint for a fresh cross-run W1/U0/P1/J1; "
            "only model weights are loaded"
        ),
    )
    parser.add_argument(
        "--parent-gate",
        type=Path,
        help="optional release gate for --parent-checkpoint (otherwise derived beside it)",
    )
    parser.add_argument(
        "--then-u0",
        action="store_true",
        help="after W1, start U0 only if the registered W1 capability gate passes",
    )
    parser.add_argument(
        "--enforce-phase-gate",
        action="store_true",
        help="keep previous-phase release gate checks",
    )
    parser.add_argument(
        "--c1-evidence",
        type=Path,
        help="JSON daily evidence produced by the registered closed-loop evaluator",
    )
    parser.add_argument(
        "--c1-evaluator",
        help="daily evaluator callback as importable.module:function",
    )
    parser.add_argument(
        "--c1-dates",
        type=Path,
        help="JSON list or {dates:[...]} containing the frozen 364-day schedule",
    )
    parser.add_argument("--c1-resume-evaluation", action="store_true")
    parser.add_argument("--c1-day-batch-size", type=int, default=1)
    parser.add_argument("--c1-max-workers", type=int, default=1)
    parser.add_argument(
        "--c1-parallel-safe",
        action="store_true",
        help="declare that the daily evaluator supports independent concurrent days",
    )
    return parser


def _load_c1_evaluator(specification: str) -> Any:
    module_name, separator, attribute = str(specification).rpartition(":")
    if not separator or not module_name or not attribute:
        raise SystemExit("--c1-evaluator must be importable.module:function")
    callback = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(callback):
        raise SystemExit("--c1-evaluator did not resolve to a callable")
    return callback


def _load_c1_dates(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("dates") if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise SystemExit("--c1-dates must contain a JSON list or {dates:[...]}")
    return [str(value)[:10] for value in values]


def _run_trainable_phase(
    runner: Any,
    phase: str,
    *,
    resume: bool,
    ignore_phase_gate: bool = False,
    ignore_formal_gates: bool = False,
    parent_checkpoint: str | Path | None = None,
    parent_gate: str | Path | None = None,
) -> Any:
    """CLI trainable phases default to no formal gate thresholding."""

    gate_callback = (
        runner.registered_phase_gate_callback
        if not ignore_formal_gates and not ignore_phase_gate
        else None
    )
    return runner.run_phase(
        phase,
        resume=resume,
        parent_checkpoint=parent_checkpoint,
        parent_gate=parent_gate,
        gate_callback=gate_callback,
        ignore_phase_gate=ignore_phase_gate,
        ignore_formal_gates=ignore_formal_gates,
    )


def _run_w1_then_u0(
    runner: Any,
    *,
    output_root: str | Path,
    run_id: str,
    resume: bool,
    parent_checkpoint: str | Path | None,
    parent_gate: str | Path | None,
) -> dict[str, Any]:
    """Run the explicit W1 gate boundary, then and only then start U0."""

    w1_result = _run_trainable_phase(
        runner,
        "w1",
        resume=resume,
        ignore_phase_gate=False,
        ignore_formal_gates=False,
        parent_checkpoint=parent_checkpoint,
        parent_gate=parent_gate,
    )
    if w1_result.next_phase_released is not True:
        raise SystemExit(
            "W1 capability gate failed; U0 was not started and no U0 "
            "artifacts were created"
        )
    _guard_run_namespace(
        output_root=output_root,
        run_id=run_id,
        phase="u0",
        resume=False,
        smoke=False,
    )
    u0_result = _run_trainable_phase(
        runner,
        "u0",
        resume=False,
        ignore_phase_gate=False,
        ignore_formal_gates=False,
    )
    return {"w1": w1_result.__dict__, "u0": u0_result.__dict__}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.c1_evidence is not None and args.c1_evaluator is not None:
        raise SystemExit("choose either --c1-evidence or --c1-evaluator")
    if args.c1_evaluator is not None and args.c1_dates is None:
        raise SystemExit("--c1-evaluator requires --c1-dates")
    if args.c1_resume_evaluation and args.phase != "c1":
        raise SystemExit("--c1-resume-evaluation requires --phase c1")
    if args.parent_gate is not None and args.parent_checkpoint is None:
        raise SystemExit("--parent-gate requires --parent-checkpoint")
    if args.parent_checkpoint is not None and (
        args.resume or args.smoke or args.phase in {"w0", "c1"}
    ):
        raise SystemExit(
            "--parent-checkpoint is only valid for a fresh W1/U0/P1/J1 phase"
        )
    if args.then_u0 and (args.phase != "w1" or args.smoke):
        raise SystemExit("--then-u0 requires a non-smoke --phase w1 invocation")
    config, repository_root, paths = load_world_moe_config(args.config)
    _assert_supported_training_lineage(config, resume=args.resume)
    _configure_torch_runtime(config, device=args.device)
    if args.resume and not str(args.run_id or "").strip():
        raise SystemExit("--resume requires an explicit --run-id")
    configured_training = config.get("training", {})
    configured_run_id = (
        configured_training.get("run_id", "")
        if isinstance(configured_training, dict)
        else ""
    )
    run_id = args.run_id or str(configured_run_id).strip()
    run_id = run_id or str(config.get("project", {}).get("run_id", "")).strip()
    run_id = run_id or _default_run_id(args.seed)
    _guard_run_namespace(
        output_root=paths["output_root"],
        run_id=run_id,
        phase=args.phase,
        resume=args.resume,
        smoke=args.smoke,
        evaluation_resume=args.c1_resume_evaluation,
    )
    from dcvpp.vpp_world_moe.data import WorldMoECorpus
    from dcvpp.vpp_world_moe.model import VPPWorldMoE

    _seed_before_construction(args.seed, device=args.device)
    model_config = WorldMoEConfig.from_mapping(config["model"])
    model = VPPWorldMoE(model_config)
    corpus = _construct_world_corpus(
        WorldMoECorpus,
        config,
        repository_root,
        smoke=args.smoke,
        paths=paths,
        model_config=model_config,
    )
    runner = WorldMoERunner(
        config=config,
        model=model,
        corpus=corpus,
        output_root=paths["output_root"],
        device=args.device,
        run_id=run_id,
        seed=args.seed,
    )
    if args.smoke:
        result = runner.smoke()
    elif args.phase == "c1":
        if args.c1_evaluator is not None:
            callback = _load_c1_evaluator(args.c1_evaluator)
            result = runner.run_c1(
                callback,
                ignore_phase_gate=False,
                dates=_load_c1_dates(args.c1_dates),
                resume_evaluation=args.c1_resume_evaluation,
                day_batch_size=args.c1_day_batch_size,
                max_workers=args.c1_max_workers,
                parallel_safe=args.c1_parallel_safe,
            ).as_dict()
        else:
            if args.c1_evidence is None:
                raise SystemExit(
                    "C1 requires --c1-evidence or --c1-evaluator with --c1-dates"
                )
            payload = json.loads(args.c1_evidence.read_text(encoding="utf-8"))
            records = (
                payload.get("records", payload)
                if isinstance(payload, dict)
                else payload
            )
            if not isinstance(records, list):
                raise SystemExit("C1 evidence must be a JSON list or {records:[...]}")
            result = runner.run_c1(
                lambda **_: records,
                ignore_phase_gate=False,
                resume_evaluation=args.c1_resume_evaluation,
            ).as_dict()
    else:
        if args.then_u0:
            result = _run_w1_then_u0(
                runner,
                output_root=paths["output_root"],
                run_id=run_id,
                resume=args.resume,
                parent_checkpoint=args.parent_checkpoint,
                parent_gate=args.parent_gate,
            )
        else:
            result = _run_trainable_phase(
                runner,
                args.phase,
                resume=args.resume,
                ignore_phase_gate=not args.enforce_phase_gate,
                ignore_formal_gates=not args.enforce_phase_gate,
                parent_checkpoint=args.parent_checkpoint,
                parent_gate=args.parent_gate,
            )
            result = result.__dict__
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
