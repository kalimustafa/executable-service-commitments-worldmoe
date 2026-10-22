"""Standalone training runner for the risk-conditioned ECMoE continuation.

This module intentionally does not register a legacy WorldMoE phase.  It
consumes the released corpus batches directly, executes one RECMoE forward and
one independent R loss per update, and writes a compact R-only lineage without
hashing or external-solver hooks.
"""

from __future__ import annotations

from contextlib import nullcontext
import csv
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import yaml

from .r_ecmoe import RECMoE, RiskConditionPacket
from .r_ecmoe_losses import (
    RECMoELossResult,
    RECMoELossWeights,
    r_ecmoe_loss,
)
from .r_ecmoe_replay import RealizedHistoricalReplay


VALIDATION_RT_ORIGINS = (0, 4, 16, 48, 95)
TRAIN_DECISION_SCHEDULE = (
    (0, 0, 96),
    (1, 0, 1),
    (1, 0, 4),
    (1, 0, 16),
    (1, 0, 96),
    (1, 4, 4),
    (1, 4, 16),
    (1, 4, 92),
    (1, 16, 16),
    (1, 16, 80),
    (1, 48, 16),
    (1, 48, 48),
    (1, 95, 1),
)


class RECMoERunnerError(RuntimeError):
    """Raised when the standalone R training lineage is inconsistent."""


@dataclass(frozen=True)
class RECMoERunnerConfig:
    """Resolved, serializable training and decision-selection contract."""

    epochs: int = 80
    ra_epochs: int = 13
    batch_size: int = 32
    validation_batch_size: int = 64
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-2
    adamw_beta2: float = 0.95
    gradient_clip_norm: float = 1.0
    warmup_updates: int = 0
    cosine_decay_updates: int = 10_000
    minimum_learning_rate_ratio: float = 0.10
    use_bf16: bool = True
    allow_tf32: bool = True
    fused_adamw: bool = True
    seed: int = 20260827
    max_train_days: int | None = None
    max_validation_days: int | None = None
    smoke_max_days: int = 4
    train_decision_schedule: tuple[tuple[int, int, int], ...] = (
        TRAIN_DECISION_SCHEDULE
    )
    validation_origins: tuple[int, ...] = VALIDATION_RT_ORIGINS
    train_log_interval_updates: int = 50
    checkpoint_interval_updates: int = 100
    validation_interval_epochs: int = 2
    sentinel_validation_days: int = 16
    full_validation_interval_epochs: int = 10
    first_full_validation_epoch: int | None = None
    enable_sentinel_validation: bool = True
    save_latest_every_epoch: bool = True
    save_only_on_improvement: bool = False
    resume_checkpoint_epochs: tuple[int, ...] = ()
    coalesce_checkpoint_roles: bool = False
    save_on_exception: bool = True
    early_stop_start_epoch: int = 0
    early_stop_patience: int = 3
    early_stop_min_relative_improvement: float = 0.001
    selection_policy: str = "legacy"
    rolling_load_decay: float = 0.90
    safe_anchor_decay_power: float = 1.0
    decision_risk_weight: float = 0.25
    decision_sla_weight: float = 1.0
    decision_constraint_weight: float = 1.0
    decision_latency_penalty: float = 0.0
    hard_residual_tolerance: float = 1.0e-5
    sla_tolerance: float = 0.05
    latency_limit_ms_per_sample: float = 1_000.0
    selection_latency_warmup: int = 5
    selection_latency_repeats: int = 30
    save_best_speed_feasible: bool = False
    loss_weights: RECMoELossWeights = field(default_factory=RECMoELossWeights)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RECMoERunnerConfig":
        raw = dict(value)
        allowed = {item.name for item in fields(cls)}
        unknown = set(raw).difference(allowed)
        if unknown:
            raise RECMoERunnerError(
                "unknown R-ECMoE runner configuration keys: "
                + ", ".join(sorted(unknown))
            )
        loss_weights = raw.get("loss_weights")
        if isinstance(loss_weights, Mapping):
            raw["loss_weights"] = RECMoELossWeights(**dict(loss_weights))
        elif loss_weights is not None and not isinstance(
            loss_weights, RECMoELossWeights
        ):
            raise RECMoERunnerError("loss_weights must be a mapping or dataclass")
        decision_schedule = raw.get("train_decision_schedule")
        if decision_schedule is not None:
            try:
                raw["train_decision_schedule"] = tuple(
                    tuple(int(item) for item in decision)
                    for decision in decision_schedule
                )
            except (TypeError, ValueError) as exc:
                raise RECMoERunnerError(
                    "train_decision_schedule must contain integer triples"
                ) from exc
        if "resume_checkpoint_epochs" in raw:
            raw["resume_checkpoint_epochs"] = tuple(raw["resume_checkpoint_epochs"])
        return cls(**raw)

    def __post_init__(self) -> None:
        if self.selection_policy not in {"legacy", "validation_score_only"}:
            raise RECMoERunnerError("selection_policy must be legacy or validation_score_only")
        integer_positive = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "validation_batch_size": self.validation_batch_size,
            "cosine_decay_updates": self.cosine_decay_updates,
            "selection_latency_repeats": self.selection_latency_repeats,
        }
        for name, value in integer_positive.items():
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise RECMoERunnerError(f"{name} must be a positive integer")
        if not 0 <= int(self.ra_epochs) <= int(self.epochs):
            raise RECMoERunnerError("ra_epochs must lie between zero and epochs")
        if self.warmup_updates < 0 or self.warmup_updates > self.cosine_decay_updates:
            raise RECMoERunnerError("warmup_updates lies outside the LR schedule")
        if (
            isinstance(self.selection_latency_warmup, bool)
            or int(self.selection_latency_warmup) != self.selection_latency_warmup
            or int(self.selection_latency_warmup) < 0
        ):
            raise RECMoERunnerError(
                "selection_latency_warmup must be a nonnegative integer"
            )
        if self.max_train_days is not None and self.max_train_days <= 0:
            raise RECMoERunnerError("max_train_days must be positive")
        if self.max_validation_days is not None and self.max_validation_days <= 0:
            raise RECMoERunnerError("max_validation_days must be positive")
        if any(
            value <= 0
            for value in (
                self.smoke_max_days,
                self.train_log_interval_updates,
                self.validation_interval_epochs,
                self.sentinel_validation_days,
                self.full_validation_interval_epochs,
            )
        ):
            raise RECMoERunnerError(
                "smoke/checkpoint/validation intervals must be positive"
            )
        for name in ("checkpoint_interval_updates", "early_stop_start_epoch"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise RECMoERunnerError(f"{name} must be a nonnegative integer")
        if self.first_full_validation_epoch is not None:
            value = self.first_full_validation_epoch
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise RECMoERunnerError("first_full_validation_epoch must be positive")
        if any(
            isinstance(value, bool) or int(value) != value or value <= 0
            for value in self.resume_checkpoint_epochs
        ) or len(set(self.resume_checkpoint_epochs)) != len(self.resume_checkpoint_epochs):
            raise RECMoERunnerError("resume_checkpoint_epochs must contain unique positive integers")
        if self.early_stop_patience <= 0:
            raise RECMoERunnerError("early_stop_patience must be positive")
        for name, values, lower, upper in (
            ("validation_origins", self.validation_origins, 0, 95),
        ):
            if not values or any(
                isinstance(item, bool) or not lower <= int(item) <= upper
                for item in values
            ):
                raise RECMoERunnerError(f"{name} has an invalid value")
        if tuple(self.validation_origins) != VALIDATION_RT_ORIGINS:
            raise RECMoERunnerError(
                "validation_origins must remain the registered RT origins "
                f"{VALIDATION_RT_ORIGINS}"
            )
        normalized_schedule = tuple(
            tuple(int(item) for item in decision)
            for decision in self.train_decision_schedule
        )
        if normalized_schedule != TRAIN_DECISION_SCHEDULE:
            raise RECMoERunnerError(
                "train_decision_schedule must match the registered DA/RT curriculum"
            )
        finite_positive = {
            "learning_rate": self.learning_rate,
            "gradient_clip_norm": self.gradient_clip_norm,
            "safe_anchor_decay_power": self.safe_anchor_decay_power,
            "latency_limit_ms_per_sample": self.latency_limit_ms_per_sample,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise RECMoERunnerError(f"{name} must be finite and positive")
        finite_nonnegative = {
            "weight_decay": self.weight_decay,
            "decision_risk_weight": self.decision_risk_weight,
            "decision_sla_weight": self.decision_sla_weight,
            "decision_constraint_weight": self.decision_constraint_weight,
            "decision_latency_penalty": self.decision_latency_penalty,
            "hard_residual_tolerance": self.hard_residual_tolerance,
            "sla_tolerance": self.sla_tolerance,
            "early_stop_min_relative_improvement": (
                self.early_stop_min_relative_improvement
            ),
        }
        for name, value in finite_nonnegative.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise RECMoERunnerError(f"{name} must be finite and nonnegative")
        if not 0.0 < float(self.adamw_beta2) < 1.0:
            raise RECMoERunnerError("adamw_beta2 must lie in (0,1)")
        if not 0.0 <= float(self.minimum_learning_rate_ratio) <= 1.0:
            raise RECMoERunnerError("minimum_learning_rate_ratio must lie in [0,1]")
        if not 0.0 <= float(self.rolling_load_decay) < 1.0:
            raise RECMoERunnerError("rolling_load_decay must lie in [0,1)")


@dataclass(frozen=True)
class RECMoEValidationResult:
    epoch: int
    stage: str
    samples: int
    metrics: Mapping[str, float | bool | int | str]
    selection_eligible: bool
    decision_value: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as target:
        target.write(
            json.dumps(_json_ready(payload), sort_keys=True, allow_nan=False) + "\n"
        )


def _append_csv(path: Path, payload: Mapping[str, Any]) -> None:
    row = {str(key): _json_ready(value) for key, value in payload.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    if exists:
        with path.open("r", encoding="utf-8", newline="") as source:
            header = next(csv.reader(source), [])
        if header != list(row):
            raise RECMoERunnerError("validation CSV schema changed within one run")
    with path.open("a", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(
            _json_ready(payload), sort_keys=True, allow_unicode=True
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _batch_tensor(batch: Mapping[str, Any], outer: str, inner: str) -> Tensor | None:
    value = batch.get(outer)
    if not isinstance(value, Mapping):
        return None
    result = value.get(inner)
    return result if isinstance(result, Tensor) else None


class RECMoERunner:
    """R-A/R-B trainer and conservative decision-value checkpoint selector."""

    checkpoint_schema = "dcvpp.r_ecmoe.runner.v1"

    def __init__(
        self,
        model: RECMoE,
        corpus: Any,
        config: RECMoERunnerConfig | Mapping[str, Any],
        output_root: str | Path,
        run_id: str,
        device: torch.device | str,
        source_paths: Sequence[str | Path],
    ) -> None:
        if not isinstance(model, nn.Module) or not callable(
            getattr(model, "configure_stage", None)
        ):
            raise RECMoERunnerError("model must implement the RECMoE stage contract")
        if not callable(getattr(corpus, "batches", None)):
            raise RECMoERunnerError("corpus must expose batches()")
        if not run_id or Path(run_id).name != run_id:
            raise RECMoERunnerError("run_id must be one nonempty path component")
        self.model = model
        self.corpus = corpus
        self.config = (
            config
            if isinstance(config, RECMoERunnerConfig)
            else RECMoERunnerConfig.from_mapping(config)
        )
        self.device = torch.device(device)
        self.run_id = str(run_id)
        self.run_root = Path(output_root).resolve() / self.run_id
        self.checkpoint_root = self.run_root / "checkpoints"
        self.source_paths = tuple(str(Path(value).resolve()) for value in source_paths)
        self.mechanism_variant = str(
            getattr(getattr(self.model, "config", None), "mechanism_variant", "full")
        )
        self.resolved_config = asdict(self.config)
        self._write_run_identity()
        self.epoch = 0
        self.global_update = 0
        self.stage = "ra"
        self.best_decision_value = math.inf
        self.best_speed_ms = math.inf
        self._rolling_route_load: Tensor | None = None
        self._frozen_cost_scale: float | None = None
        self._frozen_cost_scale_tensor: Tensor | None = None
        self.batch_in_epoch = 0
        self.epoch_completed = True
        self.early_stop_bad_epochs = 0
        self.early_stop_reference_value = math.inf
        self.best_epoch = 0
        self._in_training_update = False
        self._optimizer_step_started = False
        self._batch_update_completed = False
        self._fit_started = False
        self._batch_start_rng: Mapping[str, Any] | None = None
        # Origin tensors are constant for a corpus iterator.  Checking their
        # values on every CUDA batch would introduce one needless host sync;
        # prove each stage/origin/horizon contract once per process instead.
        self._validated_origin_contracts: set[tuple[int, int, int]] = set()

        # Production training must always cross the post-decision historical
        # replay boundary.  A small explicit test hook is accepted for unit
        # doubles, but a real RECMoE never receives an identity fallback.
        if isinstance(self.model, RECMoE):
            self.realized_replay = RealizedHistoricalReplay.from_model(self.model)
        else:
            replay_hook = getattr(self.model, "realized_historical_replay", None)
            if not callable(replay_hook):
                raise RECMoERunnerError(
                    "non-RECMoE test doubles must expose realized_historical_replay"
                )
            self.realized_replay = replay_hook

        self.model.to(self.device)
        self._optimizer_parameters, self._tuned_dynamics_names = (
            self._resolve_optimizer_union()
        )
        self.optimizer = self._build_optimizer(self._optimizer_parameters)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._learning_rate_factor
        )
        # BF16 needs no numerical scaling, but an explicit disabled scaler keeps
        # resume/checkpoint structure stable if the runtime changes later.
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.use_bf16 = (
            self.config.use_bf16
            and self.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        self.optimizer_fused = bool(
            any(bool(group.get("fused", False)) for group in self.optimizer.param_groups)
        )
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = bool(self.config.allow_tf32)
            torch.backends.cudnn.allow_tf32 = bool(self.config.allow_tf32)

    @staticmethod
    def _config_payload(value: Any) -> Any:
        if value is None:
            return None
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        if isinstance(value, Mapping):
            return dict(value)
        return {"type": type(value).__name__}

    def _write_run_identity(self) -> None:
        model_config = getattr(self.model, "config", None)
        world_model = getattr(self.model, "world_model", None)
        world_config = getattr(world_model, "config", None)
        resolved = {
            "runner": self.resolved_config,
            "r_ecmoe_model": self._config_payload(model_config),
            "released_world_model": self._config_payload(world_config),
            "source_paths": self.source_paths,
        }
        config_path = self.run_root / "config_resolved.yaml"
        if not config_path.exists():
            _atomic_yaml(config_path, resolved)
        meta_path = self.run_root / "run_meta.json"
        if not meta_path.exists():
            _atomic_json(
                meta_path,
                {
                    "schema": "dcvpp.r_ecmoe.run_meta.v1",
                    "run_id": self.run_id,
                    "created_at_utc": _utc_now(),
                    "device": str(self.device),
                    "source_paths": self.source_paths,
                    "profile": str(getattr(model_config, "profile", "unknown")),
                    "mechanism_variant": self.mechanism_variant,
                    "locked_test_opened": False,
                    "locked_2025_opened": False,
                    "external_solver": False,
                    "candidate_search": False,
                    "hash_computed": False,
                    "outcome_basis": "realized_historical_replay",
                },
            )

    def _resolve_optimizer_union(self) -> tuple[list[nn.Parameter], tuple[str, ...]]:
        self._set_stage("rb")
        union = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        tuned = tuple(
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and name.startswith("world_model.dynamics.")
        )
        self._set_stage("ra")
        if not union:
            raise RECMoERunnerError("R-ECMoE exposes no trainable parameters")
        return union, tuned

    def _build_optimizer(
        self, parameters: Sequence[nn.Parameter]
    ) -> torch.optim.Optimizer:
        kwargs = {
            "lr": float(self.config.learning_rate),
            "weight_decay": float(self.config.weight_decay),
            "betas": (0.9, float(self.config.adamw_beta2)),
        }
        if self.config.fused_adamw and self.device.type == "cuda":
            try:
                optimizer = torch.optim.AdamW(parameters, fused=True, **kwargs)
                for group in optimizer.param_groups:
                    group["fused"] = True
                return optimizer
            except (RuntimeError, TypeError):
                pass
        optimizer = torch.optim.AdamW(parameters, fused=False, **kwargs)
        for group in optimizer.param_groups:
            group["fused"] = False
        return optimizer

    def _learning_rate_factor(self, completed_updates: int) -> float:
        update = int(completed_updates) + 1
        warmup = int(self.config.warmup_updates)
        finish = int(self.config.cosine_decay_updates)
        floor = float(self.config.minimum_learning_rate_ratio)
        if warmup > 0 and update <= warmup:
            return float(update) / float(warmup)
        if update < finish and finish > warmup:
            progress = float(update - warmup) / float(finish - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor + (1.0 - floor) * cosine
        return floor

    def _autocast(self) -> Any:
        if self.use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _stage_for_epoch(self, epoch: int) -> str:
        return "ra" if int(epoch) <= int(self.config.ra_epochs) else "rb"

    def _set_stage(self, stage: str) -> None:
        if stage not in {"ra", "rb"}:
            raise RECMoERunnerError("runner supports only R-A and R-B")
        self.model.configure_stage(stage)  # type: ignore[arg-type]
        apply_mechanism_trainability = getattr(self.model, "apply_mechanism_trainability", None)
        if callable(apply_mechanism_trainability):
            apply_mechanism_trainability()
        self.stage = stage

    def _training_decision(self, epoch: int) -> tuple[int, int, int]:
        """Return market stage, issue origin and active horizon for one epoch."""

        phase_epoch = (
            int(epoch) - 1
            if int(epoch) <= int(self.config.ra_epochs)
            else int(epoch) - int(self.config.ra_epochs) - 1
        )
        decision = self.config.train_decision_schedule[
            phase_epoch % len(self.config.train_decision_schedule)
        ]
        return tuple(int(value) for value in decision)  # type: ignore[return-value]

    def _validation_decisions(self) -> tuple[tuple[int, int, int], ...]:
        return ((0, 0, 96),) + tuple(
            (1, int(origin), 96 - int(origin))
            for origin in self.config.validation_origins
        )

    def _batches(
        self,
        split: str,
        *,
        epoch: int,
        smoke: bool,
        origin: int,
        start_batch: int = 0,
        validation_scope: str = "sentinel",
    ) -> Any:
        if split not in {"train", "validation"}:
            raise RECMoERunnerError("R-ECMoE runner never opens a test split")
        train = split == "train"
        batch_size = (
            self.config.batch_size if train else self.config.validation_batch_size
        )
        max_days = (
            self.config.max_train_days if train else self.config.max_validation_days
        )
        if smoke:
            max_days = (
                self.config.smoke_max_days
                if max_days is None
                else min(max_days, self.config.smoke_max_days)
            )
        elif not train:
            if validation_scope not in {"sentinel", "full"}:
                raise RECMoERunnerError("validation_scope must be sentinel or full")
            if validation_scope == "sentinel":
                max_days = (
                    self.config.sentinel_validation_days
                    if max_days is None
                    else min(max_days, self.config.sentinel_validation_days)
                )
            else:
                # Formal selection must see the complete validation year.
                max_days = None
        return self.corpus.batches(
            split,
            batch_size,
            device=self.device,
            shuffle_seed=(self.config.seed + int(epoch)) if train else None,
            max_days=max_days,
            origin=int(origin),
            require_h96=False,
            include_next_day=False,
            start_batch=int(start_batch),
        )

    def _cost_scale(self, batch: Mapping[str, Any]) -> Tensor:
        if self._frozen_cost_scale_tensor is not None:
            return self._frozen_cost_scale_tensor
        value = batch.get("train_frozen_baseline_daily_cost")
        if not isinstance(value, Tensor) or value.ndim != 0:
            raise RECMoERunnerError(
                "first batch must expose scalar train_frozen_baseline_daily_cost"
            )
        numeric = float(value.detach().float().cpu())
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise RECMoERunnerError("train-frozen baseline daily cost must be positive")
        self._frozen_cost_scale = numeric
        self._frozen_cost_scale_tensor = torch.tensor(
            numeric, device=self.device, dtype=torch.float32
        )
        meta_path = self.run_root / "run_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["train_frozen_baseline_daily_cost"] = numeric
        _atomic_json(meta_path, meta)
        return self._frozen_cost_scale_tensor

    def _forward_loss(
        self,
        batch: Mapping[str, Any],
        *,
        source_split: str,
        stage_id: int,
        decision_origin: int,
        decision_horizon: int,
        progress: float,
        action_range: float,
        rolling_route_load: Tensor | None,
        return_diagnostics: bool,
        measure_latency: bool = False,
    ) -> tuple[Any, RECMoELossResult, float]:
        packet = batch.get("world_input")
        if packet is None or not isinstance(getattr(packet, "fast_values", None), Tensor):
            raise RECMoERunnerError("batch omitted typed world_input")
        batch_size, horizon = packet.fast_values.shape[:2]
        origin_index = getattr(packet, "origin_index", None)
        origin_contract = (
            int(stage_id),
            int(decision_origin),
            int(decision_horizon),
        )
        origin_shape_valid = isinstance(origin_index, Tensor) and origin_index.shape == (
            batch_size,
        )
        if not origin_shape_valid:
            raise RECMoERunnerError("world_input origin disagrees with decision curriculum")
        if origin_contract not in self._validated_origin_contracts:
            if not bool((origin_index == int(decision_origin)).all()):
                raise RECMoERunnerError(
                    "world_input origin disagrees with decision curriculum"
                )
            self._validated_origin_contracts.add(origin_contract)
        if stage_id not in {0, 1}:
            raise RECMoERunnerError("market stage_id must be 0=DA or 1=RT")
        if not 1 <= int(decision_horizon) <= int(horizon) - int(decision_origin):
            raise RECMoERunnerError("decision horizon exceeds the remaining timeline")
        risk = RiskConditionPacket.defaults(
            int(batch_size),
            device=self.device,
            horizon=int(decision_horizon),
            stage=int(stage_id),
            action_range=float(action_range),
        )
        da_anchor = _batch_tensor(batch, "forecast_anchors", "da")
        da_cap = _batch_tensor(batch, "forecast_residual_caps", "da")
        if da_anchor is None or da_cap is None:
            raise RECMoERunnerError("batch omitted causal DA anchors or train-frozen caps")
        if measure_latency and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter() if measure_latency else 0.0
        with self._autocast():
            output = self.model(
                packet,
                risk,
                da_anchor=da_anchor,
                da_residual_cap=da_cap,
                return_diagnostics=return_diagnostics,
            )
        if measure_latency and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        forward_latency_ms = (
            (time.perf_counter() - started) * 1_000.0 if measure_latency else 0.0
        )
        # Realized outcomes are admitted only after the causal W1/R decision
        # has fixed routing, raw actions, DA commitment and target SOC.  The
        # replay remains differentiable with respect to that fixed decision.
        output = self.realized_replay(
            output,
            batch=batch,
            risk=risk,
            source_split=source_split,
        )
        diagnostics = getattr(output, "diagnostics", None)
        if not isinstance(diagnostics, Mapping) or diagnostics.get(
            "outcome_basis"
        ) != "realized_historical_replay":
            raise RECMoERunnerError(
                "R loss must be scored by post-decision realized historical replay"
            )
        if diagnostics.get("replay_split") != source_split:
            raise RECMoERunnerError("realized replay returned the wrong data split")
        horizon_cost_scale = self._cost_scale(batch) * (
            float(decision_horizon) / 96.0
        )
        result = r_ecmoe_loss(
            output,
            cost_scale=horizon_cost_scale,
            cost_scale_source="train",
            progress=progress,
            weights=self.config.loss_weights,
            rolling_route_load=rolling_route_load,
            rolling_decay=self.config.rolling_load_decay,
            safe_anchor_decay_power=self.config.safe_anchor_decay_power,
        )
        return output, result, forward_latency_ms

    def _training_progress(self) -> float:
        if self.stage == "rb" or self.epoch > int(self.config.ra_epochs):
            return 1.0
        if int(self.config.ra_epochs) <= 1:
            return 1.0
        return min(
            1.0,
            max(
                0.0,
                float(self.epoch - 1) / float(int(self.config.ra_epochs) - 1),
            ),
        )

    def _training_action_range(self) -> float:
        if self.stage == "rb" or self.epoch > int(self.config.ra_epochs):
            return 1.0
        return 0.10 + 0.90 * self._training_progress()

    def train_update(
        self,
        batch: Mapping[str, Any],
        *,
        stage_id: int,
        decision_origin: int,
        decision_horizon: int,
    ) -> Mapping[str, float | int | str]:
        self.model.train()
        world_model = getattr(self.model, "world_model", None)
        if isinstance(world_model, nn.Module):
            world_model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        training_progress = self._training_progress()
        action_range = self._training_action_range()
        _, result, _ = self._forward_loss(
            batch,
            source_split="train",
            stage_id=stage_id,
            decision_origin=decision_origin,
            decision_horizon=decision_horizon,
            progress=training_progress,
            action_range=action_range,
            rolling_route_load=self._rolling_route_load,
            return_diagnostics=False,
        )
        self.scaler.scale(result.total).backward()
        self.scaler.unscale_(self.optimizer)
        active = [parameter for parameter in self._optimizer_parameters if parameter.requires_grad]
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            active, float(self.config.gradient_clip_norm), error_if_nonfinite=True
        )
        self._optimizer_step_started = True
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.global_update += 1
        self.batch_in_epoch += 1
        self.epoch_completed = False
        self._rolling_route_load = result.next_rolling_route_load.detach()
        self._optimizer_step_started = False
        self._batch_update_completed = True
        should_log = bool(
            self.global_update == 1
            or self.global_update % int(self.config.train_log_interval_updates) == 0
        )
        should_checkpoint = bool(
            self.config.checkpoint_interval_updates > 0
            and self.global_update % int(self.config.checkpoint_interval_updates) == 0
        )
        metrics: dict[str, float | int | str] = {
            "recorded_at_utc": _utc_now(),
            "run_id": self.run_id,
            "epoch": self.epoch,
            "global_update": self.global_update,
            "stage": self.stage,
            "stage_id": int(stage_id),
            "decision_origin": int(decision_origin),
            "horizon": int(decision_horizon),
            "safe_anchor_progress": float(training_progress),
            "action_range": float(action_range),
            "cost_scale_horizon_fraction": float(decision_horizon) / 96.0,
            "samples": int(batch["world_input"].fast_values.shape[0]),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }
        # Keep steady-state updates asynchronous.  Scientific scalar logging
        # and the associated host synchronization happen only at registered
        # log/checkpoint intervals, never on every batch.
        if should_log or should_checkpoint:
            measured = torch.stack(
                (
                    result.total.detach().float(),
                    torch.as_tensor(gradient_norm).detach().float(),
                )
            ).cpu()
            metrics["train_objective"] = float(measured[0])
            metrics["gradient_norm"] = float(measured[1])
        if should_log:
            _append_jsonl(self.run_root / "train.jsonl", metrics)
        if should_checkpoint:
            self._save_checkpoint(
                "latest.pt", {**metrics, "checkpoint_reason": "update_interval"}
            )
        return metrics

    @staticmethod
    def _hard_residual(output: Any) -> float:
        residuals = getattr(output, "constraint_residuals", None)
        if not isinstance(residuals, Mapping) or not residuals:
            return math.inf
        maxima = []
        for name, value in residuals.items():
            # Deadline backlog has its own SLA filter and must not be silently
            # promoted to an exact physical-equality gate.
            if name in {"critical_shortfall", "target_soc_tracking"}:
                continue
            if not isinstance(value, Tensor) or not bool(torch.isfinite(value).all()):
                return math.inf
            maxima.append(float(value.detach().float().abs().amax().cpu()))
        return max(maxima, default=0.0)

    @staticmethod
    def _operating_dates(batch: Mapping[str, Any], count: int) -> tuple[str, ...]:
        raw = batch.get("operating_dates")
        if not isinstance(raw, (tuple, list)) or len(raw) != int(count):
            raise RECMoERunnerError(
                "realized validation requires one operating date per sample"
            )
        dates = tuple(str(value) for value in raw)
        if any(not value.startswith("2024-") for value in dates):
            raise RECMoERunnerError(
                "R checkpoint selection is restricted to the 2024 validation year"
            )
        if len(set(dates)) != len(dates):
            raise RECMoERunnerError("a validation batch repeated an operating date")
        return dates

    @staticmethod
    def _empirical_cvar95(values: Sequence[float]) -> tuple[float, int]:
        if not values or any(not math.isfinite(float(value)) for value in values):
            raise RECMoERunnerError(
                "empirical CVaR95 requires finite realized daily costs"
            )
        tail_count = max(1, int(math.ceil(0.05 * len(values))))
        tail = sorted((float(value) for value in values), reverse=True)[:tail_count]
        return sum(tail) / float(tail_count), tail_count

    def _batch1_selection_latency(self) -> Mapping[str, float | int | str]:
        """Measure direct batch-1 h96 inference for formal checkpoint selection."""

        iterator = self.corpus.batches(
            "validation",
            1,
            device=self.device,
            shuffle_seed=None,
            max_days=1,
            origin=0,
            require_h96=False,
            include_next_day=False,
            start_batch=0,
        )
        try:
            batch = next(iter(iterator))
        except StopIteration as exc:
            raise RECMoERunnerError(
                "batch-1 selection latency benchmark found no validation sample"
            ) from exc
        packet = batch.get("world_input")
        if packet is None or not isinstance(getattr(packet, "fast_values", None), Tensor):
            raise RECMoERunnerError("latency benchmark batch omitted world_input")
        if int(packet.fast_values.shape[0]) != 1:
            raise RECMoERunnerError("selection latency must use batch=1")
        risk = RiskConditionPacket.defaults(
            1,
            device=self.device,
            horizon=96,
            stage=1,
            action_range=1.0,
        )
        da_anchor = _batch_tensor(batch, "forecast_anchors", "da")
        da_cap = _batch_tensor(batch, "forecast_residual_caps", "da")
        if da_anchor is None or da_cap is None:
            raise RECMoERunnerError("latency benchmark omitted causal DA anchors")

        def one_forward() -> None:
            with self._autocast():
                output = self.model(
                    packet,
                    risk,
                    da_anchor=da_anchor,
                    da_residual_cap=da_cap,
                    return_diagnostics=False,
                )
            del output

        warmup = int(self.config.selection_latency_warmup)
        repeats = int(self.config.selection_latency_repeats)
        with torch.inference_mode():
            for _ in range(warmup):
                one_forward()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            values: list[float] = []
            for _ in range(repeats):
                started = time.perf_counter()
                one_forward()
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                values.append((time.perf_counter() - started) * 1_000.0)
        mean = sum(values) / float(len(values))
        return {
            "measurement": "batch1_h96_direct_causal_forward",
            "batch_size": 1,
            "horizon": 96,
            "warmup": warmup,
            "repeats": repeats,
            "mean_ms": mean,
            "p50_ms": _percentile(values, 0.50),
            "p95_ms": _percentile(values, 0.95),
            "windows_per_second": 1_000.0 / max(mean, 1.0e-12),
        }

    def _decision_value(self, metrics: Mapping[str, float]) -> float:
        latency_over_budget = max(
            0.0,
            metrics.get(
                "latency_p95_ms_per_sample", metrics["latency_ms_per_sample"]
            )
            / float(self.config.latency_limit_ms_per_sample)
            - 1.0,
        )
        return (
            metrics["realized_daily_mean_cost"]
            + float(self.config.decision_risk_weight) * metrics["empirical_cvar95"]
            + float(self.config.decision_sla_weight) * metrics["worst_origin_sla"]
            + float(self.config.decision_constraint_weight)
            * metrics["constraint_residual"]
            + float(self.config.decision_latency_penalty) * latency_over_budget
        )

    def _write_validation_outputs(
        self,
        metrics: Mapping[str, Any],
        *,
        per_decision: Mapping[str, Mapping[str, Any]],
        residual_means: Mapping[str, float],
        residual_maxima: Mapping[str, float],
        route_load: Sequence[float],
        latency_values: Sequence[float],
    ) -> None:
        common = {
            "recorded_at_utc": metrics["recorded_at_utc"],
            "run_id": self.run_id,
            "epoch": metrics["epoch"],
            "stage": self.stage,
            "validation_scope": metrics["validation_scope"],
            "samples": metrics["samples"],
            "stage_id": metrics["stage_id"],
            "decision_origin": metrics["decision_origin"],
            "horizon": metrics["horizon"],
        }
        _atomic_json(
            self.run_root / "validation_summary.json",
            {"aggregate": dict(metrics), "by_decision": dict(per_decision)},
        )
        _append_csv(
            self.run_root / "decision_value.csv",
            {
                **common,
                "outcome_basis": metrics["outcome_basis"],
                "expected_operating_cost": metrics["expected_operating_cost"],
                "realized_daily_mean_cost": metrics["realized_daily_mean_cost"],
                "empirical_cvar95": metrics["empirical_cvar95"],
                "worst_5pct_mean_cost": metrics["worst_5pct_mean_cost"],
                "empirical_tail_day_count": metrics["empirical_tail_day_count"],
                "ru_upper_tail_objective": metrics["ru_upper_tail_objective"],
                "worst_origin_sla": metrics["worst_origin_sla"],
                "constraint_residual": metrics["constraint_residual"],
                "latency_penalty_input_p95": metrics[
                    "latency_p95_ms_per_sample"
                ],
                "decision_value": metrics["decision_value"],
                "selection_policy": metrics["selection_policy"],
                "selection_thresholds_passed": metrics["selection_thresholds_passed"],
                "selection_eligible": metrics["selection_eligible"],
                "formal_best_eligible": bool(
                    metrics["validation_scope"] == "full"
                    and metrics["selection_eligible"]
                ),
            },
        )
        _append_csv(
            self.run_root / "risk_service.csv",
            {
                **common,
                "tail_risk": metrics["tail_risk"],
                "empirical_cvar95": metrics["empirical_cvar95"],
                "ru_upper_tail_objective": metrics["ru_upper_tail_objective"],
                "sla_deadline": metrics["sla_deadline"],
                "worst_origin_sla": metrics["worst_origin_sla"],
                "deadline_expired_work_mean": metrics[
                    "deadline_expired_work_mean"
                ],
                "sla_passed": metrics["sla_passed"],
            },
        )
        _append_csv(
            self.run_root / "constraint_residuals.csv",
            {
                **common,
                "constraint_residual": metrics["constraint_residual"],
                "hard_residual_max": metrics["hard_residual_max"],
                "hard_residual_passed": metrics["hard_residual_passed"],
                **{
                    f"mean_{name}": value
                    for name, value in sorted(residual_means.items())
                },
                **{
                    f"max_{name}": value
                    for name, value in sorted(residual_maxima.items())
                },
            },
        )
        _append_csv(
            self.run_root / "routing_diagnostics.csv",
            {
                **common,
                "route_continuation": metrics["route_continuation"],
                "load_balance": metrics["load_balance"],
                "route_entropy": metrics["route_entropy"],
                "tail_activation_rate": metrics["tail_activation_rate"],
                **{
                    f"expert_{index}_load": float(value)
                    for index, value in enumerate(route_load)
                },
            },
        )
        speed = {
            **common,
            "latency_mean_ms_per_sample": metrics["latency_ms_per_sample"],
            "latency_p50_ms_per_sample": metrics["latency_p50_ms_per_sample"],
            "latency_p95_ms_per_sample": metrics["latency_p95_ms_per_sample"],
            "latency_limit_ms_per_sample": self.config.latency_limit_ms_per_sample,
            "latency_measurement": metrics["latency_measurement"],
            "latency_warmup": metrics["latency_warmup"],
            "latency_repeats": metrics["latency_repeats"],
            "throughput_windows_per_second": metrics[
                "throughput_windows_per_second"
            ],
            "latency_passed": metrics["latency_passed"],
        }
        _append_csv(self.run_root / "speed.csv", speed)
        _atomic_json(
            self.run_root / "inference_latency.json",
            {
                **speed,
                "measurement": metrics["latency_measurement"],
                "weighted_latency_observations": len(latency_values),
            },
        )
        _atomic_text(
            self.run_root / "summary.md",
            "\n".join(
                (
                    f"# R-ECMoE {self.run_id}",
                    "",
                    f"- Epoch: {metrics['epoch']}",
                    f"- R stage: {self.stage}",
                    f"- Validation scope: {metrics['validation_scope']}",
                    f"- DA/RT decision points: {metrics['decision_points_evaluated']}",
                    f"- Weighted samples: {metrics['samples']}",
                    f"- Decision value: {float(metrics['decision_value']):.8g}",
                    f"- Outcome basis: {metrics['outcome_basis']}",
                    f"- Realized mean/CVaR95: "
                    f"{float(metrics['realized_daily_mean_cost']):.8g} / "
                    f"{float(metrics['empirical_cvar95']):.8g}",
                    f"- Worst-origin SLA: {float(metrics['worst_origin_sla']):.8g}",
                    f"- Selection policy: {metrics['selection_policy']}",
                    f"- Diagnostic thresholds passed: {metrics['selection_thresholds_passed']}",
                    f"- Selection eligible: {metrics['selection_eligible']}",
                    f"- Pure-forward latency p95: "
                    f"{float(metrics['latency_p95_ms_per_sample']):.6g} ms/sample",
                    "",
                )
            ),
        )

    def validate(
        self,
        *,
        epoch: int | None = None,
        smoke: bool = False,
        scope: str = "sentinel",
    ) -> RECMoEValidationResult:
        current_epoch = self.epoch if epoch is None else int(epoch)
        validation_scope = "smoke" if smoke else str(scope)
        if validation_scope not in {"smoke", "sentinel", "full"}:
            raise RECMoERunnerError("validation scope must be smoke, sentinel, or full")
        self.model.eval()
        totals: dict[str, float] = {}
        residual_totals: dict[str, float] = {}
        residual_maxima: dict[str, float] = {}
        route_load_total = torch.zeros(7, dtype=torch.float64)
        samples = 0
        finite = True
        hard_residual = 0.0
        latency_values: list[float] = []
        entropy_total = 0.0
        tail_activation_total = 0.0
        deadline_expired_total = 0.0
        per_decision: dict[str, dict[str, Any]] = {}
        realized_cost_by_date: dict[str, list[float]] = {}
        local_rolling = self._rolling_route_load
        with torch.inference_mode():
            for stage_id, origin, decision_horizon in self._validation_decisions():
                label = (
                    f"{'da' if stage_id == 0 else 'rt'}_o{origin}_h{decision_horizon}"
                )
                decision_totals: dict[str, float] = {}
                decision_samples = 0
                decision_latencies: list[float] = []
                decision_hard_residual = 0.0
                decision_residual_maxima: dict[str, float] = {}
                for batch_index, batch in enumerate(
                    self._batches(
                        "validation",
                        epoch=current_epoch,
                        smoke=smoke,
                        origin=origin,
                        validation_scope=validation_scope,
                    )
                ):
                    packet = batch.get("world_input")
                    if packet is None:
                        raise RECMoERunnerError("validation batch omitted world_input")
                    count = int(packet.fast_values.shape[0])
                    output, result, elapsed_ms = self._forward_loss(
                        batch,
                        source_split="validation",
                        stage_id=stage_id,
                        decision_origin=origin,
                        decision_horizon=decision_horizon,
                        progress=1.0,
                        action_range=1.0,
                        rolling_route_load=local_rolling,
                        return_diagnostics=True,
                        measure_latency=True,
                    )
                    local_rolling = result.next_rolling_route_load
                    sample_metrics = {
                        name: float(value.detach().cpu())
                        for name, value in result.components.items()
                    }
                    latency = elapsed_ms / float(count)
                    sample_metrics["latency_ms_per_sample"] = latency
                    sample_metrics["validation_objective"] = float(
                        result.total.detach().cpu()
                    )
                    finite = finite and all(
                        math.isfinite(value) for value in sample_metrics.values()
                    )
                    batch_hard = self._hard_residual(output)
                    hard_residual = max(hard_residual, batch_hard)
                    decision_hard_residual = max(decision_hard_residual, batch_hard)
                    dates = self._operating_dates(batch, count)
                    horizon_scale = float(self._cost_scale(batch).detach().cpu()) * (
                        float(decision_horizon) / 96.0
                    )
                    realized_costs = (
                        output.differentiable_operating_cost.detach().float()
                        / horizon_scale
                    ).cpu()
                    if realized_costs.shape != (count,):
                        raise RECMoERunnerError(
                            "realized replay operating cost must retain [B]"
                        )
                    for operating_date, realized_cost in zip(
                        dates, realized_costs.tolist(), strict=True
                    ):
                        realized_cost_by_date.setdefault(operating_date, []).append(
                            float(realized_cost)
                        )
                    for name, value in sample_metrics.items():
                        totals[name] = totals.get(name, 0.0) + float(value) * count
                        decision_totals[name] = (
                            decision_totals.get(name, 0.0) + float(value) * count
                        )
                    residuals = output.constraint_residuals
                    for name, value in residuals.items():
                        per_sample = value.detach().float().abs().flatten(1).mean(1)
                        residual_totals[name] = residual_totals.get(name, 0.0) + float(
                            per_sample.sum().cpu()
                        )
                        family_maximum = float(
                            value.detach().float().abs().amax().cpu()
                        )
                        residual_maxima[name] = max(
                            residual_maxima.get(name, 0.0), family_maximum
                        )
                        decision_residual_maxima[name] = max(
                            decision_residual_maxima.get(name, 0.0), family_maximum
                        )
                    route_load_total += result.route_load.detach().double().cpu() * count
                    active = output.decision_active_mask.detach().bool()
                    active_count = active.sum().clamp_min(1)
                    route_entropy = -(
                        output.route_weights.detach().float()
                        * torch.log(output.route_weights.detach().float().clamp_min(1.0e-8))
                    ).sum(dim=-1)
                    entropy_total += float(
                        (route_entropy * active.float()).sum().cpu()
                        / float(active_count.cpu())
                    ) * count
                    tail_activation_total += float(
                        (output.tail_decision_active.detach() & active).sum().cpu()
                    ) / float(active_count.cpu()) * count
                    deadline_expired_total += float(
                        output.deadline_expired_work.detach().float().flatten(1).mean(1).sum().cpu()
                    )
                    latency_values.extend([latency] * count)
                    decision_latencies.extend([latency] * count)
                    samples += count
                    decision_samples += count
                    if smoke and batch_index == 0:
                        break
                if decision_samples <= 0:
                    raise RECMoERunnerError(f"validation produced no samples for {label}")
                per_decision[label] = {
                    "stage_id": stage_id,
                    "decision_origin": origin,
                    "horizon": decision_horizon,
                    "samples": decision_samples,
                    **{
                        name: value / float(decision_samples)
                        for name, value in decision_totals.items()
                    },
                    "hard_residual_max": decision_hard_residual,
                    "residual_max_by_family": decision_residual_maxima,
                    "latency_p95_ms_per_sample": _percentile(
                        decision_latencies, 0.95
                    ),
                }
        if samples <= 0:
            raise RECMoERunnerError("validation produced no samples")
        averaged = {name: value / float(samples) for name, value in totals.items()}
        expected_decisions = len(per_decision)
        if not realized_cost_by_date or any(
            len(values) != expected_decisions
            for values in realized_cost_by_date.values()
        ):
            raise RECMoERunnerError(
                "every validation date must have one realized replay score per decision"
            )
        realized_daily_costs = [
            sum(values) / float(len(values))
            for _, values in sorted(realized_cost_by_date.items())
        ]
        empirical_cvar95, empirical_tail_days = self._empirical_cvar95(
            realized_daily_costs
        )
        realized_daily_mean = sum(realized_daily_costs) / float(
            len(realized_daily_costs)
        )
        # Preserve the RU training component as a diagnostic, but do not let
        # its learned eta or batch composition select the formal checkpoint.
        averaged["ru_upper_tail_objective"] = averaged["upper_tail_cost"]
        averaged["expected_operating_cost"] = realized_daily_mean
        averaged["realized_daily_mean_cost"] = realized_daily_mean
        averaged["empirical_cvar95"] = empirical_cvar95
        averaged["worst_5pct_mean_cost"] = empirical_cvar95
        averaged["empirical_tail_day_count"] = float(empirical_tail_days)
        averaged["realized_validation_days"] = float(len(realized_daily_costs))
        averaged["latency_p50_ms_per_sample"] = _percentile(latency_values, 0.50)
        averaged["latency_p95_ms_per_sample"] = _percentile(latency_values, 0.95)
        averaged["hard_residual_max"] = hard_residual
        averaged["route_entropy"] = entropy_total / float(samples)
        averaged["tail_activation_rate"] = tail_activation_total / float(samples)
        averaged["deadline_expired_work_mean"] = deadline_expired_total / float(samples)
        residual_means = {
            name: value / float(samples) for name, value in residual_totals.items()
        }
        route_load = (route_load_total / float(samples)).tolist()
        worst_origin_sla = max(
            float(item["sla_deadline"]) for item in per_decision.values()
        )
        averaged["worst_origin_sla"] = worst_origin_sla
        finite = finite and math.isfinite(hard_residual) and all(
            math.isfinite(value) for value in realized_daily_costs
        )
        residual_ok = hard_residual <= float(self.config.hard_residual_tolerance)
        sla_ok = worst_origin_sla <= float(self.config.sla_tolerance)
        rb_stage_ready = self.stage == "rb"
        averaged["validation_batch_latency_mean_ms_per_sample"] = averaged[
            "latency_ms_per_sample"
        ]
        averaged["validation_batch_latency_p50_ms_per_sample"] = averaged[
            "latency_p50_ms_per_sample"
        ]
        averaged["validation_batch_latency_p95_ms_per_sample"] = averaged[
            "latency_p95_ms_per_sample"
        ]
        if validation_scope == "full":
            latency_benchmark = self._batch1_selection_latency()
            averaged["latency_ms_per_sample"] = float(
                latency_benchmark["mean_ms"]
            )
            averaged["latency_p50_ms_per_sample"] = float(
                latency_benchmark["p50_ms"]
            )
            averaged["latency_p95_ms_per_sample"] = float(
                latency_benchmark["p95_ms"]
            )
            averaged["throughput_windows_per_second"] = float(
                latency_benchmark["windows_per_second"]
            )
            latency_basis = str(latency_benchmark["measurement"])
            latency_warmup = int(latency_benchmark["warmup"])
            latency_repeats = int(latency_benchmark["repeats"])
        else:
            latency_basis = "validation_batch_forward_elapsed_divided_by_batch"
            latency_warmup = 0
            latency_repeats = len(latency_values)
            averaged["throughput_windows_per_second"] = 1_000.0 / max(
                averaged["latency_ms_per_sample"], 1.0e-12
            )
        latency_ok = averaged["latency_p95_ms_per_sample"] <= float(
            self.config.latency_limit_ms_per_sample
        )
        decision_value = self._decision_value(averaged)
        finite = bool(
            finite and math.isfinite(decision_value)
            and all(math.isfinite(float(value)) for value in averaged.values())
        )
        thresholds_passed = bool(finite and residual_ok and sla_ok and latency_ok)
        if self.config.selection_policy == "validation_score_only":
            eligible = bool(finite and rb_stage_ready and validation_scope == "full")
        else:
            eligible = bool(thresholds_passed and rb_stage_ready)
        metrics: dict[str, float | bool | int | str] = {
            "recorded_at_utc": _utc_now(),
            "run_id": self.run_id,
            "epoch": current_epoch,
            "stage": self.stage,
            "validation_scope": validation_scope,
            "samples": samples,
            "stage_id": "0+1",
            "decision_origin": "0,0,4,16,48,95",
            "horizon": "96,96,92,80,48,1",
            "decision_points_evaluated": len(per_decision),
            "rt_origins_evaluated": len(self.config.validation_origins),
            "outcome_basis": "realized_historical_replay",
            "daily_aggregation": "mean_of_six_daily_equivalent_decision_costs",
            "tail_definition": "worst_ceil_5pct_realized_validation_days",
            "sla_gate_basis": "worst_of_six_decision_origins",
            "latency_measurement": latency_basis,
            "latency_warmup": latency_warmup,
            "latency_repeats": latency_repeats,
            "rb_stage_ready_for_selection": rb_stage_ready,
            **averaged,
            "finite": finite,
            "hard_residual_passed": residual_ok,
            "sla_passed": sla_ok,
            "latency_passed": latency_ok,
            "selection_policy": self.config.selection_policy,
            "selection_thresholds_passed": thresholds_passed,
            "selection_eligible": eligible,
            "decision_value": decision_value,
        }
        _append_csv(self.run_root / "validation.csv", metrics)
        _atomic_json(self.run_root / "latest_metrics.json", metrics)
        self._write_validation_outputs(
            metrics,
            per_decision=per_decision,
            residual_means=residual_means,
            residual_maxima=residual_maxima,
            route_load=route_load,
            latency_values=latency_values,
        )
        return RECMoEValidationResult(
            epoch=current_epoch,
            stage=self.stage,
            samples=samples,
            metrics=metrics,
            selection_eligible=eligible,
            decision_value=decision_value,
        )

    def _r_state(self) -> dict[str, Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in self.model.state_dict().items()
            if not name.startswith("world_model.")
        }

    def _tuned_dynamics_state(self) -> dict[str, Tensor] | None:
        if not self._tuned_dynamics_names:
            return None
        parameters = dict(self.model.named_parameters())
        return {
            name: parameters[name].detach().cpu()
            for name in self._tuned_dynamics_names
        }

    @staticmethod
    def _rng_state() -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        }

    @staticmethod
    def _restore_rng_state(state: Mapping[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"].cpu())
        if state.get("torch_cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])

    def _checkpoint_payload(
        self, metrics: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "schema": self.checkpoint_schema,
            "phase": "r_ecmoe",
            "w1_interface_mode": (
                "single_final_risk_slot_packet_plus_parallel_response"
            ),
            "freeze_state": {
                "w0": True,
                "w1_experts": True,
                "w1_router": True,
                "w1_dynamics": True,
                "r_trainable": True,
            },
            "locked_2025_opened": False,
            "external_solver": False,
            "candidate_search": False,
            "hash_computed": False,
            "outcome_basis": metrics.get(
                "outcome_basis", "realized_historical_replay"
            ),
            "run_id": self.run_id,
            "mechanism_variant": self.mechanism_variant,
            "model_config": self._config_payload(getattr(self.model, "config", None)),
            "selection_policy": self.config.selection_policy,
            "saved_at_utc": _utc_now(),
            "epoch": self.epoch,
            "global_update": self.global_update,
            "stage": self.stage,
            "r_state": self._r_state(),
            "tuned_dynamics_state": self._tuned_dynamics_state(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng_state": self._rng_state(),
            "resolved_config": self.resolved_config,
            "metrics": dict(metrics),
            "source_paths": self.source_paths,
            "rolling_route_load": (
                None
                if self._rolling_route_load is None
                else self._rolling_route_load.detach().cpu()
            ),
            "train_frozen_cost_scale": self._frozen_cost_scale,
            "best_decision_value": self.best_decision_value,
            "best_speed_ms": self.best_speed_ms,
            "best_epoch": self.best_epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "epoch_completed": self.epoch_completed,
            "data_cursor": {
                "epoch": self.epoch,
                "next_batch": self.batch_in_epoch,
                "epoch_completed": self.epoch_completed,
                "shuffle_seed": self.config.seed + self.epoch,
            },
            "early_stop_bad_epochs": self.early_stop_bad_epochs,
            "early_stop_reference_value": self.early_stop_reference_value,
        }

    def _save_checkpoint(self, name: str, metrics: Mapping[str, Any]) -> Path:
        allowed = {
            "latest.pt",
            "best_decision_value.pt",
            "best_speed_feasible.pt",
        }
        if name not in allowed:
            raise RECMoERunnerError("unregistered R checkpoint name")
        if name == "best_speed_feasible.pt" and not self.config.save_best_speed_feasible:
            raise RECMoERunnerError("speed checkpoint was not enabled")
        path = self.checkpoint_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self._checkpoint_payload(metrics), temporary)
        os.replace(temporary, path)
        return path

    def _reuse_checkpoint(self, source: Path, name: str) -> None:
        """Assign another role without serializing the tensors/optimizer twice."""
        target = self.checkpoint_root / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        try:
            os.link(source, temporary)
        except OSError:
            # Both role paths normally share a volume; copying also supports
            # filesystems that do not expose hard links.
            shutil.copyfile(source, temporary)
        os.replace(temporary, target)

    def _select_and_save(self, validation: RECMoEValidationResult) -> bool:
        metrics = validation.metrics
        scope = str(metrics["validation_scope"])
        formal = scope == "full"
        eligible = bool(
            validation.selection_eligible and math.isfinite(validation.decision_value)
        )
        if formal and eligible:
            previous_reference = self.early_stop_reference_value
            if not math.isfinite(previous_reference):
                significant = True
            elif float(self.config.early_stop_min_relative_improvement) == 0.0:
                # Exact best-score patience: even the next representable lower
                # finite score resets the counter; ties never do.
                significant = validation.decision_value < previous_reference
            else:
                required = max(
                    1.0e-12,
                    abs(previous_reference)
                    * float(self.config.early_stop_min_relative_improvement),
                )
                significant = validation.decision_value < previous_reference - required
            if significant:
                self.early_stop_reference_value = float(validation.decision_value)
                self.early_stop_bad_epochs = 0
            elif validation.epoch > self.config.early_stop_start_epoch:
                self.early_stop_bad_epochs += 1
            else:
                self.early_stop_bad_epochs = 0
        elif (
            formal
            and self.config.first_full_validation_epoch is not None
            and validation.epoch > self.config.early_stop_start_epoch
            and math.isfinite(self.early_stop_reference_value)
        ):
            self.early_stop_bad_epochs += 1
        self.epoch_completed = True
        self.batch_in_epoch = 0
        roles = ["latest.pt"]
        if (
            formal
            and eligible
            and validation.decision_value < self.best_decision_value
        ):
            self.best_decision_value = float(validation.decision_value)
            self.best_epoch = int(validation.epoch)
            roles.append("best_decision_value.pt")
        latency = float(metrics["latency_p95_ms_per_sample"])
        if (
            formal
            and self.config.selection_policy == "legacy"
            and self.config.save_best_speed_feasible
            and eligible
            and latency < self.best_speed_ms
        ):
            self.best_speed_ms = latency
            roles.append("best_speed_feasible.pt")
        save_due = bool(
            not self.config.save_only_on_improvement
            or "best_decision_value.pt" in roles
        )
        if save_due and (self.config.coalesce_checkpoint_roles or self.config.save_only_on_improvement):
            checkpoint = self._save_checkpoint("latest.pt", metrics)
            for role in roles[1:]:
                self._reuse_checkpoint(checkpoint, role)
        elif save_due:
            for role in [*roles[1:], "latest.pt"]:
                self._save_checkpoint(role, metrics)
        return self.early_stop_bad_epochs >= int(self.config.early_stop_patience)

    def resume(self, checkpoint: str | Path | None = None) -> bool:
        path = (
            self.checkpoint_root / "latest.pt"
            if checkpoint is None
            else Path(checkpoint)
        )
        if not path.is_file():
            return False
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(payload, Mapping) or payload.get("schema") != self.checkpoint_schema:
            raise RECMoERunnerError("resume checkpoint has an incompatible schema")
        if payload.get("run_id") != self.run_id:
            raise RECMoERunnerError("resume checkpoint belongs to another run")
        stored_config = payload.get("resolved_config")
        if not isinstance(stored_config, Mapping) or asdict(
            RECMoERunnerConfig.from_mapping(stored_config)
        ) != self.resolved_config:
            raise RECMoERunnerError("resume checkpoint configuration changed")
        if payload.get("mechanism_variant", "full") != self.mechanism_variant:
            raise RECMoERunnerError("resume checkpoint mechanism variant changed")
        if "model_config" in payload and payload["model_config"] != self._config_payload(
            getattr(self.model, "config", None)
        ):
            raise RECMoERunnerError("resume checkpoint model configuration changed")
        r_state = payload.get("r_state")
        if not isinstance(r_state, Mapping) or any(
            str(name).startswith("world_model.") for name in r_state
        ):
            raise RECMoERunnerError("resume R state is malformed")
        incompatible = self.model.load_state_dict(r_state, strict=False)
        if incompatible.unexpected_keys or any(
            not name.startswith("world_model.") for name in incompatible.missing_keys
        ):
            raise RECMoERunnerError("resume R state does not match the model")
        tuned = payload.get("tuned_dynamics_state")
        if tuned is not None:
            if not isinstance(tuned, Mapping) or set(tuned) != set(
                self._tuned_dynamics_names
            ):
                raise RECMoERunnerError("resume tuned-dynamics subset changed")
            parameters = dict(self.model.named_parameters())
            with torch.no_grad():
                for name, value in tuned.items():
                    if not isinstance(value, Tensor):
                        raise RECMoERunnerError("tuned dynamics state is not tensor-valued")
                    parameters[name].copy_(value.to(parameters[name]))
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        self.epoch = int(payload["epoch"])
        self.global_update = int(payload["global_update"])
        self._set_stage(str(payload["stage"]))
        rolling = payload.get("rolling_route_load")
        self._rolling_route_load = (
            None
            if rolling is None
            else torch.as_tensor(rolling, device=self.device, dtype=torch.float32)
        )
        scale = payload.get("train_frozen_cost_scale")
        self._frozen_cost_scale = None if scale is None else float(scale)
        self._frozen_cost_scale_tensor = (
            None
            if scale is None
            else torch.tensor(float(scale), device=self.device, dtype=torch.float32)
        )
        self.best_decision_value = float(payload.get("best_decision_value", math.inf))
        self.best_speed_ms = float(payload.get("best_speed_ms", math.inf))
        self.best_epoch = int(payload.get("best_epoch", 0))
        self.batch_in_epoch = int(payload.get("batch_in_epoch", 0))
        self.epoch_completed = bool(payload.get("epoch_completed", True))
        self.early_stop_bad_epochs = int(payload.get("early_stop_bad_epochs", 0))
        self.early_stop_reference_value = float(
            payload.get("early_stop_reference_value", math.inf)
        )
        if isinstance(payload.get("rng_state"), Mapping):
            self._restore_rng_state(payload["rng_state"])
        return True

    def _validation_scope_for_epoch(self, epoch: int, *, smoke: bool) -> str | None:
        if smoke:
            return "smoke"
        first = self.config.first_full_validation_epoch
        full_due = (
            epoch % self.config.full_validation_interval_epochs == 0
            if first is None
            else epoch >= first
            and (epoch - first) % self.config.full_validation_interval_epochs == 0
        )
        if (first is None or epoch >= first) and (epoch == self.config.epochs or full_due):
            return "full"
        if (
            self.config.enable_sentinel_validation
            and (first is None or epoch >= first)
            and epoch % self.config.validation_interval_epochs == 0
        ):
            return "sentinel"
        return None

    def fit(self, *, resume: bool = False, smoke: bool = False) -> Mapping[str, Any]:
        try:
            return self._fit(resume=resume, smoke=smoke)
        except BaseException as exc:
            # Never replace the last recoverable snapshot with a potentially
            # partial optimizer update. Other failures preserve the exact next
            # batch cursor, including interruption during validation.
            saved = False
            save_error = None
            if self._fit_started and self.config.save_on_exception and not self._optimizer_step_started:
                try:
                    if (
                        self._in_training_update
                        and not self._batch_update_completed
                        and self._batch_start_rng is not None
                    ):
                        self._restore_rng_state(self._batch_start_rng)
                    self._save_checkpoint("latest.pt", {
                        "epoch": self.epoch,
                        "stage": self.stage,
                        "checkpoint_reason": "exception_or_interrupt",
                        "exception_type": type(exc).__name__,
                    })
                    saved = True
                except Exception as checkpoint_exc:
                    save_error = repr(checkpoint_exc)
            _atomic_json(self.run_root / "interruption.json", {
                "recorded_at_utc": _utc_now(),
                "epoch": self.epoch,
                "global_update": self.global_update,
                "next_batch": self.batch_in_epoch,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "resume_checkpoint_saved": saved,
                "optimizer_step_in_progress": self._optimizer_step_started,
                "checkpoint_error": save_error,
            })
            raise

    def _fit(self, *, resume: bool = False, smoke: bool = False) -> Mapping[str, Any]:
        resumed = self.resume() if resume else False
        self._fit_started = True
        resume_incomplete = bool(resumed and not self.epoch_completed)
        start_epoch = self.epoch if resume_incomplete else self.epoch + 1
        last_validation: RECMoEValidationResult | None = None
        stopped_early = False
        for epoch in range(start_epoch, int(self.config.epochs) + 1):
            self.epoch = epoch
            self._set_stage(self._stage_for_epoch(epoch))
            stage_id, decision_origin, decision_horizon = self._training_decision(epoch)
            start_batch = self.batch_in_epoch if resume_incomplete else 0
            if not resume_incomplete:
                self.batch_in_epoch = 0
            self.epoch_completed = False
            updates_this_epoch = 0
            last_train_metrics: Mapping[str, Any] | None = None
            for batch in self._batches(
                "train",
                epoch=epoch,
                smoke=smoke,
                origin=decision_origin,
                start_batch=start_batch,
            ):
                self._batch_start_rng = (
                    self._rng_state() if self.config.save_on_exception else None
                )
                self._in_training_update = True
                self._batch_update_completed = False
                last_train_metrics = self.train_update(
                    batch,
                    stage_id=stage_id,
                    decision_origin=decision_origin,
                    decision_horizon=decision_horizon,
                )
                self._in_training_update = False
                self._batch_start_rng = None
                updates_this_epoch += 1
                if smoke:
                    break
            if updates_this_epoch == 0 and not resume_incomplete:
                raise RECMoERunnerError("training produced no optimizer updates")
            resume_incomplete = False
            scope = self._validation_scope_for_epoch(epoch, smoke=smoke)
            if scope is not None:
                last_validation = self.validate(
                    epoch=epoch, smoke=smoke, scope=scope
                )
                stopped_early = self._select_and_save(last_validation)
            else:
                self.epoch_completed = True
                self.batch_in_epoch = 0
                checkpoint_metrics = dict(last_train_metrics or {})
                checkpoint_metrics.update(
                    {
                        "epoch": epoch,
                        "stage": self.stage,
                        "stage_id": stage_id,
                        "decision_origin": decision_origin,
                        "horizon": decision_horizon,
                        "checkpoint_reason": "completed_epoch_without_validation",
                    }
                )
                if (
                    self.config.save_latest_every_epoch
                    or epoch in self.config.resume_checkpoint_epochs
                    or (epoch == self.config.epochs and not self.config.save_only_on_improvement)
                ):
                    checkpoint_metrics["checkpoint_reason"] = (
                        "resume_checkpoint_epoch"
                        if epoch in self.config.resume_checkpoint_epochs
                        else "completed_epoch_without_validation"
                    )
                    self._save_checkpoint("latest.pt", checkpoint_metrics)
            if smoke:
                break
            if stopped_early:
                break
        summary = {
            "schema": "dcvpp.r_ecmoe.summary.v1",
            "recorded_at_utc": _utc_now(),
            "run_id": self.run_id,
            "mechanism_variant": self.mechanism_variant,
            "selection_policy": self.config.selection_policy,
            "optimizer_parameter_count": sum(parameter.numel() for parameter in self._optimizer_parameters),
            "resumed": resumed,
            "smoke": smoke,
            "epoch": self.epoch,
            "global_update": self.global_update,
            "stage": self.stage,
            "best_decision_value": self.best_decision_value,
            "best_speed_ms": self.best_speed_ms,
            "best_epoch": self.best_epoch,
            "stopped_early": stopped_early,
            "early_stop_bad_epochs": self.early_stop_bad_epochs,
            "latest_validation": (
                None if last_validation is None else dict(last_validation.metrics)
            ),
            "checkpoint_names": sorted(
                path.name for path in self.checkpoint_root.glob("*.pt")
            )
            if self.checkpoint_root.is_dir()
            else [],
            "source_paths": self.source_paths,
            "optimizer_fused": self.optimizer_fused,
            "bf16_enabled": self.use_bf16,
            "tf32_enabled": bool(
                self.config.allow_tf32 and self.device.type == "cuda"
            ),
        }
        _atomic_json(self.run_root / "summary.json", summary)
        return summary

    def smoke(self, *, resume: bool = False) -> Mapping[str, Any]:
        """Run exactly one optimizer update followed by one validation batch."""

        return self.fit(resume=resume, smoke=True)


__all__ = [
    "TRAIN_DECISION_SCHEDULE",
    "RECMoERunner",
    "RECMoERunnerConfig",
    "RECMoERunnerError",
    "RECMoEValidationResult",
]
