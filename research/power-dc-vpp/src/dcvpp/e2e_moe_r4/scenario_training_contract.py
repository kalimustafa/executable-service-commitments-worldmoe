"""Train-only scale and residual-cap contract for the formal R4 scenario phase.

The scenario distribution is centred on the already selected Representation
checkpoint.  Consequently, raw residual caps must be fitted *after* that
checkpoint is frozen.  The scoring scale is intentionally not re-fitted here:
it is copied byte-for-value from the registered frozen training-residual
baseline used by the independent Scenario capability gate.

Only the 543 training dates are consumed.  Validation, 2024 selection, and the
locked 2025 test are outside this module's API.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .artifact_io import atomic_json, read_json, utc_now
from .hashing import sha256_bytes, sha256_file, sha256_json
from .training_data import PreloadedR4Corpus, move_batch


SCHEMA = "dcvpp.r4.scenario-training-contract.v1"
SCENARIO_SEMANTICS_REVISION = (
    "r4_scenario_gate_aligned_raw_caps_exact_identity_cycle1"
)
BASELINE_SCHEMA = "dcvpp.r4.frozen-training-residual-sampling.v1"
BASELINE_NAME = "frozen_training_residual_sampling"
HORIZON = 96
CHANNELS = 21
TRAIN_DAYS = 543
RAW_CAP_QUANTILE = 0.99
VARIABLE_ORDER = (
    "da_z1",
    "da_z2",
    "da_z3",
    "da_z4",
    "rt_z1",
    "rt_z2",
    "rt_z3",
    "rt_z4",
    "dart_z1",
    "dart_z2",
    "dart_z3",
    "dart_z4",
    "workload",
    "pv_z1",
    "pv_z2",
    "pv_z3",
    "pv_z4",
    "pue_z1",
    "pue_z2",
    "pue_z3",
    "pue_z4",
)


class ScenarioTrainingContractError(ValueError):
    """The train-only Scenario contract is absent, malformed, or unbound."""


@dataclass(frozen=True)
class ScenarioTrainingContract:
    """Validated contract plus its immutable file binding."""

    path: Path
    artifact_sha256: str
    payload: Mapping[str, Any]

    @property
    def score_scales(self) -> Tensor:
        return torch.as_tensor(
            self.payload["scenario_residual_score_scales"], dtype=torch.float32
        )

    @property
    def raw_p99_caps(self) -> Tensor:
        return torch.as_tensor(
            self.payload["scenario_residual_raw_p99_caps"], dtype=torch.float32
        )

    @property
    def representation_checkpoint_sha256(self) -> str:
        return str(self.payload["representation_checkpoint_sha256"])

    @property
    def representation_normalization_sha256(self) -> str:
        return str(self.payload["representation_normalization_sha256"])

    @property
    def frozen_baseline_sha256(self) -> str:
        return str(self.payload["frozen_training_residual_baseline_sha256"])


def _is_sha256(value: Any) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _operating_date(record: Mapping[str, Any]) -> str:
    value = record.get("operating_date")
    text = value.isoformat() if hasattr(value, "isoformat") else str(value)
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError as exc:
        raise ScenarioTrainingContractError(
            f"invalid training operating date {value!r}"
        ) from exc


def _validate_train_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    dates = tuple(_operating_date(record) for record in records)
    if len(records) != TRAIN_DAYS or len(set(dates)) != TRAIN_DAYS:
        raise ScenarioTrainingContractError(
            "Scenario contract requires exactly 543 unique training dates"
        )
    if dates != tuple(sorted(dates)):
        raise ScenarioTrainingContractError("Scenario contract dates must be ordered")
    if dates[0] != "2022-01-01" or dates[-1] != "2023-06-30":
        raise ScenarioTrainingContractError(
            "Scenario contract may use only 2022-01-01 through 2023-06-30"
        )
    if not all("2022-01-01" <= item <= "2023-06-30" for item in dates):
        raise ScenarioTrainingContractError(
            "Scenario contract attempted to open a non-training date"
        )
    return dates


def _collate(values: Sequence[Any]) -> Any:
    first = values[0]
    if isinstance(first, Tensor):
        return torch.stack([torch.as_tensor(value) for value in values])
    if isinstance(first, Mapping):
        return {key: _collate([value[key] for value in values]) for key in first}
    return list(values)


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return torch.device("cpu") if buffer is None else buffer.device


def _states_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    for name in left:
        first, second = left[name], right[name]
        if isinstance(first, Tensor) and isinstance(second, Tensor):
            if first.shape != second.shape or first.dtype != second.dtype:
                return False
            if not torch.equal(first.detach().cpu(), second.detach().cpu()):
                return False
        elif first != second:
            return False
    return True


def _load_representation_binding(
    checkpoint: Path, model: nn.Module
) -> tuple[dict[str, Any], str]:
    if not checkpoint.is_file():
        raise ScenarioTrainingContractError(
            f"selected Representation checkpoint is missing: {checkpoint}"
        )
    try:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(checkpoint, map_location="cpu")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("model_state"), Mapping):
        raise ScenarioTrainingContractError(
            "selected Representation checkpoint is missing model_state"
        )
    payload = dict(raw)
    if (
        payload.get("phase") != "representation"
        or payload.get("checkpoint_role") != "capability_passed_selected_best"
        or payload.get("capability_gate_status") != "PASSED"
    ):
        raise ScenarioTrainingContractError(
            "Scenario contract requires the capability-passed selected Representation"
        )
    normalization_sha = str(payload.get("normalization_sha256", "")).lower()
    if not _is_sha256(normalization_sha):
        raise ScenarioTrainingContractError(
            "selected Representation normalization binding is missing"
        )
    if not _states_equal(payload["model_state"], model.state_dict()):
        raise ScenarioTrainingContractError(
            "live model is not the selected Representation checkpoint"
        )
    return payload, sha256_file(checkpoint)


def _load_frozen_score_scale(
    path: Path, *, expected_sha256: str | None
) -> tuple[Tensor, str, Mapping[str, Any]]:
    if not path.is_file():
        raise ScenarioTrainingContractError(
            f"frozen Scenario residual baseline is missing: {path}"
        )
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != str(expected_sha256).lower():
        raise ScenarioTrainingContractError(
            "frozen Scenario residual baseline SHA-256 changed"
        )
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, Mapping):
        raise ScenarioTrainingContractError("frozen Scenario residual baseline is malformed")
    if raw.get("schema") != BASELINE_SCHEMA or raw.get("name") != BASELINE_NAME:
        raise ScenarioTrainingContractError("wrong frozen Scenario residual baseline")
    scale = torch.as_tensor(raw.get("scale"), dtype=torch.float64).flatten()
    if scale.shape != (CHANNELS,) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ScenarioTrainingContractError(
            "frozen Scenario residual score scale must be 21 finite positives"
        )
    if raw.get("normalization") != "train_residual_channel_median_absolute_scale":
        raise ScenarioTrainingContractError(
            "frozen Scenario residual scale normalization changed"
        )
    return scale, digest, raw


def _corpus_sha256(
    records: Sequence[Mapping[str, Any]], dates: Sequence[str]
) -> str:
    sources: list[dict[str, Any]] = []
    for operating_date, record in zip(dates, records, strict=True):
        provenance = record.get("provenance", {})
        if not isinstance(provenance, Mapping):
            provenance = {}
        sources.append(
            {
                "date": operating_date,
                "bundle_sha256": provenance.get("bundle_sha256"),
                "settlement_sha256": provenance.get("settlement_sha256"),
                "physical_sha256": provenance.get("physical_sha256"),
            }
        )
    return sha256_json({"schema": SCHEMA, "training_sources": sources})


def _validate_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SCHEMA:
        raise ScenarioTrainingContractError("wrong Scenario training contract schema")
    if int(payload.get("correction_cycle", -1)) != 1:
        raise ScenarioTrainingContractError(
            "Scenario training contract must be correction cycle 1"
        )
    if payload.get("scenario_semantics_revision") != SCENARIO_SEMANTICS_REVISION:
        raise ScenarioTrainingContractError(
            "Scenario training contract semantics revision changed"
        )
    declarations = {
        "decoder_scale": "raw_p99_cap_times_bounded_fraction",
        "score_scale": "frozen_baseline_exact",
        "coverage_training": "soft_90_95_plus_interval",
        "formal_coverage": "exact",
    }
    for key, expected in declarations.items():
        if payload.get(key) != expected:
            raise ScenarioTrainingContractError(
                f"Scenario training contract declaration {key} changed"
            )
    if payload.get("fit_split") != "train" or int(payload.get("fit_days", -1)) != TRAIN_DAYS:
        raise ScenarioTrainingContractError("Scenario training contract is not train-only")
    if (
        payload.get("fit_date_start") != "2022-01-01"
        or payload.get("fit_date_end") != "2023-06-30"
        or payload.get("future_years_opened") is not False
    ):
        raise ScenarioTrainingContractError("Scenario training contract date boundary changed")
    if tuple(payload.get("variable_order", ())) != VARIABLE_ORDER:
        raise ScenarioTrainingContractError("Scenario training contract variable order changed")
    if payload.get("score_scale_method") != (
        "frozen_training_residual_baseline.scale_exact_"
        "train_residual_channel_median_absolute_scale"
    ):
        raise ScenarioTrainingContractError("Scenario score-scale method changed")
    if payload.get("raw_cap_method") != (
        "free_channel_absolute_selected_representation_model_centered_"
        "residual_q0.99_floor_score_scale_with_derived_rt_da_plus_dart_envelope"
    ):
        raise ScenarioTrainingContractError("Scenario raw-cap method changed")
    scales = torch.as_tensor(
        payload.get("scenario_residual_score_scales"), dtype=torch.float64
    ).flatten()
    caps = torch.as_tensor(
        payload.get("scenario_residual_raw_p99_caps"), dtype=torch.float64
    ).flatten()
    if (
        scales.shape != (CHANNELS,)
        or caps.shape != (CHANNELS,)
        or not torch.isfinite(scales).all()
        or not torch.isfinite(caps).all()
        or (scales <= 0).any()
        or (caps < scales).any()
    ):
        raise ScenarioTrainingContractError(
            "Scenario score scales/caps must be 21 finite positive ordered values"
        )
    if not torch.allclose(
        caps[4:8], caps[0:4] + caps[8:12], rtol=0.0, atol=1.0e-12
    ):
        raise ScenarioTrainingContractError(
            "Scenario RT residual caps must equal the canonical DA plus DART envelope"
        )
    for key in (
        "representation_checkpoint_sha256",
        "representation_normalization_sha256",
        "frozen_training_residual_baseline_sha256",
        "train_corpus_sha256",
        "model_centered_residual_sha256",
    ):
        if not _is_sha256(payload.get(key)):
            raise ScenarioTrainingContractError(f"Scenario contract lacks {key}")


def load_scenario_training_contract(
    path: str | Path,
    *,
    expected_representation_sha256: str | None = None,
    expected_normalization_sha256: str | None = None,
    expected_baseline_sha256: str | None = None,
) -> ScenarioTrainingContract:
    """Load one immutable contract and verify all supplied lineage bindings."""

    source = Path(path).resolve()
    if not source.is_file():
        raise ScenarioTrainingContractError(
            f"Scenario training contract is missing: {source}"
        )
    payload = read_json(source)
    _validate_payload(payload)
    bindings = (
        (
            "representation_checkpoint_sha256",
            expected_representation_sha256,
        ),
        (
            "representation_normalization_sha256",
            expected_normalization_sha256,
        ),
        (
            "frozen_training_residual_baseline_sha256",
            expected_baseline_sha256,
        ),
    )
    for key, expected in bindings:
        if expected is not None and str(payload[key]).lower() != str(expected).lower():
            raise ScenarioTrainingContractError(f"Scenario contract {key} mismatch")
    return ScenarioTrainingContract(source, sha256_file(source), payload)


def materialize_scenario_training_contract(
    model: nn.Module,
    corpus: PreloadedR4Corpus,
    representation_checkpoint: str | Path,
    frozen_baseline: str | Path,
    destination: str | Path,
    *,
    device: torch.device | None = None,
    batch_size: int = 128,
    expected_baseline_sha256: str | None = None,
) -> ScenarioTrainingContract:
    """Fit selected-Representation-centred p99 caps on exactly 543 train days."""

    if int(batch_size) < 1:
        raise ValueError("Scenario contract batch_size must be positive")
    checkpoint = Path(representation_checkpoint).resolve()
    checkpoint_payload, representation_sha = _load_representation_binding(
        checkpoint, model
    )
    normalization_sha = str(checkpoint_payload["normalization_sha256"]).lower()
    baseline_scale, baseline_sha, baseline_payload = _load_frozen_score_scale(
        Path(frozen_baseline).resolve(), expected_sha256=expected_baseline_sha256
    )
    destination_path = Path(destination).resolve()
    if destination_path.exists():
        return load_scenario_training_contract(
            destination_path,
            expected_representation_sha256=representation_sha,
            expected_normalization_sha256=normalization_sha,
            expected_baseline_sha256=baseline_sha,
        )

    # No API in this routine can request validation/selection/test records.
    records = tuple(corpus.records("train"))
    dates = _validate_train_records(records)
    corpus_sha = _corpus_sha256(records, dates)
    model_device = _model_device(model) if device is None else torch.device(device)
    if model_device != _model_device(model):
        raise ScenarioTrainingContractError(
            "contract device must equal the selected Representation model device"
        )
    was_training = model.training
    residual_chunks: list[Tensor] = []
    identity_max = 0.0
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, TRAIN_DAYS, int(batch_size)):
                selected = records[start : start + int(batch_size)]
                batch = move_batch(_collate(selected), model_device)
                batch["origin"] = torch.zeros(
                    len(selected), dtype=torch.long, device=model_device
                )
                output = model(batch)
                forecasts = output.get("forecasts") if isinstance(output, Mapping) else None
                center = forecasts.get("zonal") if isinstance(forecasts, Mapping) else None
                target = batch.get("scenario_target_zonal")
                if not isinstance(center, Tensor) or not isinstance(target, Tensor):
                    raise ScenarioTrainingContractError(
                        "selected Representation did not emit zonal forecast centres"
                    )
                if center.shape != (len(selected), HORIZON, CHANNELS):
                    raise ScenarioTrainingContractError(
                        "selected Representation zonal centre must be [batch,96,21]"
                    )
                if target.shape != center.shape:
                    raise ScenarioTrainingContractError(
                        "Scenario target and selected Representation centre differ in shape"
                    )
                if not torch.isfinite(center).all() or not torch.isfinite(target).all():
                    raise ScenarioTrainingContractError(
                        "non-finite model-centred Scenario residual input"
                    )
                residual = (target - center).detach().to(dtype=torch.float64).cpu()
                identity = (
                    center[..., 0:4] - center[..., 4:8] - center[..., 8:12]
                )
                identity_max = max(identity_max, float(identity.abs().max().cpu()))
                residual_chunks.append(residual)
    finally:
        model.train(was_training)

    residuals = torch.cat(residual_chunks, dim=0).contiguous()
    if residuals.shape != (TRAIN_DAYS, HORIZON, CHANNELS):
        raise ScenarioTrainingContractError(
            "model-centred Scenario residual tensor has the wrong shape"
        )
    absolute = residuals.abs().flatten(0, 1)
    raw_p99 = torch.quantile(absolute, RAW_CAP_QUANTILE, dim=0)
    raw_p99 = torch.maximum(raw_p99, baseline_scale)
    # RT is structurally derived from DA and DART, not sampled independently.
    # Freeze its cap as the exact triangle-inequality envelope so the artifact,
    # attached batches, generator diagnostics, and formal gate all share one
    # canonical 21-channel contract.
    raw_p99[4:8] = raw_p99[0:4] + raw_p99[8:12]
    residual_sha = sha256_bytes(residuals.numpy().tobytes(order="C"))
    train_dates_sha = sha256_json(list(dates))
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "correction_cycle": 1,
        "scenario_semantics_revision": SCENARIO_SEMANTICS_REVISION,
        "decoder_scale": "raw_p99_cap_times_bounded_fraction",
        "score_scale": "frozen_baseline_exact",
        "coverage_training": "soft_90_95_plus_interval",
        "formal_coverage": "exact",
        "created_at_utc": utc_now(),
        "fit_split": "train",
        "fit_days": TRAIN_DAYS,
        "fit_date_start": dates[0],
        "fit_date_end": dates[-1],
        "fit_dates_sha256": train_dates_sha,
        "future_years_opened": False,
        "validation_2023_h2_opened": False,
        "formal_selection_2024_opened": False,
        "locked_test_2025_opened": False,
        "variable_order": list(VARIABLE_ORDER),
        "units": "raw_target_units_by_zonal_variable",
        "score_scale_method": (
            "frozen_training_residual_baseline.scale_exact_"
            "train_residual_channel_median_absolute_scale"
        ),
        "scenario_residual_score_scales": [
            float(value) for value in baseline_scale
        ],
        "raw_cap_method": (
            "free_channel_absolute_selected_representation_model_centered_"
            "residual_q0.99_floor_score_scale_with_derived_rt_da_plus_dart_envelope"
        ),
        "raw_cap_quantile": RAW_CAP_QUANTILE,
        "scenario_residual_raw_p99_caps": [float(value) for value in raw_p99],
        "representation_checkpoint": str(checkpoint),
        "representation_checkpoint_sha256": representation_sha,
        "representation_checkpoint_epoch": int(
            checkpoint_payload.get(
                "selected_epoch", checkpoint_payload.get("epochs_completed", -1)
            )
        ),
        "representation_normalization_sha256": normalization_sha,
        "frozen_training_residual_baseline": str(Path(frozen_baseline).resolve()),
        "frozen_training_residual_baseline_sha256": baseline_sha,
        "frozen_training_residual_baseline_corpus_sha256": baseline_payload.get(
            "corpus_sha256"
        ),
        "train_corpus_sha256": corpus_sha,
        "model_centered_residual_dtype": "float64",
        "model_centered_residual_shape": [TRAIN_DAYS, HORIZON, CHANNELS],
        "model_centered_residual_sha256": residual_sha,
        "model_center_da_rt_dart_identity_max_abs": identity_max,
    }
    _validate_payload(payload)
    atomic_json(destination_path, payload)
    return load_scenario_training_contract(
        destination_path,
        expected_representation_sha256=representation_sha,
        expected_normalization_sha256=normalization_sha,
        expected_baseline_sha256=baseline_sha,
    )


def attach_scenario_training_contract(
    batch: Mapping[str, Any],
    contract: ScenarioTrainingContract,
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Attach one shared 96x21 cap envelope and immutable contract metadata."""

    result = dict(batch)
    if device is None:
        reference = next(
            (value for value in result.values() if isinstance(value, Tensor)), None
        )
        target_device = torch.device("cpu") if reference is None else reference.device
    else:
        target_device = torch.device(device)
    scales = contract.score_scales.to(device=target_device)
    caps = contract.raw_p99_caps.to(device=target_device)
    zonal_caps = caps[None, :].expand(HORIZON, CHANNELS)
    expected_metadata = {
        "schema": SCHEMA,
        "artifact_sha256": contract.artifact_sha256,
        "fit_split": "train",
        "fit_days": TRAIN_DAYS,
        "representation_checkpoint_sha256": (
            contract.representation_checkpoint_sha256
        ),
        "representation_normalization_sha256": (
            contract.representation_normalization_sha256
        ),
        "frozen_training_residual_baseline_sha256": (
            contract.frozen_baseline_sha256
        ),
        "raw_cap_quantile": RAW_CAP_QUANTILE,
        "units": "raw_target_units_by_zonal_variable",
    }
    for key, value in (
        ("scenario_residual_score_scales", scales),
        ("scenario_residual_caps_zonal", zonal_caps),
    ):
        prior = result.get(key)
        if prior is not None and (
            not isinstance(prior, Tensor)
            or prior.shape != value.shape
            or not torch.equal(prior.to(value), value)
        ):
            raise ScenarioTrainingContractError(
                f"batch already contains a conflicting {key}"
            )
        result[key] = value
    prior_sha = result.get("scenario_training_contract_sha256")
    if prior_sha is not None and str(prior_sha) != contract.artifact_sha256:
        raise ScenarioTrainingContractError(
            "batch already contains a conflicting Scenario contract SHA-256"
        )
    prior_metadata = result.get("scenario_training_contract_metadata")
    if prior_metadata is not None and prior_metadata != expected_metadata:
        raise ScenarioTrainingContractError(
            "batch already contains conflicting Scenario contract metadata"
        )
    result["scenario_training_contract_sha256"] = contract.artifact_sha256
    result["scenario_training_contract_metadata"] = expected_metadata
    return result


__all__ = [
    "CHANNELS",
    "HORIZON",
    "RAW_CAP_QUANTILE",
    "SCHEMA",
    "SCENARIO_SEMANTICS_REVISION",
    "TRAIN_DAYS",
    "VARIABLE_ORDER",
    "ScenarioTrainingContract",
    "ScenarioTrainingContractError",
    "attach_scenario_training_contract",
    "load_scenario_training_contract",
    "materialize_scenario_training_contract",
]
