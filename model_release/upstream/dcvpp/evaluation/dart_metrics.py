"""Proper scores and sign gates for canonical DA-minus-RT DART forecasts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np


CANONICAL_DART_FIELD = "dart_da_minus_rt"
INVERSE_SPREAD_FIELD = "rt_spread_rt_minus_da"
FORBIDDEN_AMBIGUOUS_SPREAD_FIELD = "rt_spread"

Reduction = Literal["mean", "none"]
TailRegion = Literal["lower", "upper", "two_sided"]


def _float_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def assert_no_ambiguous_spread(field_names: Iterable[str]) -> bool:
    """Reject the ambiguous bare ``rt_spread`` field.

    Substrings are intentionally permitted.  Only an actual field whose name
    is exactly ``rt_spread`` is forbidden by the frozen contract.
    """

    names = {str(name) for name in field_names}
    if FORBIDDEN_AMBIGUOUS_SPREAD_FIELD in names:
        raise AssertionError(
            "ambiguous field 'rt_spread' is forbidden; use "
            f"'{CANONICAL_DART_FIELD}' or '{INVERSE_SPREAD_FIELD}'"
        )
    return True


def assert_canonical_dart(
    dart_da_minus_rt: Any,
    rt_spread_rt_minus_da: Any,
) -> bool:
    """Apply the contract's exact ``np.allclose(..., equal_nan=True)`` gate."""

    canonical = _float_array(dart_da_minus_rt, name=CANONICAL_DART_FIELD)
    inverse = _float_array(rt_spread_rt_minus_da, name=INVERSE_SPREAD_FIELD)
    if canonical.shape != inverse.shape:
        raise AssertionError(
            "canonical DART and inverse spread shapes differ: "
            f"{canonical.shape} != {inverse.shape}"
        )
    # Keep this call aligned verbatim with the governing specification.  In
    # particular, NaNs are allowed only when aligned in the two exact inverses.
    if not np.allclose(canonical, -inverse, equal_nan=True):
        finite = np.isfinite(canonical) & np.isfinite(inverse)
        mismatch = int(np.count_nonzero(finite & ~np.isclose(canonical, -inverse)))
        nan_mismatch = int(np.count_nonzero(np.isnan(canonical) != np.isnan(inverse)))
        raise AssertionError(
            "DART sign gate failed: expected dart_da_minus_rt == "
            f"-rt_spread_rt_minus_da; finite_mismatches={mismatch}, "
            f"nan_mismatches={nan_mismatch}"
        )
    return True


def derive_dart_fields(da_lbmp: Any, rt_lbmp: Any) -> dict[str, np.ndarray]:
    """Derive the canonical DART field and its explicitly named inverse."""

    day_ahead, real_time = np.broadcast_arrays(
        _float_array(da_lbmp, name="da_lbmp"),
        _float_array(rt_lbmp, name="rt_lbmp"),
    )
    canonical = day_ahead - real_time
    inverse = real_time - day_ahead
    assert_canonical_dart(canonical, inverse)
    return {
        CANONICAL_DART_FIELD: canonical,
        INVERSE_SPREAD_FIELD: inverse,
    }


def _validate_quantiles(
    predictions: Any,
    quantile_levels: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    forecast = _float_array(predictions, name="quantile_predictions")
    levels = np.asarray(quantile_levels, dtype=float)
    if levels.ndim != 1 or levels.size == 0:
        raise ValueError("quantile_levels must be a nonempty one-dimensional array")
    if forecast.shape[-1] != levels.size:
        raise ValueError(
            "the final prediction axis must match quantile_levels: "
            f"{forecast.shape[-1]} != {levels.size}"
        )
    if not np.all(np.isfinite(levels)) or np.any((levels <= 0.0) | (levels >= 1.0)):
        raise ValueError("quantile_levels must be finite and strictly between 0 and 1")
    if np.any(np.diff(levels) <= 0.0):
        raise ValueError("quantile_levels must be strictly increasing")
    return forecast, levels


def pinball_loss(
    observations: Any,
    quantile_predictions: Any,
    quantile_levels: Sequence[float],
    *,
    reduction: Reduction = "mean",
) -> float | np.ndarray:
    """Compute pinball loss, retaining quantiles on the final axis."""

    forecast, levels = _validate_quantiles(quantile_predictions, quantile_levels)
    observed = _float_array(observations, name="observations")
    try:
        observed = np.broadcast_to(observed, forecast.shape[:-1])
    except ValueError as error:
        raise ValueError(
            "observations must broadcast to prediction axes excluding quantiles"
        ) from error
    residual = observed[..., None] - forecast
    loss = np.maximum(levels * residual, (levels - 1.0) * residual)
    loss = np.where(np.isfinite(residual), loss, np.nan)
    if reduction == "none":
        return loss
    if reduction != "mean":
        raise ValueError("reduction must be 'mean' or 'none'")
    if not np.any(np.isfinite(loss)):
        return float("nan")
    return float(np.nanmean(loss))


def crps_from_quantiles(
    observations: Any,
    quantile_predictions: Any,
    quantile_levels: Sequence[float],
    *,
    reduction: Reduction = "mean",
) -> float | np.ndarray:
    """Approximate CRPS as twice mean pinball loss over frozen quantiles."""

    losses = np.asarray(
        pinball_loss(
            observations,
            quantile_predictions,
            quantile_levels,
            reduction="none",
        ),
        dtype=float,
    )
    per_observation = 2.0 * np.nanmean(losses, axis=-1)
    if reduction == "none":
        return per_observation
    if reduction != "mean":
        raise ValueError("reduction must be 'mean' or 'none'")
    if not np.any(np.isfinite(per_observation)):
        return float("nan")
    return float(np.nanmean(per_observation))


def threshold_weighted_crps_from_quantiles(
    observations: Any,
    quantile_predictions: Any,
    quantile_levels: Sequence[float],
    *,
    lower_threshold: Any | None = None,
    upper_threshold: Any | None = None,
    region: TailRegion = "two_sided",
    reduction: Reduction = "mean",
) -> float | np.ndarray:
    """Approximate proper threshold-weighted CRPS from frozen quantiles.

    Indicator tail weights are implemented through the monotone chaining
    transforms ``min(x, lower)`` and ``max(x, upper)`` before calling
    :func:`crps_from_quantiles`.  Thresholds may broadcast over every
    non-quantile prediction axis.  When this proper tail component is combined
    with the untransformed global CRPS, the finite-grid objective is a discrete
    strictly-consistent quantile approximation for checkpoint comparison.
    """

    if region not in {"lower", "upper", "two_sided"}:
        raise ValueError("region must be 'lower', 'upper', or 'two_sided'")
    forecast, levels = _validate_quantiles(quantile_predictions, quantile_levels)
    observed = _float_array(observations, name="observations")
    try:
        observed = np.broadcast_to(observed, forecast.shape[:-1])
    except ValueError as error:
        raise ValueError(
            "observations must broadcast to prediction axes excluding quantiles"
        ) from error

    def broadcast_threshold(raw_threshold: Any, *, name: str) -> np.ndarray:
        try:
            threshold = np.broadcast_to(
                _float_array(raw_threshold, name=name), observed.shape
            )
        except ValueError as error:
            raise ValueError(
                f"{name} must broadcast to prediction axes excluding quantiles"
            ) from error
        if not np.all(np.isfinite(threshold)):
            raise ValueError(f"{name} must be finite")
        return threshold

    def component(
        threshold: np.ndarray, *, tail: Literal["lower", "upper"]
    ) -> np.ndarray:
        if tail == "lower":
            transformed_observed = np.minimum(observed, threshold)
            transformed_forecast = np.minimum(forecast, threshold[..., None])
        else:
            transformed_observed = np.maximum(observed, threshold)
            transformed_forecast = np.maximum(forecast, threshold[..., None])
        return np.asarray(
            crps_from_quantiles(
                transformed_observed,
                transformed_forecast,
                levels,
                reduction="none",
            ),
            dtype=float,
        )

    if region in {"lower", "two_sided"} and lower_threshold is None:
        raise ValueError("lower_threshold is required for the lower tail")
    if region in {"upper", "two_sided"} and upper_threshold is None:
        raise ValueError("upper_threshold is required for the upper tail")

    if region == "two_sided":
        lower = broadcast_threshold(lower_threshold, name="lower_threshold")
        upper = broadcast_threshold(upper_threshold, name="upper_threshold")
        if np.any(lower >= upper):
            raise ValueError("lower_threshold must be below upper_threshold")
        score = component(lower, tail="lower") + component(upper, tail="upper")
    elif region == "lower":
        score = component(
            broadcast_threshold(lower_threshold, name="lower_threshold"),
            tail="lower",
        )
    else:
        score = component(
            broadcast_threshold(upper_threshold, name="upper_threshold"),
            tail="upper",
        )

    if reduction == "none":
        return score
    if reduction != "mean":
        raise ValueError("reduction must be 'mean' or 'none'")
    return float(np.nanmean(score)) if np.any(np.isfinite(score)) else float("nan")


def emphasized_tail_threshold_weighted_crps_from_quantiles(
    observations: Any,
    quantile_predictions: Any,
    quantile_levels: Sequence[float],
    *,
    lower_extreme_threshold: Any,
    lower_tail_threshold: Any,
    upper_tail_threshold: Any,
    upper_extreme_threshold: Any,
    reduction: Reduction = "mean",
) -> float | np.ndarray:
    """Approximate proper q05/q95 plus q01/q99 tail CRPS from quantiles.

    The q05/q95 two-sided score is added to a q01/q99 two-sided score, so the
    extreme tails receive one additional unit of weight.  All thresholds are
    frozen, finite, and broadcast over non-quantile axes before the exact
    ``q01 <= q05 < q95 <= q99`` gate is applied.  Combined with global CRPS,
    this is a discrete strictly-consistent quantile approximation suitable for
    checkpoint comparison.
    """

    forecast, _ = _validate_quantiles(quantile_predictions, quantile_levels)
    target_shape = forecast.shape[:-1]

    def broadcast_threshold(raw_threshold: Any, *, name: str) -> np.ndarray:
        try:
            threshold = np.broadcast_to(
                _float_array(raw_threshold, name=name), target_shape
            )
        except ValueError as error:
            raise ValueError(
                f"{name} must broadcast to prediction axes excluding quantiles"
            ) from error
        if not np.all(np.isfinite(threshold)):
            raise ValueError(f"{name} must be finite")
        return threshold

    lower_extreme = broadcast_threshold(
        lower_extreme_threshold, name="lower_extreme_threshold"
    )
    lower_tail = broadcast_threshold(
        lower_tail_threshold, name="lower_tail_threshold"
    )
    upper_tail = broadcast_threshold(
        upper_tail_threshold, name="upper_tail_threshold"
    )
    upper_extreme = broadcast_threshold(
        upper_extreme_threshold, name="upper_extreme_threshold"
    )
    if np.any(
        (lower_extreme > lower_tail)
        | (lower_tail >= upper_tail)
        | (upper_tail > upper_extreme)
    ):
        raise ValueError("tail thresholds must satisfy q01 <= q05 < q95 <= q99")

    tail = np.asarray(
        threshold_weighted_crps_from_quantiles(
            observations,
            forecast,
            quantile_levels,
            lower_threshold=lower_tail,
            upper_threshold=upper_tail,
            region="two_sided",
            reduction="none",
        ),
        dtype=float,
    )
    extreme = np.asarray(
        threshold_weighted_crps_from_quantiles(
            observations,
            forecast,
            quantile_levels,
            lower_threshold=lower_extreme,
            upper_threshold=upper_extreme,
            region="two_sided",
            reduction="none",
        ),
        dtype=float,
    )
    score = tail + extreme
    if reduction == "none":
        return score
    if reduction != "mean":
        raise ValueError("reduction must be 'mean' or 'none'")
    return float(np.nanmean(score)) if np.any(np.isfinite(score)) else float("nan")


def _crps_single_ensemble(observed: float, samples: np.ndarray) -> float:
    finite_samples = np.asarray(samples, dtype=float)
    finite_samples = finite_samples[np.isfinite(finite_samples)]
    if not np.isfinite(observed) or finite_samples.size == 0:
        return float("nan")
    ordered = np.sort(finite_samples)
    count = ordered.size
    first_term = float(np.mean(np.abs(ordered - observed)))
    ranks = np.arange(1, count + 1, dtype=float)
    half_pairwise_term = float(
        np.sum((2.0 * ranks - count - 1.0) * ordered) / (count * count)
    )
    return first_term - half_pairwise_term


def ensemble_crps(
    observations: Any,
    ensemble: Any,
    *,
    scenario_axis: int = -1,
    reduction: Reduction = "mean",
) -> float | np.ndarray:
    """Compute empirical CRPS from scenarios without an O(S^2) matrix."""

    samples = _float_array(ensemble, name="ensemble")
    samples = np.moveaxis(samples, scenario_axis, -1)
    observed = _float_array(observations, name="observations")
    try:
        observed = np.broadcast_to(observed, samples.shape[:-1])
    except ValueError as error:
        raise ValueError(
            "observations must broadcast to ensemble axes excluding scenarios"
        ) from error
    flattened_samples = samples.reshape(-1, samples.shape[-1])
    flattened_observed = observed.reshape(-1)
    scores = np.asarray(
        [
            _crps_single_ensemble(value, row)
            for value, row in zip(flattened_observed, flattened_samples, strict=True)
        ],
        dtype=float,
    ).reshape(observed.shape)
    if reduction == "none":
        return scores
    if reduction != "mean":
        raise ValueError("reduction must be 'mean' or 'none'")
    if not np.any(np.isfinite(scores)):
        return float("nan")
    return float(np.nanmean(scores))


def threshold_weighted_crps(
    observations: Any,
    ensemble: Any,
    *,
    lower_threshold: Any | None = None,
    upper_threshold: Any | None = None,
    region: TailRegion = "two_sided",
    scenario_axis: int = -1,
    reduction: Reduction = "mean",
) -> float | np.ndarray:
    """Compute the proper threshold-weighted CRPS from ensemble draws.

    This is the Gneiting--Ranjan threshold-weighted CRPS, not an
    outcome-weighted average of ordinary CRPS values.  For indicator weight
    functions, the score equals empirical CRPS after applying an antiderivative
    of the weight: ``min(x, lower)`` for the lower tail and ``max(x, upper)``
    for the upper tail.  The two-sided score is the sum of the two proper tail
    scores.  Thresholds may vary over every non-scenario axis (for example,
    one frozen rolling-training q01/q99 pair per NYISO zone).
    """

    if region not in {"lower", "upper", "two_sided"}:
        raise ValueError("region must be 'lower', 'upper', or 'two_sided'")
    samples = np.moveaxis(_float_array(ensemble, name="ensemble"), scenario_axis, -1)
    observed = np.broadcast_to(
        _float_array(observations, name="observations"), samples.shape[:-1]
    )

    def component(raw_threshold: Any, *, tail: Literal["lower", "upper"]) -> np.ndarray:
        threshold = np.broadcast_to(
            _float_array(raw_threshold, name=f"{tail}_threshold"), observed.shape
        )
        if not np.all(np.isfinite(threshold)):
            raise ValueError(f"{tail}_threshold must be finite")
        if tail == "lower":
            transformed_observed = np.minimum(observed, threshold)
            transformed_samples = np.minimum(samples, threshold[..., None])
        else:
            transformed_observed = np.maximum(observed, threshold)
            transformed_samples = np.maximum(samples, threshold[..., None])
        return np.asarray(
            ensemble_crps(
                transformed_observed,
                transformed_samples,
                scenario_axis=-1,
                reduction="none",
            ),
            dtype=float,
        )

    if region in {"lower", "two_sided"} and lower_threshold is None:
        raise ValueError("lower_threshold is required for the lower tail")
    if region in {"upper", "two_sided"} and upper_threshold is None:
        raise ValueError("upper_threshold is required for the upper tail")
    if region == "two_sided":
        lower = np.broadcast_to(
            _float_array(lower_threshold, name="lower_threshold"), observed.shape
        )
        upper = np.broadcast_to(
            _float_array(upper_threshold, name="upper_threshold"), observed.shape
        )
        if np.any(lower >= upper):
            raise ValueError("lower_threshold must be below upper_threshold")
        score = component(lower, tail="lower") + component(upper, tail="upper")
    elif region == "lower":
        score = component(lower_threshold, tail="lower")
    else:
        score = component(upper_threshold, tail="upper")
    if reduction == "none":
        return score
    if reduction != "mean":
        raise ValueError("reduction must be 'mean' or 'none'")
    return float(np.nanmean(score)) if np.any(np.isfinite(score)) else float("nan")


def emphasized_tail_threshold_weighted_crps(
    observations: Any,
    ensemble: Any,
    *,
    lower_extreme_threshold: Any,
    lower_tail_threshold: Any,
    upper_tail_threshold: Any,
    upper_extreme_threshold: Any,
    scenario_axis: int = -1,
    reduction: Reduction = "mean",
) -> float | np.ndarray:
    """Proper score emphasizing both q05--q95 tails and q01/q99 extremes.

    The score is the sum of two proper threshold-weighted CRPS components:
    one outside the frozen q05/q95 interval and one outside q01/q99.  Thus the
    most extreme region receives an additional unit of weight without making
    the weight depend on the candidate forecast or realized test outcome.
    """

    samples = np.moveaxis(_float_array(ensemble, name="ensemble"), scenario_axis, -1)
    observed = np.broadcast_to(
        _float_array(observations, name="observations"), samples.shape[:-1]
    )
    lower_extreme = np.broadcast_to(
        _float_array(lower_extreme_threshold, name="lower_extreme_threshold"),
        observed.shape,
    )
    lower_tail = np.broadcast_to(
        _float_array(lower_tail_threshold, name="lower_tail_threshold"),
        observed.shape,
    )
    upper_tail = np.broadcast_to(
        _float_array(upper_tail_threshold, name="upper_tail_threshold"),
        observed.shape,
    )
    upper_extreme = np.broadcast_to(
        _float_array(upper_extreme_threshold, name="upper_extreme_threshold"),
        observed.shape,
    )
    if np.any(
        (lower_extreme > lower_tail)
        | (lower_tail >= upper_tail)
        | (upper_tail > upper_extreme)
    ):
        raise ValueError(
            "tail thresholds must satisfy q01 <= q05 < q95 <= q99"
        )
    tail = np.asarray(
        threshold_weighted_crps(
            observed,
            samples,
            lower_threshold=lower_tail,
            upper_threshold=upper_tail,
            region="two_sided",
            scenario_axis=-1,
            reduction="none",
        ),
        dtype=float,
    )
    extreme = np.asarray(
        threshold_weighted_crps(
            observed,
            samples,
            lower_threshold=lower_extreme,
            upper_threshold=upper_extreme,
            region="two_sided",
            scenario_axis=-1,
            reduction="none",
        ),
        dtype=float,
    )
    score = tail + extreme
    if reduction == "none":
        return score
    if reduction != "mean":
        raise ValueError("reduction must be 'mean' or 'none'")
    return float(np.nanmean(score)) if np.any(np.isfinite(score)) else float("nan")


def fit_tail_thresholds(
    training_dart: Any,
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> tuple[float, float]:
    """Fit positive/negative tail thresholds from training data only."""

    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("tail quantiles must satisfy 0 < lower < upper < 1")
    values = _float_array(training_dart, name="training_dart").ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("training_dart has no finite observations")
    lower, upper = np.quantile(values, [lower_quantile, upper_quantile])
    return float(lower), float(upper)


def tail_weighted_crps(
    per_observation_crps: Any,
    observations: Any,
    *,
    lower_threshold: float,
    upper_threshold: float,
    tail_weight: float = 2.0,
) -> float:
    """Descriptively reweight ordinary CRPS by the realized tail outcome.

    This legacy diagnostic is *not* the proper threshold-weighted CRPS.  New
    paper evidence must use :func:`threshold_weighted_crps`; this function is
    retained only so pre-amendment auxiliary artifacts remain readable.
    """

    if not np.isfinite(lower_threshold) or not np.isfinite(upper_threshold):
        raise ValueError("tail thresholds must be finite")
    if lower_threshold >= upper_threshold:
        raise ValueError("lower_threshold must be less than upper_threshold")
    if not np.isfinite(tail_weight) or tail_weight < 1.0:
        raise ValueError("tail_weight must be finite and at least one")
    score, observed = np.broadcast_arrays(
        _float_array(per_observation_crps, name="per_observation_crps"),
        _float_array(observations, name="observations"),
    )
    valid = np.isfinite(score) & np.isfinite(observed)
    if not np.any(valid):
        return float("nan")
    weights = np.ones(score.shape, dtype=float)
    weights[(observed < lower_threshold) | (observed > upper_threshold)] = tail_weight
    return float(np.sum(score[valid] * weights[valid]) / np.sum(weights[valid]))


def interval_calibration(
    observations: Any,
    lower: Any,
    upper: Any,
    *,
    nominal_coverage: float | None = None,
) -> dict[str, float | int | None]:
    """Return empirical interval coverage, width, and calibration error."""

    observed, low, high = np.broadcast_arrays(
        _float_array(observations, name="observations"),
        _float_array(lower, name="lower"),
        _float_array(upper, name="upper"),
    )
    valid = np.isfinite(observed) & np.isfinite(low) & np.isfinite(high)
    if np.any(low[valid] > high[valid]):
        raise ValueError("lower interval bounds must not exceed upper bounds")
    if nominal_coverage is not None and not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage must be between zero and one")
    if not np.any(valid):
        return {
            "count": 0,
            "coverage": None,
            "mean_width": None,
            "absolute_calibration_error": None,
        }
    coverage = float(np.mean((observed[valid] >= low[valid]) & (observed[valid] <= high[valid])))
    width = float(np.mean(high[valid] - low[valid]))
    return {
        "count": int(np.count_nonzero(valid)),
        "coverage": coverage,
        "mean_width": width,
        "absolute_calibration_error": (
            abs(coverage - nominal_coverage) if nominal_coverage is not None else None
        ),
    }


def exceedance_calibration(
    observations: Any,
    exceedance_probability: Any,
    *,
    threshold: Any,
    direction: Literal["above", "below"] = "above",
) -> dict[str, float | int | None]:
    """Score positive or negative spike probabilities with Brier/calibration error."""

    observed, probability, threshold_values = np.broadcast_arrays(
        _float_array(observations, name="observations"),
        _float_array(exceedance_probability, name="exceedance_probability"),
        _float_array(threshold, name="threshold"),
    )
    valid = (
        np.isfinite(observed)
        & np.isfinite(probability)
        & np.isfinite(threshold_values)
    )
    if np.any((probability[valid] < 0.0) | (probability[valid] > 1.0)):
        raise ValueError("exceedance probabilities must lie in [0, 1]")
    if not np.any(valid):
        return {"count": 0, "brier": None, "mean_probability": None, "event_rate": None, "calibration_error": None}
    if direction == "above":
        event = observed[valid] > threshold_values[valid]
    elif direction == "below":
        event = observed[valid] < threshold_values[valid]
    else:
        raise ValueError("direction must be 'above' or 'below'")
    probabilities = probability[valid]
    event_float = event.astype(float)
    mean_probability = float(np.mean(probabilities))
    event_rate = float(np.mean(event_float))
    return {
        "count": int(event_float.size),
        "brier": float(np.mean((probabilities - event_float) ** 2)),
        "mean_probability": mean_probability,
        "event_rate": event_rate,
        "calibration_error": abs(mean_probability - event_rate),
    }


def pit_from_ensemble(
    observations: Any,
    ensemble: Any,
    *,
    scenario_axis: int = -1,
) -> np.ndarray:
    """Return deterministic mid-rank PIT values from an empirical ensemble."""

    samples = np.moveaxis(_float_array(ensemble, name="ensemble"), scenario_axis, -1)
    observed = np.broadcast_to(
        _float_array(observations, name="observations"), samples.shape[:-1]
    )
    less = np.sum(samples < observed[..., None], axis=-1)
    equal = np.sum(samples == observed[..., None], axis=-1)
    finite = np.sum(np.isfinite(samples), axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pit = (less + 0.5 * equal) / finite
    pit[(finite == 0) | ~np.isfinite(observed)] = np.nan
    return pit


def energy_score(
    observations: Any,
    scenario_paths: Any,
    *,
    scenario_axis: int = 1,
) -> float:
    """Compute the multivariate energy score over cases and joint paths."""

    scenarios = np.moveaxis(
        _float_array(scenario_paths, name="scenario_paths"), scenario_axis, 1
    )
    observed = _float_array(observations, name="observations")
    if scenarios.ndim < 3:
        raise ValueError("scenario_paths must contain case, scenario, and event axes")
    if observed.shape[0] != scenarios.shape[0] or observed.shape[1:] != scenarios.shape[2:]:
        raise ValueError("observations must match scenario case and event axes")
    case_scores: list[float] = []
    for truth, paths in zip(observed, scenarios, strict=True):
        flat_paths = paths.reshape(paths.shape[0], -1)
        flat_truth = truth.reshape(-1)
        if not np.all(np.isfinite(flat_paths)) or not np.all(np.isfinite(flat_truth)):
            continue
        first = np.mean(np.linalg.norm(flat_paths - flat_truth, axis=1))
        pairwise = np.linalg.norm(
            flat_paths[:, None, :] - flat_paths[None, :, :], axis=2
        )
        case_scores.append(float(first - 0.5 * np.mean(pairwise)))
    return float(np.mean(case_scores)) if case_scores else float("nan")


def variogram_score(
    observations: Any,
    scenario_paths: Any,
    *,
    scenario_axis: int = 1,
    power: float = 0.5,
) -> float:
    """Compute an equally weighted variogram score for each joint path."""

    if not 0.0 < power <= 2.0:
        raise ValueError("variogram power must lie in (0, 2]")
    scenarios = np.moveaxis(
        _float_array(scenario_paths, name="scenario_paths"), scenario_axis, 1
    )
    observed = _float_array(observations, name="observations")
    if scenarios.ndim < 3 or observed.shape[0] != scenarios.shape[0] or observed.shape[1:] != scenarios.shape[2:]:
        raise ValueError("observations must match scenario case and event axes")
    scores: list[float] = []
    for truth, paths in zip(observed, scenarios, strict=True):
        truth_flat = truth.reshape(-1)
        path_flat = paths.reshape(paths.shape[0], -1)
        if not np.all(np.isfinite(truth_flat)) or not np.all(np.isfinite(path_flat)):
            continue
        if truth_flat.size < 2:
            scores.append(0.0)
            continue
        truth_variogram = np.abs(truth_flat[:, None] - truth_flat[None, :]) ** power
        forecast_variogram = np.mean(
            np.abs(path_flat[:, :, None] - path_flat[:, None, :]) ** power,
            axis=0,
        )
        upper = np.triu_indices(truth_flat.size, k=1)
        scores.append(float(np.mean((truth_variogram[upper] - forecast_variogram[upper]) ** 2)))
    return float(np.mean(scores)) if scores else float("nan")


def joint_path_diagnostics(
    observations: Any,
    scenario_paths: Any,
    *,
    scenario_axis: int = 1,
    variogram_power: float = 0.5,
    hooks: Mapping[str, Callable[[np.ndarray, np.ndarray], float]] | None = None,
) -> dict[str, float]:
    """Run core joint scores and optional frozen diagnostic hooks.

    Hooks make temporal autocorrelation, cross-zone rank correlation, spike
    duration, and joint-spike-frequency diagnostics pluggable without mixing
    source-domain assumptions into these pure scoring functions.
    """

    observed = _float_array(observations, name="observations")
    scenarios = _float_array(scenario_paths, name="scenario_paths")
    result = {
        "energy_score": energy_score(
            observed, scenarios, scenario_axis=scenario_axis
        ),
        "variogram_score": variogram_score(
            observed,
            scenarios,
            scenario_axis=scenario_axis,
            power=variogram_power,
        ),
    }
    for name, hook in sorted((hooks or {}).items()):
        if name in result:
            raise ValueError(f"joint diagnostic hook duplicates core metric: {name}")
        value = float(hook(observed.copy(), scenarios.copy()))
        result[str(name)] = value
    return result


# Explicit compatibility aliases for callers that use the contract wording.
validate_dart_sign = assert_canonical_dart
canonical_dart_fields = derive_dart_fields
crps = ensemble_crps


__all__ = [
    "CANONICAL_DART_FIELD",
    "FORBIDDEN_AMBIGUOUS_SPREAD_FIELD",
    "INVERSE_SPREAD_FIELD",
    "assert_canonical_dart",
    "assert_no_ambiguous_spread",
    "canonical_dart_fields",
    "crps",
    "crps_from_quantiles",
    "derive_dart_fields",
    "energy_score",
    "ensemble_crps",
    "emphasized_tail_threshold_weighted_crps",
    "emphasized_tail_threshold_weighted_crps_from_quantiles",
    "exceedance_calibration",
    "fit_tail_thresholds",
    "interval_calibration",
    "joint_path_diagnostics",
    "pinball_loss",
    "pit_from_ensemble",
    "tail_weighted_crps",
    "threshold_weighted_crps",
    "threshold_weighted_crps_from_quantiles",
    "validate_dart_sign",
    "variogram_score",
]
