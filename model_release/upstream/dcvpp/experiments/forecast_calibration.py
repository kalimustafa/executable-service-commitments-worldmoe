from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats


DEFAULT_QUANTILES = np.asarray([0.05, 0.10, 0.50, 0.90, 0.95], dtype="float64")


def rearrange_quantiles(values: np.ndarray) -> np.ndarray:
    """Return monotonically ordered quantiles without changing their marginals."""
    array = np.asarray(values, dtype="float64")
    if array.ndim < 2:
        raise ValueError("quantile arrays require a sample axis and a quantile axis")
    return np.sort(array, axis=-1)


def crossing_count(values: np.ndarray) -> int:
    array = np.asarray(values, dtype="float64")
    return int(np.sum(np.diff(array, axis=-1) < 0))


def _higher_quantile(values: np.ndarray, probability: float) -> float:
    finite = np.asarray(values, dtype="float64")
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("cannot calibrate from an empty residual set")
    level = min(1.0, np.ceil((len(finite) + 1) * probability) / len(finite))
    return float(np.quantile(finite, level, method="higher"))


def fit_additive_conformal_quantiles(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    minimum_samples: int = 500,
) -> dict[str, Any]:
    """Fit validation-only additive quantile corrections.

    Inputs have shape ``(sample, group..., quantile)``.  A group-specific
    correction is used only when it has the requested number of validation
    samples.  Otherwise the correction is pooled over every non-sample group;
    the fallback is explicit in the returned manifest.
    """
    truth = np.asarray(actual, dtype="float64")
    raw = rearrange_quantiles(predicted)
    levels = np.asarray(quantiles, dtype="float64")
    if raw.shape[:-1] != truth.shape or raw.shape[-1] != len(levels):
        raise ValueError("actual and quantile prediction shapes are incompatible")
    if not np.all(np.diff(levels) > 0) or levels[0] <= 0 or levels[-1] >= 1:
        raise ValueError("quantile levels must be strictly increasing inside (0,1)")
    group_shape = truth.shape[1:]
    per_group_samples = truth.shape[0]
    if per_group_samples >= minimum_samples:
        corrections = np.empty((*group_shape, len(levels)), dtype="float64")
        for index in np.ndindex(group_shape):
            selection = (slice(None), *index)
            for q_index, level in enumerate(levels):
                corrections[(*index, q_index)] = _higher_quantile(
                    truth[selection] - raw[(*selection, q_index)], float(level)
                )
        grouping = "full_group"
        effective_samples = per_group_samples
    else:
        corrections = np.empty((len(levels),), dtype="float64")
        for q_index, level in enumerate(levels):
            corrections[q_index] = _higher_quantile(
                truth.reshape(-1) - raw[..., q_index].reshape(-1), float(level)
            )
        grouping = "pooled_validation_fallback"
        effective_samples = truth.size
    return {
        "method": "validation_additive_conformal_quantiles",
        "quantiles": levels.tolist(),
        "grouping": grouping,
        "minimum_samples": int(minimum_samples),
        "effective_samples": int(effective_samples),
        "corrections": corrections,
        "crossings_before": crossing_count(predicted),
        "locked_test_used": False,
    }


def apply_additive_conformal_quantiles(
    predicted: np.ndarray,
    calibrator: dict[str, Any],
) -> np.ndarray:
    raw = rearrange_quantiles(predicted)
    correction = np.asarray(calibrator["corrections"], dtype="float64")
    if correction.ndim == 1:
        calibrated = raw + correction
    else:
        calibrated = raw + correction[None, ...]
    return rearrange_quantiles(calibrated)


def quantile_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    spike_threshold: float | None = None,
) -> dict[str, Any]:
    truth = np.asarray(actual, dtype="float64")
    forecast = rearrange_quantiles(predicted)
    levels = np.asarray(quantiles, dtype="float64")
    if forecast.shape[:-1] != truth.shape or forecast.shape[-1] != len(levels):
        raise ValueError("actual and forecast quantile shapes are incompatible")
    error = truth[..., None] - forecast
    pinball = np.maximum(levels * error, (levels - 1.0) * error)
    weights = np.ones(len(levels), dtype="float64")
    weights /= weights.sum()
    median_index = int(np.argmin(np.abs(levels - 0.5)))
    median = forecast[..., median_index]
    point_error = median - truth
    denominator = float(np.abs(truth).sum())
    coverage: dict[str, float] = {}
    widths: dict[str, float] = {}
    calibration_errors: list[float] = []
    for lower, upper in ((0.05, 0.95), (0.10, 0.90)):
        lo_index = int(np.argmin(np.abs(levels - lower)))
        hi_index = int(np.argmin(np.abs(levels - upper)))
        nominal = upper - lower
        observed = float(
            np.mean((truth >= forecast[..., lo_index]) & (truth <= forecast[..., hi_index]))
        )
        key = int(round(nominal * 100))
        coverage[f"coverage_{key}"] = observed
        widths[f"mean_width_{key}"] = float(
            np.mean(forecast[..., hi_index] - forecast[..., lo_index])
        )
        calibration_errors.append(abs(observed - nominal))
    result: dict[str, Any] = {
        "n": int(truth.size),
        "mae": float(np.mean(np.abs(point_error))),
        "rmse": float(np.sqrt(np.mean(np.square(point_error)))),
        "mean_error": float(np.mean(point_error)),
        "wape": float(np.abs(point_error).sum() / denominator) if denominator else None,
        "mean_weighted_pinball": float(np.mean(np.sum(pinball * weights, axis=-1))),
        "mean_pinball": float(pinball.mean()),
        "crps_quantile_approximation": float(2.0 * pinball.mean()),
        "absolute_calibration_error": float(np.mean(calibration_errors)),
        "quantile_crossing_count": crossing_count(predicted),
        "quantile_crossing_rate": float(
            np.mean(np.any(np.diff(np.asarray(predicted), axis=-1) < 0, axis=-1))
        ),
        "pinball_by_quantile": {
            f"q{int(round(level * 100)):02d}": float(pinball[..., index].mean())
            for index, level in enumerate(levels)
        },
        **coverage,
        **widths,
    }
    if spike_threshold is not None:
        high_index = int(np.argmin(np.abs(levels - 0.95)))
        observed_spike = truth >= float(spike_threshold)
        predicted_spike = forecast[..., high_index] >= float(spike_threshold)
        true_positive = int(np.sum(observed_spike & predicted_spike))
        false_positive = int(np.sum(~observed_spike & predicted_spike))
        false_negative = int(np.sum(observed_spike & ~predicted_spike))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        result.update(
            {
                "spike_threshold_training_only": float(spike_threshold),
                "spike_precision": float(precision),
                "spike_recall": float(recall),
                "spike_f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
            }
        )
    return result


def negative_binomial_quantiles(
    mean: np.ndarray,
    dispersion: np.ndarray,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> np.ndarray:
    mu = np.asarray(mean, dtype="float64").clip(1e-8)
    size = np.asarray(dispersion, dtype="float64").clip(1e-8)
    probability = size / (size + mu)
    values = [stats.nbinom.ppf(float(q), size, probability) for q in quantiles]
    return np.moveaxis(np.asarray(values, dtype="float64"), 0, -1)


def negative_binomial_metrics(
    actual: np.ndarray,
    mean: np.ndarray,
    dispersion: np.ndarray,
) -> dict[str, float | int]:
    truth = np.asarray(actual, dtype="float64").clip(0)
    mu = np.asarray(mean, dtype="float64").clip(1e-8)
    size = np.asarray(dispersion, dtype="float64").clip(1e-8)
    probability = size / (size + mu)
    nll = -stats.nbinom.logpmf(np.rint(truth), size, probability)
    finite = np.isfinite(nll)
    error = mu - truth
    saturated = np.where(truth > 0, truth * np.log(np.maximum(truth, 1e-12) / mu), 0.0)
    deviance = 2.0 * (saturated - (truth + size) * np.log((truth + size) / (mu + size)))
    return {
        "n": int(truth.size),
        "negative_binomial_nll": float(nll[finite].mean()),
        "negative_binomial_deviance": float(np.mean(np.maximum(deviance, 0.0))),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def positive_volume_quantiles(
    count_mean: np.ndarray,
    count_dispersion: np.ndarray,
    parameters: dict[str, np.ndarray],
    *,
    distribution: str,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> np.ndarray:
    mu = np.asarray(count_mean, dtype="float64").clip(1e-8)
    size = np.asarray(count_dispersion, dtype="float64").clip(1e-8)
    p_zero = np.power(size / (size + mu), size)
    output = []
    for q in quantiles:
        conditional_q = np.clip((float(q) - p_zero) / np.maximum(1.0 - p_zero, 1e-12), 0, 1)
        if distribution == "lognormal":
            values = stats.lognorm.ppf(
                conditional_q,
                s=np.asarray(parameters["sigma"], dtype="float64").clip(1e-8),
                scale=np.exp(np.asarray(parameters["log_mu"], dtype="float64")),
            )
        elif distribution == "gamma":
            values = stats.gamma.ppf(
                conditional_q,
                a=np.asarray(parameters["shape"], dtype="float64").clip(1e-8),
                scale=1.0 / np.asarray(parameters["rate"], dtype="float64").clip(1e-8),
            )
        else:
            raise ValueError("volume distribution must be lognormal or gamma")
        output.append(np.where(float(q) <= p_zero, 0.0, np.nan_to_num(values, posinf=0.0)))
    return rearrange_quantiles(np.moveaxis(np.asarray(output), 0, -1))


def positive_volume_nll(
    actual: np.ndarray,
    parameters: dict[str, np.ndarray],
    *,
    distribution: str,
) -> float:
    truth = np.asarray(actual, dtype="float64")
    positive = truth > 0
    if not positive.any():
        return 0.0
    if distribution == "lognormal":
        log_probability = stats.lognorm.logpdf(
            truth[positive],
            s=np.asarray(parameters["sigma"])[positive].clip(1e-8),
            scale=np.exp(np.asarray(parameters["log_mu"])[positive]),
        )
    elif distribution == "gamma":
        log_probability = stats.gamma.logpdf(
            truth[positive],
            a=np.asarray(parameters["shape"])[positive].clip(1e-8),
            scale=1.0 / np.asarray(parameters["rate"])[positive].clip(1e-8),
        )
    else:
        raise ValueError("volume distribution must be lognormal or gamma")
    finite = np.isfinite(log_probability)
    return float(-log_probability[finite].mean())
