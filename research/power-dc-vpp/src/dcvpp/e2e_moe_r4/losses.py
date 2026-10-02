"""Solver-free R4 differentiable objectives and raw-unit capability metrics.

Training surrogates in this module never replace the independent runtime audit.
In particular, formal CVaR is recomputed from the twenty particle rollout
costs; it is not inferred by a free scalar head.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


SCENARIO_COUNT = 20
MIN_DISTRIBUTION_SAMPLES = 2


def normalize_weights(weights: Tensor) -> Tensor:
    """Validate and normalize a variable-size Monte Carlo sample axis.

    ``SCENARIO_COUNT`` remains the legacy N=20 compatibility setting.  The V6
    continuous distribution is not parameterized by that value: N is read
    from ``weights.shape[1]`` and may differ between training, cheap
    validation, formal evaluation, and sampling-budget ablations.
    """

    if weights.ndim != 2 or weights.shape[-1] < MIN_DISTRIBUTION_SAMPLES:
        raise ValueError("weights must have shape [batch, N] with N >= 2")
    total = weights.sum(dim=-1, keepdim=True)
    finite = torch.isfinite(weights).all()
    nonnegative = ~(weights < 0).any()
    positive_mass = (total > 0).all()
    # Keep validation on device until one combined decision.  The previous
    # three independent Python truth tests forced three CUDA synchronizations
    # every time a scenario score normalized the same learned weights.
    if not (finite & nonnegative & positive_mass):
        # Preserve the public error distinction on the exceptional path; the
        # extra synchronizations here do not affect valid training batches.
        if not (finite & nonnegative):
            raise ValueError("scenario weights must be finite and nonnegative")
        raise ValueError("each scenario-weight row must have positive mass")
    return weights / total


def weighted_cvar(costs: Tensor, weights: Tensor, alpha: float = 0.95) -> Tensor:
    """Exact forward CVaR of a weighted empirical distribution for any N."""

    if costs.ndim != 2 or costs.shape[-1] < MIN_DISTRIBUTION_SAMPLES:
        raise ValueError("costs must have shape [batch, N] with N >= 2")
    if costs.shape != weights.shape:
        raise ValueError("costs and weights must have identical shapes")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if not torch.isfinite(costs).all():
        raise ValueError("particle costs must be finite")
    weights = normalize_weights(weights).to(costs)
    order = torch.argsort(costs, dim=-1)
    sorted_costs = torch.gather(costs, 1, order)
    sorted_weights = torch.gather(weights, 1, order)
    lower = torch.as_tensor(alpha, dtype=costs.dtype, device=costs.device)
    previous_cdf = torch.cat(
        (torch.zeros_like(sorted_weights[:, :1]), sorted_weights.cumsum(1)[:, :-1]),
        dim=1,
    )
    upper_cdf = previous_cdf + sorted_weights
    tail_mass = (upper_cdf - torch.maximum(previous_cdf, lower)).clamp_min(0.0)
    return (tail_mass * sorted_costs).sum(dim=-1) / (1.0 - alpha)


def differentiable_cvar(
    costs: Tensor,
    weights: Tensor,
    eta: Tensor,
    alpha: float = 0.95,
) -> Tensor:
    """Rockafellar--Uryasev surrogate used only for gradient training."""

    if costs.ndim != 2 or costs.shape[-1] < MIN_DISTRIBUTION_SAMPLES:
        raise ValueError("costs must have shape [batch, N] with N >= 2")
    if costs.shape != weights.shape:
        raise ValueError("costs and weights must have identical shapes")
    weights = normalize_weights(weights).to(costs)
    eta = eta.reshape(costs.shape[0]).to(costs)
    excess = F.relu(costs - eta[:, None])
    return eta + (weights * excess).sum(dim=-1) / (1.0 - alpha)


def weighted_energy_score(
    particles: Tensor,
    weights: Tensor,
    target: Tensor,
) -> Tensor:
    """Weighted joint-trajectory energy score, returned per batch member."""

    weights = _check_particles(particles, weights, target)
    flat = particles.flatten(2)
    observation = target.flatten(1)
    first = (
        weights
        * torch.linalg.vector_norm(flat - observation[:, None, :], dim=-1)
    ).sum(-1)
    pairwise = torch.cdist(flat, flat)
    second = 0.5 * (
        weights[:, :, None] * weights[:, None, :] * pairwise
    ).sum((1, 2))
    return first - second


def weighted_marginal_crps(
    particles: Tensor,
    weights: Tensor,
    target: Tensor,
) -> Tensor:
    """Weighted empirical marginal CRPS, returned per batch member."""

    weights = _check_particles(particles, weights, target)
    flat = particles.flatten(2)
    observation = target.flatten(1)
    first = (weights[:, :, None] * (flat - observation[:, None]).abs()).sum(1)

    # For sorted x_i with mass w_i,
    #
    #   1/2 sum_ij w_i w_j |x_i-x_j|
    #       = sum_i w_i x_i (2 F_i - w_i - sum_j w_j).
    #
    # This is the same weighted empirical CRPS, but it avoids materializing
    # [batch,20,20,96,21].  At B=512 that obsolete pairwise tensor alone was
    # about 1.54 GiB in fp32 (before autograd storage).  The sorted form is
    # O(B * 20 * features) memory instead of O(B * 20**2 * features).
    sorted_samples, order = flat.sort(dim=1)
    sorted_weights = weights[:, :, None].expand_as(flat).gather(1, order)
    cumulative = sorted_weights.cumsum(dim=1)
    total_mass = weights.sum(dim=1, keepdim=True)[:, :, None]
    second = (
        sorted_weights
        * sorted_samples
        * (2.0 * cumulative - sorted_weights - total_mass)
    ).sum(1)
    return (first - second).mean(1)


def weighted_variogram_score(
    particles: Tensor,
    weights: Tensor,
    target: Tensor,
    *,
    lags: tuple[int, ...] = (1, 4, 16),
    power: float = 0.5,
) -> Tensor:
    """Exact temporal weighted variogram score for formal evaluation.

    This is the literal score registered by the capability gate.  For
    ``power < 1`` its derivative is singular when a generated increment is
    exactly zero, so gradient training must use
    :func:`weighted_variogram_score_stable` instead.  Formal evaluation runs
    without backward and therefore has no reason to perturb the registered
    score at ties.
    """

    return _weighted_variogram_score(
        particles,
        weights,
        target,
        lags=lags,
        power=power,
        stable_epsilon=None,
    )


def weighted_variogram_score_stable(
    particles: Tensor,
    weights: Tensor,
    target: Tensor,
    *,
    lags: tuple[int, ...] = (1, 4, 16),
    power: float = 0.5,
    epsilon: float = 1.0e-6,
) -> Tensor:
    """Finite-gradient variogram surrogate used only during training.

    The zero-centred Charbonnier continuation has value zero at coincident
    increments and converges to the exact ``abs(delta) ** power`` away from
    the declared ``epsilon`` neighbourhood.  Keeping this API separate from
    :func:`weighted_variogram_score` prevents the formal evaluator from
    silently reporting the numerical training continuation as the exact
    scientific metric.
    """

    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be one finite positive scalar")
    return _weighted_variogram_score(
        particles,
        weights,
        target,
        lags=lags,
        power=power,
        stable_epsilon=float(epsilon),
    )


def _weighted_variogram_score(
    particles: Tensor,
    weights: Tensor,
    target: Tensor,
    *,
    lags: tuple[int, ...],
    power: float,
    stable_epsilon: float | None,
) -> Tensor:
    """Shared exact/stable variogram implementation."""

    weights = _check_particles(particles, weights, target)
    return _weighted_variogram_score_normalized(
        particles,
        weights,
        target,
        lags=lags,
        power=power,
        stable_epsilon=stable_epsilon,
    )


def _weighted_variogram_score_normalized(
    particles: Tensor,
    weights: Tensor,
    target: Tensor,
    *,
    lags: tuple[int, ...],
    power: float,
    stable_epsilon: float | None,
) -> Tensor:
    """Variogram core for already shape-checked normalized weights."""

    if particles.ndim < 4:
        raise ValueError("particles need an explicit temporal dimension")
    if not math.isfinite(float(power)) or float(power) <= 0.0:
        raise ValueError("power must be one finite positive scalar")
    values: list[Tensor] = []
    for lag in lags:
        if int(lag) <= 0:
            raise ValueError("variogram lags must be positive integers")
        if lag >= particles.shape[2]:
            continue
        realized = (target[:, lag:] - target[:, :-lag]).abs().pow(power)
        generated_delta = particles[:, :, lag:] - particles[:, :, :-lag]
        if stable_epsilon is not None and power < 1.0:
            epsilon = float(stable_epsilon)
            generated = (
                generated_delta.square() + epsilon**2
            ).pow(power / 2.0) - epsilon**power
        else:
            generated = generated_delta.abs().pow(power)
        expected = (weights[:, :, None, None] * generated).sum(1)
        values.append((realized - expected).square().flatten(1).mean(1))
    if not values:
        return particles.flatten(1).sum(1) * 0.0
    return torch.stack(values, dim=1).mean(1)


def empirical_coverage(
    particles: Tensor,
    weights: Tensor,
    target: Tensor,
    level: float,
) -> tuple[Tensor, Tensor]:
    """Return per-batch coverage and interval width for a central interval."""

    normalized_weights = _check_particles(particles, weights, target)
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    lower, upper = _weighted_quantiles(
        particles,
        normalized_weights,
        ((1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0),
    )
    coverage = ((target >= lower) & (target <= upper)).float().flatten(1).mean(1)
    width = (upper - lower).flatten(1).mean(1)
    return coverage, width


def weighted_interval_score(
    particles: Tensor,
    weights: Tensor,
    target: Tensor,
    level: float,
) -> Tensor:
    """Proper central-interval score, returned per batch member.

    The interval endpoints use the same discrete weighted quantiles as the
    formal coverage audit.  Sorting and endpoint selection are piecewise
    differentiable with respect to the selected particles, while the interval
    score supplies gradients both inside and outside the interval.  This is a
    training surrogate; :func:`empirical_coverage` remains the exact reported
    formal metric.
    """

    normalized_weights = _check_particles(particles, weights, target)
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    alpha = 1.0 - float(level)
    lower, upper = _weighted_quantiles(
        particles,
        normalized_weights,
        (alpha / 2.0, 1.0 - alpha / 2.0),
    )
    width = upper - lower
    score = (
        width
        + (2.0 / alpha) * F.relu(lower - target)
        + (2.0 / alpha) * F.relu(target - upper)
    )
    return score.flatten(1).mean(1)


def smooth_empirical_coverage(
    particles: Tensor,
    weights: Tensor,
    target: Tensor,
    level: float,
    *,
    temperature: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """Differentiable central-interval coverage surrogate.

    All inputs are expected to be in the frozen residual-score space shared
    with the formal gate.  ``temperature`` is therefore dimensionless.  The
    product of two logistic soft indicators approaches the exact inside-
    interval indicator as the temperature tends to zero and has gradients
    with respect to both selected interval boundaries and the target.
    """

    normalized_weights = _check_particles(particles, weights, target)
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be one finite positive scalar")
    alpha = 1.0 - float(level)
    lower, upper = _weighted_quantiles(
        particles,
        normalized_weights,
        (alpha / 2.0, 1.0 - alpha / 2.0),
    )
    tau = float(temperature)
    soft_inside = torch.sigmoid((target - lower) / tau) * torch.sigmoid(
        (upper - target) / tau
    )
    coverage = soft_inside.flatten(1).mean(1)
    width = (upper - lower).flatten(1).mean(1)
    return coverage, width


def weighted_quantile(samples: Tensor, weights: Tensor, quantile: float) -> Tensor:
    """Discrete weighted quantile over a variable Monte Carlo sample axis."""

    if samples.ndim < 3 or samples.shape[1] < MIN_DISTRIBUTION_SAMPLES:
        raise ValueError("samples must have shape [batch, N, ...] with N >= 2")
    if weights.shape != samples.shape[:2]:
        raise ValueError("weights must match the first two sample dimensions")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    weights = normalize_weights(weights).to(samples)
    return _weighted_quantiles(samples, weights, (quantile,))[0]


def _weighted_quantiles(
    samples: Tensor,
    normalized_weights: Tensor,
    quantiles: tuple[float, ...],
) -> tuple[Tensor, ...]:
    """Evaluate several quantiles from one scenario-axis sort.

    ``normalized_weights`` is deliberately an internal precondition.  It lets
    central-interval coverage share both the sort and weight validation for
    its lower/upper endpoints while the public ``weighted_quantile`` contract
    remains unchanged.
    """

    flat = samples.flatten(2)
    sorted_samples, indices = flat.sort(dim=1)
    sorted_weights = (
        normalized_weights[:, :, None].expand_as(flat).gather(1, indices)
    )
    cumulative = sorted_weights.cumsum(1)
    output_shape = samples.shape[:1] + samples.shape[2:]
    results: list[Tensor] = []
    for quantile in quantiles:
        positions = (cumulative < quantile).sum(1).clamp_max(samples.shape[1] - 1)
        result = sorted_samples.gather(1, positions[:, None]).squeeze(1)
        results.append(result.reshape(output_shape))
    return tuple(results)


def effective_sample_size(weights: Tensor) -> Tensor:
    weights = normalize_weights(weights)
    return 1.0 / weights.square().sum(dim=-1)


def mean_pairwise_particle_distance(particles: Tensor) -> Tensor:
    if particles.ndim < 3 or particles.shape[1] < MIN_DISTRIBUTION_SAMPLES:
        raise ValueError("particles must have shape [batch, N, ...] with N >= 2")
    flat = particles.flatten(2)
    distance = torch.cdist(flat, flat) / math.sqrt(max(flat.shape[-1], 1))
    mask = ~torch.eye(
        particles.shape[1], dtype=torch.bool, device=particles.device
    )
    return distance[:, mask].mean(1)


def uniform_sample_weights(samples: Tensor) -> Tensor:
    """Return equal weights without coupling model capacity to sample count."""

    if samples.ndim < 3 or samples.shape[1] < MIN_DISTRIBUTION_SAMPLES:
        raise ValueError("samples must have shape [batch, N, ...] with N >= 2")
    return samples.new_full(
        samples.shape[:2], 1.0 / float(samples.shape[1])
    )


def weighted_tail_pinball_score(
    samples: Tensor,
    weights: Tensor,
    target: Tensor,
    *,
    quantiles: tuple[float, ...] = (0.05, 0.95),
) -> Tensor:
    """Proper multi-quantile tail score, reduced to one scalar per day."""

    if not quantiles or any(not 0.0 < float(q) < 1.0 for q in quantiles):
        raise ValueError("tail quantiles must lie strictly inside (0, 1)")
    values: list[Tensor] = []
    for quantile in quantiles:
        forecast = weighted_quantile(samples, weights, float(quantile))
        error = target - forecast
        pinball = torch.maximum(
            float(quantile) * error,
            (float(quantile) - 1.0) * error,
        )
        values.append(pinball.flatten(1).mean(1))
    return torch.stack(values, dim=-1).mean(-1)


def smooth_coverage_calibration_loss(
    samples: Tensor,
    weights: Tensor,
    target: Tensor,
    level: float,
    *,
    temperature: float = 0.05,
) -> Tensor:
    """Per-day differentiable absolute calibration error at one level."""

    coverage, _ = smooth_empirical_coverage(
        samples, weights, target, level, temperature=temperature
    )
    return (coverage - float(level)).abs()


def sample_center_scale_score(
    samples: Tensor,
    target: Tensor,
    *,
    center: Tensor | None = None,
    scale: Tensor | None = None,
    minimum_scale: float = 1.0e-3,
) -> Tensor:
    """Laplace proper score for the conditional center/scale warm-up stage.

    Explicit distribution parameters are preferred.  The empirical mean and
    standard deviation are a reparameterized fallback for small architecture
    tests.  Inputs are expected in the same frozen score space as the gate.
    """

    if samples.ndim < 3 or samples.shape[1] < MIN_DISTRIBUTION_SAMPLES:
        raise ValueError("samples must have shape [batch, N, ...] with N >= 2")
    if target.shape != samples.shape[:1] + samples.shape[2:]:
        raise ValueError("target shape must equal samples with sample axis removed")
    if not math.isfinite(float(minimum_scale)) or float(minimum_scale) <= 0.0:
        raise ValueError("minimum_scale must be finite and positive")
    location = samples.mean(1) if center is None else torch.as_tensor(center).to(samples)
    dispersion = (
        samples.std(1, unbiased=False) if scale is None else torch.as_tensor(scale).to(samples)
    )
    if location.shape != target.shape or dispersion.shape != target.shape:
        raise ValueError("center and scale must have target shape")
    dispersion = dispersion.clamp_min(float(minimum_scale))
    score = dispersion.log() + (target - location).abs() / dispersion
    return score.flatten(1).mean(1)


def dart_identity_loss(samples: Tensor) -> Tensor:
    """Raw-unit DA-RT-DART closure loss for zonal V6 samples."""

    if samples.ndim != 4 or samples.shape[-1] != 21:
        raise ValueError("zonal samples must have shape [batch, N, 96, 21]")
    residual = samples[..., 0:4] - samples[..., 4:8] - samples[..., 8:12]
    return residual.square().flatten(1).mean(1)


def diversity_floor_loss(
    samples: Tensor, *, minimum_distance: float = 0.05
) -> Tensor:
    """Collapse penalty with no incentive to inflate an already adequate spread."""

    if not math.isfinite(float(minimum_distance)) or float(minimum_distance) <= 0.0:
        raise ValueError("minimum_distance must be finite and positive")
    distance = mean_pairwise_particle_distance(samples)
    return F.relu(float(minimum_distance) - distance).square()


def distribution_proper_loss_terms(
    samples: Tensor,
    target: Tensor,
    *,
    weights: Tensor | None = None,
    center: Tensor | None = None,
    scale: Tensor | None = None,
    identity_samples: Tensor | None = None,
    coverage_temperature: float = 0.05,
    diversity_floor: float = 0.05,
    validate_inputs: bool = True,
) -> dict[str, Tensor]:
    """Complete V6 Distribution curriculum terms for arbitrary sample N.

    The returned keys are stable across S1--S5.  Curriculum stages change only
    preregistered scalar weights; they never silently change metric formulas.
    """

    sample_weights = uniform_sample_weights(samples) if weights is None else weights
    if validate_inputs:
        sample_weights = _check_particles(samples, sample_weights, target)
    else:
        if (
            samples.ndim < 3
            or samples.shape[1] < MIN_DISTRIBUTION_SAMPLES
            or target.shape != samples.shape[:1] + samples.shape[2:]
            or sample_weights.shape != samples.shape[:2]
        ):
            raise ValueError("invalid variable-N Distribution sample shapes")
        sample_weights = sample_weights / sample_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-12)

    flat = samples.flatten(2)
    observation = target.flatten(1)
    energy_first = (
        sample_weights
        * torch.linalg.vector_norm(flat - observation[:, None], dim=-1)
    ).sum(-1)
    energy_second = 0.5 * (
        sample_weights[:, :, None]
        * sample_weights[:, None, :]
        * torch.cdist(flat, flat)
    ).sum((1, 2))
    energy = energy_first - energy_second

    crps_first = (
        sample_weights[:, :, None] * (flat - observation[:, None]).abs()
    ).sum(1)
    sorted_samples, order = flat.sort(dim=1)
    sorted_weights = sample_weights[:, :, None].expand_as(flat).gather(1, order)
    cumulative = sorted_weights.cumsum(1)
    total_mass = sample_weights.sum(dim=1, keepdim=True)[:, :, None]
    crps_second = (
        sorted_weights
        * sorted_samples
        * (2.0 * cumulative - sorted_weights - total_mass)
    ).sum(1)
    crps = (crps_first - crps_second).mean(1)

    lower90, upper90, lower95, upper95 = _weighted_quantiles(
        samples, sample_weights, (0.05, 0.95, 0.025, 0.975)
    )

    def interval_score(lower: Tensor, upper: Tensor, alpha: float) -> Tensor:
        score = (
            upper
            - lower
            + (2.0 / alpha) * F.relu(lower - target)
            + (2.0 / alpha) * F.relu(target - upper)
        )
        return score.flatten(1).mean(1)

    interval90 = interval_score(lower90, upper90, 0.10)
    interval95 = interval_score(lower95, upper95, 0.05)
    tau = float(coverage_temperature)

    def coverage_error(lower: Tensor, upper: Tensor, level: float) -> Tensor:
        soft_inside = torch.sigmoid((target - lower) / tau) * torch.sigmoid(
            (upper - target) / tau
        )
        return (soft_inside.flatten(1).mean(1) - level).abs()

    tail_values: list[Tensor] = []
    for quantile, forecast in ((0.05, lower90), (0.95, upper90)):
        error = target - forecast
        tail_values.append(
            torch.maximum(
                quantile * error, (quantile - 1.0) * error
            ).flatten(1).mean(1)
        )
    return {
        "center_scale": sample_center_scale_score(
            samples, target, center=center, scale=scale
        ),
        "energy_score": energy,
        "marginal_crps": crps,
        "coverage90": coverage_error(lower90, upper90, 0.90),
        "coverage95": coverage_error(lower95, upper95, 0.95),
        "interval": 0.5 * (interval90 + interval95),
        "tail": torch.stack(tail_values, dim=-1).mean(-1),
        "dependence": _weighted_variogram_score_normalized(
            samples,
            sample_weights,
            target,
            lags=(1, 4, 16),
            power=0.5,
            stable_epsilon=1.0e-6,
        ),
        "diversity": diversity_floor_loss(
            samples, minimum_distance=diversity_floor
        ),
        "identity": dart_identity_loss(
            samples if identity_samples is None else identity_samples
        ),
    }


def router_balance_loss(probabilities: Tensor, top_indices: Tensor) -> Tensor:
    """Switch-style token-router balance loss for eight experts."""

    if probabilities.shape[-1] != 8 or top_indices.shape[-1] != 2:
        raise ValueError("R4 requires eight experts and Top-2 routing")
    probs = probabilities.reshape(-1, 8)
    dispatch = F.one_hot(top_indices.reshape(-1, 2), 8).float().sum(1) / 2.0
    importance = probs.mean(0)
    load = dispatch.mean(0)
    return (8.0 * (importance * load).sum() - 1.0).clamp_min(0.0)


def router_stability_loss(current: Tensor, previous: Tensor | None) -> Tensor:
    if previous is None:
        return current.sum() * 0.0
    if current.shape != previous.shape:
        raise ValueError("adjacent router probabilities must have equal shapes")
    current = current.clamp_min(1.0e-8)
    previous = previous.detach().clamp_min(1.0e-8)
    return F.kl_div(current.log(), previous, reduction="batchmean")


def two_settlement_cost(
    da_schedule: Tensor,
    grid: Tensor,
    da_price: Tensor,
    rt_price: Tensor,
    *,
    interval_hours: float = 0.25,
    additional_cost: Tensor | float = 0.0,
) -> Tensor:
    """Raw-unit daily two-settlement cost; no batch-dependent normalization."""

    da_schedule, grid, da_price, rt_price = torch.broadcast_tensors(
        da_schedule, grid, da_price, rt_price
    )
    cost = (
        da_schedule * da_price + (grid - da_schedule) * rt_price
    ) * interval_hours
    return cost.flatten(1).sum(1) + torch.as_tensor(additional_cost).to(cost)


def fixed_scale_cost_loss(cost: Tensor, training_scale: Tensor | float) -> Tensor:
    """Normalize with one frozen train-only scale, never the current batch."""

    scale = torch.as_tensor(training_scale, dtype=cost.dtype, device=cost.device)
    if scale.numel() != 1 or not torch.isfinite(scale) or scale <= 0:
        raise ValueError("training cost scale must be one finite positive scalar")
    return (cost / scale).mean()


def forecast_raw_metrics(prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
    """Per-day raw-unit MAE/RMSE used by the representation gate."""

    prediction, target = torch.broadcast_tensors(prediction, target)
    error = prediction - target
    return {
        "mae": error.abs().flatten(1).mean(1),
        "rmse": error.square().flatten(1).mean(1).sqrt(),
    }


def compose_weighted_loss(
    terms: Mapping[str, Tensor], weights: Mapping[str, float]
) -> Tensor:
    unknown = set(weights) - set(terms)
    if unknown:
        raise KeyError(f"loss weights have no matching term: {sorted(unknown)}")
    values = [terms[name] * float(weight) for name, weight in weights.items()]
    if not values:
        raise ValueError("at least one weighted term is required")
    return torch.stack([value.mean() for value in values]).sum()


def _check_particles(particles: Tensor, weights: Tensor, target: Tensor) -> Tensor:
    if particles.ndim < 3 or particles.shape[1] < MIN_DISTRIBUTION_SAMPLES:
        raise ValueError("particles must have shape [batch, N, ...] with N >= 2")
    if target.shape != particles.shape[:1] + particles.shape[2:]:
        raise ValueError("target shape must equal particles with scenario axis removed")
    if weights.shape != particles.shape[:2]:
        raise ValueError("weights must match the first two particle dimensions")
    normalized_weights = normalize_weights(weights).to(particles)
    # One host decision instead of one per tensor keeps the valid CUDA path
    # asynchronous for longer without weakening the finite-value contract.
    finite = torch.isfinite(particles).all() & torch.isfinite(target).all()
    if not finite:
        raise ValueError("particles and target must be finite")
    return normalized_weights


__all__ = [
    "SCENARIO_COUNT",
    "MIN_DISTRIBUTION_SAMPLES",
    "compose_weighted_loss",
    "dart_identity_loss",
    "distribution_proper_loss_terms",
    "diversity_floor_loss",
    "differentiable_cvar",
    "effective_sample_size",
    "empirical_coverage",
    "fixed_scale_cost_loss",
    "forecast_raw_metrics",
    "mean_pairwise_particle_distance",
    "normalize_weights",
    "router_balance_loss",
    "router_stability_loss",
    "smooth_empirical_coverage",
    "smooth_coverage_calibration_loss",
    "sample_center_scale_score",
    "two_settlement_cost",
    "weighted_cvar",
    "weighted_energy_score",
    "weighted_interval_score",
    "weighted_marginal_crps",
    "weighted_quantile",
    "weighted_tail_pinball_score",
    "weighted_variogram_score",
    "weighted_variogram_score_stable",
    "uniform_sample_weights",
]
