from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import norm, rankdata

from .contracts import FrozenForecastArtifact, FrozenPhysicalArtifact, ScenarioBundle


SCENARIO_SCHEMA_VERSION = "gaussian_copula_rank_v2"


@dataclass(frozen=True)
class DependenceRegime:
    name: str
    kappa: float


DEFAULT_REGIMES = (
    DependenceRegime("independent", 0.0),
    DependenceRegime("kappa_0_3", 0.3),
    DependenceRegime("kappa_0_6", 0.6),
)


def stable_seed(*parts: object, modulus: int = 2**32 - 1) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def _gaussian_copula_resample(
    pool: np.ndarray,
    rng: np.random.Generator,
    *,
    block_name: str,
    count: int = 100,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rank-fit a shrinkage Gaussian copula and preserve empirical marginals.

    The frozen forecast pools are built from complete validation-residual day
    blocks.  Flattening keeps every zone/horizon coordinate in the same block;
    the efficient factor representation samples the shrinkage covariance
    without forming or factorizing an unstable d-by-d matrix.
    """

    values = np.asarray(pool, dtype=float)
    if values.ndim < 2 or len(values) < count:
        raise ValueError(
            f"{block_name} scenario marginal contains {len(values)} rows; {count} required"
        )
    flat = values.reshape(len(values), -1)
    n, dimension = flat.shape
    ranks = rankdata(flat, axis=0, method="average")
    gaussian = norm.ppf((ranks - 0.5) / n)
    gaussian -= gaussian.mean(axis=0, keepdims=True)
    scale = gaussian.std(axis=0, ddof=1, keepdims=True)
    constant = scale[0] < 1e-12
    scale[:, constant] = 1.0
    gaussian /= scale
    gaussian[:, constant] = 0.0
    regularization = float(0.05 + 0.20 * dimension / (n + dimension))
    common_scores = rng.standard_normal((count, n))
    independent = rng.standard_normal((count, dimension))
    latent = (
        np.sqrt(1.0 - regularization)
        * common_scores
        @ gaussian
        / np.sqrt(max(n - 1, 1))
        + np.sqrt(regularization) * independent
    )

    generated = np.empty((count, dimension), dtype=float)
    marginal_error = 0.0
    for column in range(dimension):
        target_order = np.argsort(latent[:, column], kind="stable")
        if n == count:
            marginal = np.sort(flat[:, column], kind="stable")
        else:
            probabilities = (np.arange(count, dtype=float) + 0.5) / count
            marginal = np.quantile(flat[:, column], probabilities, method="linear")
        generated[target_order, column] = marginal
        reference_quantiles = np.quantile(flat[:, column], (0.05, 0.5, 0.95))
        generated_quantiles = np.quantile(
            generated[:, column], (0.05, 0.5, 0.95)
        )
        denominator = max(float(np.ptp(flat[:, column])), 1e-12)
        marginal_error = max(
            marginal_error,
            float(np.max(np.abs(reference_quantiles - generated_quantiles)))
            / denominator,
        )

    sampled_dimensions = np.linspace(
        0, dimension - 1, min(dimension, 32), dtype=int
    )
    source_ranks = rankdata(flat[:, sampled_dimensions], axis=0)
    generated_ranks = rankdata(generated[:, sampled_dimensions], axis=0)
    varying = (np.std(source_ranks, axis=0) > 1e-12) & (
        np.std(generated_ranks, axis=0) > 1e-12
    )
    if int(np.sum(varying)) >= 2:
        source_rank_correlation = np.corrcoef(source_ranks[:, varying], rowvar=False)
        generated_rank_correlation = np.corrcoef(
            generated_ranks[:, varying], rowvar=False
        )
    else:
        source_rank_correlation = np.asarray([[1.0]])
        generated_rank_correlation = np.asarray([[1.0]])
    correlation_error = float(
        np.nanmean(np.abs(source_rank_correlation - generated_rank_correlation))
    )
    return generated.reshape((count, *values.shape[1:])), {
        "method": "gaussian_copula_from_validation_residual_ranks",
        "block": block_name,
        "effective_sample_size": int(n),
        "dimension": int(dimension),
        "regularization": "dimension_aware_diagonal_shrinkage",
        "regularization_strength": regularization,
        "temporal_blocks_preserved": True,
        "empirical_inverse_cdf": True,
        "maximum_normalized_q05_q50_q95_marginal_error": marginal_error,
        "sampled_rank_correlation_mean_absolute_error": correlation_error,
    }


def _adverse_reorder(
    block: np.ndarray,
    adverse_score: np.ndarray,
    common: np.ndarray,
    kappa: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Reorder complete rows, preserving the empirical marginal exactly."""

    if not 0.0 <= kappa <= 1.0:
        raise ValueError("kappa must be between zero and one")
    noise = rng.standard_normal(len(block))
    latent = kappa * common + np.sqrt(max(0.0, 1.0 - kappa**2)) * noise
    target_order = np.argsort(latent, kind="stable")
    source_order = np.argsort(adverse_score, kind="stable")
    reordered = np.empty_like(block)
    reordered[target_order] = block[source_order]
    return reordered


def _standardized_features(
    rt_price: np.ndarray,
    workload: np.ndarray,
    pv: np.ndarray,
    physical_multiplier: np.ndarray,
) -> np.ndarray:
    # Hourly/zone detail is retained. Equal block scaling prevents a large block
    # from dominating the Euclidean reduction solely because it has more cells.
    blocks = []
    for values in (rt_price, workload, pv, physical_multiplier):
        flat = values.reshape(len(values), -1).astype(float)
        center = flat.mean(axis=0)
        scale = flat.std(axis=0)
        scale[scale < 1e-12] = 1.0
        normalized = (flat - center) / scale
        blocks.append(normalized / np.sqrt(normalized.shape[1]))
    return np.concatenate(blocks, axis=1)


def _farthest_first_reduce(features: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n = len(features)
    if not 1 <= count <= n:
        raise ValueError("reduced scenario count must not exceed the initial count")
    centroid = features.mean(axis=0)
    first = int(np.argmin(np.sum((features - centroid) ** 2, axis=1)))
    selected = [first]
    distance = np.sum((features - features[first]) ** 2, axis=1)
    distance[first] = -1.0
    while len(selected) < count:
        candidate = int(np.argmax(distance))
        selected.append(candidate)
        candidate_distance = np.sum((features - features[candidate]) ** 2, axis=1)
        distance = np.minimum(distance, candidate_distance)
        distance[np.asarray(selected, dtype=int)] = -1.0
    representatives = features[np.asarray(selected)]
    squared = ((features[:, None, :] - representatives[None, :, :]) ** 2).sum(axis=2)
    mapping = np.argmin(squared, axis=1)
    weights = np.bincount(mapping, minlength=count).astype(float) / n
    mse = float(np.mean(np.min(squared, axis=1)))
    return np.asarray(selected, dtype=int), mapping.astype(int), weights, mse


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probability: float
) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(sorted_weights) / sorted_weights.sum()
    return float(sorted_values[min(np.searchsorted(cumulative, probability), len(values) - 1)])


def _weighted_correlation(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float) / np.sum(weights)
    center = np.sum(weights[:, None] * values, axis=0)
    centered = values - center
    covariance = (centered * weights[:, None]).T @ centered
    scale = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    return covariance / np.outer(scale, scale)


def _lag_one_autocorrelation(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape[1] < 2:
        return np.zeros(len(values), dtype=float)
    left = values[:, :-1] - values[:, :-1].mean(axis=1, keepdims=True)
    right = values[:, 1:] - values[:, 1:].mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(left**2, axis=1) * np.sum(right**2, axis=1))
    return np.divide(
        np.sum(left * right, axis=1),
        denominator,
        out=np.zeros(len(values), dtype=float),
        where=denominator > 1e-12,
    )


def _longest_event_duration(mask: np.ndarray) -> np.ndarray:
    durations = np.zeros(len(mask), dtype=float)
    for index, row in enumerate(np.asarray(mask, dtype=bool)):
        current = 0
        longest = 0
        for value in row:
            current = current + 1 if value else 0
            longest = max(longest, current)
        durations[index] = longest
    return durations


def _small_nomination_gap(
    physical: FrozenPhysicalArtifact,
    full_workload: np.ndarray,
    full_pv: np.ndarray,
    full_physical: np.ndarray,
    reduced_workload: np.ndarray,
    reduced_pv: np.ndarray,
    reduced_physical: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    def net_load(
        workload: np.ndarray, pv: np.ndarray, multiplier: np.ndarray
    ) -> np.ndarray:
        idle = np.sum(multiplier * physical.idle_mw[None, :], axis=1)
        marginal = np.mean(
            multiplier * physical.work_mw_per_unit[None, :], axis=1
        )
        return np.maximum(
            idle[:, None]
            + marginal[:, None] * workload
            - pv.sum(axis=1),
            0.0,
        )

    full = net_load(full_workload, full_pv, full_physical)
    reduced = net_load(reduced_workload, reduced_pv, reduced_physical)
    full_schedule = np.quantile(full, 0.5, axis=0)
    reduced_schedule = np.asarray(
        [
            _weighted_quantile(reduced[:, step], probabilities, 0.5)
            for step in range(reduced.shape[1])
        ]
    )
    oracle_objective = float(np.mean(np.abs(full - full_schedule[None, :])))
    reduced_decision_objective = float(
        np.mean(np.abs(full - reduced_schedule[None, :]))
    )
    gap = reduced_decision_objective - oracle_objective
    return {
        "method": "small_l1_net_load_nomination_problem",
        "full_distribution_optimal_objective_mean_absolute_mw": oracle_objective,
        "reduced_decision_evaluated_on_full_objective_mean_absolute_mw": reduced_decision_objective,
        "decision_value_gap_mean_absolute_mw": gap,
        "decision_value_gap_fraction": gap / max(oracle_objective, 1e-12),
        "counterfactual_proxy": True,
    }


def _reduction_diagnostics(
    physical: FrozenPhysicalArtifact,
    forecast: FrozenForecastArtifact,
    *,
    full_rt: np.ndarray,
    full_workload: np.ndarray,
    full_pv: np.ndarray,
    full_physical: np.ndarray,
    selected: np.ndarray,
    probabilities: np.ndarray,
    reduction_mse: float,
    copula: Mapping[str, Any],
) -> dict[str, Any]:
    reduced_rt = full_rt[selected]
    reduced_workload = full_workload[selected]
    reduced_pv = full_pv[selected]
    reduced_physical = full_physical[selected]
    full_features = _standardized_features(
        full_rt, full_workload, full_pv, full_physical
    )
    reduced_features = full_features[selected]
    distances = np.sqrt(
        np.maximum(
            ((full_features[:, None, :] - reduced_features[None, :, :]) ** 2).sum(
                axis=2
            ),
            0.0,
        )
    )
    representative_distances = np.sqrt(
        np.maximum(
            (
                (
                    reduced_features[:, None, :]
                    - reduced_features[None, :, :]
                )
                ** 2
            ).sum(axis=2),
            0.0,
        )
    )
    energy_score = float(
        np.mean(distances @ probabilities)
        - 0.5
        * np.sum(
            probabilities[:, None]
            * probabilities[None, :]
            * representative_distances
        )
    )

    def summary(rt: np.ndarray, workload: np.ndarray, pv: np.ndarray, multiplier: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                rt.mean(axis=1),
                workload,
                pv.mean(axis=1),
                np.repeat(multiplier.mean(axis=1)[:, None], workload.shape[1], axis=1),
            ]
        )

    full_summary = summary(full_rt, full_workload, full_pv, full_physical)
    reduced_summary = full_summary[selected]
    dimension = full_summary.shape[1]
    pair_rng = np.random.default_rng(20260809)
    left = pair_rng.integers(0, dimension, size=min(512, dimension * 2))
    right = pair_rng.integers(0, dimension, size=len(left))
    valid = left != right
    left, right = left[valid], right[valid]
    full_variogram = np.mean(
        np.abs(full_summary[:, left] - full_summary[:, right]) ** 0.5, axis=0
    )
    reduced_variogram = np.sum(
        probabilities[:, None]
        * np.abs(reduced_summary[:, left] - reduced_summary[:, right]) ** 0.5,
        axis=0,
    )
    variogram_score = float(np.mean((full_variogram - reduced_variogram) ** 2))

    full_blocks = (
        full_rt.mean(axis=(1, 2)),
        full_workload.mean(axis=1),
        -full_pv.mean(axis=(1, 2)),
        full_physical.mean(axis=1),
    )
    reduced_blocks = tuple(block[selected] for block in full_blocks)
    full_ranks = np.column_stack([rankdata(block) for block in full_blocks])
    reduced_ranks = np.column_stack([rankdata(block) for block in reduced_blocks])
    full_varying = np.std(full_ranks, axis=0) > 1e-12
    full_rank_correlation = np.eye(full_ranks.shape[1], dtype=float)
    if int(np.sum(full_varying)) >= 2:
        full_rank_correlation[np.ix_(full_varying, full_varying)] = np.corrcoef(
            full_ranks[:, full_varying], rowvar=False
        )
    reduced_rank_correlation = _weighted_correlation(reduced_ranks, probabilities)
    rank_error = float(
        np.nanmean(np.abs(full_rank_correlation - reduced_rank_correlation))
    )

    full_temporal = (
        full_rt.mean(axis=1),
        full_workload,
        full_pv.mean(axis=1),
    )
    reduced_temporal = tuple(block[selected] for block in full_temporal)
    autocorrelation_by_block: dict[str, Any] = {}
    for name, full_block, reduced_block in zip(
        ("rt_price", "workload", "pv"), full_temporal, reduced_temporal
    ):
        full_acf = float(np.mean(_lag_one_autocorrelation(full_block)))
        reduced_acf = float(
            np.dot(probabilities, _lag_one_autocorrelation(reduced_block))
        )
        autocorrelation_by_block[name] = {
            "full_lag1": full_acf,
            "reduced_lag1": reduced_acf,
            "absolute_error": abs(full_acf - reduced_acf),
        }
    autocorrelation_error = float(
        np.mean(
            [record["absolute_error"] for record in autocorrelation_by_block.values()]
        )
    )

    flat_full = np.concatenate(
        [
            full_rt.reshape(len(full_rt), -1),
            full_workload.reshape(len(full_workload), -1),
            full_pv.reshape(len(full_pv), -1),
            full_physical.reshape(len(full_physical), -1),
        ],
        axis=1,
    )
    flat_reduced = flat_full[selected]
    lower, upper = np.quantile(flat_full, (0.05, 0.95), axis=0)
    marginal_coverage = float(
        np.mean(
            np.sum(
                probabilities[:, None]
                * ((flat_reduced >= lower) & (flat_reduced <= upper)),
                axis=0,
            )
        )
    )
    pinball_scores = []
    for quantile in (0.05, 0.50, 0.95):
        predicted = np.asarray(
            [
                _weighted_quantile(flat_reduced[:, column], probabilities, quantile)
                for column in range(flat_reduced.shape[1])
            ]
        )
        error = flat_full - predicted[None, :]
        pinball_scores.append(
            float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))
        )

    full_spread = full_rt.mean(axis=1) - forecast.da_price.mean(axis=0)[None, :]
    reduced_spread = full_spread[selected]
    full_pv_time = full_pv.mean(axis=1)
    reduced_pv_time = full_pv_time[selected]
    workload_threshold = float(np.quantile(full_workload, 0.80))
    spread_threshold = float(np.quantile(full_spread, 0.80))
    pv_threshold = float(np.quantile(full_pv_time, 0.20))
    full_events = (
        (full_workload >= workload_threshold)
        & (full_spread >= spread_threshold)
        & (full_pv_time <= pv_threshold)
    )
    reduced_events = (
        (reduced_workload >= workload_threshold)
        & (reduced_spread >= spread_threshold)
        & (reduced_pv_time <= pv_threshold)
    )
    full_rate = float(full_events.mean())
    reduced_rate = float(
        np.dot(probabilities, reduced_events.mean(axis=1))
    )
    full_duration = float(np.mean(_longest_event_duration(full_events)))
    reduced_duration = float(
        np.dot(probabilities, _longest_event_duration(reduced_events))
    )
    physical_center = full_physical.mean(axis=0)
    physical_dispersion = float(
        np.mean(np.std(full_physical, axis=0, ddof=0))
    )
    physical_extrapolation = float(
        np.mean(
            (reduced_physical < full_physical.min(axis=0))
            | (reduced_physical > full_physical.max(axis=0))
        )
    )
    return {
        "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
        "within_domain_dependence": dict(copula),
        "cross_domain_nominal_coupling": "product_coupling_after_separate_within_domain_rank_copulas",
        "marginal_coverage_90": marginal_coverage,
        "mean_quantile_score_q05_q50_q95": float(np.mean(pinball_scores)),
        "quantile_scores": {
            "q05": pinball_scores[0],
            "q50": pinball_scores[1],
            "q95": pinball_scores[2],
        },
        "energy_score": energy_score,
        "variogram_score": variogram_score,
        "rank_correlation_mean_absolute_error": rank_error,
        "autocorrelation_mean_absolute_error": autocorrelation_error,
        "autocorrelation_by_block": autocorrelation_by_block,
        "extreme_event": {
            "definition": "high_workload_and_high_rt_spread_and_low_pv_using_full_pool_80_80_20_thresholds",
            "full_cooccurrence_rate": full_rate,
            "reduced_cooccurrence_rate": reduced_rate,
            "absolute_rate_error": abs(full_rate - reduced_rate),
            "full_expected_longest_duration_intervals": full_duration,
            "reduced_expected_longest_duration_intervals": reduced_duration,
            "absolute_duration_error_intervals": abs(full_duration - reduced_duration),
        },
        "physical_transfer_support": {
            "central_multiplier_by_zone": physical_center.tolist(),
            "central_vs_ensemble_mean_dispersion": physical_dispersion,
            "reduced_extrapolation_fraction": physical_extrapolation,
            "climate_support_distance": None,
            "climate_extrapolation_rate": None,
        },
        "full_vs_reduced_distribution_distance": float(np.sqrt(max(reduction_mse, 0.0))),
        "reduction_mse_standardized_probability_distance": float(reduction_mse),
        "decision_value": _small_nomination_gap(
            physical,
            full_workload,
            full_pv,
            full_physical,
            reduced_workload,
            reduced_pv,
            reduced_physical,
            probabilities,
        ),
    }


def build_scenario_bundle(
    physical: FrozenPhysicalArtifact,
    forecast: FrozenForecastArtifact,
    *,
    regime: DependenceRegime,
    reduced_count: int,
    study_seed: int,
    decision_value_evaluation: bool = False,
) -> ScenarioBundle:
    """Build 100 draws and reduce them without refitting either frozen artifact."""

    if reduced_count not in {10, 20, 40, 80} and not (
        reduced_count == 100 and decision_value_evaluation
    ):
        raise ValueError(
            "scenario sensitivity supports 10, 20, 40, or 80 representatives; "
            "100 is reserved for decision-value evaluation"
        )
    # The initial 100-draw population is shared across reduction-count
    # sensitivities.  Only the deterministic reduction step changes with K.
    # All dependence regimes and K-reductions start from the same marginal
    # 100-draw population.  Only the declared rank coupling and deterministic
    # reduction differ, so sensitivity contrasts are paired rather than being
    # contaminated by different random marginal draws.
    seed = stable_seed(SCENARIO_SCHEMA_VERSION, study_seed, forecast.operating_date)
    rng = np.random.default_rng(seed)
    price, price_copula = _gaussian_copula_resample(
        forecast.rt_price_pool, rng, block_name="nyiso_rt_validation_residual_block"
    )
    workload, workload_copula = _gaussian_copula_resample(
        forecast.workload_pool, rng, block_name="nlr_workload_validation_residual_block"
    )
    pv, pv_copula = _gaussian_copula_resample(
        forecast.pv_pool, rng, block_name="nyiso_pv_validation_day_block"
    )
    physical_multiplier, physical_copula = _gaussian_copula_resample(
        physical.physical_multiplier_pool,
        rng,
        block_name="nlr_physical_parameter_residual_block",
    )

    common = rng.standard_normal(100)
    price = _adverse_reorder(price, price.mean(axis=(1, 2)), common, regime.kappa, rng)
    workload = _adverse_reorder(workload, workload.mean(axis=1), common, regime.kappa, rng)
    pv = _adverse_reorder(pv, -pv.mean(axis=(1, 2)), common, regime.kappa, rng)
    physical_multiplier = _adverse_reorder(
        physical_multiplier,
        physical_multiplier.mean(axis=1),
        common,
        regime.kappa,
        rng,
    )

    features = _standardized_features(price, workload, pv, physical_multiplier)
    selected, mapping, probabilities, mse = _farthest_first_reduce(features, reduced_count)
    copula_diagnostics = {
        "primary_method": "gaussian_copula_estimated_from_validation_residual_ranks",
        "regularization": "dimension_aware_diagonal_shrinkage",
        "temporal_block_policy": "complete_horizon_rows_retained_before_rank_fit",
        "cross_domain_parameters_estimated": False,
        "blocks": {
            "nyiso_rt": price_copula,
            "nlr_workload": workload_copula,
            "nyiso_pv": pv_copula,
            "nlr_physical": physical_copula,
        },
    }
    diagnostics = _reduction_diagnostics(
        physical,
        forecast,
        full_rt=price,
        full_workload=workload,
        full_pv=pv,
        full_physical=physical_multiplier,
        selected=selected,
        probabilities=probabilities,
        reduction_mse=mse,
        copula=copula_diagnostics,
    )
    diagnostics["adverse_rank_coupling"] = {
        "estimated": False,
        "kappa": float(regime.kappa),
        "method": "common_latent_rank_iman_conover_reordering",
    }
    bundle = ScenarioBundle(
        operating_date=forecast.operating_date,
        regime=regime.name,
        kappa=regime.kappa,
        initial_count=100,
        reduced_count=reduced_count,
        zones=forecast.zones,
        timestamps_utc=forecast.timestamps_utc,
        da_price=forecast.da_price,
        rt_price=price[selected],
        workload=workload[selected],
        pv=pv[selected],
        physical_multiplier=physical_multiplier[selected],
        probabilities=probabilities,
        selected_parent_ids=selected,
        full_parent_mapping=mapping,
        reduction_mse=mse,
        seed=seed,
        full_parent_components=np.full(100, regime.name, dtype="U32"),
        full_parent_source_ids=np.arange(100, dtype=int),
        generation_diagnostics=diagnostics,
        decision_value_evaluation=bool(decision_value_evaluation),
    )
    bundle.validate()
    return bundle


def build_scenario_family(
    physical: FrozenPhysicalArtifact,
    forecast: FrozenForecastArtifact,
    *,
    study_seed: int,
    reduced_counts: Iterable[int] = (10, 20, 40, 80),
    regimes: Iterable[DependenceRegime] = DEFAULT_REGIMES,
) -> dict[tuple[str, int], ScenarioBundle]:
    return {
        (regime.name, count): build_scenario_bundle(
            physical,
            forecast,
            regime=regime,
            reduced_count=int(count),
            study_seed=study_seed,
        )
        for regime in regimes
        for count in reduced_counts
    }


def mix_nominal_and_stress(
    nominal: ScenarioBundle,
    stress: ScenarioBundle,
    *,
    stress_mass: float = 0.20,
    study_seed: int,
) -> ScenarioBundle:
    """Create the B7 mixture at the same registered sensitivity count."""

    nominal.validate()
    stress.validate()
    if nominal.regime != "independent" or stress.kappa <= 0:
        raise ValueError("the mixture requires an independent nominal bundle and an adverse stress bundle")
    if nominal.operating_date != stress.operating_date or nominal.reduced_count != stress.reduced_count:
        raise ValueError("nominal and stress bundle day/count must match")
    if not 0.0 <= stress_mass < 1.0:
        raise ValueError("stress_mass must be in [0, 1)")
    count = nominal.reduced_count
    stress_count = (
        0
        if stress_mass == 0
        else min(count - 1, max(1, int(round(count * stress_mass))))
    )
    nominal_count = count - stress_count
    seed = stable_seed(study_seed, nominal.operating_date, stress.regime, count, stress_mass, "mixture")
    rng = np.random.default_rng(seed)

    def choose(probabilities: np.ndarray, size: int) -> np.ndarray:
        probabilities = probabilities / probabilities.sum()
        return np.sort(rng.choice(len(probabilities), size=size, replace=False, p=probabilities))

    nominal_ids = choose(nominal.probabilities, nominal_count)
    stress_ids = choose(stress.probabilities, stress_count)
    nominal_weights = nominal.probabilities[nominal_ids]
    stress_weights = stress.probabilities[stress_ids]
    probability_parts = [
        (1.0 - stress_mass) * nominal_weights / nominal_weights.sum()
    ]
    if stress_count:
        probability_parts.append(
            stress_mass * stress_weights / stress_weights.sum()
        )
    probabilities = np.concatenate(probability_parts)

    def combine(field: str) -> np.ndarray:
        return np.concatenate([getattr(nominal, field)[nominal_ids], getattr(stress, field)[stress_ids]], axis=0)

    nominal_parent_count = int(round((1.0 - stress_mass) * 100))
    stress_parent_count = 100 - nominal_parent_count
    nominal_source_ids = np.sort(
        rng.choice(100, size=nominal_parent_count, replace=False)
    )
    stress_source_ids = np.sort(
        rng.choice(100, size=stress_parent_count, replace=False)
    )

    def map_component(
        source: ScenarioBundle,
        chosen_reduced_ids: np.ndarray,
        source_parent_ids: np.ndarray,
        offset: int,
    ) -> np.ndarray:
        if len(source_parent_ids) == 0:
            return np.asarray([], dtype=int)
        if len(chosen_reduced_ids) == 0:
            raise ValueError("a nonempty mixture component requires a representative")
        features = _standardized_features(
            source.rt_price,
            source.workload,
            source.pv,
            source.physical_multiplier,
        )
        chosen = features[chosen_reduced_ids]
        squared = ((features[:, None, :] - chosen[None, :, :]) ** 2).sum(axis=2)
        reduced_to_chosen = np.argmin(squared, axis=1)
        source_reduced = source.full_parent_mapping[source_parent_ids]
        return offset + reduced_to_chosen[source_reduced]

    nominal_mapping = map_component(
        nominal, nominal_ids, nominal_source_ids, 0
    )
    stress_mapping = map_component(
        stress, stress_ids, stress_source_ids, nominal_count
    )
    full_parent_mapping = np.concatenate([nominal_mapping, stress_mapping])
    full_parent_components = np.concatenate(
        [
            np.full(nominal_parent_count, "nominal", dtype="U32"),
            np.full(stress_parent_count, f"stress:{stress.regime}", dtype="U32"),
        ]
    )
    full_parent_source_ids = np.concatenate(
        [nominal_source_ids, stress_source_ids]
    )

    def weighted_diagnostic(key: str) -> float | None:
        left = nominal.generation_diagnostics.get(key)
        right = stress.generation_diagnostics.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return float((1.0 - stress_mass) * left + stress_mass * right)
        return None

    nominal_event = nominal.generation_diagnostics.get("extreme_event", {})
    stress_event = stress.generation_diagnostics.get("extreme_event", {})
    mixture_event: dict[str, Any] = {
        "definition": nominal_event.get("definition", stress_event.get("definition")),
        "component_weighting": "declared_stress_probability_mass",
    }
    for key in (
        "full_cooccurrence_rate",
        "reduced_cooccurrence_rate",
        "full_expected_longest_duration_intervals",
        "reduced_expected_longest_duration_intervals",
    ):
        if isinstance(nominal_event.get(key), (int, float)) and isinstance(
            stress_event.get(key), (int, float)
        ):
            mixture_event[key] = float(
                (1.0 - stress_mass) * nominal_event[key]
                + stress_mass * stress_event[key]
            )
    if {
        "full_cooccurrence_rate",
        "reduced_cooccurrence_rate",
    }.issubset(mixture_event):
        mixture_event["absolute_rate_error"] = abs(
            mixture_event["full_cooccurrence_rate"]
            - mixture_event["reduced_cooccurrence_rate"]
        )
    if {
        "full_expected_longest_duration_intervals",
        "reduced_expected_longest_duration_intervals",
    }.issubset(mixture_event):
        mixture_event["absolute_duration_error_intervals"] = abs(
            mixture_event["full_expected_longest_duration_intervals"]
            - mixture_event["reduced_expected_longest_duration_intervals"]
        )
    generation_diagnostics = {
        "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
        "within_domain_dependence": {
            "nominal": nominal.generation_diagnostics.get("within_domain_dependence"),
            "stress": stress.generation_diagnostics.get("within_domain_dependence"),
        },
        "stress_mixture": {
            "estimated": False,
            "declared_stress_mass": float(stress_mass),
            "nominal_representative_count": int(nominal_count),
            "stress_representative_count": int(stress_count),
            "full_parent_population_count": 100,
            "full_parent_mapping_complete": bool((full_parent_mapping >= 0).all()),
            "parent_component_counts": {
                "nominal": nominal_parent_count,
                "stress": stress_parent_count,
            },
        },
        "marginal_coverage_90": weighted_diagnostic("marginal_coverage_90"),
        "mean_quantile_score_q05_q50_q95": weighted_diagnostic(
            "mean_quantile_score_q05_q50_q95"
        ),
        "energy_score": weighted_diagnostic("energy_score"),
        "variogram_score": weighted_diagnostic("variogram_score"),
        "rank_correlation_mean_absolute_error": weighted_diagnostic(
            "rank_correlation_mean_absolute_error"
        ),
        "autocorrelation_mean_absolute_error": weighted_diagnostic(
            "autocorrelation_mean_absolute_error"
        ),
        "extreme_event": mixture_event,
        "physical_transfer_support": nominal.generation_diagnostics.get(
            "physical_transfer_support", {}
        ),
        "full_vs_reduced_distribution_distance": weighted_diagnostic(
            "full_vs_reduced_distribution_distance"
        ),
        "reduction_mse_standardized_probability_distance": float(
            (1.0 - stress_mass) * nominal.reduction_mse
            + stress_mass * stress.reduction_mse
        ),
        "decision_value": {
            "method": "declared_mixture_weighted_component_small_l1_net_load_nomination_gap",
            "decision_value_gap_mean_absolute_mw": (
                (1.0 - stress_mass)
                * float(
                    nominal.generation_diagnostics.get("decision_value", {}).get(
                        "decision_value_gap_mean_absolute_mw", 0.0
                    )
                )
                + stress_mass
                * float(
                    stress.generation_diagnostics.get("decision_value", {}).get(
                        "decision_value_gap_mean_absolute_mw", 0.0
                    )
                )
            ),
            "counterfactual_proxy": True,
        },
    }

    bundle = ScenarioBundle(
        operating_date=nominal.operating_date,
        regime=stress.regime,
        kappa=stress.kappa,
        initial_count=100,
        reduced_count=count,
        zones=nominal.zones,
        timestamps_utc=nominal.timestamps_utc,
        da_price=nominal.da_price,
        rt_price=combine("rt_price"),
        workload=combine("workload"),
        pv=combine("pv"),
        physical_multiplier=combine("physical_multiplier"),
        probabilities=probabilities,
        selected_parent_ids=np.concatenate(
            [nominal.selected_parent_ids[nominal_ids], 100 + stress.selected_parent_ids[stress_ids]]
        ),
        full_parent_mapping=full_parent_mapping,
        reduction_mse=float((1.0 - stress_mass) * nominal.reduction_mse + stress_mass * stress.reduction_mse),
        seed=seed,
        mixture_stress_mass=stress_mass,
        nominal_component_count=nominal_count,
        stress_component_count=stress_count,
        full_parent_components=full_parent_components,
        full_parent_source_ids=full_parent_source_ids,
        generation_diagnostics=generation_diagnostics,
    )
    bundle.validate()
    return bundle


def scenario_diagnostics(
    bundle: ScenarioBundle,
    *,
    support_distance_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bundle.validate()
    adverse = np.column_stack(
        [
            bundle.rt_price.mean(axis=(1, 2)),
            bundle.workload.mean(axis=1),
            -bundle.pv.mean(axis=(1, 2)),
            bundle.physical_multiplier.mean(axis=1),
        ]
    )
    ranks = np.column_stack([np.argsort(np.argsort(adverse[:, j])) for j in range(adverse.shape[1])])
    varying = np.std(ranks, axis=0) > 1e-12
    if int(np.sum(varying)) >= 2:
        partial = np.corrcoef(ranks[:, varying], rowvar=False)
        dependence = np.eye(ranks.shape[1], dtype=float)
        dependence[np.ix_(varying, varying)] = partial
    else:
        dependence = np.eye(ranks.shape[1], dtype=float)
    off_diagonal = dependence[np.triu_indices(4, 1)]
    thresholds = np.quantile(adverse, 0.8, axis=0)
    extreme_weight = float(bundle.probabilities[(adverse >= thresholds).all(axis=1)].sum())
    stored = dict(bundle.generation_diagnostics)
    physical_support = dict(stored.get("physical_transfer_support", {}))
    if isinstance(support_distance_evidence, Mapping):
        distances = support_distance_evidence.get("support_distance_by_zone", {})
        out_of_support = support_distance_evidence.get(
            "out_of_support_fraction_by_zone", {}
        )
        if isinstance(distances, Mapping) and distances:
            finite_distances = [
                float(value)
                for value in distances.values()
                if value is not None and np.isfinite(float(value))
            ]
            if finite_distances:
                physical_support["climate_support_distance"] = {
                    str(key): float(value) for key, value in distances.items()
                }
                physical_support["mean_climate_support_distance"] = float(
                    np.mean(finite_distances)
                )
                physical_support["maximum_climate_support_distance"] = float(
                    np.max(finite_distances)
                )
        if isinstance(out_of_support, Mapping) and out_of_support:
            physical_support["climate_extrapolation_rate"] = float(
                np.mean([float(value) for value in out_of_support.values()])
            )
        physical_support["support_evidence_source"] = support_distance_evidence.get(
            "source"
        )
    stored["physical_transfer_support"] = physical_support
    record = {
        "operating_date": bundle.operating_date,
        "regime": bundle.regime,
        "kappa": bundle.kappa,
        "initial_count": bundle.initial_count,
        "reduced_count": bundle.reduced_count,
        "seed": bundle.seed,
        "reduction_mse": bundle.reduction_mse,
        "mixture_stress_mass": bundle.mixture_stress_mass,
        "nominal_component_count": bundle.nominal_component_count,
        "stress_component_count": bundle.stress_component_count,
        "mean_pairwise_rank_correlation": float(np.nanmean(off_diagonal)),
        "all_four_upper_quintile_probability": extreme_weight,
        "full_parent_mapping_complete": bool(
            len(bundle.full_parent_mapping) == bundle.initial_count
            and (bundle.full_parent_mapping >= 0).all()
            and (bundle.full_parent_mapping < bundle.reduced_count).all()
        ),
        "full_parent_mapping_count": int(len(bundle.full_parent_mapping)),
        "full_parent_component_counts": {
            str(component): int(np.sum(bundle.full_parent_components == component))
            for component in np.unique(bundle.full_parent_components)
        },
        "evidence_role": bundle.evidence_role,
    }
    record.update(stored)
    return record
