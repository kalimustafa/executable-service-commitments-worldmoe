from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dcvpp.core import atomic_json, sha256_file


def _farthest_first(matrix: np.ndarray, k: int) -> np.ndarray:
    standardized = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-6)
    selected = [int(np.argmin(np.linalg.norm(standardized - standardized.mean(axis=0), axis=1)))]
    while len(selected) < k:
        distance = np.min(
            np.stack([np.linalg.norm(standardized - standardized[idx], axis=1) for idx in selected]),
            axis=0,
        )
        distance[selected] = -1
        selected.append(int(np.argmax(distance)))
    return np.asarray(selected, dtype=int)


def build_demo_scenarios(
    market_path: Path,
    workload_path: Path,
    artifact_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    seed = int(config.get("seed", 20260809))
    rng = np.random.default_rng(seed + 71)
    initial = int(config["scenarios"].get("initial", 20))
    reduced = int(config["scenarios"].get("reduced", 5))
    market = pd.read_parquet(market_path)
    workload = pd.read_parquet(workload_path)
    sites = sorted(market["zone"].unique())
    last_times = sorted(market["timestamp_utc"].unique())[-96:]
    base_price = (
        market[market["timestamp_utc"].isin(last_times)]
        .pivot(index="timestamp_utc", columns="zone", values="rt_lbmp")[sites]
        .sort_index()
        .to_numpy(dtype=float)
        .T
    )
    base_work = workload.sort_values("timestamp_utc")["requested_work"].tail(96).to_numpy(dtype=float)
    base_work = np.maximum(base_work, 0.1)
    base_pv = np.maximum(0, np.sin(np.pi * (np.arange(96) - 24) / 48))[None, :] * np.ones((len(sites), 1))
    prices, works, pvs, physical = [], [], [], []
    latent = rng.normal(size=initial)
    for w in range(initial):
        rank = (np.argsort(np.argsort(latent))[w] + 0.5) / initial
        adverse = (rank - 0.5) * 2.0
        price = base_price + rng.normal(0, 4.0, base_price.shape) + 7.0 * adverse
        work = np.maximum(0, base_work * (1.0 + 0.12 * rng.normal(size=96) + 0.18 * adverse))
        pv = np.clip(base_pv * (1.0 + 0.08 * rng.normal(size=base_pv.shape) - 0.20 * adverse), 0, None)
        prices.append(price)
        works.append(work)
        pvs.append(pv)
        physical.append(1.0 + 0.04 * rng.normal() + 0.05 * adverse)
    prices_a = np.asarray(prices)
    works_a = np.asarray(works)
    pvs_a = np.asarray(pvs)
    physical_a = np.asarray(physical)
    flat = np.concatenate([prices_a.reshape(initial, -1), works_a, pvs_a.reshape(initial, -1), physical_a[:, None]], axis=1)
    selected = _farthest_first(flat, reduced)
    selected_flat = flat[selected]
    distance = ((flat[:, None, :] - selected_flat[None, :, :]) ** 2).mean(axis=2)
    parents = np.argmin(distance, axis=1)
    weights = np.bincount(parents, minlength=reduced).astype(float) / initial
    scenario_dir = artifact_root / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = scenario_dir / "demo_reduced_scenarios.npz"
    np.savez_compressed(
        scenario_path,
        prices=prices_a[selected],
        workload=works_a[selected],
        pv=pvs_a[selected],
        physical_multiplier=physical_a[selected],
        probabilities=weights,
        selected_parent_ids=selected,
        full_parent_mapping=parents,
        sites=np.asarray(sites),
    )
    diag = pd.DataFrame(
        {
            "metric": ["initial_count", "reduced_count", "probability_sum", "full_to_reduced_mse"],
            "value": [initial, reduced, weights.sum(), float(np.mean(np.min(distance, axis=1)))],
            "profile": "demo",
            "synthetic": True,
            "evidence_role": "derived",
        }
    )
    diag_path = artifact_root / "tables/scenario_diagnostics.csv"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag.to_csv(diag_path, index=False)
    manifest = {
        "profile": "demo",
        "synthetic": True,
        "blocks": ["synthetic_nyiso_market_weather", "synthetic_nlr_workload", "synthetic_physical_transfer"],
        "cross_domain_nominal": "product_of_marginals",
        "adverse_coupling": {"estimated": False, "kappa": [0.0, 0.3, 0.6]},
        "initial": initial,
        "reduced": reduced,
        "probabilities": weights.tolist(),
        "scenario_sha256": sha256_file(scenario_path),
    }
    manifest_path = artifact_root / "manifests/scenario_manifest.json"
    atomic_json(manifest_path, manifest)
    return {"scenario": scenario_path, "diagnostics": diag_path, "manifest": manifest_path}


def load_scenarios(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

