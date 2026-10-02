from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TASK = Path(__file__).resolve().parents[1]
WORKSPACE = TASK.parent
ENGINE = WORKSPACE / "ExecutableServiceCommitments_Revision_20260928" / "engine"
ARCHIVE = WORKSPACE / "R2_PAPER_INTEGRATION_20260923"
EXTENSION = WORKSPACE / "ServiceCommitment_1to5_20260927"
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ARCHIVE))
sys.path.insert(0, str(EXTENSION))
sys.path.insert(0, str(TASK / "scripts"))

import src  # noqa: E402
import vendor  # noqa: E402

src.__path__.append(str(EXTENSION / "src"))
vendor.__path__.append(str(EXTENSION / "vendor"))
vendor.__path__.append(str(ARCHIVE / "vendor"))
from src import context as ctx  # noqa: E402
from isone_context_adapter import install as install_isone_context  # noqa: E402
from vendor.forecast_provider import ForecastProvider  # noqa: E402
from vendor import paper_controls  # noqa: E402


def run_portfolio(portfolio: str) -> list[dict]:
    task_results = TASK / "outputs" / "baselines" / "runs"
    task_trajectories = task_results / "trajectories"
    task_results.mkdir(parents=True, exist_ok=True)
    task_trajectories.mkdir(parents=True, exist_ok=True)
    ctx.RESULTS = task_results
    ctx.TRAJECTORIES = task_trajectories
    install_isone_context(ctx, TASK, portfolio, "neutral_mean")
    arrivals, reference, shared = ctx.configure(2025, "Eagle")
    c = ctx.c
    c.DATA = task_trajectories
    paper_controls.H.c = c
    notices = pd.read_csv(TASK / "data_processed" / "isone_da_publication_notices.csv")
    notices["notice_posted_at_utc"] = pd.to_datetime(notices["notice_posted_at_utc"], format="mixed", utc=True)
    local_dates = c.times.tz_convert("America/New_York").strftime("%Y-%m-%d")
    publication = notices.set_index("operating_date").notice_posted_at_utc
    publication_by_step = pd.DatetimeIndex([publication.loc[d] for d in local_dates]).asi8
    rows = []

    # Published-DA myopic control: the official DA LMP becomes rankable only
    # after the target operating-date's actual ISO-NE notice time.
    da_provider = ForecastProvider(c.times, c.da, c.train_rt, kind="published_da")
    da_provider.post_issue = publication_by_step
    da_rank = da_provider.ranked_slots(24, "point")
    name = f"ISO-NE_{portfolio}_published_DA_myopic_w24"
    book, stats = paper_controls.myopic_execute(arrivals, 24, da_rank, capacity=1.0, hedge=True)
    row = ctx.measure(name, book, reference, arrivals, shared, dict(
        experiment="ISO-NE direct baseline", market="ISO-NE", portfolio=portfolio,
        trace="Eagle", seed=0, window_h=24, policy="published_DA_myopic",
        publication_source="official per-date ISO-NE DA notice timestamps",
        baseline_training=False,
    ), late=stats["late_IT_mwh"], extra={"baseline_training": False, "parameter_updates_during_evaluation": False})
    rows.append(row)

    # Causal lagged-RT control: each target quarter uses the same UTC quarter
    # 24 h earlier, which is observed before the current action. Startup slots
    # use the neutral frozen NYISO training RT median supplied by the adapter.
    lag = 96
    history = np.broadcast_to(c.train_rt[None, :], c.rt.shape).copy()
    if c.N > lag:
        history[lag:] = c.rt[:-lag]
    rt_provider = ForecastProvider(c.times, c.da, c.train_rt, kind="published_da")
    rt_provider.post = np.repeat(history[:, :, None], 7, axis=2)
    rt_provider.post_issue = np.full(c.N, np.iinfo(np.int64).min, dtype=np.int64)
    rt_rank = rt_provider.ranked_slots(24, "point")
    name = f"ISO-NE_{portfolio}_lagged_RT_myopic_w24"
    book, stats = paper_controls.myopic_execute(arrivals, 24, rt_rank, capacity=1.0, hedge=True)
    row = ctx.measure(name, book, reference, arrivals, shared, dict(
        experiment="ISO-NE direct baseline", market="ISO-NE", portfolio=portfolio,
        trace="Eagle", seed=0, window_h=24, policy="lagged_RT_24h_myopic",
        lag_hours=24, baseline_training=False,
    ), late=stats["late_IT_mwh"], extra={"baseline_training": False, "parameter_updates_during_evaluation": False})
    rows.append(row)
    return rows


def main() -> None:
    rows = []
    for portfolio in ("ISO-NE-SOUTH", "ISO-NE-NORTH"):
        rows.extend(run_portfolio(portfolio))
    pd.DataFrame(rows).to_csv(TASK / "outputs" / "baselines" / "zero_shot_isone_direct_baselines.csv", index=False)
    print({"baseline_rows": len(rows), "policies": sorted({row["policy"] for row in rows})})


if __name__ == "__main__":
    main()
