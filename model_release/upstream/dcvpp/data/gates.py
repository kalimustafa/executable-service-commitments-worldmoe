from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dcvpp.core import EVIDENCE_ROLES


class GateFailure(RuntimeError):
    pass


def validate_demo_domains(paths: dict[str, Path]) -> dict[str, Any]:
    market = pd.read_parquet(paths["market"])
    physical = pd.read_parquet(paths["physical"])
    workload = pd.read_parquet(paths["workload"])
    for name, frame in (("market", market), ("physical", physical), ("workload", workload)):
        if frame.empty:
            raise GateFailure(f"{name} is empty")
        if not frame["synthetic"].all() or not (frame["profile"] == "demo").all():
            raise GateFailure(f"{name} demo labels are invalid")
        invalid = set(frame["evidence_role"].dropna().unique()) - EVIDENCE_ROLES
        if invalid:
            raise GateFailure(f"{name} has invalid evidence roles: {sorted(invalid)}")
    if not ((market["da_lbmp"] < 0).any() or (market["rt_lbmp"] < 0).any()):
        raise GateFailure("negative price fixture is missing")
    clocks = {
        "market": str(pd.to_datetime(market["timestamp_utc"], utc=True).min()),
        "physical": str(pd.to_datetime(physical["timestamp_utc"], utc=True).min()),
        "workload": str(pd.to_datetime(workload["timestamp_utc"], utc=True).min()),
    }
    if len(set(clocks.values())) != 3:
        raise GateFailure("demo domains must use distinct clocks")
    forbidden = {c for c in market.columns if c.lower().startswith("xcel")}
    if forbidden:
        raise GateFailure(f"Xcel fields leaked into market training table: {sorted(forbidden)}")
    return {
        "status": "passed",
        "rows": {"market": len(market), "physical": len(physical), "workload": len(workload)},
        "clocks": clocks,
        "negative_prices_preserved": True,
        "cross_domain_training_join": False,
    }


def chronological_split(frame: pd.DataFrame, time_column: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(time_column, kind="stable").reset_index(drop=True)
    n = len(ordered)
    a, b = int(0.60 * n), int(0.80 * n)
    train, val, test = ordered.iloc[:a], ordered.iloc[a:b], ordered.iloc[b:]
    if train.empty or val.empty or test.empty:
        raise GateFailure("chronological split produced an empty segment")
    if train[time_column].max() > val[time_column].min() or val[time_column].max() > test[time_column].min():
        raise GateFailure("chronological split order failed")
    return train, val, test

