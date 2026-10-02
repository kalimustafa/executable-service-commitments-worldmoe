"""Audit causal DA publication and RT-history availability on 20 fixed dates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TASK = Path(__file__).resolve().parents[1]
DATA = TASK / "data_processed"
OUT = TASK / "data_processed/information_availability_audit.csv"


def main() -> None:
    notices_path = DATA / "isone_da_publication_notices.csv"
    decisions_path = DATA / "decision_feature_availability.csv"
    if not notices_path.exists() or not decisions_path.exists():
        raise FileNotFoundError("Official DA notice timestamps and decision availability rows are required.")
    notices = pd.read_csv(notices_path)
    decisions = pd.read_csv(decisions_path)
    notices["operating_date"] = pd.to_datetime(notices["operating_date"]).dt.date
    notices["notice_posted_at_utc"] = pd.to_datetime(notices["notice_posted_at_utc"], format="mixed", utc=True)
    decisions["decision_timestamp_utc"] = pd.to_datetime(decisions["decision_timestamp_utc"], format="mixed", utc=True)
    required = {
        "operating_date", "decision_timestamp_utc", "da_used",
        "rt_history_last_interval_end_utc", "forecast_issue_timestamp_utc",
    }
    missing = required - set(decisions.columns)
    if missing:
        raise ValueError(f"Decision availability audit is missing columns: {sorted(missing)}")
    decisions["operating_date"] = pd.to_datetime(decisions["operating_date"]).dt.date
    decisions["rt_history_last_interval_end_utc"] = pd.to_datetime(
        decisions["rt_history_last_interval_end_utc"], format="mixed", utc=True,
    )
    decisions["forecast_issue_timestamp_utc"] = pd.to_datetime(
        decisions["forecast_issue_timestamp_utc"], format="mixed", utc=True,
    )
    merged = decisions.merge(notices[["operating_date", "notice_posted_at_utc"]], on="operating_date", validate="many_to_one")
    seed = 20260928
    sampled_dates = sorted(pd.Series(merged["operating_date"].unique()).sample(n=20, random_state=seed).tolist())
    sample = merged[merged["operating_date"].isin(sampled_dates)].copy()
    sample["da_published_at_decision"] = sample["decision_timestamp_utc"] >= sample["notice_posted_at_utc"]
    sample["da_used_before_publication"] = sample["da_used"].astype(bool) & ~sample["da_published_at_decision"]
    sample["rt_history_is_strictly_lagged"] = (
        (sample["rt_history_last_interval_end_utc"] < sample["forecast_issue_timestamp_utc"])
        & (sample["forecast_issue_timestamp_utc"] <= sample["decision_timestamp_utc"])
    )
    sample["future_price_leak"] = (
        sample["da_used_before_publication"] | ~sample["rt_history_is_strictly_lagged"]
    )
    sample.to_csv(OUT, index=False)
    if sample["future_price_leak"].astype(bool).any():
        raise RuntimeError("Future market-price leakage detected.")
    print({"status": "PASS", "sampled_dates": len(sampled_dates), "rows": len(sample), "seed": seed})


if __name__ == "__main__":
    main()
