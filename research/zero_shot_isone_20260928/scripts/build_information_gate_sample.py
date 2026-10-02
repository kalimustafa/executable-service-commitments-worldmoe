from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


TASK = Path(__file__).resolve().parents[1]
DATA = TASK / "data_processed"
TZ = ZoneInfo("America/New_York")
SEED = 20260928


def main() -> None:
    notices = pd.read_csv(DATA / "isone_da_publication_notices.csv")
    notices["notice_posted_at_utc"] = pd.to_datetime(notices["notice_posted_at_utc"], format="mixed", utc=True)
    candidates = sorted(notices["operating_date"].astype(str).tolist())
    rng = np.random.default_rng(SEED)
    sampled = sorted(rng.choice(candidates, size=20, replace=False).tolist())
    rows = []
    for operating_date in sampled:
        notice = notices.loc[notices.operating_date.astype(str).eq(operating_date), "notice_posted_at_utc"].iloc[0]
        floor_slot = notice.floor("15min")
        before = floor_slot if floor_slot < notice else floor_slot - pd.Timedelta(minutes=15)
        after = before + pd.Timedelta(minutes=15)
        day = pd.Timestamp(operating_date)
        issue_local = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=4, minutes=55)).tz_localize(TZ)
        history_end_local = (day - pd.Timedelta(days=1)).tz_localize(TZ)
        for decision in (before, after):
            rows.append({
                "operating_date": operating_date,
                "decision_timestamp_utc": decision.isoformat(),
                "da_used": bool(decision >= notice),
                "da_required_notice_utc": notice.isoformat() if decision >= notice else "",
                "rt_history_last_interval_end_utc": history_end_local.tz_convert("UTC").isoformat(),
                "forecast_issue_timestamp_utc": issue_local.tz_convert("UTC").isoformat(),
                "official_notice_url": notices.loc[notices.operating_date.astype(str).eq(operating_date), "official_notice_url"].iloc[0],
            })
    pd.DataFrame(rows).to_csv(DATA / "decision_feature_availability.csv", index=False)
    (TASK / "logs" / "information_gate_sample.json").write_text(json.dumps({"seed": SEED, "sampled_operating_dates": sampled, "rows": len(rows)}, indent=2), encoding="utf-8")
    print({"sampled_dates": len(sampled), "quarter_hour_decisions": len(rows), "seed": SEED})


if __name__ == "__main__":
    main()
