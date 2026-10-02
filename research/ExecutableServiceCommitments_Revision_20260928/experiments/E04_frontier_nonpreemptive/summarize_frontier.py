"""Collect newly executed job and matched-fluid comparisons."""
from pathlib import Path
import json
import pandas as pd

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
rows=[]
for window in (24,30,48):
    path=HERE/f'Frontier_2025_w{window}_summary.json'
    if not path.exists():continue
    data=json.loads(path.read_text())
    rows.append(dict(window_h=window,jobs=data['jobs'],saving_usd=data['saving_usd'],
        saving_pct=data['saving_pct'],matched_fluid_saving_usd=data['matched_fluid_saving_usd'],
        job_to_fluid_saving_ratio=data['job_to_fluid_saving_ratio'],
        late_jobs=data['late_jobs'],late_job_it_mwh=data['late_job_it_mwh'],
        closing_jobs=data['closing_jobs'],closing_job_it_mwh=data['closing_job_it_mwh'],
        reference_late_jobs=data['reference_late_jobs'],
        reference_closing_jobs=data['reference_closing_jobs'],
        cvar95_usd=data['cvar95_usd'],execution_seconds=data['execution_seconds'],
        source_json=str(path)))
frame=pd.DataFrame(rows)
frame.to_csv(HERE/'frontier_results.csv',index=False)
frame.to_csv(BASE/'source_data/extfig1_frontier.csv',index=False)
print(frame[['window_h','saving_usd','matched_fluid_saving_usd','job_to_fluid_saving_ratio','late_jobs','closing_jobs']].to_string(index=False))
