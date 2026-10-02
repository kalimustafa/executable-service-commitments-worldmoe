"""Compare the frozen homogeneous 24 h book with fixed mixed contracts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from audit_hetero_book import audit

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
WORKSPACE=BASE.parent
CURRENT=WORKSPACE/'R2_JointDecision_20260927/results'
ENGINE=BASE/'engine/results'
EPOCH={20260920:6,20260921:2,20260922:0}
with np.load(WORKSPACE/'external_validation/data/forcing_2025.npz',allow_pickle=False) as f:
    arrivals=f['eagle_arrivals']
    k=f['k']

rows=[]
for seed,epoch in EPOCH.items():
    base=f'CdfR2_Eagle_2025_s{seed}_ep{epoch:02d}_coupled_cdf'
    with np.load(CURRENT/'trajectories'/f'{base}_complete_book.npz') as f:book=f['compute']
    original=json.loads((CURRENT/f'{base}.json').read_text())
    for priority in ('fifo','edf'):
        metrics=audit(book,arrivals,k,priority)
        rows.append(dict(seed=seed,pathway='H0 homogeneous 24 h request '+priority,
            annual_saving_usd=original['saving_usd'],saving_pct=original['saving_pct'],
            cvar95_usd=original['cvar95_usd'],shifted_it_mwh=original['shifted_IT_mwh'],
            source_book=str(CURRENT/'trajectories'/f'{base}_complete_book.npz'),**metrics))
    h1=ENGINE/f'{base}_w24_mixed_h1.json'
    if h1.exists():
        result=json.loads(h1.read_text())
        h1_book=ENGINE/'trajectories'/f'{base}_w24_mixed_h1_complete_book.npz'
        with np.load(h1_book) as f:mixed_book=f['compute']
        metrics=audit(mixed_book,arrivals,k,'edf')
        result.update(metrics)
        result['late_IT_mwh']=metrics['late_work_it_mwh']
        result.pop('late_backlog_it_mwh',None)
        h1.write_text(json.dumps(result,indent=2))
        rows.append(dict(seed=seed,pathway='H1 mixed deadline aware WorldMoE edf',
            annual_saving_usd=result['saving_usd'],saving_pct=result['saving_pct'],
            cvar95_usd=result['cvar95_usd'],shifted_it_mwh=result['shifted_IT_mwh'],
            source_book=str(h1_book),**metrics))
frame=pd.DataFrame(rows)
frame.to_csv(HERE/'mixed_deadline_results.csv',index=False)
frame.to_csv(BASE/'source_data/extfig1_heterogeneous_deadlines.csv',index=False)
print(frame.groupby('pathway')[['annual_saving_usd','late_work_it_mwh','late_backlog_mwh_quarters','closing_it_mwh','p95_delay_h']].mean().round(4).to_string())
