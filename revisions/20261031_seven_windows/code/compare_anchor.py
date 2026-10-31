"""One necessary 24 h reduction check for the shared extension implementation."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'ExecutableServiceCommitments_Revision_20260928/engine/results'
rows=[]
for seed,ep in [(20260920,6),(20260921,2),(20260922,0)]:
    old=f'CdfR2_Eagle_2025_s{seed}_ep{ep:02d}_coupled_cdf_w24'
    new='WindowExtension20261031_'+old
    a=json.loads((OLD/(old+'.json')).read_text());b=json.loads((ROOT/'results'/(new+'.json')).read_text())
    checks={}
    for suffix in ['_complete_book.npz','_trajectory.npz']:
        x=np.load(OLD/'trajectories'/(old+suffix));y=np.load(ROOT/'results/trajectories'/(new+suffix))
        checks[suffix]={k:bool(np.array_equal(x[k],y[k])) for k in x.files}
    rows.append({'seed':seed,'historical_saving_usd':a['saving_usd'],'new_anchor_saving_usd':b['saving_usd'],
        'saving_difference_usd':b['saving_usd']-a['saving_usd'],'checkpoint_hash_equal':a['forecast_checkpoint_sha256']==b['forecast_checkpoint_sha256'],
        'array_equal':checks,'all_arrays_bitwise_equal':all(v for check in checks.values() for v in check.values())})
result={'scope':'One full-year three-checkpoint reduction check after shared runner extension','rows':rows,
        'all_three_anchors_bitwise_equal':all(r['all_arrays_bitwise_equal'] for r in rows)}
(ROOT/'results/anchor_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
if not result['all_three_anchors_bitwise_equal']:raise SystemExit('Anchor differs; inspect before interpreting window contrasts.')
