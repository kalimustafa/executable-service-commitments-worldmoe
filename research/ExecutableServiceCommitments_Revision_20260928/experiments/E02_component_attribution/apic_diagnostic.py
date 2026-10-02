"""Same-checkpoint APIC residual diagnostic: forecast branch and placement."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
WORKSPACE=BASE.parent
ENGINE=BASE/'engine'
sys.path.insert(0,str(ENGINE))
from src.models.coupled_rollout import ArchitectedR2QPHead

torch.set_num_threads(1)
checkpoint=ENGINE/'checkpoints/CdfR2_seed20260920_epoch06.pt'
state=torch.load(checkpoint,map_location='cpu',weights_only=False)
model=ArchitectedR2QPHead()
model.load_state_dict(state['state_dict'])
model.eval().requires_grad_(False)
with np.load(ENGINE/'cache/structured_market2025_s20260920_ep10.npz') as z:
    hours=z['hours'];raw=z['rt_1'].astype(np.float64);latent=z['latent_1']
with torch.inference_mode():
    _,shift=model.market_update(torch.from_numpy(latent.reshape(-1,128)))
shift=shift.numpy().reshape(raw.shape[0],raw.shape[1])
full=raw+shift[:,:,None]
with np.load(WORKSPACE/'external_validation/data/forcing_2025.npz') as f:
    times=pd.DatetimeIndex(pd.to_datetime(f['times'],utc=True))
    truth=f['rt'];k=f['k']
hi=np.searchsorted(hours,times.floor('h').asi8)
if not np.array_equal(hours[hi],times.floor('h').asi8):raise RuntimeError('Forecast time alignment')
tau=np.array([.05,.1,.25,.5,.75,.9,.95])
def score(q):
    error=truth[:,:,None]-q[hi]
    pinball=float(np.maximum(tau*error,(tau-1)*error).mean())
    coverage=float(100*((truth>=q[hi,:,0])&(truth<=q[hi,:,-1])).mean())
    violations=int((np.diff(q,axis=-1)<-1e-8).sum())
    return pinball,coverage,violations

root=ENGINE/'results'
on='CdfR2_Eagle_2025_s20260920_ep06_coupled_cdf_w24'
off=on+'_apic_off'
on_result=json.loads((root/f'{on}.json').read_text())
off_result=json.loads((root/f'{off}.json').read_text())
with np.load(root/'trajectories'/f'{on}_complete_book.npz') as z:book_on=z['compute']
with np.load(root/'trajectories'/f'{off}_complete_book.npz') as z:book_off=z['compute']
changed=int((np.abs(book_on-book_off)>1e-6).sum())
reassigned=float(.5*.25*np.sum(np.abs(book_on-book_off)*k))
rows=[]
for label,q,data in [('on',full,on_result),('off',raw,off_result)]:
    pinball,coverage,ordering=score(q)
    rows.append(dict(apic_residual=label,seed=20260920,post_rt_pinball_usd_mwh=pinball,
        coverage90_pct=coverage,quantile_ordering_violations=ordering,
        annual_saving_usd=data['saving_usd'],late_it_mwh=data['late_IT_mwh'],
        closing_it_mwh=data['closing_it_mwh'],changed_quarter_hour_site_actions=changed,
        reassigned_it_mwh_between_paths=reassigned,
        note='Terminal learned RT-quantile shift only. Upstream W0/W1 and R2 APIC remain active; retained R2 forecast Parquet permits direct post-DA identity checking.'))
frame=pd.DataFrame(rows)
frame.to_csv(HERE/'terminal_market_shift_raw.csv',index=False)
correct=frame.rename(columns={'apic_residual':'final_market_shift'})
correct['upstream_w0w1_r2_apic_unchanged']=True
correct['interpretation']='Only the final learned market residual shift is zeroed; W0/W1 and R2 APIC remain active in both branches.'
correct.to_csv(BASE/'source_data/final_market_shift_diagnostic.csv',index=False)
print(frame.to_string(index=False))
