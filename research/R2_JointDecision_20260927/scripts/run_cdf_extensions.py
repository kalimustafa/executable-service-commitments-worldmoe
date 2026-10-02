"""Same-checkpoint evidence for innovations 2 and 3; no training or reselection."""
import sys, json, subprocess, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT, EXTENSION
import numpy as np
import pandas as pd
import src, vendor
src.__path__.append(str(EXTENSION/'src'))
vendor.__path__.append(str(EXTENSION/'vendor'))
sys.path.append(str(EXTENSION))
from src import context as ctx
from src.controllers import project_grid

seeds=[20260920,20260921,20260922]
for seed in seeds:
    path=ROOT/'results'/('cdf_primary_complete.json' if seed==20260920 else f'cdf_complete_s{seed}.json')
    summary=json.loads(path.read_text())
    assert summary['status']=='complete' and summary['best_restored']

for seed in seeds:
    path=ROOT/'results'/('cdf_selection.json' if seed==20260920 else f'cdf_selection_s{seed}.json')
    selection=json.loads(path.read_text());epoch=selection['refinement_epoch']
    native=ROOT/'results'/f'CdfR2_Eagle_2025_s{seed}_ep{epoch:02d}_coupled_cdf.json'
    assert native.exists()
    subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/evaluate_scenario_recourse.py'),
                    '--year','2025','--seed',str(seed),'--cdf-epoch',str(epoch),
                    '--eta','0','--safety','0.9','--current-coupling','--continuous-cost','--post-only'],
                   cwd=ROOT,check=True)

arr, reference, shared=ctx.configure(2025)
ctx.RESULTS=ROOT/'results';ctx.TRAJECTORIES=ROOT/'results/trajectories';ctx.c.DATA=ctx.TRAJECTORIES
scenarios=json.loads((EXTENSION/'configs/protocol.json').read_text())['grid_constraints']
reference_cache={}
for scenario in scenarios:
    kw={k:scenario[k] for k in ['grid_fraction','ramp_fraction']}
    ref,_=project_grid(reference,arr,**kw)
    reference_cache[scenario['name']]=(ref,ctx.fifo_lateness(ref,arr,24))

rows=[]
for seed in seeds:
    selection=json.loads((ROOT/'results'/('cdf_selection.json' if seed==20260920 else f'cdf_selection_s{seed}.json')).read_text())
    epoch=selection['refinement_epoch'];base=f'CdfR2_Eagle_2025_s{seed}_ep{epoch:02d}_coupled_cdf'
    with np.load(ROOT/'results/trajectories'/f'{base}_complete_book.npz') as z:requests=z['compute']
    original=json.loads((ROOT/'results'/f'{base}.json').read_text())
    for scenario in scenarios:
        kw={k:scenario[k] for k in ['grid_fraction','ramp_fraction']}
        ref,ref_late=reference_cache[scenario['name']]
        for guard in [False,True]:
            experiment='CdfI3G' if guard else 'CdfI3'
            name=f'{experiment}_Eagle_2025_s{seed}_w24_{scenario["name"]}'
            path=ROOT/'results'/f'{name}.json'
            if path.exists():rows.append(json.loads(path.read_text()));continue
            started=time.perf_counter()
            book,extra=project_grid(requests,arr,deadline_guard=guard,window=24,**kw)
            extra.update(reference_late_it_mwh=ref_late,seconds=time.perf_counter()-started,
                         source_request_result=base+'.json',request_forecast_checkpoint_sha256=selection['forecast_checkpoint_sha256'],
                         intervention_scope='causal projection of the frozen CdfR2 request stream; execution debt retained; no retraining and no feedback into the unconstrained neural request generator')
            row=ctx.measure(name,book,ref,arr,shared,dict(experiment=experiment,year=2025,trace='Eagle',seed=seed,
                            window_h=24,scenario=scenario['name'],role='frozen-policy electrical stress analysis',**kw),extra=extra)
            row['delta_cost_vs_unconstrained_usd']=row['adjusted_cost_usd']-original['adjusted_cost_usd']
            path.write_text(json.dumps(row,indent=2));rows.append(row)
pd.DataFrame(rows).to_csv(ROOT/'results/cdf_grid_constraints.csv',index=False)
import torch
from src.models.coupled_rollout import ArchitectedR2QPHead
torch.set_num_threads(1)
with np.load(ROOT/'cache/structured_market2025_s20260920_ep10.npz') as z:
    hours=z['hours'];market=z['rt_1'];latent=z['latent_1']
indices=np.searchsorted(hours,ctx.c.times.floor('h').asi8)
assert np.array_equal(hours[indices],ctx.c.times.floor('h').asi8)
truth=ctx.c.rt
forecast_rows=[];tau=np.array([.05,.1,.25,.5,.75,.9,.95])
for seed in seeds:
    selected=json.loads((ROOT/'results'/('cdf_selection.json' if seed==20260920 else f'cdf_selection_s{seed}.json')).read_text())
    epoch=selected['refinement_epoch'];model=ArchitectedR2QPHead()
    model.load_state_dict(torch.load(ROOT/'checkpoints'/f'CdfR2_seed{seed}_epoch{epoch:02d}.pt',map_location='cpu',weights_only=False)['state_dict'])
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        _,shift=model.market_update(torch.from_numpy(latent.reshape(-1,128)))
    q=market+shift.numpy().reshape(len(hours),4,1);q=q[indices]
    error=truth[:,:,None]-q
    forecast_rows.append(dict(seed=seed,epoch=epoch,post_rt_pinball_usd_mwh=float(np.maximum(tau*error,(tau-1)*error).mean()),
                              post_rt_median_mae_usd_mwh=float(np.abs(truth-q[:,:,3]).mean()),
                              post_rt_coverage90_pct=float(100*((truth>=q[:,:,0])&(truth<=q[:,:,-1])).mean()),
                              post_rt_interval_width_usd_mwh=float((q[:,:,-1]-q[:,:,0]).mean()),
                              quarter_hour_zone_targets=int(truth.size),
                              forecast_scope='queue-independent APIC-preserving RT quantiles; excludes task-only price bias and opportunity score'))
pd.DataFrame(forecast_rows).to_csv(ROOT/'results/worldmoe_final_forecast_scores.csv',index=False)
print('CDF EXTENSIONS COMPLETE',len(rows),'electrical cases; three information interventions',flush=True)
