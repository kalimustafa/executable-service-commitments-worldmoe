"""Distributional OOD screen using existing actions, not R2 test fitting."""
from pathlib import Path
import json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT.parent;REPO=ROOT.parents[1]/'power-dc-vpp'
DT=.25;K=1.639059066772461;IDLE=.23069976270198822

def describe(name,values,reference=None):
    out=[]
    for metric,v in values.items():
        v=np.asarray(v,float).ravel();v=v[np.isfinite(v)]
        q=np.quantile(v,[.001,.01,.5,.95,.99,.999])
        r=dict(source=name,metric=metric,n=len(v),mean=float(v.mean()),minimum=float(v.min()),maximum=float(v.max()),
            p001=float(q[0]),p01=float(q[1]),p50=float(q[2]),p95=float(q[3]),p99=float(q[4]),p999=float(q[5]))
        if reference is not None:
            a=np.asarray(reference[metric]).ravel();lo,hi=np.quantile(a,[.001,.999]);span=max(hi-lo,1e-6)
            r.update(outside_training_998pct_fraction=float(((v<lo)|(v>hi)).mean()),
                far_outside_training_envelope_fraction=float(((v<lo-.1*span)|(v>hi+.1*span)).mean()))
        out.append(r)
    return out

def metrics(compute,grid,queue,pv_used,*,ramp_axis):
    return dict(compute_power_mw=compute*K,utilization=compute,
        quarter_hour_ramp_mw=np.abs(np.diff(compute*K,axis=ramp_axis)),queue_work_units=queue,
        queue_clearance_time_h=queue*DT,pv_used_mw=pv_used,grid_import_mw=grid)

def main():
    cached=np.load(ROOT/'cache/development_physical_reference.npz',allow_pickle=False);groups={};rows=[];summaries=[]
    for split in ['train','validation']:
        keep=cached['splits']==split;dates=cached['dates'][keep]
        # The native state row t contains the interval t-1 action. The daily
        # boundary interpolation is intentionally excluded from measured data.
        compute=cached['compute'][keep,1:];grid=cached['grid'][keep,1:];queue=cached['queue'][keep,1:]
        multipliers=[]
        base=REPO/'artifacts/dart_value_select/counterfactual_inputs/v1/selection/daily_bundles'
        for date in dates:
            with np.load(base/(str(date)+'.npz'),allow_pickle=False) as raw:multipliers.append(raw['physical_multiplier'].copy())
        load=np.asarray(multipliers)[:,None]*(IDLE+K*compute)
        pv_used=np.minimum(cached['pv'][keep,:95],np.maximum(load-grid,0.))
        values=metrics(compute,grid,queue,pv_used,ramp_axis=1);groups[split]=values
        rows+=describe('W1_'+split,values,groups.get('train') if split!='train' else None)
        summaries.append(dict(source='W1_'+split,days=int(keep.sum()),action_intervals_per_day=95,
            shifted_it_mwh=None,shifted_energy_scope='not a continuous conserved-work replay: native teacher expires work and resets each day',
            service_timing_scope='queue/capacity clearance-time diagnostic, not cohort waiting time'))
    with np.load(PAPER/'revision_20260918/annual_continuous_2025/data/annual_forcing_2025.npz') as force:
        forcing=force['actual_21_hourly_pricing'];times=pd.to_datetime(force['timestamps_utc'],utc=True)
    arr=forcing[:,12,None]*np.ones((1,4))/4;pue=forcing[:,17:21];pv=forcing[:,13:17]
    cases={'original_learned_RASC':'legacy_rasc','EF_7h':'boundary_1_w7_robust','EF_12h':'original_w12_robust','EF_24h':'original_w24_robust'}
    old=PAPER/'revision_20260920_market_expansion'
    annual=pd.concat([pd.read_csv(old/'results/physical_screen_results.csv'),pd.read_csv(old/'deep_results/boundary_refinement_results.csv')],ignore_index=True).set_index('name')
    for name,file in cases.items():
        z=np.load(old/'data'/(file+'_trajectory.npz'));compute=z['compute'];grid=z['grid']
        queue=np.maximum(np.cumsum(arr-compute,axis=0),0.)
        pv_used=np.clip(pue*(IDLE+K*compute)-grid,0,pv)
        values=metrics(compute,grid,queue,pv_used,ramp_axis=0);rows+=describe(name,values,groups['train'])
        row=annual.loc[file]
        summaries.append(dict(source=name,days=365,action_intervals=35040,shifted_it_mwh=float(row.shifted_IT_mwh),
            shifted_energy_scope='positive IT reduction relative to continuous earliest service; closing work retained',
            service_timing_scope='queue/capacity clearance-time diagnostic; cohort waiting is separately reported in policy results'))
    pd.DataFrame(rows).to_csv(ROOT/'w1_ood_metrics.csv',index=False)
    pd.DataFrame(summaries).to_csv(ROOT/'results/w1_ood_energy_and_timing.csv',index=False)
    frame=pd.DataFrame(rows);control=frame[frame.source.str.startswith('EF_')&frame.metric.isin(['compute_power_mw','utilization','quarter_hour_ramp_mw','grid_import_mw'])]
    note=dict(status='distribution_complete_predictive_diagnostic_pending',
        far_envelope_definition='outside training 0.1%-99.9% envelope by more than 10% of its span',
        control_far_fraction_max=float(control.far_outside_training_envelope_fraction.max()),
        source_2025_role='existing frozen-policy action traces only; no R2 fitting or calibration from these outcomes',
        no_action_adapter_trained=True,
        teacher_semantics='Native 49-column state carries previous-interval execution; overdue work expires and days reset. This differs from continuous EF obligations.')
    (ROOT/'results/w1_ood_distribution_summary.json').write_text(json.dumps(note,indent=2));print(json.dumps(note,indent=2))
if __name__=='__main__':main()
