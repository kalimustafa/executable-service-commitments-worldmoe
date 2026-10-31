"""Causal nonpreemptive Frontier execution from frozen WorldMoE slot scores.

Jobs stay at their retained fixed site for their full rounded runtime. The
decision at release uses only the final WorldMoE score issued at that step,
within its 24 h valuation horizon. Later starts use earliest feasible fallback.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
from numba import njit

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
WORKSPACE=BASE.parent
ENGINE=BASE/'engine/results'
sys.path.insert(0,str(WORKSPACE/'ServiceCommitment_1to5_20260927'))
from src import context as ctx

@njit(cache=True)
def first_fit(book,release,site,duration,units):
    for start in range(release,book.shape[0]-duration+1):
        feasible=True
        for j in range(duration):
            if book[start+j,site]+units>1.+1e-10:
                feasible=False
                break
        if feasible:return start
    return -1

@njit(cache=True)
def choose_start(book,release,site,duration,units,latest,scores):
    earliest=first_fit(book,release,site,duration,units)
    if earliest<0 or earliest>latest:return earliest
    last=min(latest,release+len(scores)-duration)
    if last<earliest or not np.isfinite(scores[0]):return earliest
    best=earliest
    best_score=1e100
    for start in range(earliest,last+1):
        feasible=True
        score=0.
        for j in range(duration):
            if book[start+j,site]+units>1.+1e-10:
                feasible=False
                break
            score+=scores[start-release+j]
        if feasible and score<best_score-1e-9:
            best=start
            best_score=score
    return best

@njit(cache=True)
def add_job(book,start,site,duration,units):
    for j in range(duration):book[start+j,site]+=units

def schedule(jobs,window,score=None):
    c=ctx.c
    book=np.zeros((c.N+8192,4),dtype=np.float64)
    starts=np.empty(len(jobs),dtype=np.int32)
    data={key:jobs[key].to_numpy() for key in ['submit_time','release_step','duration_steps','site','units']}
    begin=time.perf_counter()
    for i in range(len(jobs)):
        t=int(data['release_step'][i]);z=int(data['site'][i]);d=int(data['duration_steps'][i]);u=float(data['units'][i])
        latest=int((int(data['submit_time'][i])+window*3600)//900)-d
        start=first_fit(book,t,z,d,u) if score is None else choose_start(book,t,z,d,u,latest,score[t,z])
        if start<0:raise RuntimeError(f'Calendar exhausted at job {int(jobs.job_id.iloc[i])}')
        starts[i]=start
        add_job(book,start,z,d,u)
        if (i+1)%100000==0:print('JOBS',window,'earliest' if score is None else 'WorldMoE',i+1,len(jobs),'seconds',round(time.perf_counter()-begin,1),flush=True)
    return book,starts,time.perf_counter()-begin

def outcomes(jobs,starts,window,n):
    frame=jobs[['job_id','submit_time','requested_nodes','runtime','release_step','duration_steps','site','power_mw']].copy()
    frame['start_step']=starts
    frame['end_step_exclusive']=starts+jobs.duration_steps.to_numpy(dtype=np.int32)
    frame['completion_seconds']=frame.end_step_exclusive.to_numpy(dtype=np.int64)*900
    frame['deadline_seconds']=jobs.submit_time.to_numpy(dtype=np.int64)+window*3600
    frame['late']=frame.completion_seconds>frame.deadline_seconds
    frame['closing']=frame.end_step_exclusive>n
    frame['it_mwh']=frame.power_mw*frame.duration_steps*.25
    frame['start_delay_h']=(frame.start_step.to_numpy(dtype=np.int64)*900-frame.submit_time.to_numpy(dtype=np.int64))/3600
    return frame

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--window',type=int,choices=[24,30,48],required=True)
    args=p.parse_args()
    window=args.window
    jobs=pd.read_parquet(HERE/'mapped_jobs_2025.parquet')
    _,_,shared=ctx.configure(2025,'Frontier')
    c=ctx.c
    with np.load(HERE/'matched_job_energy_arrivals.npz') as z:arr=z['arrivals']
    ctx.RESULTS=ENGINE
    ctx.TRAJECTORIES=ENGINE/'trajectories'
    c.DATA=ctx.TRAJECTORIES
    base=f'CdfR2_Frontier_2025_s20260920_ep06_coupled_cdf_w{window}'
    score_file=ENGINE/f'{base}_slot_score.npz'
    with np.load(score_file) as z:score=z['score']
    if score.shape!=(c.N,4,96):raise RuntimeError('Score cache shape mismatch')
    missing=~np.isfinite(score[jobs.release_step.to_numpy(dtype=np.intp),jobs.site.to_numpy(dtype=np.intp),0])
    if missing.any():raise RuntimeError(f'Missing causal WorldMoE scores for {missing.sum()} jobs')
    ref_book,ref_starts,ref_seconds=schedule(jobs,window,None)
    book,starts,seconds=schedule(jobs,window,score)
    ref=outcomes(jobs,ref_starts,window,c.N)
    final=outcomes(jobs,starts,window,c.N)
    ref.to_parquet(HERE/f'Frontier_2025_w{window}_earliest_jobs.parquet',index=False)
    final.to_parquet(HERE/f'Frontier_2025_w{window}_worldmoe_jobs.parquet',index=False)
    late=float(final.loc[final.late,'it_mwh'].sum())
    name=f'Frontier_2025_w{window}_worldmoe_nonpreemptive'
    row=ctx.measure(name,book,ref_book,arr,shared,dict(experiment='FrontierNonpreemptive',year=2025,trace='Frontier',seed=20260920,window_h=window,role='frozen WorldMoE score, causal job placement'),late=late,extra=dict(
        jobs=len(jobs),late_jobs=int(final.late.sum()),late_job_it_mwh=late,
        reference_late_jobs=int(ref.late.sum()),reference_late_job_it_mwh=float(ref.loc[ref.late,'it_mwh'].sum()),
        closing_jobs=int(final.closing.sum()),closing_job_it_mwh=float(final.loc[final.closing,'it_mwh'].sum()),
        reference_closing_jobs=int(ref.closing.sum()),execution_seconds=seconds,
        reference_seconds=ref_seconds,valuation_horizon_h=24,score_source=str(score_file),
        scheduling_rule='At release, choose the lowest-score feasible contiguous start within the issued 24 h valuation horizon and contract; use earliest feasible fallback outside the horizon'))
    fluid=json.loads((ENGINE/f'{base}.json').read_text())
    row['matched_fluid_saving_usd']=fluid['saving_usd']
    row['job_to_fluid_saving_ratio']=row['saving_usd']/fluid['saving_usd'] if fluid['saving_usd'] else None
    row['matched_fluid_it_mwh']=fluid['IT_energy_mwh']
    (HERE/f'Frontier_2025_w{window}_summary.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    print(json.dumps({k:row[k] for k in ['jobs','saving_usd','matched_fluid_saving_usd','job_to_fluid_saving_ratio','late_jobs','closing_jobs']},indent=2),flush=True)

if __name__=='__main__':main()
