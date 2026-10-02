"""Selected W1 and its released old R, with three continuous service clocks."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,pickle,sys,time
os.environ.update(PYTHONUTF8='1',CUDA_LAUNCH_BLOCKING='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT.parent;REPO=ROOT.parents[1]/'power-dc-vpp'
sys.path[:0]=[str(REPO/'src'),str(PAPER/'revision_20260919_nopenalty/experiment'),str(PAPER/'revision_20260920_service_frontier/experiment')]
import pyarrow.dataset
import numpy as np,pandas as pd,torch
from no_storage_adapter import NoStorageAdapter
from input_cache import CacheAwareFeedback
from collections import OrderedDict
from neural_adapter import FrozenResources
from window_core import WindowState,snapshot,physical_row,deliver,reference_action,DT,FEE
from forecast_adapter import ForecastAdapter
from dcvpp.vpp_world_moe.config import _read_config_with_base
from dcvpp.vpp_world_moe.r_ecmoe_runtime import RECMoEInferenceRuntime
from dcvpp.vpp_world_moe.r_ecmoe_sealed_corpus import build_sealed_final_corpus
from prepare_frozen_cache import save

N=35040;WINDOWS=(7,12,24);O=ROOT/'neural_2025';O.mkdir(exist_ok=True)
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x
def flat(row):
    out={}
    for k,v in row.items():
        if isinstance(v,np.ndarray):out.update({f'{k}_{z}':float(x) for z,x in enumerate(v.ravel())})
        elif np.isscalar(v) or v is None:out[k]=v
    return out
def load():
    start=time.perf_counter();s=json.loads((ROOT/'selected_w0_w1.json').read_text())
    cfg=_read_config_with_base(Path(s['config']));rp=REPO/'configs/multiseed_20260910_20240829_r_full.yaml'
    rc=_read_config_with_base(rp);cp=REPO/'artifacts/r_ecmoe_multiseed_20260910/r/multiseed-20260910-seed20240829-full/checkpoints/best_decision_value.pt'
    runtime=RECMoEInferenceRuntime.from_w1_checkpoint(cfg,rc,s['w1']['checkpoint'],'cuda',r_checkpoint_path=cp)
    corpus=build_sealed_final_corpus(config=cfg,repository_root=REPO,frozen_evaluation_marker=json.loads((ROOT/'cache/frozen_evaluation_marker.json').read_text()))
    records={str(r['operating_date']):r for r in corpus.records('locked_test',require_h96=False)}
    rec=dict(checkpoint=str(cp),w1_checkpoint=s['w1']['checkpoint'],config_source=str(rp))
    resources=FrozenResources(runtime.model,corpus,rec,cfg,rc,torch.device('cuda'),records,sorted(records),time.perf_counter()-start)
    return resources
def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    started=time.perf_counter();resources=load();names=[f'h{w}' for w in WINDOWS]
    adapter=NoStorageAdapter(resources=resources)
    # The cache constructor checks the legacy names; afterwards use three
    # independent service-clock histories, all executing the same old R rule.
    adapter.policies=tuple(names);adapter.histories={}
    for name in names:
        history=CacheAwareFeedback(resources,policy_id=name);history.history=OrderedDict();history.snapshots=OrderedDict();adapter.histories[name]=history
    adapter.clear_input_cache();prices=ForecastAdapter()
    z=np.load(PAPER/'revision_20260918/annual_continuous_2025/data/annual_forcing_2025.npz')
    forcing=z['actual_21_hourly_pricing'];times=pd.to_datetime(z['timestamps_utc'],utc=True)
    checkpoint=O/'checkpoint.pkl'
    if checkpoint.exists():
        prior=pickle.loads(checkpoint.read_bytes());states=prior['states'];refs=prior['refs'];previous=prior['previous'];c=prior['c'];totals=prior['totals']
        adapter.import_history(prior['history'])
    else:
        opening=adapter.initial_state_once(times[0]);c={k:np.asarray(v,float) for k,v in opening[names[0]]['constraints'].items()}
        states={f'h{w}':WindowState(window_steps=w*4) for w in WINDOWS};refs={f'h{w}':WindowState(window_steps=w*4) for w in WINDOWS}
        previous={k:np.zeros(4) for k in states};totals={k:0. for k in states}
    first=states[names[0]].next_step
    queue_price=prices.terminal_rt_median_usd_per_mwh*DT*c['pue_max']*c['work_power_per_unit']
    save(O/'protocol.json',dict(model='Old_R_action',seed=20240829,R_epoch=78,windows_h=WINDOWS,steps=N,continuous=True,initializations=1,
        gate='unconditional original learned action',hourly_neural_refresh=True,physical_step_minutes=15,independent_batch=3,
        frozen=True,backbone_trainable=0,action_head_retrained=False,legacy_action_rows_preserved_in_source=True,
        penalties=0,neural_checkpoint=resources.record,deadline_priority_override=False,
        boundary='deadline clock feedback and accounting; old learned controller is not a hard-deadline executor'))
    save(O/'process.json',dict(status='running',pid=os.getpid(),start_step=first,load_seconds=time.perf_counter()-started))
    pending=[];chunk=first;began=time.perf_counter()
    for step in range(first,N):
        t=times[step];origin=t.hour*4+t.minute//15
        proposal=adapter.forecast(t,{k:snapshot(s,previous[k]) for k,s in states.items()},horizon_steps=min(96-origin,N-step))
        physical=physical_row(forcing[step],c);requests=adapter.assemble_actual(t,{k:True for k in names},physical);qda=np.asarray(requests['shared_qda'],float)
        for w in WINDOWS:
            k=f'h{w}';state=states[k];refstate=refs[k]
            ref=refstate.apply(step,physical,reference_action(physical,refstate,c),c,forcing[step,:4],forcing[step,4:8],qda)
            action=deliver(requests['policies'][k],physical,state,c)
            row=state.apply(step,physical,action,c,forcing[step,:4],forcing[step,4:8],qda);after=snapshot(state,action['compute'])
            adapter.observe(k,t,soc_after=np.zeros(4),queue_after=state.queue,grid=action['grid'],compute=action['compute'],charge=np.zeros(4),discharge=np.zeros(4),pv_used=action['pv_used'],arrived=physical['arrivals'],expired=row['expired_work'],deadline_fraction_after=after['deadline_fraction'],pv_curtailed=physical['pv']-action['pv_used'])
            previous[k]=action['compute'].copy();fee=float(FEE*DT*np.abs(ref['grid']-row['grid']).sum());value=ref['raw_cost']-row['raw_cost']-fee
            liability=float(np.sum(queue_price*(state.queue-refstate.queue)));totals[k]+=value
            pending.append(flat(dict(**row,timestamp_utc=t.isoformat(),window_h=w,model='Old_R_action',reference_grid=ref['grid'],reference_compute=ref['compute'],reference_queue=refstate.queue,reference_raw_cost=ref['raw_cost'],reference_expired_work=ref['expired_work'],fee=fee,value=value,service_liability=liability)))
        nxt=step+1
        if nxt%672==0 or nxt==N:
            pd.DataFrame(pending).to_parquet(O/f'ledger_{chunk:05d}_{nxt:05d}.parquet',index=False);pending=[];chunk=nxt
            payload=dict(states=states,refs=refs,previous=previous,c=c,totals=totals,history=adapter.export_history())
            temp=checkpoint.with_suffix('.tmp');temp.write_bytes(pickle.dumps(payload,protocol=pickle.HIGHEST_PROTOCOL));temp.replace(checkpoint)
        if nxt%96==0 or nxt==N:
            elapsed=time.perf_counter()-began
            status=dict(status='running' if nxt<N else 'complete',step=nxt,total=N,elapsed_seconds=elapsed,seconds_per_step=elapsed/max(nxt-first,1),estimated_remaining_seconds=(N-nxt)*elapsed/max(nxt-first,1))
            save(O/'progress.json',status);print(json.dumps(status),flush=True)
    summaries={k:dict(cash_value=totals[k],closing_liability=float(np.sum(queue_price*(s.queue-refs[k].queue))),final_state=s.snapshot(),late_work=s.expired,late_served_work=s.late_served,arrived_work=s.arrived,served_work=s.served,initializations=s.initialization_count) for k,s in states.items()}
    save(O/'completed.json',plain(dict(status='complete',steps=N,policies=summaries,constraints=c,queue_price=queue_price,seconds=time.perf_counter()-started)))
    save(O/'process.json',dict(status='complete',pid=os.getpid()))
if __name__=='__main__':main()
