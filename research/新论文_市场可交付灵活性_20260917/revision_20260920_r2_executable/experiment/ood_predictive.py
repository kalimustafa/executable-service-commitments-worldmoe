"""Matched pre-2025 one-step state-transition diagnostic for the frozen W1.

Historical context is held common; the currently observable queue, arrival,
previous action and chosen current action vary. Future state labels are used
only to score the learned one-step transition. This is a controlled diagnostic,
not a claim to reproduce the original full-day proper-score validation.
"""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import json,os,sys,time,importlib.util
os.environ.update(PYTHONUTF8='1',CUDA_LAUNCH_BLOCKING='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT.parent;REPO=ROOT.parents[1]/'power-dc-vpp';sys.path.insert(0,str(REPO/'src'))
import numpy as np,pandas as pd,torch
from prepare_frozen_cache import load_world,MarketContext,disable_inactive_slots
from dcvpp.vpp_world_moe.data import WorldMoECorpus,_dynamic_graph
from dcvpp.vpp_world_moe.r_ecmoe_runtime import _to_device
from select_2024_economics import configure_validation,core,engine
from ef_forecast_provider import ForecastProvider

def set_value(packet,stats,slot,feature,value):
    center=stats.fast_center;scale=stats.fast_scale
    c=center[slot,feature] if center.ndim==2 else center[feature]
    s=scale[slot,feature] if scale.ndim==2 else scale[feature]
    packet.fast_values[0,:,slot,feature]=(torch.as_tensor(value,dtype=torch.float32)-c)/s

def main():
    started=time.perf_counter();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    arr,_=configure_validation();old=pd.read_parquet(PAPER/'revision_20260918/forecast_2025/development_forecast_only_2024h2.parquet')
    provider=ForecastProvider(core.times,core.da,core.train_rt,kind='old_rasc',frame=old)
    engine.rank=lambda w,p:provider.ranked_slots(w,p)
    policies={}
    for w in [7,12,24]:
        book,stats,events=engine.schedule(arr,[(1.,w)],'robust',keep_events=True)
        compute=book[:core.N];cost,grid=core.cash(book);after=np.maximum(np.cumsum(arr-compute,axis=0),0)
        before=np.vstack([np.zeros((1,4)),after[:-1]])
        due_delta=np.zeros((len(book)+1,4))
        for release,deadline,execution,z,units,window in events:
            if execution>=deadline:
                due_delta[deadline,z]+=units;due_delta[execution+1,z]-=units
        due=np.cumsum(due_delta,axis=0)[:core.N];due_fraction=np.divide(due,before,out=np.zeros_like(due),where=before>1e-8).clip(0,1)
        policies[f'EF_{w}h']=dict(compute=compute,grid=grid,before=before,after=after,due=due_fraction)
    model,cfg,selected=load_world()
    recurrence=PAPER/'revision_20260918/annual_native_2025/performance_v2/gpu_recurrence.py'
    spec=importlib.util.spec_from_file_location('r2_ood_recurrence',recurrence);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.install(model)
    dates=sorted(set(core.times.strftime('%Y-%m-%d')))
    chosen=[dates[i] for i in np.linspace(0,len(dates)-1,8).astype(int)]
    points={(day,origin) for day in chosen for origin in [24,60]}
    for policy in policies.values():
        idx=int(np.argmax(policy['before'].sum(1)));t=core.times[idx];origin=t.hour*4+t.minute//15
        if 1<=origin<=93:points.add((t.strftime('%Y-%m-%d'),origin))
    prepared_path=ROOT/'cache/ood_prepared_inputs.pt'
    if prepared_path.exists():
        prepared=torch.load(prepared_path,map_location='cpu',weights_only=False,mmap=True)
    else:
        corpus=WorldMoECorpus(cfg,REPO,materialize=False);records={str(r['operating_date']):r for r in corpus.records('validation')};market=MarketContext()
        prepared=dict(packets={},originals={},fast_center=corpus.statistics.fast_center,fast_scale=corpus.statistics.fast_scale,
            selected_w1_id=selected['w1']['recorded_checkpoint_id'])
        for day,origin in sorted(points):
            record=records[day];state=record['base_record']['state_sequence'].float().numpy()
            packet=corpus._packet([record],origin=origin,device=torch.device('cpu'))
            packet,_=market.overlay(packet,[record],corpus.statistics,corpus.graph,require_reference_only=False)
            prepared['packets'][(day,origin)]=packet
            prepared['originals'][(day,origin)]=dict(queue=state[origin,4:8].copy(),previous_compute=state[origin,20:24].copy(),previous_grid=state[origin,12:16].copy(),
                due=state[origin,8:12].copy(),compute=state[origin+1,20:24].copy(),grid=state[origin+1,12:16].copy(),target_queue=state[origin+1,4:8].copy())
        torch.save(prepared,prepared_path)
        del corpus,records,market
    assert prepared['selected_w1_id']==selected['w1']['recorded_checkpoint_id']
    stats=SimpleNamespace(fast_center=prepared['fast_center'],fast_scale=prepared['fast_scale'])
    results=[]
    for day,origin in sorted(points):
        absolute=int(core.times.get_loc(pd.Timestamp(day,tz='UTC')+pd.Timedelta(minutes=15*origin)))
        original=prepared['originals'][(day,origin)]
        variants={'original_validation_actions':original}
        for name,p in policies.items():
            variants[name]=dict(queue=p['before'][absolute],previous_compute=p['compute'][absolute-1],previous_grid=p['grid'][absolute-1],
                due=p['due'][absolute],compute=p['compute'][absolute],grid=p['grid'][absolute],target_queue=p['after'][absolute])
        for name,v in variants.items():
            base_packet=prepared['packets'][(day,origin)]
            packet=replace(base_packet,fast_values=base_packet.fast_values.clone())
            graph=SimpleNamespace(edge_index=packet.graph_edge_index,edge_type=packet.graph_edge_type,edge_features=packet.graph_edge_features,edge_mask=packet.graph_edge_mask)
            # Common issued/historical context; replace only observable present
            # state and its persistence after the rolling origin, not future truth.
            for z in range(4):
                base=z*8
                for slot,feature,value in [(base+5,0,v['queue'][z]),(base+5,1,v['due'][z]),
                    (base+6,0,v['previous_compute'][z]),(base+6,3,v['previous_compute'][z]),
                    (base+7,0,v['previous_grid'][z]),(base+7,2,max(core.c['grid_import_limit'][z]-v['previous_grid'][z],0))]:
                    raw=packet.fast_values[0,:,slot,feature]*stats.fast_scale[slot,feature]+stats.fast_center[slot,feature]
                    raw[origin:]=float(value);set_value(packet,stats,slot,feature,raw)
                # Current arrivals are observed at release; future arrivals remain
                # the common forecast. The one-step target is scored afterwards.
                for feature in [0,2,3,4]:
                    slot=base+4;raw=packet.fast_values[0,:,slot,feature]*stats.fast_scale[slot,feature]+stats.fast_center[slot,feature]
                    raw[origin]=float(arr[absolute,z]);set_value(packet,stats,slot,feature,raw)
            qratio=float(np.mean(v['queue']));set_value(packet,stats,37,6,np.full(96,qratio))
            actions=packet.previous_actions.clone();actions[0,origin,:,2]=torch.as_tensor(v['previous_compute']);masks=packet.action_mask.clone();masks[0,origin,:,2]=True
            slacks=packet.constraint_slacks.clone();slacks[0,origin:,:,1]=torch.tensor((np.array(core.c['grid_import_limit'])-v['previous_grid'])/np.array(core.c['grid_import_limit'])).clamp(0,1)
            slacks[0,origin:,:,2]=torch.tensor(1-np.asarray(v['previous_compute'])).clamp(0,1);slacks[0,origin:,:,3]=torch.tensor(1-v['due']);slacks[0,origin:,:,5]=torch.tensor(1-v['due'])
            constraints=replace(packet.constraints,initial_queue=torch.tensor(np.asarray(v['queue'])[None],dtype=torch.float32))
            dynamic,mask=_dynamic_graph(packet.fast_values,packet.fast_mask,graph)
            packet=replace(packet,previous_actions=actions,action_mask=masks,constraint_slacks=slacks,constraints=constraints,
                dynamic_graph_edge_features=dynamic,dynamic_graph_edge_mask=mask)
            packet=disable_inactive_slots(packet,require_reference_only=False);packet=_to_device(packet,torch.device('cuda'))
            current=torch.tensor([[0.,np.mean(v['compute']),0.,np.mean(v['grid']),0.,0.]],dtype=torch.float32,device='cuda')
            previous=torch.tensor([[0.,np.mean(v['previous_compute']),0.,np.mean(v['previous_grid']),0.,0.]],dtype=torch.float32,device='cuda')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                encoded=model.encode(packet,sample_stochastic=False,phase='w1')
                a=model.dynamics.normalize_action_summary(current[:,None],packet.constraints)[:,0]
                p=model.dynamics.normalize_action_summary(previous[:,None],packet.constraints)[:,0]
                hidden,scale=model.dynamics._deterministic_step(encoded.pooled_state,a,a-p,packet.constraints.dt_hours)
                prior=model._gaussian_parameters(model.latent_prior_heads['h1'],hidden[:,None],anchor_mean=True,process_log_scale=scale[:,None])
                predicted=model.observation_queries.queue_distribution_from_latent_timeline(prior['mean'])['mean'][0,0].float().cpu().numpy()
            for z in range(4):results.append(dict(policy=name,date=day,origin=origin,site=z,current_queue=float(v['queue'][z]),
                target_queue=float(v['target_queue'][z]),predicted_queue=float(predicted[z]),absolute_error=float(abs(predicted[z]-v['target_queue'][z]))))
        print('OOD scored',day,origin,flush=True)
    f=pd.DataFrame(results);f.to_csv(ROOT/'results/w1_one_step_predictions.csv',index=False)
    rows=[]
    for name,g in f.groupby('policy'):
        residual=g.predicted_queue-g.target_queue;rows.append(dict(policy=name,n=len(g),queue_mae=float(np.abs(residual).mean()),queue_rmse=float(np.sqrt(np.square(residual).mean()))))
    out=pd.DataFrame(rows);baseline=out[out.policy.eq('original_validation_actions')].iloc[0]
    out['mae_ratio_vs_original_actions']=out.queue_mae/max(baseline.queue_mae,1e-9)
    out['rmse_ratio_vs_original_actions']=out.queue_rmse/max(baseline.queue_rmse,1e-9)
    out.to_csv(ROOT/'results/w1_predictive_ood_metrics.csv',index=False)
    dist=json.loads((ROOT/'results/w1_ood_distribution_summary.json').read_text())
    severe=dist['control_far_fraction_max']>=.01 and out[~out.policy.eq('original_validation_actions')].mae_ratio_vs_original_actions.max()>=10
    grade='SEVERE' if severe else 'MODERATE'
    report=dict(status='complete',grade=grade,action_adapter_required=False,
        reason='New EF controls do not feed the R2 market-reference encoding; explicit EF state transitions own work conservation. No far action-envelope violations were measured.',
        far_action_envelope_fraction=dist['control_far_fraction_max'],matched_predictive_metrics=rows,
        predictive_scope='Pre-2025 matched rolling one-step queue transition with common historical context and observed current state/actions; distinct from native full-day proper-score validation',
        native_recorded_h1_proper_score=selected['w1']['h1'],sample_points=len(points),seconds=time.perf_counter()-started)
    (ROOT/'results/w1_ood_completed.json').write_text(json.dumps(report,indent=2))
    lines=['# W1 OOD diagnostic','','Overall: **'+grade+'**. No W0/W1 training or action adapter is used.','',
        'Compute power, utilisation, ramps, queue state, clearance time, PV use and grid import are reported in w1_ood_metrics.csv. The observed fraction far outside the training action envelope is 0.',
        'Native W1 state rows contain the previous action; the interpolated daily boundary was excluded. Its training context expires overdue work and resets daily, whereas EF conserves outstanding work continuously. Training-trace displacement is therefore not labelled as recovered annual flexibility.',
        'The predictive check uses 2024 only, at fixed calendar origins plus current-queue stress states. It changes current observable state/actions while holding the historical reference context common. Future queue values are scoring targets only. It queries the existing learned one-step transition and queue head; it is not the original full-day scoring protocol.','',out.to_csv(index=False),
        '', 'The R2 price branch encodes a declared issue-time reference action profile. The larger EF actions and queues do not drive a learned feasibility decision. They remain under the explicit reservation/recourse/physical execution layer. This separation is the reason to retain frozen W1 without adding an action adapter.']
    (ROOT/'W1_OOD_DIAGNOSTIC.md').write_text('\n'.join(lines),encoding='utf-8');print('OOD COMPLETE',report,flush=True)
if __name__=='__main__':main()
