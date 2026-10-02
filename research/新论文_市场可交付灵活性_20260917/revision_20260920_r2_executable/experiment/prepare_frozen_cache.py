"""Cache causal development representations from the selected frozen world model.

Only 2022--2024 world inputs are opened here. Market-context tokens are rebuilt
from completed lagged prices. Weather is queried at the pre-DA cutoff instead
of the native delivery-day rolling origin. No old/new learned action is used.
"""
from pathlib import Path
from dataclasses import replace
import argparse,copy,gc,importlib.util,json,os,sys,time
os.environ.update(PYTHONUTF8='1',CUDA_LAUNCH_BLOCKING='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT.parent;REPO=ROOT.parents[1]/'power-dc-vpp';sys.path.insert(0,str(REPO/'src'))
import pyarrow.dataset
import numpy as np,pandas as pd,torch
from dcvpp.vpp_world_moe.config import _read_config_with_base,WorldMoEConfig
from dcvpp.vpp_world_moe.model import VPPWorldMoE,load_plannerless_released_w1_state
from dcvpp.vpp_world_moe.data import (WorldMoECorpus,_dynamic_graph,SOURCE_ISSUE_MARKET,QUALITY_FROZEN_DERIVED,
    WEATHER_FEATURES,NYISO_WEATHER_SITES,WEATHER_TEMPERATURE_CENTER_K,WEATHER_TEMPERATURE_SCALE_K,WEATHER_SHORTWAVE_SCALE_WM2)

TZ='America/New_York';ZONES=['WEST','CAPITL','N.Y.C.','LONGIL'];INACTIVE=(1,9,17,25)
PRICE_PANEL=PAPER/'experiments/gate_hourly_panel.parquet'
CACHE_SCHEMA='r2.causal_issue_reference_context.v1'
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8');temp.replace(p)
def cutoff(day):
    return (pd.Timestamp(day).normalize()-pd.Timedelta(days=1)).tz_localize(TZ).tz_convert('UTC')
def generation(day):
    return (pd.Timestamp(day).normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=4,minutes=55)).tz_localize(TZ).tz_convert('UTC')

def disable_inactive_slots(packet,*,require_reference_only=True):
    changed={}
    for prefix in ['fast','medium','slow']:
        v=getattr(packet,prefix+'_values');m=getattr(packet,prefix+'_mask')
        if v is not None:
            v=v.clone();v[...,list(INACTIVE),:]=0;v[...,38,1]=0;changed[prefix+'_values']=v
        if m is not None:
            m=m.clone();m[...,list(INACTIVE)]=False;changed[prefix+'_mask']=m
    actions=packet.previous_actions.clone();actions[...,1]=0
    masks=packet.action_mask.clone();masks[...,1]=False
    # At origin zero only the explicitly issued reference DA quantity is known.
    if require_reference_only:assert not masks[...,2:].any()
    slacks=packet.constraint_slacks.clone();slacks[...,0]=0;slacks[...,6:8]=0
    incident=torch.zeros(packet.graph_edge_index.shape[-1],dtype=torch.bool,device=actions.device)
    for slot in INACTIVE:incident|=(packet.graph_edge_index[0]==slot)|(packet.graph_edge_index[1]==slot)
    for key in ['graph_edge_mask','dynamic_graph_edge_mask']:
        value=getattr(packet,key)
        if value is not None:value=value.clone();value[...,incident]=False;changed[key]=value
    c=packet.constraints
    limits=replace(c,battery_power_limit=torch.zeros_like(c.battery_power_limit),
        battery_energy_capacity=torch.zeros_like(c.battery_energy_capacity),initial_soc=torch.zeros_like(c.initial_soc),
        migration_capacity=torch.zeros_like(c.migration_capacity),migration_allowed=torch.zeros_like(c.migration_allowed))
    return replace(packet,**changed,previous_actions=actions,action_mask=masks,constraint_slacks=slacks,constraints=limits)

def causal_weather(self,record,*,origin):
    assert origin==0
    d=str(record['operating_date']);asof=cutoff(d)
    times=pd.date_range(pd.Timestamp(d,tz='UTC'),periods=96,freq='15min')
    result=self.weather_store.query_causal_array_as_of(asof.to_pydatetime(),valid_times=tuple(t.to_pydatetime() for t in times),
        source=self.weather_source,zones=tuple(NYISO_WEATHER_SITES),features=WEATHER_FEATURES)
    raw=torch.from_numpy(result.values);available=torch.from_numpy(result.mask).bool()
    available[...,0]&=raw[...,0].ge(180)&raw[...,0].le(340)
    available[...,1]&=raw[...,1].ge(-5)&raw[...,1].le(1600)
    complete=available.all(dim=(-1,-2))
    values=torch.cat(((raw[...,0]-WEATHER_TEMPERATURE_CENTER_K)/WEATHER_TEMPERATURE_SCALE_K,
        raw[...,1].clamp_min(0)/WEATHER_SHORTWAVE_SCALE_WM2),dim=-1).float()
    return torch.where(complete[:,None],values,0.),complete

class MarketContext:
    def __init__(self,evaluation_panel=None):
        columns=['timestamp_utc','zone','date','split','feature_last_interval_end_utc','da_lbmp_lag2','rt_integrated_lag2','rt_integrated_mean7']
        frame=pd.read_parquet(PRICE_PANEL,columns=columns)
        assert frame.date.max()<='2024-12-31'
        self.columns=['da_lbmp_lag2','rt_integrated_lag2','rt_integrated_mean7']
        self.imputation=frame[frame['split'].eq('train')].groupby('zone')[self.columns].median().reindex(ZONES).to_numpy()
        if evaluation_panel is not None:
            assert (ROOT/'EVALUATION_FREEZE.json').exists()
            future=pd.read_parquet(evaluation_panel)
            future=future[future.date.ge('2025-01-01')].copy();future['split']='evaluation_2025'
            frame=pd.concat([frame,future[columns]],ignore_index=True)
        self.frame=frame.set_index(['timestamp_utc','zone']).sort_index();self.missing_cells=0
    def overlay(self,packet,records,statistics,graph,*,require_reference_only=True):
        raw=[];last=[]
        for record in records:
            d=str(record['operating_date']);times=pd.date_range(pd.Timestamp(d,tz='UTC'),periods=96,freq='15min').floor('h')
            ix=pd.MultiIndex.from_product([times,ZONES],names=['timestamp_utc','zone'])
            q=self.frame.reindex(ix)
            ends=pd.to_datetime(q.feature_last_interval_end_utc,utc=True)
            assert (ends.dropna()<=cutoff(d)).all()
            values=q[self.columns].to_numpy().reshape(96,4,3)
            missing=~np.isfinite(values);self.missing_cells+=int(missing.sum())
            values=np.where(missing,self.imputation[None],values)
            da=values[...,0];rt=values[...,1];sigma=np.maximum(np.abs(rt-values[...,2]),1.)
            raw.append(np.stack([da,rt,da-rt,sigma,rt-1.28155*sigma,rt+1.28155*sigma,np.abs(da-rt)],axis=-1))
            last.append(ends.max().isoformat() if ends.notna().any() else None)
        values=torch.as_tensor(np.stack(raw),dtype=torch.float32,device=packet.fast_values.device)
        x=packet.fast_values.clone();center=statistics.fast_center.to(x.device);scale=statistics.fast_scale.to(x.device)
        if center.ndim==1:center=center[None].expand(40,-1)
        if scale.ndim==1:scale=scale[None].expand(40,-1)
        for z in range(4):
            slot=z*8;x[:,:,slot,:7]=(values[:,:,z]-center[slot,:7])/scale[slot,:7]
        for slot,raw_global in [(32,torch.stack([values[...,0].mean(2),values[...,1].mean(2),values[...,2].mean(2),values[...,3].mean(2)],dim=-1)),
            (33,torch.stack([values[...,3].amax(2),values[...,2].abs().amax(2),values[...,2].std(2,unbiased=False)],dim=-1)),
            (37,torch.stack([values[...,3].mean(2),values[...,2].abs().mean(2)],dim=-1))]:
            count=raw_global.shape[-1];x[:,:,slot,:count]=(raw_global-center[slot,:count])/scale[slot,:count]
        dynamic,mask=_dynamic_graph(x,packet.fast_mask,graph)
        packet=replace(packet,fast_values=x,dynamic_graph_edge_features=dynamic.to(x.device),dynamic_graph_edge_mask=mask.to(x.device))
        return disable_inactive_slots(packet,require_reference_only=require_reference_only),last

def load_world():
    selected=json.loads((ROOT/'selected_w0_w1.json').read_text(encoding='utf-8'))
    cfg=_read_config_with_base(Path(selected['config']))
    cfg['data']['migration']['enabled']=False
    cfg['model']['gradient_checkpointing']=False
    model=VPPWorldMoE(WorldMoEConfig.from_mapping(cfg['model']),include_legacy_planner=False)
    payload=torch.load(selected['w1']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    load_plannerless_released_w1_state(model,payload['model_state']);del payload
    model.eval().requires_grad_(False);model.to('cuda')
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==0
    return model,cfg,selected

def reduce_features(model,state,tokenized):
    diag=state.router_diagnostics[-1];risk=37;ids=diag.topk_indices[:,:,risk]
    weights=diag.topk_weights[:,:,risk].float()
    moe=model.backbone.blocks[max(model.config.moe_block_indices)].moe
    effective=diag.captured_pre_state[:,:,None].float()+torch.sigmoid(moe.output_gate.float())*diag.captured_selected_expert_states.float()
    bank=[];priors=[]
    for expert in range(7):
        w=weights*(ids==expert);mass=w.sum((1,2));priors.append(mass/weights.shape[1])
        bank.append((effective*w[...,None]).sum((1,2))/mass[:,None].clamp_min(1e-8))
    local=[0,8,16,24]
    fields=dict(w0_local=tokenized[:,:,local].float().mean(1),w1_local=state.slots[:,:,local].float().mean(1),
        w1_global=state.pooled_timeline.float().mean(1),expert_bank=torch.stack(bank,dim=1),expert_prior=torch.stack(priors,dim=1),
        expert_ids=ids,expert_weights=weights,
        tail_context=(diag.captured_tail_state.float()*diag.tail_active_mask[...,None]).mean(1)*moe.tail_residual_scale,
        tail_probability=diag.tail_route_probabilities.float().mean(1),regime=state.regime_probabilities[:,:,risk].float().mean(1),
        log_scale=state.stochastic_log_scale[:,:,risk].float().mean(1))
    return {key:value.detach().cpu().numpy().astype(np.int16 if key=='expert_ids' else np.float16) for key,value in fields.items()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=int,default=12);parser.add_argument('--max-days',type=int,default=0)
    args=parser.parse_args();start=time.perf_counter();torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    model,cfg,selected=load_world()
    recurrence=PAPER/'revision_20260918/annual_native_2025/performance_v2/gpu_recurrence.py'
    spec=importlib.util.spec_from_file_location('r2_frozen_recurrence',recurrence);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.install(model)
    market=MarketContext()
    corpus=WorldMoECorpus(cfg,REPO,materialize=False)
    import types
    corpus._weather_context=types.MethodType(causal_weather,corpus)
    records=sorted(corpus.records('trainval',require_h96=False),key=lambda r:str(r['operating_date']))
    if args.max_days:records=records[:args.max_days]
    save(ROOT/'cache/development_cache_manifest.json',dict(status='running',schema=CACHE_SCHEMA,selected_seed=selected['seed'],
        selected_w1_id=selected['w1']['recorded_checkpoint_id'],expected_days=len(records),no_2025_opened=True,
        market_context='completed lag-2 DA/RT profiles; dispersion abs(lag2 RT - mean7 RT), floor USD 1',
        world_context='origin-zero explicitly issued reference actions; no executed future actions',
        weather_asof='D-1 00:00 America/New_York',generation='D-1 04:55 America/New_York',
        inactive_slots=list(INACTIVE),representation_summary='daily issued-context summaries; R2 target-hour features supply forward lead and calendar',
        pretrained_parameters_trainable=0))
    captures={}
    hook=model.tokenizer.register_forward_hook(lambda m,ins,out:captures.update(tokens=out[0]))
    pieces=[];times=[];sources=[];inference_seconds=0.
    for offset in range(0,len(records),args.batch):
        batch=records[offset:offset+args.batch];file=ROOT/'cache'/f'development_{offset:05d}_{offset+len(batch):05d}.npz'
        dates=[str(r['operating_date']) for r in batch]
        for r in batch:
            assert pd.Timestamp(r['issue_time_utc'])<=cutoff(str(r['operating_date']))
            for date in r['medium_source_dates']+r['slow_source_dates']:
                if date is not None:assert pd.Timestamp(str(date),tz='UTC')+pd.Timedelta(days=1)<=cutoff(str(r['operating_date']))
        if file.exists():
            data=dict(np.load(file,allow_pickle=False));assert data['dates'].tolist()==dates
        else:
            packet=corpus._packet(batch,origin=0,device=torch.device('cpu'))
            packet,last=market.overlay(packet,batch,corpus.statistics,corpus.graph)
            assert not packet.fast_mask[...,list(INACTIVE)].any()
            from dcvpp.vpp_world_moe.r_ecmoe_runtime import _to_device
            packet=_to_device(packet,torch.device('cuda'))
            t=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                state=model.encode(packet,sample_stochastic=False,phase='w1',capture_expert_context=True)
                data=reduce_features(model,state,captures.pop('tokens'))
            torch.cuda.synchronize();inference_seconds+=time.perf_counter()-t
            data.update(dates=np.asarray(dates),source_issue=np.asarray([str(r['issue_time_utc']) for r in batch]),
                price_cutoff=np.asarray([cutoff(d).isoformat() for d in dates]),generation=np.asarray([generation(d).isoformat() for d in dates]))
            np.savez_compressed(file,**data);del packet,state
        pieces.append(file.name);times+=dates
        if offset==0 or (offset//args.batch)%10==0:
            elapsed=time.perf_counter()-start
            print(json.dumps(dict(completed_days=offset+len(batch),total_days=len(records),elapsed_seconds=round(elapsed,1),
                inference_seconds=round(inference_seconds,1),gpu_allocated_mib=round(torch.cuda.memory_allocated()/1048576,1))),flush=True)
            save(ROOT/'cache/progress.json',dict(completed_days=offset+len(batch),total_days=len(records),elapsed_seconds=elapsed,inference_seconds=inference_seconds))
    hook.remove()
    teacher=[]
    for r in records:
        base=r['base_record'];target=base['scenario_target_zonal'].cpu().numpy();state=base['state_sequence'].cpu().numpy()
        teacher.append(dict(date=str(r['operating_date']),split=str(r['split']),compute=state[:,20:24],grid=state[:,12:16],
            queue=state[:,4:8],arrivals=target[:,12,None]*np.ones((1,4))/4,pv=target[:,13:17],pue=target[:,17:21]))
    np.savez_compressed(ROOT/'cache/development_physical_reference.npz',dates=np.asarray([x['date'] for x in teacher]),
        splits=np.asarray([x['split'] for x in teacher]),**{key:np.stack([x[key] for x in teacher]) for key in ['compute','grid','queue','arrivals','pv','pue']})
    manifest=json.loads((ROOT/'cache/development_cache_manifest.json').read_text(encoding='utf-8'))
    manifest.update(status='complete' if not args.max_days else 'probe_complete',completed_days=len(times),shards=pieces,
        first_date=min(times),last_date=max(times),seconds=time.perf_counter()-start,inference_seconds=inference_seconds,
        price_context_imputed_cells=market.missing_cells,total_frozen_parameters=sum(p.numel() for p in model.parameters()))
    save(ROOT/'cache/development_cache_manifest.json',manifest);print('CACHE COMPLETE',manifest['seconds'],flush=True)
if __name__=='__main__':main()
