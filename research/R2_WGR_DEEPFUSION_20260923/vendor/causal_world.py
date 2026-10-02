"""Minimal causal helper copy; no original mains or evaluation data loaders."""
from dataclasses import replace
import numpy as np
import pandas as pd
import torch
from src.runtime import ROOT, PAPER
from dcvpp.vpp_world_moe.data import (_dynamic_graph, WEATHER_FEATURES, NYISO_WEATHER_SITES,
    WEATHER_TEMPERATURE_CENTER_K, WEATHER_TEMPERATURE_SCALE_K, WEATHER_SHORTWAVE_SCALE_WM2)
TZ='America/New_York'
ZONES=['WEST','CAPITL','N.Y.C.','LONGIL']
INACTIVE=(1,9,17,25)
PRICE_PANEL=PAPER/'experiments/gate_hourly_panel.parquet'
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
            raise RuntimeError('This branch has no locked-evaluation input mode')
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
