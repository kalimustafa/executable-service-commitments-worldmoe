"""Load the full frozen WorldMoE chain and execute one conserved-service action.

The default synthetic fixture exercises interfaces, not paper performance.
Supply --input and its --metadata JSON for your own issue-legal tensors.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time
from dataclasses import fields
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
from dcvpp.vpp_world_moe.config import WorldMoEConfig
from dcvpp.vpp_world_moe.model import VPPWorldMoE
from dcvpp.vpp_world_moe.packets import WorldInputPacket,WorldConstraintPacket
from src.models.r2_base import R2Base
from src.models.profile_residual import R2ProfileResidual
from src.models.structured_decision import StructuredDecision
from src.models.coupled_rollout import ArchitectedR2QPHead
from src.models.scenario_recourse import select_action
from vendor.r2_market import R2Config

PAIR={20260920:6,20260921:2,20260922:0}
STATE_FIELDS=['booked_fraction','candidate_increment','pv_prior_mw','pue_ratio','horizon_fraction','log_arrival_over_4','log_backlog_over_4','occupancy','market_available','physical_available','da_published','deadline_slack','log_due_4h_over_4','log_due_8h_over_4','log_due_12h_over_4','log_due_24h_over_4','pv_log_adjustment','history_ready','log_previous_pv','log_previous_issued_pv','clock_sin','clock_cos']

def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def tensor(a,device,dtype=torch.float32):return torch.as_tensor(a,dtype=dtype,device=device)

def demo_inputs(normal,world_normal):
    """Deterministic synthetic inputs at the training median; no provider records."""
    center=np.asarray(world_normal['fast_slot_feature_center'],np.float32)
    x={key:np.broadcast_to(center,(1,n,40,8)).copy() for key,n in [('fast_raw',96),('medium_raw',4),('slow_raw',2)]}
    x['market_raw']=np.broadcast_to(np.asarray(normal['feature_center'],np.float32),(24,4,len(normal['feature_names']))).copy()
    names=normal['feature_names']
    for h in range(24):
        for z in range(4):
            for name,val in [('hour',h),('dow',2),('month',1),('utc_offset_hours',-5)]:
                if name in names:x['market_raw'][h,z,names.index(name)]=val
            for j,zone in enumerate(['WEST','CAPITL','N.Y.C.','LONGIL']):
                if 'zone_'+zone in names:x['market_raw'][h,z,names.index('zone_'+zone)]=float(j==z)
    x['queue']=np.zeros((96,4),np.float64);x['queue'][0]=.05;x['queue'][15]=.8;x['queue'][95]=1.2
    x['da_anchor']=np.full((24,4),45.,np.float32)
    x['lead_hours']=np.broadcast_to(np.arange(24,dtype=np.float32)[:,None]+19,(24,4)).copy()
    x['previous_pv']=np.full(4,.2);x['previous_pue']=np.full(4,1.15)
    x['reference_queue']=np.full(4,1.2)
    x['settlement_da_price']=np.full(4,45.);x['settlement_rt_price']=np.full(4,50.)
    x['settlement_pv']=np.full(4,.2);x['settlement_pue']=np.full(4,1.15);x['da_position_mw']=np.full(4,1.)
    meta={'fixture':'synthetic_interface_test','decision_at':'2025-01-15T16:00:00Z','available_at':{'world_observations':'2025-01-15T15:45:00Z','market_features':'2025-01-15T15:45:00Z','physical_observations':'2025-01-15T15:45:00Z','da_anchor':'2025-01-14T16:00:00Z'},'da_published':True,'world_available':True,'physical_available':True,'world_values_are_raw':True,'clock_utc_hour':16.,'terminal_guard':False}
    return x,meta

def issue_guard(meta):
    decision=np.datetime64(meta['decision_at'].replace('Z',''))
    required=['world_observations','market_features','physical_observations']
    if meta['da_published']:required.append('da_anchor')
    for key in required:
        available=np.datetime64(meta['available_at'][key].replace('Z',''))
        if available>decision:raise ValueError(f'{key}: available_at exceeds decision_at')
    # Supplied timestamps enforce the input contract, not retrospective source-vintage proof.

def make_packet(x,stats,device):
    center=np.asarray(stats['fast_slot_feature_center'],np.float32)
    scale=np.asarray(stats['fast_slot_feature_scale'],np.float32)
    values={key:tensor((x[key+'_raw']-center)/np.maximum(scale,1e-8),device) for key in ['fast','medium','slow']}
    masks={key:tensor(x.get(key+'_mask',np.ones(values[key].shape[:-1],bool)),device,torch.bool) for key in values}
    # The synthetic topology is deliberately simple; a caller may supply the
    # actual registered graph arrays for a real observation packet.
    edges=np.asarray(x.get('graph_edge_index',np.stack([np.arange(40)]*2)),np.int64);nedge=edges.shape[1]
    ones=torch.ones(1,4,device=device);zero=torch.zeros_like(ones)
    c=WorldConstraintPacket(battery_power_limit=zero,battery_energy_capacity=zero,initial_soc=zero,compute_capacity=ones,initial_queue=zero,pv_capacity=1.5*ones,grid_import_limit=3.*ones,grid_export_limit=zero,migration_capacity=torch.zeros(1,4,4,device=device),migration_allowed=torch.zeros(1,4,4,dtype=torch.bool,device=device),pue_min=ones,pue_max=1.5*ones,idle_power=.23069976270198822*ones,work_power_per_unit=1.639059066772461*ones,charge_efficiency=ones,discharge_efficiency=ones,dt_hours=torch.full((1,),.25,device=device))
    for f in fields(c):
        if 'constraint_'+f.name in x:
            object.__setattr__(c,f.name,tensor(x['constraint_'+f.name],device,torch.bool if f.name=='migration_allowed' else torch.float32))
    fast=values['fast'];n=fast.shape[1]
    packet=WorldInputPacket(fast_values=fast,fast_mask=masks['fast'],medium_values=values['medium'],medium_mask=masks['medium'],slow_values=values['slow'],slow_mask=masks['slow'],entity_ids=tensor(np.r_[np.tile(np.arange(8),4),np.arange(8,16)],device,torch.long),node_ids=tensor(np.r_[np.repeat(np.arange(4),8),np.full(8,4)],device,torch.long),source_ids=tensor(x.get('source_ids',np.zeros((1,n,40))),device,torch.long),quality_ids=tensor(x.get('quality_ids',np.zeros((1,n,40))),device,torch.long),graph_edge_index=tensor(edges,device,torch.long),graph_edge_type=tensor(x.get('graph_edge_type',np.zeros(nedge)),device,torch.long),graph_edge_features=tensor(x.get('graph_edge_features',np.zeros((nedge,6))),device),graph_edge_mask=tensor(x.get('graph_edge_mask',np.ones(nedge,bool)),device,torch.bool),previous_actions=tensor(x.get('previous_actions',np.zeros((1,n,4,6))),device),action_mask=tensor(x.get('action_mask',np.zeros((1,n,4,6),bool)),device,torch.bool),constraint_slacks=tensor(x.get('constraint_slacks',np.zeros((1,n,4,8))),device),constraints=c,origin_index=torch.zeros(1,dtype=torch.long,device=device),dt_hours=torch.full((1,n),.25,device=device))
    return packet.validate(WorldMoEConfig()),c

def reduce_world(world,state,capture):
    diag=state.router_diagnostics[-1];ids=diag.topk_indices[:,:,37];weights=diag.topk_weights[:,:,37].float()
    moe=world.backbone.blocks[max(world.config.moe_block_indices)].moe
    effective=diag.captured_pre_state[:,:,None].float()+torch.sigmoid(moe.output_gate.float())*diag.captured_selected_expert_states.float()
    bank=[];prior=[]
    for expert in range(7):
        w=weights*(ids==expert);mass=w.sum((1,2));prior.append(mass/weights.shape[1]);bank.append((effective*w[...,None]).sum((1,2))/mass[:,None].clamp_min(1e-8))
    result={k:capture[k] for k in ['early','middle','late']}
    result.update(site=state.slots[:,:,[0,8,16,24]].float().mean(1),w0_local=capture['tokens'][:,:,[0,8,16,24]].float().mean(1),w1_local=state.slots[:,:,[0,8,16,24]].float().mean(1),w1_global=state.pooled_timeline.float().mean(1),expert_bank=torch.stack(bank,1),expert_prior=torch.stack(prior,1),tail_context=(diag.captured_tail_state.float()*diag.tail_active_mask[...,None]).mean(1)*moe.tail_residual_scale,tail_probability=diag.tail_route_probabilities.float().mean(1),regime=state.regime_probabilities[:,:,37].float().mean(1),log_scale=state.stochastic_log_scale[:,:,37].float().mean(1))
    # Original feature cache stores FP16; retain that bridge quantization.
    return {k:v.half().float() for k,v in result.items()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path);parser.add_argument('--metadata',type=Path);parser.add_argument('--seed',type=int,choices=list(PAIR),default=20260920);parser.add_argument('--device',default='cpu');parser.add_argument('--output',type=Path,default=ROOT/'example_result.json');parser.add_argument('--write-demo',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(0)
    started=time.perf_counter();device=torch.device(args.device)
    normal=load_json(ROOT/'configs/r2_normalization.json');world_normal=load_json(ROOT/'configs/world_normalization.json')
    if args.input:
        if not args.metadata:parser.error('--metadata is required with --input')
        with np.load(args.input,allow_pickle=False) as z:x=dict(z)
        meta=load_json(args.metadata)
    else:x,meta=demo_inputs(normal,world_normal)
    issue_guard(meta)
    if args.write_demo:
        np.savez_compressed(ROOT/'examples/synthetic_input.npz',**x);(ROOT/'examples/synthetic_metadata.json').write_text(json.dumps(meta,indent=2))
    config=WorldMoEConfig.from_mapping(load_json(ROOT/'configs/world_model.json'))
    world=VPPWorldMoE(config,include_legacy_planner=False).to(device)
    world.load_state_dict(torch.load(ROOT/'weights/world_w1_inference.pt',map_location=device,weights_only=True),strict=True);world.eval().requires_grad_(False)
    packet,constraints=make_packet(x,world_normal,device);capture={};hooks=[]
    for key,index in zip(['early','middle','late'],[2,5,9]):
        hooks.append(world.backbone.blocks[index].register_forward_hook(lambda m,i,o,key=key:capture.update({key:o[0].float().mean(1)})))
    hooks.append(world.tokenizer.register_forward_hook(lambda m,i,o:capture.update(tokens=o[0])))
    with torch.inference_mode():
        state=world.encode(packet,sample_stochastic=False,phase='w1',capture_expert_context=True)
        physical=world.query_observations(state,constraints);features=reduce_world(world,state,capture)
    for h in hooks:h.remove()
    upstream_parameters=sum(p.numel() for p in world.parameters())
    market_state=torch.load(ROOT/'weights/StructuredR2_seed20260920_epoch10.pt',map_location=device,weights_only=True)
    base=R2Base(R2Config(numerical_features=len(normal['feature_names']),use_world=False,seed=20260920),da_center=market_state['r2.base.da_center'],da_scale=market_state['r2.base.da_scale'],dart_center=market_state['r2.base.dart_center'],dart_scale=market_state['r2.base.dart_scale'])
    market=StructuredDecision(R2ProfileResidual(base),market_state['target_scales'],market_state['historical_rt'],market_state['pue_reference']).to(device)
    market.load_state_dict(market_state,strict=True);market.eval().requires_grad_(False)
    raw=x['market_raw'];n_hours=raw.shape[0]
    if n_hours!=24:raise ValueError('Minimal one-decision interface requires a 24-hour forecast grid; full replay handles DST profiles with released profile_mask logic')
    normalized=np.clip((np.where(np.isfinite(raw),raw,np.array(normal['feature_center']))-np.array(normal['feature_center']))/np.array(normal['feature_scale']),-20,20).astype(np.float32)
    zones=tensor(np.tile(np.arange(4),24),device,torch.long);rows=96
    batch={k:(v[:,zones].squeeze(0) if k in ['w0_local','w1_local'] else v.expand(rows,*v.shape[1:])) for k,v in features.items()}
    profile=np.zeros((rows,25,len(normal['feature_names'])),np.float32);profile_da=np.zeros((rows,25),np.float32);profile_mask=np.zeros((rows,25),bool)
    for i,z in enumerate(zones.cpu().numpy()):profile[i,:24]=normalized[:,z];profile_da[i,:24]=x['da_anchor'][:,z];profile_mask[i,:24]=True
    batch.update(numerical=tensor(normalized.reshape(rows,-1),device),zone=zones,lead_hours=tensor(x['lead_hours'].reshape(-1),device),world_available=torch.full((rows,),float(meta['world_available']),device=device),profile_numerical=tensor(profile,device),profile_da=tensor(profile_da,device),profile_mask=tensor(profile_mask,device,torch.bool),profile_position=tensor(np.repeat(np.arange(24),4),device,torch.long))
    publication=torch.full((rows,),float(meta['da_published']),device=device);anchor=tensor(x['da_anchor'].reshape(-1),device)
    decision=ArchitectedR2QPHead().to(device);epoch=PAIR[args.seed]
    decision.load_state_dict(torch.load(ROOT/'weights'/f'CdfR2_seed{args.seed}_epoch{epoch:02d}.pt',map_location=device,weights_only=True),strict=True);decision.eval().requires_grad_(False)
    with torch.inference_mode():
        out_market=market(batch,publication,anchor)
        q=out_market['rt'].reshape(24,4,7).repeat_interleave(4,dim=0).permute(1,0,2)
        lat=out_market['decision_latent'].reshape(24,4,128).repeat_interleave(4,dim=0).permute(1,0,2)
        pue_ref=market.pue_reference.detach().cpu().numpy();pv_prior=physical.pv[0].float().cpu().numpy();pue_prior=physical.pue[0].float().cpu().numpy()
        if pv_prior.shape!=(96,4):raise ValueError(f'Expected 96-slot physical decode, got {pv_prior.shape}')
        if not meta.get('market_available',True):
            q=market.historical_rt[:,None,None].expand_as(q);lat=torch.zeros_like(lat)
        if not meta['physical_available']:
            import datetime
            day=datetime.datetime.fromisoformat(meta['decision_at'].replace('Z','+00:00'))
            leapday=datetime.date(2020,day.month,day.day).timetuple().tm_yday-1
            with np.load(ROOT/'configs/solar_seasonal_envelope.npz') as envelope:pv_prior=envelope['median'][leapday].copy()
            pue_prior=np.broadcast_to(pue_ref,(96,4)).copy()
        queue=np.asarray(x['queue'],np.float64).copy();horizon=np.arange(96)
        pue=np.clip(pue_prior+(x['previous_pue']-pue_prior[0])*np.exp(-horizon[:,None]/32),1,1.5);pue[0]=x['previous_pue']
        pv_prior[0]=x['previous_pv'];states=np.zeros((4,96,22),np.float32)
        for z in range(4):
            bins=np.array([queue[:h*4,z].sum() for h in [4,8,12,24]]);clock=(meta.get('clock_utc_hour',0)+horizon/4)%24
            states[z]=np.column_stack([np.zeros(96),np.full(96,min(.05,queue[:,z].sum())),pv_prior[:,z],pue[:,z]/pue_ref[z],horizon/96,np.zeros(96),np.full(96,np.log1p(queue[:,z].sum())/4),np.zeros(96),np.ones(96),np.ones(96),np.full(96,float(meta['da_published'])),1-horizon/96,np.broadcast_to(np.log1p(bins)/4,(96,4)),np.zeros(96),np.ones(96),np.full(96,np.log1p(x['previous_pv'][z])),np.full(96,np.log1p(pv_prior[0,z])),np.sin(2*np.pi*clock/24),np.cos(2*np.pi*clock/24)])
        states[:,:,8]=float(meta.get('market_available',True))
        output=decision(tensor(states,device),lat,q,tensor(pue_ref,device))
    probs=np.array([.075,.1,.2,.25,.2,.1,.075]);pv=output['pv_quantiles'].cpu().numpy().transpose(0,2,1).astype(float);prices=output['conditional_price'].cpu().numpy().transpose(0,2,1).astype(float);op=output['opportunity'].cpu().numpy().astype(float)
    with np.load(ROOT/'configs/causal_observer.npz') as observer:residual=observer['residual_quantiles']
    slot=int(meta.get('clock_utc_hour',0)*4)%96
    original_queue=queue.copy();actions=[]
    for z in range(4):
        # Same current-scenario recoupling and constrained action kernel.
        prior=pv[z,:,0].copy();prior_sd=np.sqrt(np.dot(probs,(prior-np.dot(probs,prior))**2))
        pv[z,:,0]=np.clip(x['previous_pv'][z]+residual[slot,z]*max(pv_prior[0,z],.1),0,1.5)
        observed=pv[z,:,0];observed_sd=np.sqrt(np.dot(probs,(observed-np.dot(probs,observed))**2))
        rt=output['rt_quantiles'][z,0].cpu().numpy().astype(float);rt_sd=np.sqrt(np.dot(probs,(rt-np.dot(probs,rt))**2))
        slope=float(output['cross_moment_rho'][z,0])*rt_sd/max(prior_sd,observed_sd,1e-5)
        prices[z,:,0]=float(output['rt_decision'][z,0])+slope*(observed-np.dot(probs,observed))
        cap=float(min(1.,(constraints.grid_import_limit[0,z].item()/constraints.pue_max[0,z].item()-.23069976270198822)/1.639059066772461))
        reference=np.minimum(1.,np.maximum(float(x['reference_queue'][z])-horizon,0.))
        action=min(cap,queue[:,z].sum()) if meta.get('terminal_guard') else float(select_action(queue[:,z].copy(),pv[z].copy(),prices[z].copy(),probs,pue[:,z].copy(),reference,.23069976270198822,1.639059066772461,op[z].copy(),float(pue_ref[z]),cap,0.,.9,False,True,False))
        remaining=action
        for due in range(96):take=min(queue[due,z],remaining);queue[due,z]-=take;remaining-=take
        actions.append(action-remaining)
    actions=np.array(actions);residual_balance=original_queue.sum(0)-actions-queue.sum(0)
    if np.max(np.abs(residual_balance))>1e-9 or np.min(queue)<-1e-9:raise AssertionError('Service ledger not conserved')
    if np.max(queue[0])>1e-8:raise AssertionError('Current deadline missed by input queue/action')
    result={'fixture':meta.get('fixture','user_inputs'),'scientific_performance_result':False,'seed':args.seed,'archive_epoch':epoch,'author_display_epoch':2 if args.seed==20260922 else epoch,'upstream_parameters':upstream_parameters,'decision_parameters':sum(p.numel() for p in decision.parameters()),'action_work_units':actions.tolist(),'served_it_mwh':(actions*1.639059066772461*.25).tolist(),'remaining_work_units':queue.sum(0).tolist(),'ledger_max_abs_residual':float(np.max(np.abs(residual_balance))),'current_overdue_work':queue[0].tolist(),'apic_post_quantile_identity_max_abs':float((out_market['da']-torch.flip(out_market['dart'],[-1])-out_market['rt']).abs().max()) if meta['da_published'] else None,'seconds':time.perf_counter()-started,'input_time_contract':'all supplied feature availability times <= decision_at; historical source vintages remain caller responsibility','loaded_weights':[v['path'] for v in load_json(ROOT/'model_manifest.json')['weights'] if 'CdfR2' not in v['path'] or str(args.seed) in v['path']]}
    if 'settlement_rt_price' in x:
        imports=np.maximum(0,x['settlement_pue']*(.23069976270198822+1.639059066772461*actions)-x['settlement_pv'])
        result['ex_post_settlement_usd']=(.25*(x['da_position_mw']*x['settlement_da_price']+(imports-x['da_position_mw'])*x['settlement_rt_price'])).tolist()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
