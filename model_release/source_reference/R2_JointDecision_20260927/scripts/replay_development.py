"""Single-seed 2024 replay with deterministic, persistent, feasible reservations.

The neural correction is computed once per arriving cohort, as in its training
surrogate. The executor then updates the analytical PV/grid marginal cost as
capacity is consumed. No realized future outcome is supplied to either part.
"""
import sys,argparse,json,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT,EXTENSION,PREVIOUS,WORKSPACE
import numpy as np,pandas as pd,torch
from torch import nn
from numba import njit
from collections import defaultdict
from src.models.structured_decision import StructuredHead,CachedStructuredHead,structured_allocation
from src.models.coupled_rollout import ArchitectedR2Head,ArchitectedR2QPHead
from src.models.feasible_qp import qp_allocation
from src.models.distributional_r2 import DistributionalR2Head
import src,vendor
src.__path__.append(str(EXTENSION/'src'));vendor.__path__.append(str(EXTENSION/'vendor'))
sys.path.append(str(EXTENSION))
from src import context as ctx

class Correction(nn.Module):
    def __init__(self,state):
        super().__init__()
        self.state_encoder=nn.Sequential(nn.Linear(16,64),nn.GELU(),nn.LayerNorm(64),nn.Linear(64,64),nn.GELU())
        self.value_head=nn.Sequential(nn.Linear(199,128),nn.GELU(),nn.LayerNorm(128),nn.Linear(128,64),nn.GELU(),nn.Linear(64,2))
        self.load_state_dict({k:v for k,v in state.items() if k.startswith(('state_encoder.','value_head.'))})
    def forward(self,state,latent,market):
        encoded=self.state_encoder(state)
        head=self.value_head(torch.cat([latent,encoded,market/100.],-1))
        return 80*torch.tanh(head[:,0])+40*torch.tanh(head[:,1])

@njit(cache=True)
def distribute(book,price,pv,ratio,physical,correction,amount,pue_ref,limit,quantum=.05):
    assigned=np.zeros(96);remaining=amount
    while remaining>1e-10:
        best=-1;minimum=1e30
        for j in range(limit):
            free=1-book[j]
            if free<1e-10:continue
            delta=min(quantum,remaining,free)
            covered=0.
            if physical[j]:
                load=ratio[j]*pue_ref*(.23069976270198822+1.639059066772461*book[j])
                increment=ratio[j]*pue_ref*1.639059066772461*delta
                covered=min(1.,max(0.,(pv[j]-load)/max(increment,1e-12)))
            score=ratio[j]*((1-covered)*price[j]-covered)+correction[j]+(2. if j>0 else 0.)
            if score<minimum:minimum=score;best=j
        if best<0:break
        take=min(remaining,1-book[best],quantum)
        book[best]+=take;assigned[best]+=take;remaining-=take
    return assigned,remaining

@njit(cache=True)
def soft_distribute(book,price,pv,ratio,physical,correction,amount,pue_ref,limit):
    cap=np.zeros(96);scores=np.zeros(96);allocation=np.zeros(96)
    for j in range(limit):
        cap[j]=max(0.,1-book[j]);delta=max(1e-5,min(.05,amount,max(cap[j],1e-5)))
        covered=0.
        if physical[j]:
            load=ratio[j]*pue_ref*(.23069976270198822+1.639059066772461*book[j])
            inc=ratio[j]*pue_ref*1.639059066772461*delta
            covered=min(1.,max(0.,(pv[j]-load)/max(inc,1e-12)))
        scores[j]=ratio[j]*((1-covered)*price[j]-covered)+correction[j]+(2. if j else 0.)
    for step in range(8):
        left=np.maximum(0.,cap-allocation);need=max(0.,amount-allocation.sum());logits=np.full(96,-1e6)
        for j in range(limit):
            if left[j]>1e-9:logits[j]=-scores[j]/2.+np.log(left[j])
        probability=np.exp(logits-logits.max());probability/=probability.sum()
        allocation+=np.minimum(need*probability,left)
    left=np.maximum(0.,cap-allocation);need=min(max(0.,amount-allocation.sum()),left.sum())
    allocation+=need*left/max(left.sum(),1e-12)
    book+=allocation
    return allocation,max(0.,amount-allocation.sum())

def replay(seed,epoch,weight,allocation_mode='marginal',structured=False,year=2024,fast=False,solar_hours=0.,deep=False,qp=0.,oracle='none',distribution=False):
    name=f'JointR2_Eagle_{year}_s{seed}_ep{epoch:02d}_a{weight:g}'+('_soft' if allocation_mode=='soft' else '')
    if structured:name=f'StructuredR2_Eagle_{year}_s{seed}_ep{epoch:02d}'
    if fast:name+='_fast_check'
    if solar_hours:name+=f'_solar{solar_hours:g}h'
    if deep:
        if seed!=20260920:raise RuntimeError('Only the primary seed is authorized until its final 80% gate passes')
        family='DistR2' if distribution else ('QPR2' if qp else 'DeepR2')
        if year==2025:
            frozen=json.loads((ROOT/'results'/f'primary_freeze_{family}.json').read_text())
            checkpoint=ROOT/'checkpoints'/f'{family}_seed{seed}_epoch{epoch:02d}.pt'
            assert frozen['selected_epoch']==epoch and hashlib.sha256(checkpoint.read_bytes()).hexdigest()==frozen['checkpoint_sha256']
        name=f'{family}_Eagle_{year}_s{seed}_ep{epoch:02d}'
    if oracle!='none':
        assert year==2024 and deep and not distribution,'Ideal-information diagnostics are restricted to primary development'
        name+=f'_oracle_{oracle}'
    output=ROOT/'results'/f'{name}.json'
    if output.exists():print('CACHED',name,flush=True);return json.loads(output.read_text())
    if year==2025 and not deep:
        frozen=json.loads((ROOT/'results'/f'structured_complete_s{seed}.json').read_text())
        assert structured and frozen['best_epoch']==epoch and frozen['best_restored']
    arr,reference,shared=ctx.configure(year);c=ctx.c;n=c.N;times=c.times
    family=('DistR2' if distribution else ('QPR2' if qp else 'DeepR2')) if deep else ('StructuredR2' if structured else 'JointR2')
    checkpoint=ROOT/'checkpoints'/f'{family}_seed{seed}_epoch{epoch:02d}.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cache_epoch=10 if deep else epoch
    with np.load(ROOT/'cache'/f'{"structured_" if structured else ""}market{year}_s{seed}_ep{cache_epoch:02d}.npz') as z:
        hours=z['hours'];pue_ref=z['pue_reference'];historical=z['historical_rt']
        hi=np.searchsorted(hours,times.floor('h').asi8)
        assert (hours[hi]==times.floor('h').asi8).all()
        market={gate:np.concatenate([z[f'rt_{gate}'][hi],np.broadcast_to(historical[None,:,None],(96,4,7))]).astype(np.float32) for gate in [0,1]}
        latent={gate:np.concatenate([z[f'latent_{gate}'][hi],np.zeros((96,4,128))]).astype(np.float32) for gate in [0,1]}
    local=times.tz_convert('America/New_York').tz_localize(None).normalize()
    pre=np.r_[(local-pd.Timedelta(days=1)+pd.Timedelta(hours=4,minutes=55)).tz_localize('America/New_York').tz_convert('UTC').asi8,np.full(96,np.iinfo(np.int64).max)]
    post=np.r_[(local-pd.Timedelta(days=1)+pd.Timedelta(hours=11)).tz_localize('America/New_York').tz_convert('UTC').asi8,np.full(96,np.iinfo(np.int64).max)]
    physical_file=PREVIOUS/'cache/world_physical_forecasts.npz' if year==2024 else ctx.ARCHIVE/'cache/final_physical_priors.npz'
    with np.load(physical_file) as z:
        # Physical packet slots and supervised physical references use UTC days.
        dates={str(d):i for i,d in enumerate(z['dates'])}
        di=np.array([dates.get(d,-1) for d in times.strftime('%Y-%m-%d')]);valid=di>=0
        slot=times.hour.to_numpy()*4+times.minute.to_numpy()//15
        pv=np.concatenate([z['issue_pv'][di.clip(0),slot]*valid[:,None],np.zeros((96,4))])
        pue=np.concatenate([z['issue_pue'][di.clip(0),slot],np.broadcast_to(pue_ref,(96,4))])
        issue=pd.DatetimeIndex(pd.to_datetime(z['issue_utc'],utc=True)).asi8
        issue=np.r_[np.where(valid,issue[di.clip(0)],np.iinfo(np.int64).max),np.full(96,np.iinfo(np.int64).max)]
    use_fast=fast or (ROOT/'configs/fast_executor_enabled.json').exists()
    if deep:
        correction=(DistributionalR2Head if distribution else (ArchitectedR2QPHead if qp else ArchitectedR2Head))();correction.load_state_dict(state['state_dict']);correction.eval().requires_grad_(False)
        torch.backends.mha.set_fastpath_enabled(False)
        correction=torch.jit.freeze(torch.jit.trace(correction,(torch.zeros(4,96,23 if distribution else 22),torch.zeros(4,96,128),torch.zeros(4,96,7),torch.tensor(pue_ref)),strict=False,check_trace=False))
        use_fast=False
    elif structured:
        head_state={k.removeprefix('structured.'):v for k,v in state['state_dict'].items() if k.startswith('structured.')}
        if use_fast:
            correction=CachedStructuredHead(head_state).eval().requires_grad_(False)
            with torch.inference_mode():
                projected={gate:correction.project_context(torch.from_numpy(latent[gate]),torch.from_numpy(market[gate])).numpy() for gate in [0,1]}
                fallback=correction.project_context(torch.zeros(4,128),torch.tensor(np.broadcast_to(historical[:,None],(4,7)).copy())).numpy()
            correction=torch.jit.freeze(torch.jit.trace(correction,(torch.zeros(4,96,22),torch.zeros(4,96,96),torch.zeros(4,96,7),torch.tensor(pue_ref)),strict=False))
        else:
            correction=StructuredHead();correction.load_state_dict(head_state);correction.eval().requires_grad_(False)
            correction=torch.jit.freeze(torch.jit.trace(correction,(torch.zeros(4,96,22),torch.zeros(4,96,128),torch.zeros(4,96,7),torch.tensor(pue_ref)),strict=False))
    else:
        correction=Correction(state['state_dict']).eval().requires_grad_(False)
        correction=torch.jit.freeze(torch.jit.trace(correction,(torch.zeros(384,16),torch.zeros(384,128),torch.zeros(384,7))))
    obs_cum=np.vstack([np.zeros((1,4)),np.cumsum(c.pv,axis=0)])
    prior_cum=np.vstack([np.zeros((1,4)),np.cumsum(pv[:n],axis=0)])
    if solar_hours:
        from src.data.solar_nowcast import prepare,forecast
        solar_prepared=prepare(times,c.pv)
    book=np.zeros((n+8192,4));events=[defaultdict(list) for _ in range(4)];pending=[defaultdict(float) for _ in range(4)]
    h=np.arange(96);late=0.;started=time.perf_counter();neural_calls=0
    with torch.inference_mode():
        for t in range(n):
            for z in range(4):
                for due,units in events[z].pop(t-1,[]):
                    pending[z][due]-=units
                    if pending[z][due]<1e-8:pending[z].pop(due,None)
            active=np.flatnonzero(arr[t]>1e-10)
            if not len(active):continue
            ids=t+h;now=times.asi8[t];available=pre[ids]<=now;published=post[ids]<=now;physical=(issue[ids]<=now)&available
            physical_source=physical.copy()
            if solar_hours:
                solar_pv=forecast(t,ids,pv[ids],physical_source,solar_prepared,solar_hours)
                physical=np.ones(96,dtype=bool)
            q=np.where(published[:,None,None],market[1][ids],market[0][ids])
            q=np.where(available[:,None,None],q,historical[None,:,None])
            lat=np.where(published[:,None,None],latent[1][ids],latent[0][ids])*available[:,None,None]
            st=np.zeros((len(active),96,16),np.float32)
            ratios=np.where(physical_source[:,None],pue[ids]/pue_ref,1.)
            for a,z in enumerate(active):
                amount=arr[t,z];due=t+95;pending[z][due]+=amount;booked=book[ids,z]
                bins=np.array([sum(v for d,v in pending[z].items() if d<=t+hours*4-1) for hours in [4,8,12,24]])
                st[a]=np.column_stack([booked,np.minimum(np.maximum(1-booked,1e-5),min(.05,amount)),pv[ids,z]*physical,ratios[:,z],h/96,
                    np.full(96,np.log1p(amount)/4),np.full(96,np.log1p(sum(pending[z].values()))/4),np.full(96,booked.mean()),available,physical,published,1-h/96,
                    np.broadcast_to(np.log1p(bins)/4,(96,4))])
                if solar_hours:st[a,:,2]=solar_pv[:,z]
            if structured:
                begin=max(0,t-16);rat=np.clip(np.log((obs_cum[t]-obs_cum[begin]+.2)/(prior_cum[t]-prior_cum[begin]+.2)),-np.log(4),np.log(4))
                if t==0:rat[:]=0
                clock=(times[t].hour+times[t].minute/60+h/4)%24
                additional=np.zeros((len(active),96,6),np.float32)
                for a,z in enumerate(active):
                    additional[a]=np.column_stack([rat[z]*np.exp(-h/16)/2,np.full(96,min(t,16)/16),np.full(96,np.log1p(c.pv[t-1,z]) if t else 0),np.full(96,np.log1p(pv[t-1,z]) if t else 0),np.sin(2*np.pi*clock/24),np.cos(2*np.pi*clock/24)])
                st=np.concatenate([st,additional],-1)
                if solar_hours:st[:,:,16]=0.
                if distribution:
                    assert solar_hours>0
                    st=np.concatenate([st,solar_prepared[0][ids][:,active].T[:,:,None].astype(np.float32)],-1)
                if use_fast:
                    fixed=np.where(published[:,None,None],projected[1][ids],projected[0][ids]);fixed=np.where(available[:,None,None],fixed,fallback[None])
                    network_context=fixed[:,active].transpose(1,0,2).copy()
                else:network_context=lat[:,active].transpose(1,0,2).copy()
                decoded=correction(torch.from_numpy(st),torch.from_numpy(network_context),torch.from_numpy(q[:,active].transpose(1,0,2).copy()),torch.tensor(pue_ref[active]))
                if oracle!='none':
                    # Intervene only on cost inputs after the native network
                    # forward. Realized future values never enter its features.
                    inside=np.minimum(ids,n-1);booked=torch.from_numpy(st[:,:,0]);cap=(1-booked).clamp_min(0)
                    use_physics=oracle in ['physics','both'];use_price=oracle in ['price','both']
                    pv_scenarios=torch.tensor(np.broadcast_to(c.pv[inside][:,active].T[:,:,None],(len(active),96,7)).copy(),dtype=torch.float32) if use_physics else decoded['pv_quantiles']
                    pue_hat=torch.tensor(c.pue[inside][:,active].T.copy(),dtype=torch.float32) if use_physics else decoded['physical_pue']
                    price_hat=torch.tensor(c.rt[inside][:,active].T.copy(),dtype=torch.float32) if use_price else decoded['rt_decision']
                    load=pue_hat*(.23069976270198822+1.639059066772461*booked)
                    knots=torch.minimum(((pv_scenarios-load[:,:,None])/(pue_hat[:,:,None]*1.639059066772461)).clamp_min(0),cap[:,:,None])
                    boundaries=torch.cat([torch.zeros_like(cap[:,:,None]),knots,cap[:,:,None]],-1);ow=(boundaries[:,:,1:]-boundaries[:,:,:-1]).clamp_min(0)
                    mid=(boundaries[:,:,1:]+boundaries[:,:,:-1])*.5;threshold=load[:,:,None]+pue_hat[:,:,None]*1.639059066772461*mid
                    probabilities=torch.tensor([.075,.1,.2,.25,.2,.1,.075])
                    coverage=((pv_scenarios[:,:,:,None]>threshold[:,:,None,:]).float()*probabilities[None,None,:,None]).sum(2)
                    ratio=pue_hat/torch.tensor(pue_ref[active])[:,None];oc=ratio[:,:,None]*((1-coverage)*price_hat[:,:,None]-coverage)
                    average=(oc*ow).sum(-1)/cap.clamp_min(1e-8);oc=torch.where((price_hat<-1)[:,:,None],average[:,:,None],oc)
                    fees=torch.ones_like(cap)*2;fees[:,0]=0
                    decoded['segment_widths']=ow;decoded['segment_scores']=oc+decoded['opportunity'][:,:,None]+fees[:,:,None]
                widths=decoded['segment_widths'].clone();widths[:,min(96,n-t):]=0
                amount_tensor=torch.tensor(arr[t,active],dtype=torch.float32)
                allowed_limits=np.full(len(active),min(96,n-t),dtype=np.int64)
                if distribution and t<n-4:
                    guard_limit=min(96,n-4-t)
                    guarded=widths.clone();guarded[:,guard_limit:]=0
                    can_guard=guarded.sum((1,2))+1e-7>=amount_tensor
                    widths=torch.where(can_guard[:,None,None],guarded,widths)
                    allowed_limits=np.where(can_guard.numpy(),guard_limit,allowed_limits)
                assigned_all=(qp_allocation(widths,decoded['segment_scores'],amount_tensor,qp) if qp else structured_allocation(widths,decoded['segment_scores'],amount_tensor)).numpy().astype(np.float64)
                if distribution and t>=n-4:
                    # The final hour serves only arrived work in earliest free
                    # slots; no future arrival or price realization is consulted.
                    for a,z in enumerate(active):
                        free=np.maximum(1-book[ids,z],0.);free[min(96,n-t):]=0
                        assigned_all[a]=np.minimum(free,np.maximum(float(arr[t,z])-np.r_[0.,np.cumsum(free[:-1])],0.))
                neural_calls+=1
            elif weight:
                extra=correction(torch.from_numpy(st.reshape(-1,16)),torch.from_numpy(lat[:,active].transpose(1,0,2).reshape(-1,128).copy()),torch.from_numpy(q[:,active].transpose(1,0,2).reshape(-1,7).copy())).numpy().reshape(len(active),96)*weight
                neural_calls+=1
            else:extra=np.zeros((len(active),96))
            for a,z in enumerate(active):
                if structured:
                    assigned=np.minimum(assigned_all[a],np.maximum(0.,1-book[ids,z]));amount=float(arr[t,z])
                    if assigned.sum()>amount:assigned*=amount/assigned.sum()
                    rest=max(0.,amount-assigned.sum())
                    # Resolve FP32 segment-allocation roundoff deterministically.
                    if rest<1e-5:
                        free=np.maximum(0.,1-book[ids,z]-assigned);free[int(allowed_limits[a]):]=0
                        for j in np.flatnonzero(free>0):
                            take=min(rest,free[j]);assigned[j]+=take;rest-=take
                            if rest<1e-12:break
                    book[ids,z]+=assigned
                else:
                    executor=soft_distribute if allocation_mode=='soft' else distribute
                    assigned,rest=executor(book[t:t+96,z],q[:,z,3].astype(np.float64),pv[ids,z],ratios[:,z],physical,extra[a].astype(np.float64),float(arr[t,z]),float(pue_ref[z]),min(96,n-t))
                for j in np.flatnonzero(assigned>1e-10):events[z][t+int(j)].append((t+95,float(assigned[j])))
                if rest>1e-8:
                    targets=np.arange(t+96,len(book));free=np.maximum(1-book[targets,z],0.)
                    take=np.minimum(free,np.maximum(rest-np.r_[0.,np.cumsum(free[:-1])],0.))
                    assert abs(take.sum()-rest)<1e-7
                    book[targets,z]+=take;late+=rest*c.DT*c.k[z]
                    for slot,units in zip(targets[take>1e-10],take[take>1e-10]):events[z][int(slot)].append((t+95,float(units)))
            if t%8640==0:print('REPLAY',name,t,n,'seconds',round(time.perf_counter()-started,1),flush=True)
    ctx.RESULTS=ROOT/'results';ctx.TRAJECTORIES=ROOT/'results/trajectories';ctx.TRAJECTORIES.mkdir(exist_ok=True);c.DATA=ctx.TRAJECTORIES
    extra=dict(seconds=time.perf_counter()-started,neural_calls=neural_calls,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),correction_weight=weight,price_training_years=[2021,2022,2023],physical_training_years=[2022,2023],parameter_updates=False,physical_prior_mapping='UTC packet day and quarter-hour, matching supervised state bank',allocation_mode=allocation_mode,allocation='persistent, fixed per-cohort correction; soft mode matches the capacity-capped differentiable training surrogate, marginal mode uses .05-unit analytical updates',model_epoch=epoch)
    if structured:extra.update(allocation_mode='structured',allocation='deterministic segment sorting; identical hard forward rule in training and execution; exact physical cash account retained',physical_observation_lag_quarters=1,cached_context_inference=use_fast)
    if solar_hours:extra.update(solar_nowcast_hours=solar_hours,unissued_physics_fallback='training-only seasonal PV median and PUE reference',solar_envelope_training_years=[2022,2023])
    if deep:extra.update(architecture='dual-stream cross attention, hourly temporal interaction, workload queries, joint price-PV cross moments',training_rollout_arrivals=16,market_source='frozen StructuredR2 primary epoch 10 plus trained APIC-preserving market-only shift')
    if qp:extra.update(allocation_mode='differentiable_qp',qp_regularization=qp,gradient='implicit derivative of the executed box/equality QP; no straight-through gradient',candidate_specific_query=True)
    if distribution:extra.update(pv_distribution='eight clear-sky attenuation states with learned causal probabilities',price_quantile_valuation='learned task-specific weights; native forecasts remain a separate APIC branch',training_objective='aggregate episode US-dollar difference with one training-only normalization constant',terminal_completion_buffer_quarters=4,terminal_guard='reserve the final hour from earlier claims when feasible, then use earliest service for arrived work; fixed settlement and liability unchanged')
    if oracle!='none':extra.update(oracle_diagnostic=oracle,deployable=False,eligible_for_80_gate=False,diagnostic_description='Realized future cost input intervention; not a forecast or achieved policy')
    row=ctx.measure(name,book,reference,arr,shared,dict(experiment='StructuredR2' if structured else 'JointR2',year=year,trace='Eagle',seed=seed,window_h=24,role='selection' if year==2024 else 'exploratory_frozen_evaluation'),late=late,extra=extra)
    bound=json.loads((EXTENSION/'results'/f'I1_Eagle_{year}_w24.json').read_text())
    row['capture_lower_fraction']=row['saving_usd']/bound['maximum_saving_upper_usd'];row['target_80_usd']=.8*bound['maximum_saving_upper_usd']
    output.write_text(json.dumps(row,indent=2))
    print('COMPLETE',name,'SAVING',row['saving_usd'],'CAPTURE',row['capture_lower_fraction'],'LATE',late,'CLOSING',row['closing_it_mwh'],flush=True)
    return row

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=20260920);p.add_argument('--epochs',type=int,nargs='+');p.add_argument('--weight',type=float,default=1.);p.add_argument('--allocation',choices=['marginal','soft'],default='marginal');p.add_argument('--structured',action='store_true');p.add_argument('--year',type=int,choices=[2024,2025],default=2024);p.add_argument('--fast-check',action='store_true');p.add_argument('--solar-hours',type=float,default=0.);p.add_argument('--deep',action='store_true');p.add_argument('--qp',type=float,default=0.);p.add_argument('--oracle',choices=['none','price','physics','both'],default='none');p.add_argument('--distribution',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    epochs=args.epochs or json.loads((ROOT/'results'/f'primary_checkpoint_candidates_s{args.seed}.json').read_text())['epochs']
    for epoch in epochs:replay(args.seed,epoch,args.weight,args.allocation,args.structured,args.year,args.fast_check,args.solar_hours,args.deep,args.qp,args.oracle,args.distribution)
