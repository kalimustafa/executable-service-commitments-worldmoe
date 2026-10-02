"""Causal neural forecasts plus nonanticipative physical-scenario continuation.

Only present service is executed. Future plans are hypothetical scenario
continuations, not claimed knowledge of future prices or physical outcomes.
"""
import sys,argparse,json,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT,WORKSPACE,PREVIOUS,EXTENSION,ARCHIVE
import numpy as np,pandas as pd,torch
import src,vendor
src.__path__.append(str(EXTENSION/'src'));vendor.__path__.append(str(EXTENSION/'vendor'));sys.path.append(str(EXTENSION))
from src import context as ctx
from src.models.coupled_rollout import ArchitectedR2QPHead
class NoMarketResidual(ArchitectedR2QPHead):
    def market_update(self,latent):
        adapted,shift=super().market_update(latent)
        return adapted,torch.zeros_like(shift)
from src.models.continuation_r2 import ContinuationR2Head
from src.models.affine_continuous import AffineForecastR2Head
from src.models.scenario_recourse import select_action,action_value_curves
from src.data.solar_nowcast import prepare,forecast
from src.data.causal_observer import fit_observer

p=argparse.ArgumentParser();p.add_argument('--eta',type=float,default=1.);p.add_argument('--year',type=int,choices=[2024,2025],default=2024);p.add_argument('--window',type=int,choices=[12,16,24],required=True);p.add_argument('--safety',type=float,default=.9);p.add_argument('--epoch',type=int);p.add_argument('--seed',type=int,default=20260920);p.add_argument('--cdf-epoch',type=int);p.add_argument('--cash-epoch',type=int);p.add_argument('--value-epoch',type=int);p.add_argument('--physical-epoch',type=int);p.add_argument('--horizon-epoch',type=int);p.add_argument('--action-epoch',type=int);p.add_argument('--tail-epoch',type=int);p.add_argument('--moment-epoch',type=int);p.add_argument('--shape-epoch',type=int);p.add_argument('--regime',action='store_true');p.add_argument('--exact-action',action='store_true');p.add_argument('--current-coupling',action='store_true');p.add_argument('--continuous-cost',action='store_true');p.add_argument('--mass-aware',action='store_true');p.add_argument('--native-mean',action='store_true');p.add_argument('--opportunity-weight',type=float,default=1.);p.add_argument('--cost-bias-weight',type=float,default=1.);p.add_argument('--post-only',action='store_true');args=p.parse_args()
seed=args.seed
if seed!=20260920:
    authorization=json.loads((ROOT/'results/replication_authorization.json').read_text())
    assert authorization['authorized'] and seed in authorization['seeds']
    assert (authorization['selected_family']=='CdfR2' and args.cdf_epoch is not None) or (authorization['selected_family']=='CashR2' and args.cash_epoch is not None)
assert sum(v is not None for v in [args.value_epoch,args.physical_epoch,args.horizon_epoch,args.action_epoch,args.tail_epoch,args.moment_epoch,args.shape_epoch])+int(args.regime)<=1
if args.continuous_cost:assert args.eta==0 and all(v is None for v in [args.value_epoch,args.physical_epoch,args.horizon_epoch,args.action_epoch,args.tail_epoch,args.moment_epoch,args.shape_epoch]) and not args.regime
if args.cdf_epoch is not None or args.cash_epoch is not None:assert args.continuous_cost
assert not (args.cdf_epoch is not None and args.cash_epoch is not None)
if args.mass_aware:assert args.continuous_cost
if args.year==2025:
    selected_family=next((label for label,value in [('shape',args.shape_epoch),('moment',args.moment_epoch),('tail',args.tail_epoch),('action',args.action_epoch),('horizon',args.horizon_epoch),('physical',args.physical_epoch),('continuation',args.value_epoch)] if value is not None),'recourse')
    if args.regime:selected_family='regime'
    if args.cdf_epoch is not None:selected_family='cdf'
    if args.cash_epoch is not None:selected_family='cash'
    selection_file=ROOT/'results'/(f'{selected_family}_selection.json' if seed==20260920 else f'{selected_family}_selection_s{seed}.json')
    selected=json.loads(selection_file.read_text());assert selected['eta']==args.eta and selected['safety']==args.safety
    assert selected.get('opportunity_weight',1.)==args.opportunity_weight and selected.get('cost_bias_weight',1.)==args.cost_bias_weight
    assert selected.get('exact_action',False)==args.exact_action
    assert selected.get('current_coupling',False)==args.current_coupling
    assert selected.get('continuous_cost',False)==args.continuous_cost
    assert selected.get('mass_aware',False)==args.mass_aware
    assert selected.get('native_mean',False)==args.native_mean
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.mha.set_fastpath_enabled(False)
name=f'RecourseR2_Eagle_{args.year}_s{seed}_eta{args.eta:g}_k{args.safety:g}'
if args.epoch is not None:name+=f'_ep{args.epoch:02d}'
if args.value_epoch is not None:name=f'ValueR2_Eagle_{args.year}_s20260920_ep{args.value_epoch:02d}'
if args.physical_epoch is not None:name=f'PhysicalR2_Eagle_{args.year}_s20260920_ep{args.physical_epoch:02d}'
if args.horizon_epoch is not None:name=f'HorizonR2_Eagle_{args.year}_s20260920_ep{args.horizon_epoch:02d}'
if args.action_epoch is not None:name=f'ActionR2_Eagle_{args.year}_s20260920_ep{args.action_epoch:02d}'
if args.tail_epoch is not None:name=f'TailR2_Eagle_{args.year}_s20260920_ep{args.tail_epoch:02d}'
if args.moment_epoch is not None:name=f'MomentR2_Eagle_{args.year}_s20260920_ep{args.moment_epoch:02d}'
if args.shape_epoch is not None:name=f'ShapeR2_Eagle_{args.year}_s20260920_ep{args.shape_epoch:02d}'
if args.regime:name=f'RegimeR2_Eagle_{args.year}_s20260920'
if args.cdf_epoch is not None:name=f'CdfR2_Eagle_{args.year}_s{seed}_ep{args.cdf_epoch:02d}'
if args.cash_epoch is not None:name=f'CashR2_Eagle_{args.year}_s{seed}_ep{args.cash_epoch:02d}'
if args.opportunity_weight!=1. or args.cost_bias_weight!=1.:name+=f'_op{args.opportunity_weight:g}_b{args.cost_bias_weight:g}'
if args.exact_action:name+='_exact'
if args.current_coupling:name+='_coupled'
if args.continuous_cost:name+='_cdf'
if args.mass_aware:name+='_mass'
if args.native_mean:name+='_native_mean'
if args.post_only:name+='_postonly'
name+=f'_w{args.window}'
name+='_apic_off'
saved=ROOT/'results'/f'{name}.json'
if saved.exists():print(saved.read_text());raise SystemExit
arr,reference,shared=ctx.configure(args.year);c=ctx.c;n=c.N;times=c.times;horizon=np.arange(96)
source=ROOT/'checkpoints'/('QPR2_seed20260920_best.pt' if args.epoch is None else f'QPR2_seed20260920_epoch{args.epoch:02d}.pt')
if args.cdf_epoch is not None:source=ROOT/'checkpoints'/f'CdfR2_seed{seed}_epoch{args.cdf_epoch:02d}.pt'
if args.cash_epoch is not None:source=ROOT/'checkpoints'/f'CashR2_seed{seed}_epoch{args.cash_epoch:02d}.pt'
if args.value_epoch is not None:source=ROOT/'checkpoints'/f'ContinuationR2_seed20260920_epoch{args.value_epoch:02d}.pt'
state=torch.load(source,map_location='cpu',weights_only=False)
if args.year==2025:
    assert hashlib.sha256(source.read_bytes()).hexdigest()==selected['forecast_checkpoint_sha256']
    assert hashlib.sha256((ROOT/'cache/causal_observer.npz').read_bytes()).hexdigest()==selected['observer_sha256']
model=AffineForecastR2Head() if args.cash_epoch is not None else (ContinuationR2Head(state['base_eta']) if args.value_epoch is not None else NoMarketResidual());model.load_state_dict(state['state_dict']);model.eval().requires_grad_(False)
model=torch.jit.freeze(torch.jit.trace(model,(torch.zeros(4,96,22),torch.zeros(4,96,128),torch.zeros(4,96,7),torch.ones(4)*1.08866477),strict=False,check_trace=False))
action_model=None
if args.action_epoch is not None:
    from src.models.action_value_r2 import ActionValueDecoder
    action_path=ROOT/'checkpoints'/f'ActionR2_seed20260920_epoch{args.action_epoch:02d}.pt'
    action_state=torch.load(action_path,map_location='cpu',weights_only=False);action_hash=hashlib.sha256(action_path.read_bytes()).hexdigest()
    assert action_state['forecast_checkpoint_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    if args.year==2025:assert selected['action_checkpoint_sha256']==action_hash
    action_model=ActionValueDecoder(action_state['dollar_scale'],action_state['bound']);action_model.load_state_dict(action_state['state_dict']);action_model.eval().requires_grad_(False)
    action_model=torch.jit.freeze(torch.jit.trace(action_model,(torch.zeros(1,192),torch.zeros(1,24,64),torch.zeros(1,64),torch.zeros(1,64),torch.zeros(1,64)),check_trace=False))
with np.load(ROOT/'cache'/f'structured_market{args.year}_s20260920_ep10.npz') as z:
    hours=z['hours'];hi=np.searchsorted(hours,times.floor('h').asi8);assert (hours[hi]==times.floor('h').asi8).all();pue_ref=z['pue_reference'];historical=z['historical_rt']
    market={gate:np.concatenate([z[f'rt_{gate}'][hi],np.broadcast_to(historical[None,:,None],(96,4,7))]).astype(np.float32) for gate in [0,1]}
    latent={gate:np.concatenate([z[f'latent_{gate}'][hi],np.zeros((96,4,128))]).astype(np.float32) for gate in [0,1]}
tail_correction=None
native_mean_delta=None
if args.native_mean:
    from src.data.native_market_mean import prepare_native_mean
    with np.load(prepare_native_mean(args.year)) as z:native_mean_delta={gate:z[f'delta_{gate}'] for gate in [0,1]}
if args.tail_epoch is not None or args.moment_epoch is not None or args.shape_epoch is not None:
    from src.models.tail_moment_r2 import TailMomentR2,MomentCompletionR2
    from src.models.shape_moment_r2 import ShapeMomentR2
    tail_tag,tail_epoch,tail_prefix,tail_class=next(item for item in [('shape',args.shape_epoch,'ShapeR2',ShapeMomentR2),('moment',args.moment_epoch,'MomentR2',MomentCompletionR2),('tail',args.tail_epoch,'TailR2',TailMomentR2)] if item[1] is not None)
    completion=tail_tag!='tail'
    tail_path=ROOT/'checkpoints'/f'{tail_prefix}_seed20260920_epoch{tail_epoch:02d}.pt';tail_hash=hashlib.sha256(tail_path.read_bytes()).hexdigest()
    tail_state=torch.load(tail_path,map_location='cpu',weights_only=False)
    assert tail_state['forecast_checkpoint_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    if args.year==2025:assert selected[f'{tail_tag}_checkpoint_sha256']==tail_hash
    tail_cache=ROOT/'cache'/f'{tail_tag}_correction_{args.year}_s20260920_ep{tail_epoch:02d}.npz'
    if not tail_cache.exists():
        tail_model=tail_class().cuda();tail_model.load_state_dict(tail_state['state_dict']);tail_model.eval().requires_grad_(False)
        tail_core=ArchitectedR2QPHead().cuda();tail_core.load_state_dict(state['state_dict']);tail_core.eval().requires_grad_(False)
        arrays={}
        with torch.inference_mode():
            for gate in [0,1]:
                lat_flat=latent[gate].reshape(-1,128);q_flat=market[gate].reshape(-1,7);zone_flat=np.tile(np.arange(4),n+96);outputs=[]
                for a in range(0,len(q_flat),8192):
                    adapted,shift=tail_core.market_update(torch.tensor(lat_flat[a:a+8192],device='cuda'))
                    q_tail=torch.tensor(q_flat[a:a+8192],device='cuda')+shift[:,None]
                    outputs.append(tail_model(adapted,q_tail,torch.tensor(zone_flat[a:a+8192],device='cuda'),torch.full((len(q_tail),),gate,device='cuda'))['mean_correction'].cpu().numpy())
                arrays[f'correction_{gate}']=np.concatenate(outputs).reshape(n+96,4)
        np.savez_compressed(tail_cache,**arrays,checkpoint_sha256=tail_hash);del tail_model,tail_core,arrays
    with np.load(tail_cache) as z:
        assert str(z['checkpoint_sha256'])==tail_hash
        tail_correction={gate:z[f'correction_{gate}'] for gate in [0,1]}
if args.regime:
    regime_path=ROOT/'checkpoints/RegimeR2_seed20260920.pt';regime_hash=hashlib.sha256(regime_path.read_bytes()).hexdigest();regime_state=torch.load(regime_path,map_location='cpu',weights_only=False)
    assert regime_state['forecast_checkpoint_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    if args.year==2025:assert selected['regime_checkpoint_sha256']==regime_hash
    regime_cache=ROOT/'cache'/f'regime_mean_{args.year}_s20260920.npz'
    if not regime_cache.exists():
        from src.data.regime_memory import market_history,memory_features,export_correction
        feedback,feedback_mean,_=market_history(include_2025=args.year==2025);feedback_features=memory_features(feedback,feedback_mean)
        export_correction(feedback,feedback_features,regime_state['weights'],args.year,regime_hash)
        del feedback,feedback_mean,feedback_features
    with np.load(regime_cache) as z:
        assert str(z['checkpoint_sha256'])==regime_hash
        tail_correction={gate:z[f'correction_{gate}'] for gate in [0,1]}
local=times.tz_convert('America/New_York').tz_localize(None).normalize()
pre=np.r_[(local-pd.Timedelta(days=1)+pd.Timedelta(hours=4,minutes=55)).tz_localize('America/New_York').tz_convert('UTC').asi8,np.full(96,np.iinfo(np.int64).max)]
post=np.r_[(local-pd.Timedelta(days=1)+pd.Timedelta(hours=11)).tz_localize('America/New_York').tz_convert('UTC').asi8,np.full(96,np.iinfo(np.int64).max)]
physical_file=PREVIOUS/'cache/world_physical_forecasts.npz' if args.year==2024 else ARCHIVE/'cache/final_physical_priors.npz'
with np.load(physical_file) as z:
    lookup={str(d):i for i,d in enumerate(z['dates'])};di=np.array([lookup.get(d,-1) for d in times.strftime('%Y-%m-%d')]);valid=di>=0;slot=times.hour.to_numpy()*4+times.minute.to_numpy()//15
    prior=np.concatenate([z['issue_pv'][di.clip(0),slot]*valid[:,None],np.zeros((96,4))]);prior_pue=np.concatenate([z['issue_pue'][di.clip(0),slot],np.broadcast_to(pue_ref,(96,4))])
    issue_values=pd.DatetimeIndex(pd.to_datetime(z['issue_utc'],utc=True)).asi8;issue=np.r_[np.where(valid,issue_values[di.clip(0)],np.iinfo(np.int64).max),np.full(96,np.iinfo(np.int64).max)]
solar=prepare(times,c.pv);fit_observer()
with np.load(ROOT/'cache/causal_observer.npz') as z:alpha=z['alpha'];residual=z['residual_quantiles']
physical_quantiles=None
if args.physical_epoch is not None:
    physical_path=ROOT/'checkpoints'/f'PhysicalR2_seed20260920_epoch{args.physical_epoch:02d}.pt'
    physical_hash=hashlib.sha256(physical_path.read_bytes()).hexdigest()
    if args.year==2025:assert selected['physical_checkpoint_sha256']==physical_hash
    physical_cache=ROOT/'cache'/f'physical_observer_{args.year}_s20260920_ep{args.physical_epoch:02d}.npz'
    if not physical_cache.exists():
        from src.models.causal_physical_encoder import CausalPhysicalEncoder
        from src.data.physical_observer_data import annual_data
        physical_data=annual_data(args.year);physical_model=CausalPhysicalEncoder().cuda()
        physical_model.load_state_dict(torch.load(physical_path,map_location='cpu',weights_only=False)['state_dict']);physical_model.eval().requires_grad_(False)
        outputs=[]
        with torch.inference_mode():
            for a in range(0,n,4096):
                inputs=[torch.as_tensor(physical_data[k][a:a+4096],device='cuda') for k in ['history','base_quantiles','upper','clock']]
                outputs.append(physical_model(*inputs)['quantiles'].cpu().numpy())
        np.savez_compressed(physical_cache,quantiles=np.concatenate(outputs),checkpoint_sha256=physical_hash,history_lag=1)
        del physical_model,physical_data,outputs
    with np.load(physical_cache) as z:
        assert str(z['checkpoint_sha256'])==physical_hash and int(z['history_lag'])==1
        physical_quantiles=z['quantiles'].astype(np.float64)
    assert physical_quantiles.shape==(n,4,7)
horizon_quantiles=None
if args.horizon_epoch is not None:
    horizon_path=ROOT/'checkpoints'/f'HorizonR2_seed20260920_epoch{args.horizon_epoch:02d}.pt'
    horizon_hash=hashlib.sha256(horizon_path.read_bytes()).hexdigest()
    if args.year==2025:assert selected['horizon_checkpoint_sha256']==horizon_hash
    horizon_cache=ROOT/'cache'/f'physical_horizon_{args.year}_s20260920_ep{args.horizon_epoch:02d}.npz'
    if not horizon_cache.exists():
        from src.models.causal_physical_encoder import HorizonPhysicalEncoder
        from src.data.physical_horizon_data import build
        horizon_data=build(str(args.year));horizon_model=HorizonPhysicalEncoder().cuda()
        horizon_model.load_state_dict(torch.load(horizon_path,map_location='cpu',weights_only=False)['state_dict']);horizon_model.eval().requires_grad_(False)
        outputs=np.empty((n,4,96,7),np.float32)
        with torch.inference_mode():
            for a in range(0,n,512):
                outputs[a:a+512]=horizon_model(torch.as_tensor(horizon_data['history'][a:a+512],device='cuda'),torch.as_tensor(horizon_data['future'][a:a+512],device='cuda'))['quantiles'].cpu().numpy()
        np.savez_compressed(horizon_cache,quantiles=outputs,checkpoint_sha256=horizon_hash,history_lag=1)
        del horizon_model,horizon_data,outputs
    with np.load(horizon_cache) as z:
        assert str(z['checkpoint_sha256'])==horizon_hash and int(z['history_lag'])==1
        horizon_quantiles=z['quantiles']
    assert horizon_quantiles.shape==(n,4,96,7)
probabilities=np.array([.075,.1,.2,.25,.2,.1,.075]);queue=np.zeros((96,4));ref_queue=np.zeros(4);book=np.zeros((n+8192,4))
capacity=np.minimum(1.,(np.asarray(c.c['grid_import_limit'])/np.asarray(c.c['pue_max'])-c.idle)/c.k)
capacity=np.minimum(capacity,1.)
start=time.perf_counter()
with torch.inference_mode():
    for t in range(n):
        if t:
            overdue=queue[0].copy();queue[:-1]=queue[1:].copy();queue[-1]=0;queue[0]+=overdue
        due=min(args.window*4-1,n-1-t);queue[due]+=arr[t];ref_queue=np.maximum(ref_queue-1,0)+arr[t]
        active=np.flatnonzero(queue.sum(0)>1e-10)
        if not len(active):continue
        ids=t+horizon;now=times.asi8[t];available=pre[ids]<=now;published=post[ids]<=now;physical=(issue[ids]<=now)&available
        if args.post_only:
            available=available&published;physical=physical&published
        q=np.where(published[:,None,None],market[1][ids],market[0][ids]);q=np.where(available[:,None,None],q,historical[None,:,None])
        lat=np.where(published[:,None,None],latent[1][ids],latent[0][ids])*available[:,None,None]
        pvhat=forecast(t,ids,prior[ids],physical,solar,6.)
        puehat=np.where(physical[:,None],prior_pue[ids],pue_ref[None])
        if t:
            bias=c.pue[t-1]-prior_pue[t-1]
            puehat=np.clip(puehat+np.exp(-horizon[:,None]/32)*bias,1.,np.asarray(c.c['pue_max']))
            puehat[0]=np.clip(c.pue[t-1],1.,np.asarray(c.c['pue_max']))
            scaled=c.pv[t-1]*np.clip(solar[0][t]/np.maximum(solar[0][t-1],.02),0,2)
            current_pv=np.clip(c.pv[t-1]+alpha*(scaled-c.pv[t-1]),0,1.5);pvhat[0]=current_pv
        else:current_pv=pvhat[0]
        state_array=np.zeros((len(active),96,22),np.float32);clock=(times[t].hour+times[t].minute/60+horizon/4)%24
        for a,z in enumerate(active):
            total=queue[:,z].sum();bins=np.array([queue[:hours*4,z].sum() for hours in [4,8,12,24]])
            state_array[a]=np.column_stack([np.zeros(96),np.full(96,min(.05,total)),pvhat[:,z],puehat[:,z]/pue_ref[z],horizon/96,np.full(96,np.log1p(arr[t,z])/4),np.full(96,np.log1p(total)/4),np.zeros(96),available,np.ones(96),published,1-horizon/96,np.broadcast_to(np.log1p(bins)/4,(96,4)),np.zeros(96),np.full(96,min(t,16)/16),np.full(96,np.log1p(c.pv[t-1,z]) if t else 0),np.full(96,np.log1p(prior[t-1,z]) if t else 0),np.sin(2*np.pi*clock/24),np.cos(2*np.pi*clock/24)])
        out=model(torch.from_numpy(state_array),torch.from_numpy(lat[:,active].transpose(1,0,2).copy()),torch.from_numpy(q[:,active].transpose(1,0,2).copy()),torch.tensor(pue_ref[active]))
        pv_scenarios=out['pv_quantiles'].numpy().transpose(0,2,1).astype(np.float64)
        price_scenarios=out['conditional_price'].numpy().transpose(0,2,1).astype(np.float64);opportunity=out['opportunity'].numpy().astype(np.float64)*args.opportunity_weight
        if horizon_quantiles is not None:
            pv_scenarios=horizon_quantiles[t,active].transpose(0,2,1).astype(np.float64)
            pv_mean=(pv_scenarios*probabilities[None,:,None]).sum(1,keepdims=True)
            pv_std=np.sqrt(((pv_scenarios-pv_mean)**2*probabilities[None,:,None]).sum(1,keepdims=True))
            rt=out['rt_quantiles'].numpy();rt_mean=(rt*probabilities).sum(-1)
            rt_std=np.sqrt(((rt-rt_mean[:,:,None])**2*probabilities).sum(-1))
            price_scenarios=out['rt_decision'].numpy()[:,None,:]+out['cross_moment_rho'].numpy()[:,None,:]*rt_std[:,None,:]*(pv_scenarios-pv_mean)/np.maximum(pv_std,1e-5)
        native_mean=(out['rt_quantiles'].numpy()*probabilities).sum(-1)
        correction=out['rt_decision'].numpy()-native_mean
        price_scenarios+=(args.cost_bias_weight-1)*correction[:,None,:]
        tail_delta=np.zeros((96,4))
        if tail_correction is not None:
            tail_delta=np.where(published[:,None],tail_correction[1][ids],tail_correction[0][ids])*available[:,None]
            price_scenarios+=tail_delta[:,active].T[:,None,:]
        moment_delta=np.zeros((len(active),96))
        if native_mean_delta is not None:
            delta=np.where(published[:,None],native_mean_delta[1][ids],native_mean_delta[0][ids])*available[:,None]
            factor=np.ones((len(active),96))
            if args.cash_epoch is not None:
                original=q[:,active].transpose(1,0,2)
                span=original[:,:,-1]-original[:,:,0];new_span=out['rt_quantiles'].numpy()[:,:,-1]-out['rt_quantiles'].numpy()[:,:,0]
                factor=np.divide(new_span,span,out=np.ones_like(span),where=span>1e-6)
            moment_delta=delta[:,active].T*factor
            price_scenarios+=moment_delta[:,None,:]
        for a,z in enumerate(active):
            h=min(96,n-t)
            if t:
                pv_scenarios[a,:,0]=np.clip(current_pv[z]+residual[int(times[t].hour*4+times[t].minute//15),z]*max(solar[0][t,z],.1),0,1.5)
                if physical_quantiles is not None:pv_scenarios[a,:,0]=physical_quantiles[t,z]
                # This is an expectation under forecast scenarios; no realized
                # current or future spot-price residual is observed by the policy.
                price_scenarios[a,:,0]=float(native_mean[a,0]+args.cost_bias_weight*correction[a,0]+tail_delta[0,z]+moment_delta[a,0])
                if args.current_coupling:
                    prior_pv=out['pv_quantiles'].numpy()[a,0].astype(float);prior_mean=np.dot(probabilities,prior_pv)
                    prior_sigma=np.sqrt(np.dot(probabilities,(prior_pv-prior_mean)**2))
                    observed_pv=pv_scenarios[a,:,0];observed_mean=np.dot(probabilities,observed_pv)
                    observed_sigma=np.sqrt(np.dot(probabilities,(observed_pv-observed_mean)**2))
                    rt_values=out['rt_quantiles'].numpy()[a,0].astype(float)
                    price_sigma=np.sqrt(np.dot(probabilities,(rt_values-np.dot(probabilities,rt_values))**2))
                    slope=float(out['cross_moment_rho'][a,0])*price_sigma/max(prior_sigma,observed_sigma,1e-5)
                    price_scenarios[a,:,0]+=slope*(observed_pv-observed_mean)
            reference_known=np.minimum(1.,np.maximum(ref_queue[z]-np.arange(h),0.))
            if t>=n-4:service=min(capacity[z],queue[:,z].sum())
            else:
                eta=float(out['continuation_eta'][a]) if args.value_epoch is not None else args.eta
                values=(queue[:h,z],pv_scenarios[a,:,:h].copy(),price_scenarios[a,:,:h].copy(),probabilities,puehat[:h,z].copy(),reference_known,float(c.idle[z]),float(c.k[z]),opportunity[a,:h].copy(),float(pue_ref[z]),float(capacity[z]))
                if action_model is None:service=select_action(*values,eta,args.safety,args.exact_action,args.continuous_cost,args.mass_aware)
                else:
                    candidates,base_cost,alternative=action_value_curves(*values,args.safety,True,args.exact_action)
                    _,correction_value=action_model(out['continuation_features'][a:a+1],out['policy_tokens'][a:a+1],torch.tensor(candidates[None],dtype=torch.float32),torch.tensor(base_cost[None],dtype=torch.float32),torch.tensor(alternative[None],dtype=torch.float32))
                    scores=base_cost+correction_value.numpy()[0];service=float(candidates[scores<=scores.min()+1e-9].max())
            remaining=service
            for d in np.flatnonzero(queue[:,z]>1e-12):
                take=min(queue[d,z],remaining);queue[d,z]-=take;remaining-=take
                if remaining<1e-12:break
            book[t,z]=service-remaining
        if t%8640==0:print('SCENARIO RECOURSE',args.eta,t,n,'seconds',round(time.perf_counter()-start,1),flush=True)
for z in range(4):
    remaining=queue[:,z].sum();t=n
    while remaining>1e-9:
        take=min(capacity[z],remaining);book[t,z]=take;remaining-=take;t+=1
ctx.RESULTS=ROOT/'results';ctx.TRAJECTORIES=ROOT/'results/trajectories';ctx.TRAJECTORIES.mkdir(exist_ok=True);c.DATA=ctx.TRAJECTORIES
row=ctx.measure(name,book,reference,arr,shared,dict(experiment='ScenarioRecourse',year=args.year,trace='Eagle',seed=seed,window_h=args.window,role='selection' if args.year==2024 else 'exploratory_frozen_evaluation'),extra=dict(seconds=time.perf_counter()-start,recourse_fraction=args.eta,safety_capacity_fraction=args.safety,physical_observation_lag_quarters=1,forecast_scenarios=7,forecast_epoch=state['epoch'],forecast_parameters_updated=False,forecast_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),controller_training_scope='structural decision-layer probe using the trained forecast/state-cost module',nonanticipativity='one common current action; hypothetical future continuations use only forecast scenarios',independent_future_RT_shocks_revealed=False,known_reference='causal earliest-service queue from already arrived work',terminal_guard='earliest service for known work in the final hour',adaptive_lock_rule=False))
bound=json.loads((EXTENSION/'results'/f'I1_Eagle_{args.year}_w{args.window}.json').read_text());row['capture_lower_fraction']=row['saving_usd']/bound['maximum_saving_upper_usd'];row['target_80_usd']=.8*bound['maximum_saving_upper_usd'];row['certified_savings_ceiling_usd']=bound['maximum_saving_upper_usd'];saved.write_text(json.dumps(row,indent=2));print('SCENARIO COMPLETE',json.dumps({k:row[k] for k in ['saving_usd','capture_lower_fraction','late_IT_mwh','closing_it_mwh']}),flush=True)
row.update(parameters_updated_during_evaluation=False,prepublication_pathway_suppressed=args.post_only,information_intervention='neural market/state-cost availability and issued physical priors suppressed before target DA publication; causal lagged observations and seasonal fallback retained' if args.post_only else 'native publication-aware pathway')
row.update(opportunity_weight=args.opportunity_weight,cost_bias_weight=args.cost_bias_weight,exact_action=args.exact_action,current_coupling=args.current_coupling,continuous_cost=args.continuous_cost,mass_aware=args.mass_aware,native_mean=args.native_mean,action_search='all current/future cost breakpoints' if args.exact_action else '0.025 grid plus current physical and reference knots')
row.update(apic_market_residual_enabled=False,apic_scope='Frozen checkpoint diagnostic: disable only the learned APIC-preserving market residual shift, retain common physical allocator and settlement; not an isolated trained architectural effect')
if args.native_mean:row.update(price_cost_mean='integrated native DA marginal minus integrated native DART marginal, transformed by the trained APIC location-scale adapter; not the compressed seven-RT-atom mean')
if args.mass_aware:row.update(continuation_curve_capacity='min(physical slot capacity, total known residual work); redundant feasible bound for adaptive numerical resolution')
if args.continuous_cost:row.update(future_cost_integration='analytic clipped-Gaussian grid and price-PV cross moments; piecewise convex planning approximation')
if args.cdf_epoch is not None:row.update(cdf_training_epoch=args.cdf_epoch,forecast_parameters_updated=args.cdf_epoch>0,controller_training_scope='R2 trained through differentiable Gaussian-CDF marginal costs and conserved 16-cohort QP rollouts; deployment uses analytic continuation costs and the deterministic current-action feasibility layer')
if args.cash_epoch is not None:row.update(cash_training_epoch=args.cash_epoch,forecast_parameters_updated=args.cash_epoch>0,controller_training_scope='R2 with a positive APIC-compatible market-distribution scale, continuous physical costs and differentiable conserved QP rollouts; aggregate training USD objective; deterministic current-action execution')
if args.physical_epoch is not None:
    row.update(physical_head_epoch=args.physical_epoch,physical_checkpoint_sha256=physical_hash,physical_head_architecture='three causal dilated temporal blocks, cross-site attention, zero-initialized ordered location-scale residual',physical_forecast_queue_independent=True,physical_head_training_years=[2022,2023],controller_training_scope='original trained R2 forecast/state-cost module plus a supervised causal physical distribution head; no claim of differentiating through this current-action controller')
if args.horizon_epoch is not None:
    row.update(horizon_head_epoch=args.horizon_epoch,horizon_checkpoint_sha256=horizon_hash,physical_head_architecture='causal multi-scale temporal convolution, cross-site attention, lead embedding, asymmetric ordered quantile decoder',physical_forecast_queue_independent=True,physical_head_training_years=[2022,2023],controller_training_scope='frozen R2 market/state-cost module plus a supervised causal 96-slot PV distribution head; economic selection on 2024; no gradient-through-current-action claim')
if args.action_epoch is not None:
    row.update(action_head_epoch=args.action_epoch,action_checkpoint_sha256=action_hash,action_head_architecture='candidate-action queries attend to shared R2 temporal tokens; bounded residual value decoder',action_head_training_years=[2022,2023],controller_training_scope='attention-based action-value head trained with differentiable finite-action expected-cost loss and realized costs of forecast-selected future plans; frozen core forecast representations')
if args.tail_epoch is not None or args.moment_epoch is not None or args.shape_epoch is not None:
    row.update(tail_head_epoch=tail_epoch,tail_checkpoint_sha256=tail_hash,central_moment_completed=completion,tail_head_architecture='bounded central-quantile interpolation and two tail first-moment heads' if completion else 'two nonnegative conditional tail-excess first-moment heads',tail_head_training_years=[2021,2022,2023],seven_quantile_forecasts_unchanged=True,controller_training_scope='forecast-preserving completion of expected electricity cost; moment heads trained by conditional-mean scoring, actual annual economics used only for 2024 selection')
if args.regime:
    row.update(regime_checkpoint_sha256=regime_hash,regime_head_training_years=[2021,2022,2023],seven_quantile_forecasts_unchanged=True,feedback_rule='target local date minus two days; completed errors known before registered pre-DA issue',evaluation_year_weight_updates=False,controller_training_scope='convex multiscale market-error memory with fixed pre-2024 trained weights; deterministic updating of lagged observations; independent constraints and settlement')
saved.write_text(json.dumps(row,indent=2))
if args.value_epoch is not None:
    row.update(value_head_epoch=args.value_epoch,forecast_epoch=state['forecast_epoch'],controller_training_scope='R2 continuation head trained on 2022-2023 known-work decision-regret targets; shared representations and forecasting heads frozen',forecast_parameters_updated=False)
    saved.write_text(json.dumps(row,indent=2))
