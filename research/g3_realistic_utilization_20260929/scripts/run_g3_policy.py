"""Causal neural forecasts plus nonanticipative physical-scenario continuation.

Only present service is executed. Future plans are hypothetical scenario
continuations, not claimed knowledge of future prices or physical outcomes.
"""
import os,sys,argparse,json,time
from pathlib import Path
sys.dont_write_bytecode=True
TASK_ROOT=Path(__file__).resolve().parents[1]
WORKSPACE_G3=Path(__file__).resolve().parents[2]
ENGINE_ROOT=WORKSPACE_G3/'ExecutableServiceCommitments_Revision_20260928'/'engine'
os.environ.setdefault('NUMBA_CACHE_DIR',str(TASK_ROOT/'outputs'/'numba_cache'))
sys.path.insert(0,str(TASK_ROOT/'scripts'))
sys.path.insert(0,str(ENGINE_ROOT))
from src.runtime import ROOT,WORKSPACE,PREVIOUS,EXTENSION,ARCHIVE
import numpy as np,pandas as pd,torch
import src,vendor
src.__path__.append(str(EXTENSION/'src'));vendor.__path__.append(str(EXTENSION/'vendor'));sys.path.append(str(EXTENSION))
from src import context as ctx
from src.models.coupled_rollout import ArchitectedR2QPHead
from src.models.continuation_r2 import ContinuationR2Head
from src.models.affine_continuous import AffineForecastR2Head
from src.models.scenario_recourse import select_action,action_value_curves
from src.data.solar_nowcast import prepare,forecast
from src.data.causal_observer import fit_observer
from g3_executor import execute_rigid_flexible_earliest,deadline_miss_mwh,closing_work_mwh

p=argparse.ArgumentParser()
p.add_argument('--case-id',required=True)
p.add_argument('--total-utilization',type=float,required=True)
p.add_argument('--flex-share',type=float,required=True)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--smoke-days',type=int,default=0)
p.add_argument('--site',type=int,default=0)
p.add_argument('--eta',type=float,default=0.)
p.add_argument('--year',type=int,choices=[2024,2025],default=2025)
p.add_argument('--window',type=int,choices=[24],default=24)
p.add_argument('--safety',type=float,default=.9)
p.add_argument('--seed',type=int,default=20260920)
p.add_argument('--cdf-epoch',type=int,required=True)
p.add_argument('--cash-epoch',type=int)
p.add_argument('--value-epoch',type=int)
p.add_argument('--physical-epoch',type=int)
p.add_argument('--horizon-epoch',type=int)
p.add_argument('--action-epoch',type=int)
p.add_argument('--tail-epoch',type=int)
p.add_argument('--moment-epoch',type=int)
p.add_argument('--shape-epoch',type=int)
p.add_argument('--regime',action='store_true')
p.add_argument('--exact-action',action='store_true')
p.add_argument('--current-coupling',action='store_true')
p.add_argument('--continuous-cost',action='store_true')
p.add_argument('--mass-aware',action='store_true')
p.add_argument('--native-mean',action='store_true')
p.add_argument('--opportunity-weight',type=float,default=1.)
p.add_argument('--cost-bias-weight',type=float,default=1.)
p.add_argument('--post-only',action='store_true')
args=p.parse_args()
if args.year!=2025 or args.window!=24: raise ValueError('G3 uses the frozen 2025, 24 h contract only.')
if not (0.<args.total_utilization<1. and 0.<args.flex_share<1.): raise ValueError('G3 utilization and flexible share must be fractions in (0,1).')
if not 0<=args.site<4: raise ValueError('site index must be 0..3')
expected_epoch={20260920:6,20260921:2,20260922:0}
seed=args.seed
if seed not in expected_epoch or args.cdf_epoch!=expected_epoch[seed]:
    raise ValueError('G3 only accepts the three preselected CdfR2 checkpoints and epochs.')
if any(v is not None for v in [args.cash_epoch,args.value_epoch,args.physical_epoch,args.horizon_epoch,
        args.action_epoch,args.tail_epoch,args.moment_epoch,args.shape_epoch]) or args.regime:
    raise ValueError('G3 is locked to the existing selected CdfR2 path.')
if args.eta!=0. or args.safety!=.9 or not args.current_coupling or not args.continuous_cost:
    raise ValueError('G3 must retain the already selected frozen controller settings.')
if args.exact_action or args.mass_aware or args.native_mean or args.post_only or args.opportunity_weight!=1. or args.cost_bias_weight!=1.:
    raise ValueError('G3 does not accept controller or information-gate variants.')
seed=args.seed
if seed!=20260920:
    authorization=json.loads((ROOT/'results/replication_authorization.json').read_text(encoding='utf-8-sig'))
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
    selected=json.loads(selection_file.read_text(encoding='utf-8-sig'));assert selected['eta']==args.eta and selected['safety']==args.safety
    assert selected['family']=='CdfR2' and selected['seed']==seed and selected['refinement_epoch']==args.cdf_epoch
    assert selected.get('opportunity_weight',1.)==args.opportunity_weight and selected.get('cost_bias_weight',1.)==args.cost_bias_weight
    assert selected.get('exact_action',False)==args.exact_action
    assert selected.get('current_coupling',False)==args.current_coupling
    assert selected.get('continuous_cost',False)==args.continuous_cost
    assert selected.get('mass_aware',False)==args.mass_aware
    assert selected.get('native_mean',False)==args.native_mean
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.mha.set_fastpath_enabled(False)
name=f'{args.case_id}_s{seed}_U{args.total_utilization:.2f}_F{args.flex_share:.2f}_w24'
if args.smoke_days:name+=f'_smoke{args.smoke_days}d_z{args.site}'
OUTDIR=args.output_dir.resolve();OUTDIR.mkdir(parents=True,exist_ok=True)
TRAJECTORIES=OUTDIR/'trajectories';TRAJECTORIES.mkdir(parents=True,exist_ok=True)
saved=OUTDIR/f'{name}.json'
if saved.exists():
    existing=json.loads(saved.read_text(encoding='utf-8'))
    print('EXISTING G3 RESULT',json.dumps({k:existing.get(k) for k in ['case_id','seed','status','saving_usd','structurally_feasible','policy_feasible']}),flush=True)
    raise SystemExit(0)

# Load the official NYISO 2025 forcing, then form the frozen rigid/flexible split.
eagle_arrivals,_,shared=ctx.configure(args.year);c=ctx.c
base_n=int(c.N);site_mask=np.ones(4,dtype=bool)
if args.smoke_days:
    if args.smoke_days<1:raise ValueError('smoke_days must be positive')
    n=min(int(args.smoke_days)*96,base_n);site_mask[:]=False;site_mask[args.site]=True
    eagle_arrivals=eagle_arrivals[:n].copy();shared=shared[:n].copy()
    c.times=c.times[:n];c.N=n
    for key in ['da','rt','pv','pue']:
        value=getattr(c,key)[:n].copy();value[:,~site_mask]=0.;setattr(c,key,value)
    eagle_arrivals[:,~site_mask]=0.
    c.k=np.asarray(c.k).copy();c.k[~site_mask]=0.
    c.idle=np.asarray(c.idle).copy();c.idle[~site_mask]=0.
    c.day_index,c.days=pd.factorize(c.times.tz_convert('America/New_York').date,sort=True)
else:n=base_n
times=c.times;horizon=np.arange(96);dt=float(c.DT)
# The official normalized variable capacity is capacity=1 at the four sites;
# retain the formal physical-limit calculation for both annual and smoke runs.
capacity_full=np.minimum(1.,(np.asarray(c.c['grid_import_limit'])/np.asarray(c.c['pue_max'])-c.idle)/np.maximum(c.k,1e-12))
capacity_full=np.maximum(np.minimum(capacity_full,1.),0.)
capacity_full[~site_mask]=0.
capacity_energy=dt*n*float(np.sum(capacity_full*c.k))
eagle_energy=dt*float(np.sum(eagle_arrivals*c.k[None,:]))
if capacity_energy<=0 or eagle_energy<=0:raise RuntimeError('G3 workload/capacity energy is empty.')
target_total_energy=args.total_utilization*capacity_energy
target_flexible_energy=args.flex_share*target_total_energy
target_rigid_energy=(1.-args.flex_share)*target_total_energy
scale_flexible=target_flexible_energy/eagle_energy
flex_arrivals=eagle_arrivals*scale_flexible
rigid_mean_utilization=args.total_utilization*(1.-args.flex_share)
flexible_mean_utilization=args.total_utilization*args.flex_share
rigid_service=rigid_mean_utilization*capacity_full
rigid_arrivals=np.broadcast_to(rigid_service,(n,4)).copy()
flexible_capacity=np.maximum(capacity_full-rigid_service,0.)
total_arrivals=rigid_arrivals+flex_arrivals
rigid_mwh=dt*float(np.sum(rigid_arrivals*c.k[None,:]))
flexible_mwh=dt*float(np.sum(flex_arrivals*c.k[None,:]))
total_mwh=rigid_mwh+flexible_mwh
realized_utilization=total_mwh/capacity_energy
if abs(target_rigid_energy+target_flexible_energy-target_total_energy)>1e-8*max(1.,target_total_energy):
    raise RuntimeError('G3 rigid/flexible target-energy split is not conserved.')
if abs(total_mwh-target_total_energy)>1e-8*max(1.,target_total_energy):
    raise RuntimeError('G3 realized workload energy does not match target utilization.')
if np.any(rigid_service>capacity_full+1e-10):raise RuntimeError('Rigid service exceeds variable capacity.')
zero_rigid=np.zeros_like(rigid_arrivals)
reference_flexible_book,joined,left=execute_rigid_flexible_earliest(
    zero_rigid,flex_arrivals,flexible_capacity,int(args.window*4))
if not np.allclose(joined,flex_arrivals,rtol=1e-10,atol=1e-9):raise RuntimeError('Earliest reference flexible-arrival mismatch.')
reference=reference_flexible_book.copy()
reference[:n]+=rigid_service[None,:]
reference_late=deadline_miss_mwh(reference,total_arrivals,c.k,dt,int(args.window*4))
reference_closing=closing_work_mwh(reference,n,c.k,dt)
structurally_feasible=reference_late<=1e-8
reference_flexible_served=dt*float(np.sum(reference_flexible_book[:n]*c.k[None,:]))
reference_cash,_=c.cash(reference)
reference_liability=dt*float(np.sum(reference[n:]*c.k[None,:]*np.max(c.c['pue_max'])*c.train_rt[None,:]))
reference_cost=float(reference_cash.sum()+shared.sum()+reference_liability)
flex_common=dict(case_id=args.case_id,seed=seed,total_utilization_target=args.total_utilization,
    total_utilization_realized=realized_utilization,flexible_share=args.flex_share,
    rigid_mean_utilization=rigid_mean_utilization,flexible_mean_utilization=flexible_mean_utilization,
    total_variable_it_mwh=total_mwh,rigid_it_mwh=rigid_mwh,flexible_released_it_mwh=flexible_mwh,
    reference_flexible_served_it_mwh=reference_flexible_served,reference_cost_usd=reference_cost,
    reference_late_it_mwh=reference_late,reference_closing_it_mwh=reference_closing,
    structurally_feasible=bool(structurally_feasible),service_window_h=args.window,
    mean_residual_flexible_capacity_fraction=float(np.sum(flexible_capacity*c.k)/max(np.sum(capacity_full*c.k),1e-12)),
    matched_reference_uses_residual_capacity_through_terminal_continuation=True,
    sites_used=np.flatnonzero(site_mask).astype(int).tolist(),
    checkpoint=f'CdfR2_seed{seed}_epoch{args.cdf_epoch:02d}.pt',checkpoint_epoch=args.cdf_epoch,
    parameter_updates_during_evaluation=False,training_or_optimizer_used=False,
    rigid_profile='constant proportional capacity occupancy',
    flexible_profile='Eagle arrival tensor multiplied by one annual-energy scale',
    flex_share_definition='released flexible variable-IT energy / total variable-IT energy')
if not structurally_feasible:
    result={**flex_common,'policy_cost_usd':'NA','saving_usd':'NA','saving_pct_total_cost':'NA',
        'saving_per_flexible_mwh':'NA','shifted_flexible_it_mwh':'NA','saving_per_shifted_mwh':'NA',
        'policy_flexible_served_it_mwh':'NA','policy_late_it_mwh':'NA','late_flexible_it_mwh':'NA',
        'policy_closing_it_mwh':'NA','cvar95_usd':'NA','maximum_grid_import_mw':'NA',
        'maximum_recovery_import_mw':'NA','policy_feasible':False,
        'status':'STRUCTURALLY_INFEASIBLE_NO_POLICY_REPLAY',
        'notes':'Matched earliest-service reference misses the 24 h contract; no policy economics are reported as delivered value.'}
    saved.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(0)

arr=flex_arrivals;reference=reference;capacity=flexible_capacity
idle_with_rigid=c.idle+rigid_service*c.k
ctx.RESULTS=OUTDIR;ctx.TRAJECTORIES=TRAJECTORIES;ctx.TRAJECTORIES.mkdir(parents=True,exist_ok=True);c.DATA=TRAJECTORIES
source=ROOT/'checkpoints'/f'CdfR2_seed{seed}_epoch{args.cdf_epoch:02d}.pt'
state=torch.load(source,map_location='cpu',weights_only=False)
model=AffineForecastR2Head() if args.cash_epoch is not None else (ContinuationR2Head(state['base_eta']) if args.value_epoch is not None else ArchitectedR2QPHead());model.load_state_dict(state['state_dict']);model.eval().requires_grad_(False)
if any(parameter.requires_grad for parameter in model.parameters()):raise RuntimeError('Frozen G3 inference received trainable parameters.')
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
solar=prepare(times,c.pv)
observer_cache=ROOT/'cache/causal_observer.npz'
if not observer_cache.exists():raise FileNotFoundError(f'Frozen observer cache unavailable: {observer_cache}')
with np.load(observer_cache,allow_pickle=False) as z:alpha=z['alpha'];residual=z['residual_quantiles']
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
capacity=flexible_capacity
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
            state_array[a]=np.column_stack([np.full(96,1.-capacity[z]),np.full(96,min(.05,total)),pvhat[:,z],puehat[:,z]/pue_ref[z],horizon/96,np.full(96,np.log1p(arr[t,z])/4),np.full(96,np.log1p(total)/4),np.zeros(96),available,np.ones(96),published,1-horizon/96,np.broadcast_to(np.log1p(bins)/4,(96,4)),np.zeros(96),np.full(96,min(t,16)/16),np.full(96,np.log1p(c.pv[t-1,z]) if t else 0),np.full(96,np.log1p(prior[t-1,z]) if t else 0),np.sin(2*np.pi*clock/24),np.cos(2*np.pi*clock/24)])
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
                values=(queue[:h,z],pv_scenarios[a,:,:h].copy(),price_scenarios[a,:,:h].copy(),probabilities,puehat[:h,z].copy(),reference_known,float(idle_with_rigid[z]),float(c.k[z]),opportunity[a,:h].copy(),float(pue_ref[z]),float(capacity[z]))
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
flexible_policy_book=book.copy()
policy_book=flexible_policy_book.copy();policy_book[:n]+=rigid_service[None,:]
policy_late=deadline_miss_mwh(policy_book,total_arrivals,c.k,dt,int(args.window*4))
policy_closing=closing_work_mwh(policy_book,n,c.k,dt)
policy_feasible=policy_late<=1e-8
reference_flexible_book=np.maximum(reference[:n]-rigid_service[None,:],0.)
policy_flexible_in_year=np.maximum(policy_book[:n]-rigid_service[None,:],0.)
residual_capacity_violation=float(np.maximum(flexible_policy_book[:n]-flexible_capacity[None,:],0.).max())
total_capacity_violation=float(np.maximum(policy_book[:n]-capacity_full[None,:],0.).max())
if residual_capacity_violation>1e-8 or total_capacity_violation>1e-8:
    raise RuntimeError(f'G3 policy used occupied capacity: flex={residual_capacity_violation}, total={total_capacity_violation}')
if np.max(np.abs(reference[:n]-rigid_service[None,:]-reference_flexible_book))>1e-8:
    raise RuntimeError('G3 reference did not preserve the shared rigid service.')
row=ctx.measure(name,policy_book,reference,total_arrivals,shared,
    dict(experiment='G3 realistic utilization',year=args.year,trace='Eagle rigid background + flexible arrivals',
         seed=seed,window_h=args.window,role='frozen annual evaluation'),late=policy_late,
    extra=dict(case_id=args.case_id,total_utilization_target=args.total_utilization,
        total_utilization_realized=realized_utilization,flexible_share=args.flex_share,
        rigid_mean_utilization=rigid_mean_utilization,flexible_mean_utilization=flexible_mean_utilization,
        rigid_it_mwh=rigid_mwh,flexible_released_it_mwh=flexible_mwh,
        reference_late_it_mwh=reference_late,reference_closing_it_mwh=reference_closing,
        policy_late_it_mwh=policy_late,policy_closing_it_mwh=policy_closing,
        structurally_feasible=bool(structurally_feasible),policy_feasible=bool(policy_feasible),
        parameters_updated_during_evaluation=False,optimizer_created=False,backward_called=False,
        checkpoint=f'CdfR2_seed{seed}_epoch{args.cdf_epoch:02d}.pt',
        epoch=args.cdf_epoch,controller_eta=args.eta,safety=args.safety,
        current_coupling=args.current_coupling,continuous_cost=args.continuous_cost,
        physical_headroom_feature='state[:, :, 0] = 1 - residual flexible capacity',
        decision_idle='base idle + rigid IT load; residual capacity passed to frozen executor',
        policy_flexible_capacity_violation= residual_capacity_violation,
        policy_total_capacity_violation=total_capacity_violation,
        reference_and_policy_same_rigid_service=True,
        reference_and_policy_same_flexible_arrivals=True,
        same_physical_forcing_and_settlement=True,
        causal_future_price_gates_retained=True,
        inference_mode=True,
        rigid_profile='constant proportional capacity occupancy',
        flexible_profile='scaled Eagle quarter-hour arrival tensor',
        work_allocation='rigid immediate service then flexible current-action queue',
        information_gate='official existing causal/publication-aware pathway'))

# Save auditable economics and service metrics on top of the formal settlement row.
policy_cash,policy_grid=c.cash(policy_book);reference_cash,reference_grid=c.cash(reference)
fee_q=c.FEE*dt*np.abs(policy_grid-reference_grid).sum(1)
reference_liability=dt*float(np.sum(reference[n:]*c.k[None,:]*np.max(c.c['pue_max'])*c.train_rt[None,:]))
policy_liability=dt*float(np.sum(policy_book[n:]*c.k[None,:]*np.max(c.c['pue_max'])*c.train_rt[None,:]))
daily_shared=shared.copy() if np.ndim(shared)==1 else shared.sum(1)
daily_ref_q=reference_cash+daily_shared
daily_pol_q=policy_cash+daily_shared+fee_q
daily_ref_cost=np.bincount(c.day_index,weights=daily_ref_q,minlength=len(c.days))
daily_policy_cost=np.bincount(c.day_index,weights=daily_pol_q,minlength=len(c.days))
daily_ref_cost[-1]+=reference_liability;daily_policy_cost[-1]+=policy_liability
daily_saving=daily_ref_cost-daily_policy_cost
release_cum=np.cumsum(total_arrivals,axis=0);before_release=release_cum-total_arrivals
served_cum=np.cumsum(policy_book,axis=0);due_index=np.arange(n)+int(args.window*4)-1
cohort_late=np.maximum(release_cum-np.maximum(before_release,served_cum[due_index]),0.)
late_per_q=dt*np.sum(cohort_late*c.k[None,:],axis=1)
daily_late=np.bincount(c.day_index,weights=late_per_q,minlength=len(c.days))
flex_served_q=dt*np.sum(flexible_policy_book[:n]*c.k[None,:],axis=1)
flex_release_q=dt*np.sum(flex_arrivals*c.k[None,:],axis=1)
rigid_q=dt*np.sum(rigid_arrivals*c.k[None,:],axis=1)
daily_flex_served=np.bincount(c.day_index,weights=flex_served_q,minlength=len(c.days))
daily_flex_release=np.bincount(c.day_index,weights=flex_release_q,minlength=len(c.days))
daily_rigid=np.bincount(c.day_index,weights=rigid_q,minlength=len(c.days))
outstanding=dt*np.cumsum((flex_arrivals-flexible_policy_book[:n])*c.k[None,:],axis=0).sum(axis=1)
daily_outstanding=np.array([outstanding[np.flatnonzero(c.day_index==d)[-1]] for d in range(len(c.days))])
daily_policy_closing=np.zeros(len(c.days));daily_reference_closing=np.zeros(len(c.days))
daily_policy_closing[-1]=policy_closing;daily_reference_closing[-1]=reference_closing
daily_rows=[]
for d,day in enumerate(c.days):
    daily_rows.append(dict(date=str(day),case_id=args.case_id,seed=seed,total_utilization=args.total_utilization,
        flex_share=args.flex_share,reference_cost_usd=float(daily_ref_cost[d]),policy_cost_usd=float(daily_policy_cost[d]),
        daily_saving_usd=float(daily_saving[d]),rigid_energy_mwh=float(daily_rigid[d]),
        flexible_energy_mwh=float(daily_flex_release[d]),flexible_served_mwh=float(daily_flex_served[d]),
        late_it_mwh=float(daily_late[d]),closing_it_mwh=float(daily_policy_closing[d]),
        reference_closing_it_mwh=float(daily_reference_closing[d]),
        outstanding_flexible_work_mwh=float(daily_outstanding[d])))
daily_path=OUTDIR/f'{name}_daily.csv';pd.DataFrame(daily_rows).to_csv(daily_path,index=False,encoding='utf-8-sig')
shifted_flexible=dt*float(np.sum(np.maximum(reference[:n]-policy_book[:n],0.)*c.k[None,:]))
row.update({**flex_common,'reference_cost_usd':float(row['reference_cost_usd']),
    'policy_cost_usd':float(row['adjusted_cost_usd']),'saving_usd':float(row['saving_usd']),
    'saving_pct_total_cost':100.*float(row['saving_usd'])/max(float(row['reference_cost_usd']),1e-12),
    'saving_per_flexible_mwh':float(row['saving_usd'])/max(flexible_mwh,1e-12),
    'shifted_flexible_it_mwh':shifted_flexible,
    'saving_per_shifted_mwh':float(row['saving_usd'])/shifted_flexible if shifted_flexible>1e-12 else 'NA',
    'policy_flexible_served_it_mwh':float(daily_flex_served.sum()),
    'late_flexible_it_mwh':float(policy_late),'policy_late_it_mwh':float(policy_late),
    'policy_closing_it_mwh':float(policy_closing),'reference_late_it_mwh':float(reference_late),
    'reference_closing_it_mwh':float(reference_closing),'cvar95_usd':float(row.get('mtm_daily_loss_cvar95_usd',row.get('daily_loss_cvar95_usd',np.nan))),
    'maximum_grid_import_mw':float(row['max_grid_mw']),
    'maximum_recovery_import_mw':float(row.get('max_fleet_recovery_mw',0.)),
    'policy_flexible_capacity_violation':residual_capacity_violation,
    'policy_total_capacity_violation':total_capacity_violation,
    'reference_and_policy_same_rigid_service':True,
    'reference_and_policy_same_flexible_arrivals':True,
    'same_physical_forcing_and_settlement':True,
    'causal_future_price_gates_retained':True,
    'structurally_feasible':bool(structurally_feasible),'policy_feasible':bool(policy_feasible),
    'status':'POLICY_REPLAY_COMPLETE','daily_results_file':str(daily_path.relative_to(TASK_ROOT)),
    'parameters_updated_during_evaluation':False,'forecast_parameters_updated':False,
    'checkpoint':f'CdfR2_seed{seed}_epoch{args.cdf_epoch:02d}.pt',
    'checkpoint_epoch':args.cdf_epoch,'optimizer_created':False,'backward_called':False,
    'saving_per_flexible_mwh_retention_pct':'NA','smoke_test':bool(args.smoke_days)} )
if not np.isclose(float(daily_saving.sum()),float(row['saving_usd']),rtol=1e-8,atol=1e-5):
    raise RuntimeError(f'Daily settlement does not add to annual saving: {daily_saving.sum()} vs {row["saving_usd"]}')
saved.write_text(json.dumps(row,indent=2),encoding='utf-8')
print('G3 POLICY COMPLETE',json.dumps({k:row[k] for k in ['case_id','seed','saving_usd','saving_pct_total_cost','saving_per_flexible_mwh','late_flexible_it_mwh','policy_closing_it_mwh','policy_feasible']}),flush=True)
