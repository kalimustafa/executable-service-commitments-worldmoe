"""Forecast-only features and pre-2024 known-work decision-regret targets."""
import sys,json,argparse,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT,LEGACY,PREVIOUS,ARCHIVE
import numpy as np,pandas as pd,torch
from src.models.coupled_rollout import ArchitectedR2QPHead
from src.models.scenario_recourse import action_value_curves,realized_action_costs,forecast_policy_action_costs
from src.data.solar_nowcast import calendar_index

p=argparse.ArgumentParser();p.add_argument('--epoch',type=int,required=True);p.add_argument('--safety',type=float,default=.9);p.add_argument('--policy-targets',action='store_true');args=p.parse_args()
prefix='action' if args.policy_targets else 'continuation'
torch.set_num_threads(2);torch.backends.cuda.matmul.fp32_precision='ieee';start=time.perf_counter()
source=ROOT/'checkpoints'/f'QPR2_seed20260920_epoch{args.epoch:02d}.pt'
model=ArchitectedR2QPHead().cuda();model.load_state_dict(torch.load(source,map_location='cpu',weights_only=False)['state_dict']);model.eval().requires_grad_(False)
with np.load(ROOT/'cache/rollout_forecast_features.npz') as z:
    evidence={k:z[k] for k in ['rt_0','rt_1','decision_latent_0','decision_latent_1','pue_reference','historical_rt']}
pref=evidence['pue_reference'];historical=evidence['historical_rt']
cons=json.loads((ARCHIVE/'configs/economic_constraints.json').read_text())['constraints'];idle=np.array(cons['idle_power']);power=np.array(cons['work_power_per_unit']);pmax=np.array(cons['pue_max']);capacity=np.minimum(1.,(np.array(cons['grid_import_limit'])/pmax-idle)/power)
with np.load(LEGACY/'cache/development_physical_reference.npz') as z:dates=z['dates'];pv_source=z['pv'];pue_source=z['pue'];arr_source=z['arrivals']
with np.load(PREVIOUS/'cache/world_physical_forecasts.npz') as z:prior_pue=z['issue_pue']
date_lookup={str(d):i for i,d in enumerate(dates)}
with np.load(ROOT/'cache/solar_seasonal_envelope.npz') as z:upper=z['upper']
with np.load(ROOT/'cache/causal_observer.npz') as z:alpha=z['alpha'];residual=z['residual_quantiles']
begin=pd.Timestamp('2022-01-01',tz='UTC');end=pd.Timestamp('2025-01-01',tz='UTC');n=int((end-begin).total_seconds()/900);arrival_grid=np.zeros((n,4))
for d,day in enumerate(dates):
    timestamp=pd.Timestamp(str(day),tz='UTC')
    if begin<=timestamp<end:
        j=int((timestamp-begin).total_seconds()/900);arrival_grid[j:j+96]=arr_source[d]
reference_queue=np.zeros_like(arrival_grid);pending=np.zeros(4)
for t in range(n):
    if t==int((pd.Timestamp('2024-01-01',tz='UTC')-begin).total_seconds()/900):pending[:]=0
    pending=np.maximum(pending-1,0)+arrival_grid[t];reference_queue[t]=pending
weights=np.array([.075,.1,.2,.25,.2,.1,.075]);h=np.arange(96)
for role in ['train','selection']:
    output=ROOT/'cache'/f'{prefix}_qp{args.epoch:02d}_{role}.npz'
    if output.exists():print('CACHED',output.name,flush=True);continue
    with np.load(ROOT/'cache'/f'rollout_{role}.npz') as z:data={k:z[k] for k in z.files}
    if args.policy_targets:
        work=data['claims'].sum(1)[:,:96].astype(float);work[:,95]+=data['arrivals'][:,0]
        feasible=(work.cumsum(1)<=capacity[data['zone'],None]*np.arange(1,97)+1e-7).all(1)
        (ROOT/'configs'/f'action_training_feasibility_{role}.json').write_text(json.dumps(dict(original_states=len(feasible),retained_states=int(feasible.sum()),excluded_infeasible_states=int((~feasible).sum()),criterion='known-work cumulative deadline demand fits available capacity; annual evaluation workloads remain unchanged'),indent=2))
        data={k:v[feasible] for k,v in data.items()}
    origins=pd.DatetimeIndex(pd.to_datetime(data['origins'],utc=True));count=len(origins)
    if role=='train':assert set(origins.year)=={2022,2023}
    else:assert set(origins.year)=={2024}
    states=data['states'][:,0].copy();queues=data['claims'].sum(1)[:,:96].astype(float);queues[:,95]+=data['arrivals'][:,0]
    refs=np.zeros((count,96));actual_refs=np.zeros((count,96));now_pv=np.zeros(count);now_res=np.zeros((count,7));has_previous=np.zeros(count,bool)
    for i,t in enumerate(origins):
        zone=int(data['zone'][i]);q=queues[i];state=states[i]
        state[:,0]=0;state[:,1]=min(.05,q.sum());state[:,5]=np.log1p(data['arrivals'][i,0])/4;state[:,6]=np.log1p(q.sum())/4;state[:,7]=0
        for j,hours in enumerate([4,8,12,24]):state[:,12+j]=np.log1p(q[:hours*4].sum())/4
        prev=t-pd.Timedelta(minutes=15);d=date_lookup.get(prev.strftime('%Y-%m-%d'))
        if d is not None:
            ps=prev.hour*4+prev.minute//15;previous=pv_source[d,ps,zone];previous_pue=pue_source[d,ps,zone];bias=previous_pue-prior_pue[d,ps,zone]
            puehat=np.clip(state[:,3]*pref[zone]+np.exp(-h/32)*bias,1.,pmax[zone]);puehat[0]=np.clip(previous_pue,1.,pmax[zone]);state[:,3]=puehat/pref[zone]
            cs=t.hour*4+t.minute//15;day=calendar_index(pd.DatetimeIndex([t]))[0];previous_day=calendar_index(pd.DatetimeIndex([prev]))[0]
            high=upper[day,cs,zone];previous_high=upper[previous_day,ps,zone]
            now_pv[i]=np.clip(previous+alpha[zone]*(previous*np.clip(high/max(previous_high,.02),0,2)-previous),0,1.5)
            now_res[i]=residual[cs,zone]*max(high,.1);state[0,2]=now_pv[i];has_previous[i]=True
        j=int((t-begin).total_seconds()/900);refs[i]=np.minimum(1.,np.maximum(reference_queue[j,zone]-h,0.))
        actual_refs[i]=np.minimum(1.,reference_queue[j+h,zone])
    packed={key:[] for key in ['features','tokens','pv','prices','opportunity','current_mean']}
    with torch.inference_mode():
        for a in range(0,count,64):
            b=min(count,a+64);ids=data['row_ids'][a:b,:96];state=states[a:b];pub=state[:,:,10,None].astype(bool);available=state[:,:,8,None];zone=data['zone'][a:b]
            market=np.where(pub,evidence['rt_1'][ids],evidence['rt_0'][ids])*available+historical[zone,None,None]*(1-available)
            latent=np.where(pub,evidence['decision_latent_1'][ids],evidence['decision_latent_0'][ids])*available
            out=model(torch.tensor(state,device='cuda'),torch.tensor(latent,device='cuda'),torch.tensor(market,device='cuda'),torch.tensor(pref[zone],device='cuda'))
            for key,value in [('features',out['continuation_features']),('tokens',out['policy_tokens']),('pv',out['pv_quantiles']),('prices',out['conditional_price']),('opportunity',out['opportunity']),('current_mean',out['rt_decision'][:,0])]:packed[key].append(value.cpu().numpy())
    packed={key:np.concatenate(value) for key,value in packed.items()}
    base=np.zeros((count,64),np.float32);adjustment=base.copy();truth=base.copy();actions=base.copy();mask=np.zeros((count,64),bool)
    for i in range(count):
        zone=int(data['zone'][i]);pv=packed['pv'][i].T.astype(float).copy();prices=packed['prices'][i].T.astype(float).copy();pue=states[i,:,3].astype(float)*pref[zone]
        if has_previous[i]:pv[:,0]=np.clip(now_pv[i]+now_res[i],0,1.5);prices[:,0]=packed['current_mean'][i]
        x,a,b=action_value_curves(queues[i],pv,prices,weights,pue,refs[i],idle[zone],power[zone],packed['opportunity'][i].astype(float),float(pref[zone]),capacity[zone],args.safety,True)
        outcomes=data['outcomes'][i,:96].astype(float)
        if args.policy_targets:
            y=forecast_policy_action_costs(x,queues[i],pv,prices,weights,pue,refs[i],packed['opportunity'][i].astype(float),outcomes[:,0],outcomes[:,1],outcomes[:,2],actual_refs[i],idle[zone],power[zone],float(pref[zone]),capacity[zone],args.safety)
            y=y-y[0]+a[0]
        else:y=realized_action_costs(x,queues[i],outcomes[:,0],outcomes[:,1],outcomes[:,2],refs[i],idle[zone],power[zone],float(pref[zone]),capacity[zone],args.safety)
        m=len(x);base[i,:m]=a;adjustment[i,:m]=b;truth[i,:m]=y;actions[i,:m]=x;mask[i,:m]=True
    np.savez_compressed(output,features=packed['features'],tokens=packed['tokens'],base=base,adjustment=adjustment,truth=truth,actions=actions,mask=mask,origins=data['origins'])
    print('CONTINUATION DATA',role,count,'valid actions',int(mask.sum()),'seconds',round(time.perf_counter()-start,1),flush=True)
(ROOT/'configs'/f'{prefix}_data_qp{args.epoch:02d}.json').write_text(json.dumps(dict(forecast_epoch=args.epoch,safety=args.safety,training_years=[2022,2023],selection_year=2024,forecast_parameters_updated=False,features='frozen shared R2 market, physical and workload-query representation; causal observer and known-work state',targets='realized cost of plans selected only from forecast information, with realized reference fee labels' if args.policy_targets else 'realized pre-2024 known-work current cost plus a perfect-information future-completion surrogate; future realizations are labels only',scope='decision-focused value head; the forecast heads stay frozen'),indent=2))
