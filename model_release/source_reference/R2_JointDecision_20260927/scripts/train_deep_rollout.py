"""Train the redesigned R2 architecture through sequential, conserved claims."""
import sys,json,time,random,csv,subprocess,shutil,argparse,hashlib,atexit,msvcrt
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT
import torch,numpy as np
import pandas as pd
import torch.nn.functional as F
from src.models.coupled_rollout import ArchitectedR2Head,ArchitectedR2QPHead
from src.models.structured_decision import structured_allocation
from src.models.feasible_qp import qp_allocation
from src.models.distributional_r2 import DistributionalR2Head
from src.models.continuous_decision import ContinuousR2Head
from src.models.affine_continuous import AffineContinuousR2Head
from src.data.solar_nowcast import calendar_index

p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=18);p.add_argument('--batch',type=int,default=32);p.add_argument('--qp',type=float,default=0.);p.add_argument('--distribution',action='store_true');p.add_argument('--continuous',action='store_true');p.add_argument('--cash-refine',action='store_true');p.add_argument('--seed',type=int,default=20260920);args=p.parse_args()
if args.cash_refine:args.continuous=True
assert not (args.distribution and args.continuous)
if (args.distribution or args.continuous) and not args.qp:args.qp=2.
prefix='CashR2' if args.cash_refine else ('CdfR2' if args.continuous else ('DistR2' if args.distribution else ('QPR2' if args.qp else 'DeepR2')));log_label='cash' if args.cash_refine else ('cdf' if args.continuous else ('distribution' if args.distribution else ('qp' if args.qp else 'deep')));feature_dim=23 if args.distribution else 22
seed=args.seed
if seed!=20260920:
    authorization=json.loads((ROOT/'results/replication_authorization.json').read_text())
    assert authorization['authorized'] and seed in authorization['seeds'] and prefix==authorization['selected_family']
torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed);torch.set_num_threads(3)
torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.fp32_precision='tf32'
with np.load(ROOT/'cache/rollout_forecast_features.npz') as z:
    evidence={k:torch.as_tensor(z[k],device='cuda') for k in z.files if k not in ['dates','timestamps']}
    assert set(np.array([d[:4] for d in z['dates'][z['train_indices']]]))=={'2021','2022','2023'}
def load(role):
    with np.load(ROOT/'cache'/f'rollout_{role}.npz') as z:
        keep=np.ones(len(z['zone']),bool)
        if role=='train':
            initial=z['claims'].sum(-1);common_free=np.maximum(1-initial[:,15:96],0).sum(1)
            keep=z['arrivals'].sum(1)<=common_free-1e-4
            if seed==20260920:(ROOT/'configs/deep_training_feasibility.json').write_text(json.dumps(dict(original_episodes=len(keep),retained_episodes=int(keep.sum()),excluded=int((~keep).sum()),criterion='Total new work fits the free intersection of all 16 service windows; a sufficient guarantee under any within-window placement',evaluation_workloads_changed=False,prototype_epoch1_invalid=True),indent=2))
        arrays={k:z[k][keep] for k in z.files if k!='origins'}
        if args.distribution:
            origins=pd.DatetimeIndex(pd.to_datetime(z['origins'][keep],utc=True));start=pd.Timestamp('2022-01-01',tz='UTC')
            times=pd.date_range(start,pd.Timestamp('2025-01-02',tz='UTC'),freq='15min',inclusive='left');day=calendar_index(times);slot=times.hour.to_numpy()*4+times.minute.to_numpy()//15
            with np.load(ROOT/'cache/solar_seasonal_envelope.npz') as envelope:upper=envelope['upper'][day,slot]
            origins_idx=((origins.asi8-start.value)//(900*10**9)).astype(np.int64)
            offsets=np.arange(16)[:,None]+np.arange(96)[None,:]
            value=upper[origins_idx[:,None,None]+offsets[None],arrays['zone'][:,None,None]]
            arrays['states']=np.concatenate([arrays['states'],value[:,:,:,None]],-1).astype(np.float32)
        return {k:torch.as_tensor(v,device='cuda') for k,v in arrays.items()}
train=load('train');validation=load('selection');K=16;L=111
old=torch.load(ROOT/'checkpoints/StructuredR2_seed20260920_best.pt',map_location='cpu',weights_only=False)
initial_state={k.removeprefix('structured.'):v for k,v in old['state_dict'].items() if k.startswith('structured.')}
model=(AffineContinuousR2Head() if args.cash_refine else (ContinuousR2Head() if args.continuous else (DistributionalR2Head() if args.distribution else (ArchitectedR2QPHead if args.qp else ArchitectedR2Head)(initial_state)))).cuda()
cash_parent=None
if args.cash_refine:
    complete=json.loads((ROOT/'results/cdf_primary_complete.json').read_text());assert complete['status']=='complete' and complete['best_restored']
    cash_parent=ROOT/'checkpoints/CdfR2_seed20260920_best.pt'
    model.load_parent(torch.load(cash_parent,map_location='cpu',weights_only=False)['state_dict'])
elif args.continuous:
    model.load_state_dict(torch.load(ROOT/'checkpoints/QPR2_seed20260920_epoch10.pt',map_location='cpu',weights_only=False)['state_dict'])
elif args.distribution:
    model.load_pretrained(torch.load(ROOT/'checkpoints/QPR2_seed20260920_best.pt',map_location='cpu',weights_only=False)['state_dict'])
elif args.qp:
    model.load_state_dict(torch.load(ROOT/'checkpoints/DeepR2_seed20260920_best.pt',map_location='cpu',weights_only=False)['state_dict'])
if args.distribution:
    new=[];old_params=[]
    for name,value in model.named_parameters():(new if name.startswith(('pv_distribution.','quantile_attention.','distribution_embedding.')) else old_params).append(value)
    optimizer=torch.optim.AdamW([dict(params=old_params,lr=1e-4),dict(params=new,lr=5e-4)],weight_decay=1e-4,fused=True)
else:optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4,fused=True)
economic_normalizer=float((train['arrivals'].sum(1)*1.639059066772461*evidence['pue_reference'][train['zone']]*.25*50).mean())
if args.distribution:(ROOT/'configs/distribution_objective.json').write_text(json.dumps(dict(seed=seed,work_conservation=True,economic_objective='episode aggregate US-dollar cost difference divided by one training-only constant; no per-work-unit normalization',normalizer_usd=economic_normalizer,pv_objective='discrete-distribution CRPS',price_valuation_weights='task-specific valuation, not advertised forecast probabilities',pv_distribution_queue_independent=True),indent=2))
if args.cash_refine:
    protocol_name='cash_refinement_protocol.json' if seed==20260920 else f'cash_refinement_protocol_s{seed}.json'
    (ROOT/'configs'/protocol_name).write_text(json.dumps(dict(seed=seed,batch=args.batch,work_conservation=True,economic_objective='aggregate episode USD difference divided by a single training-only constant',normalizer_usd=economic_normalizer,market_scale='exp(0.2*tanh(head(market-only R2 representation))); positive and initialized to one',apic_pre='DA_new=s*DA; DART_new=s*DART-shift; RT_new=s*RT+shift',apic_post='DA_anchor unchanged; DART_new=s*DART+(1-s)*DA_anchor-shift; RT_new=s*RT+shift',parent_checkpoint=str(cash_parent),parent_sha256=hashlib.sha256(cash_parent.read_bytes()).hexdigest(),training_years=[2021,2022,2023],physical_training_years=[2022,2023],selection_year=2024,optimizer_reinitialized_for_new_objective=True),indent=2))
global_price_reference=torch.load(ROOT/'checkpoints/StructuredR2_seed20260920_epoch00.pt',map_location='cpu',weights_only=False)['metrics']['price_score']
tau=torch.tensor([.05,.1,.25,.5,.75,.9,.95],device='cuda');started=time.perf_counter();logs=[]
def pin(q,y):
    e=y[:,None]-q;return torch.maximum(tau*e,(tau-1)*e).mean(-1)
def price_objective(ids,gate):
    latent=torch.where(gate[:,None].bool(),evidence['decision_latent_1'][ids],evidence['decision_latent_0'][ids])
    adapted,shift=model.market_update(latent)
    q={key:torch.where(gate[:,None].bool(),evidence[f'{key}_1'][ids],evidence[f'{key}_0'][ids]) for key in ['da','dart','rt']}
    y=evidence['labels'][ids];scale=evidence['target_scale'][evidence['zone'][ids]]
    da=q['da'];dart=q['dart']-shift[:,None];rt=q['rt']+shift[:,None];anchor=shift.square().mean()/100
    if args.cash_refine:
        factor=model.market_scale(adapted)
        da=torch.where(gate[:,None].bool(),q['da'],q['da']*factor[:,None])
        dart=q['dart']*factor[:,None]+(gate*(1-factor)*y[:,0]-shift)[:,None]
        rt=q['rt']*factor[:,None]+shift[:,None]
        anchor=anchor+5*factor.log().square().mean()
    loss=.5*pin(da,y[:,0])/scale[:,0]+pin(dart,y[:,1])/scale[:,1]+.5*pin(rt,y[:,2])/scale[:,2]
    return loss.mean(),anchor

def rollout(group,indices,training):
    fixed=group['states'][indices];row_ids=group['row_ids'][indices];claims=group['claims'][indices];arrivals=group['arrivals'][indices];actual=group['outcomes'][indices];zone=group['zone'][indices]
    ref=claims.sum(-1).detach().clone();pv_terms=[];cost_terms=[];penalties=[];feasibility=[]
    offsets=torch.arange(K,device='cuda')[:,None]+torch.arange(96,device='cuda')[None,:]
    all_ids=row_ids[:,offsets];published=fixed[:,:,:,10,None].bool();available=fixed[:,:,:,8,None]
    market=torch.where(published,evidence['rt_1'][all_ids],evidence['rt_0'][all_ids])*available+evidence['historical_rt'][zone,None,None,None]*(1-available)
    latent=torch.where(published,evidence['decision_latent_1'][all_ids],evidence['decision_latent_0'][all_ids])*available
    encoded=model.encode_evidence(fixed.reshape(-1,96,feature_dim),latent.reshape(-1,96,128),market.reshape(-1,96,7))
    encoded=[x.reshape(len(indices),K,*x.shape[1:]) for x in encoded]
    for step in range(K):
        amount=arrivals[:,step];ids=row_ids[:,step:step+96];state=fixed[:,step].clone()
        book=claims.sum(-1)[:,step:step+96];pending=claims[:,step:,:].sum(1)
        state[:,:,0]=book;state[:,:,1]=torch.minimum((1-book).clamp_min(1e-5),torch.minimum(amount,torch.ones_like(amount)*.05)[:,None]).clamp_min(1e-5)
        state[:,:,5]=torch.log1p(amount)[:,None]/4;state[:,:,6]=torch.log1p(pending.sum(1)+amount)[:,None]/4;state[:,:,7]=book.mean(1,keepdim=True)
        for j,hours in enumerate([4,8,12,24]):
            due=pending[:,:min(L,step+hours*4)].sum(1)+(amount if hours==24 else 0)
            state[:,:,12+j]=torch.log1p(due)[:,None]/4
        out=model.decode_evidence(state,[x[:,step] for x in encoded],evidence['pue_reference'][zone])
        allocated=qp_allocation(out['segment_widths'],out['segment_scores'],amount,args.qp) if args.qp else structured_allocation(out['segment_widths'],out['segment_scores'],amount,straight_through=training)
        feasibility.append((allocated.sum(1)-amount).abs().max())
        addition=F.pad(allocated,(step,L-step-96));deadline=F.one_hot(torch.full((len(indices),),step+95,device='cuda',dtype=torch.long),L).float()
        claims=claims+addition[:,:,None]*deadline[:,None,:]
        with torch.no_grad():
            capacity=(1-ref[:,step:step+96]).clamp_min(0);take=torch.minimum(capacity,(amount[:,None]-(capacity.cumsum(1)-capacity)).clamp_min(0));ref=ref+F.pad(take,(step,L-step-96))
        truth=actual[:,step:step+96];pv=truth[:,:,1]
        if args.distribution:
            support=out['pv_support'];probability=out['pv_probabilities']
            first=(probability*(support-pv[:,:,None]).abs()).sum(-1)
            pair=((support[:,:,:,None]-support[:,:,None,:]).abs()*probability[:,:,:,None]*probability[:,:,None,:]).sum((-1,-2))
            pv_terms.append((first-.5*pair).mean())
        else:
            error=pv[:,:,None]-out['pv_quantiles'];pv_terms.append(torch.maximum(tau*error,(tau-1)*error).mean())
        delta=state[:,:,1];widths=out['segment_widths'];takes=torch.minimum(widths,(delta[:,:,None]-(widths.cumsum(-1)-widths)).clamp_min(0))
        predicted=(takes*out['physical_segment_cost']).sum(-1)/delta
        inc=truth[:,:,2]*1.639059066772461*delta;load=truth[:,:,2]*(.23069976270198822+1.639059066772461*book)
        coverage=((pv-load)/inc.clamp_min(1e-8)).clamp(0,1)
        target=truth[:,:,2]/evidence['pue_reference'][zone,None]*((1-coverage)*truth[:,:,0]-coverage)
        valid=(1-book)>1e-5;cost_terms.append((F.smooth_l1_loss(predicted/50,target/50,reduction='none')*valid).sum()/valid.sum().clamp_min(1))
        penalty=.002*out['opportunity'].square().mean()+.005*out['cross_moment_rho'].square().mean()
        if args.distribution:penalty=penalty+.01*out['valuation_kl'].mean()
        penalties.append(penalty)
    policy=claims.sum(-1);rt,pv,pue=actual[:,:,0],actual[:,:,1],actual[:,:,2]
    load=pue*(.23069976270198822+1.639059066772461*policy);refload=pue*(.23069976270198822+1.639059066772461*ref)
    used=torch.minimum(pv,load);refused=torch.minimum(pv,refload);grid=load-used;refgrid=refload-refused
    difference=((rt*(grid-refgrid)+(refused-used)+(grid-refgrid).abs()).sum(1)*.25)
    scale=(arrivals.sum(1)*1.639059066772461*evidence['pue_reference'][zone]*.25*50).clamp_min(1e-5)
    objective=difference.mean()/economic_normalizer if (args.distribution or args.cash_refine) else (difference/scale).mean()
    return objective,torch.stack(pv_terms).mean(),torch.stack(cost_terms).mean(),torch.stack(penalties).mean(),torch.stack(feasibility).max()

@torch.no_grad()
def validate(epoch):
    model.eval();price=[]
    for gate in [0,1]:
        for ids in evidence['selection_indices'].split(4096):
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=price_objective(ids,torch.full((len(ids),),float(gate),device='cuda'))
            price.append(loss*len(ids))
    decision=[];pv=[];error=0.
    for ids in torch.arange(len(validation['zone']),device='cuda').split(16):
        with torch.autocast('cuda',dtype=torch.bfloat16):value,pl,_,_,err=rollout(validation,ids,False)
        decision.append(value*len(ids));pv.append(pl*len(ids));error=max(error,float(err))
    result=dict(epoch=epoch,price_score=float(torch.stack(price).sum()/(2*len(evidence['selection_indices']))),rollout_cost=float(torch.stack(decision).sum()/len(validation['zone'])),allocation_mass_error=error)
    result['pv_crps' if args.distribution else 'pv_pinball']=float(torch.stack(pv).sum()/len(validation['zone']))
    return result

def save(epoch,metrics):
    path=ROOT/'checkpoints'/f'{prefix}_seed{seed}_epoch{epoch:02d}.pt'
    if epoch==0 and path.exists():
        previous=torch.load(path,map_location='cpu',weights_only=False)['state_dict']
        if all(torch.equal(v.cpu(),previous[k]) for k,v in model.state_dict().items()):return path
    torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),metrics=metrics,epoch=epoch,seed=seed,architecture=type(model).__name__,qp_regularization=args.qp,original_market_source='StructuredR2_seed20260920_epoch10',training_years=[2021,2022,2023],physical_training_years=[2022,2023],rollout_steps=K),path);return path
class TrainingGpuSlot:
    """Avoid WDDM context contention; annual CPU replay runs outside this slot."""
    def __init__(self):
        self.file=(ROOT/'logs/cdf_gpu_training.lock').open('a+b');self.held=False
        self.file.seek(0,2)
        if self.file.tell()==0:self.file.write(b'0');self.file.flush()
        atexit.register(self.release)
    def acquire(self):
        while not self.held:
            self.file.seek(0)
            try:msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1);self.held=True
            except OSError:time.sleep(.25)
    def release(self):
        if self.held:
            self.file.seek(0);msvcrt.locking(self.file.fileno(),msvcrt.LK_UNLCK,1);self.held=False

gpu_slot=TrainingGpuSlot()

def _economics(epoch):
    if args.continuous:
        target=ROOT/'results'/f'{prefix}_Eagle_2024_s{seed}_ep{epoch:02d}_coupled_cdf.json'
        probe=ROOT/'results/RecourseR2_Eagle_2024_s20260920_eta0_k0.9_ep10_coupled_cdf.json'
        if args.cash_refine:
            parent_epoch=json.loads((ROOT/'results/cdf_primary_complete.json').read_text())['best_epoch']
            probe=ROOT/'results'/f'CdfR2_Eagle_2024_s20260920_ep{parent_epoch:02d}_coupled_cdf.json'
        if epoch==0 and probe.exists():
            old_state=torch.load(cash_parent if args.cash_refine else ROOT/'checkpoints/QPR2_seed20260920_epoch10.pt',map_location='cpu',weights_only=False)['state_dict']
            assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in old_state.items())
            if args.cash_refine:assert not model.price_scale.weight.detach().count_nonzero() and not model.price_scale.bias.detach().count_nonzero()
            row=json.loads(probe.read_text());row.update(name=target.stem,seed=seed,baseline_reused_from=probe.name,baseline_identity='identical pretrained initialization and deterministic execution; no trained seed-dependent change at epoch zero',forecast_epoch=0,cdf_training_epoch=0,baseline_forecast_checkpoint_sha256=row['forecast_checkpoint_sha256'],forecast_checkpoint_sha256=hashlib.sha256((ROOT/'checkpoints'/f'{prefix}_seed{seed}_epoch00.pt').read_bytes()).hexdigest())
            target.write_text(json.dumps(row,indent=2));return row
        subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/evaluate_scenario_recourse.py'),'--eta','0','--safety','0.9','--cash-epoch' if args.cash_refine else '--cdf-epoch',str(epoch),'--current-coupling','--continuous-cost','--seed',str(seed)],check=True,cwd=ROOT)
        return json.loads(target.read_text())
    command=[sys.executable,'-B','-u',str(ROOT/'scripts/replay_development.py'),'--deep','--structured','--epochs',str(epoch),'--solar-hours','6']
    if args.qp:command+=['--qp',str(args.qp)]
    if args.distribution:command+=['--distribution']
    subprocess.run(command,check=True,cwd=ROOT)
    return json.loads((ROOT/'results'/f'{prefix}_Eagle_2024_s{seed}_ep{epoch:02d}.json').read_text())

def economics(epoch):
    torch.cuda.synchronize();torch.cuda.empty_cache();gpu_slot.release()
    try:return _economics(epoch)
    finally:gpu_slot.acquire()

gpu_slot.acquire()
initial=validate(0);save(0,initial);row=economics(0);best=row['saving_usd'];best_epoch=0;stale=0;stop_anchor=best
print('DEEP INITIAL',json.dumps(dict(**initial,saving_usd=best,parameters=sum(p.numel() for p in model.parameters()))),flush=True)
for epoch in range(1,args.epochs+1):
    model.train();epoch_started=time.perf_counter();order=torch.randperm(len(train['zone']),device='cuda');totals=[]
    for step,indices in enumerate(order.split(args.batch)):
        optimizer.zero_grad(set_to_none=True)
        ids=evidence['train_indices'][torch.randint(0,len(evidence['train_indices']),(2048,),device='cuda')];gate=torch.randint(0,2,(len(ids),),device='cuda').float()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            price,anchor=price_objective(ids,gate);value,pv,cost,penalty,mass=rollout(train,indices,True)
            loss=price+value+2*pv+.2*cost+penalty+.1*anchor
        if step%16==0:
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite deep rollout objective')
            if mass>2e-5:raise RuntimeError('Deep training allocation dropped work')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True)
        if step%16==0 and not torch.isfinite(norm):raise RuntimeError('Nonfinite deep rollout gradient')
        optimizer.step();totals.append(torch.stack([price.detach(),value.detach(),pv.detach(),cost.detach(),mass.detach()]))
        if step%25==0:print('DEEP TRAIN STEP',epoch,step,'elapsed',round(time.perf_counter()-epoch_started,1),flush=True)
    means=torch.stack(totals).mean(0).cpu().tolist();metrics=validate(epoch)
    metrics.update(price_train=means[0],rollout_train=means[1],pv_train=means[2],marginal_cost_train=means[3],training_mass_error=means[4],train_validate_seconds=time.perf_counter()-epoch_started,economic_saving_usd=None,best_epoch=best_epoch,stale=stale)
    save(epoch,metrics)
    if epoch%2==0:
        row=economics(epoch);feasible=row['late_IT_mwh']<1e-8 and row['closing_it_mwh']<1e-8 and metrics['price_score']<=global_price_reference*1.02
        if feasible and row['saving_usd']>best:best=row['saving_usd'];best_epoch=epoch
        if feasible and row['saving_usd']>stop_anchor+50:stop_anchor=row['saving_usd'];stale=0
        else:stale+=1
        metrics.update(economic_saving_usd=row['saving_usd'],best_epoch=best_epoch,stale=stale)
    logs.append(metrics)
    with (ROOT/'logs'/f'{log_label}_training_seed{seed}.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(logs[0]));writer.writeheader();writer.writerows(logs)
    print('DEEP EPOCH',json.dumps(metrics),flush=True)
    if epoch>=6 and stale>=3:break
best_path=ROOT/'checkpoints'/f'{prefix}_seed{seed}_epoch{best_epoch:02d}.pt';model.load_state_dict(torch.load(best_path,map_location='cpu',weights_only=False)['state_dict']);shutil.copyfile(best_path,ROOT/'checkpoints'/f'{prefix}_seed{seed}_best.pt')
bound=json.loads((ROOT.parent/'ServiceCommitment_1to5_20260927/results/I1_Eagle_2024_w24.json').read_text())['maximum_saving_upper_usd']
result=dict(status='complete',seed=seed,epochs=epoch,best_epoch=best_epoch,best_restored=True,early_stopped=epoch<args.epochs,best_2024_savings_usd=best,capture_fraction=best/bound,target_80_usd=.8*bound,target_reached=best>=.8*bound,only_primary_after_gate_correction=seed==20260920,replication_authorized=seed!=20260920,shared_initialization='QPR2_seed20260920_epoch10.pt' if args.continuous and not args.cash_refine else None,qp_regularization=args.qp,global_prediction_score_reference=global_price_reference,seconds=time.perf_counter()-started)
summary_name=f'{log_label}_primary_complete.json' if seed==20260920 else f'{log_label}_complete_s{seed}.json'
(ROOT/'results'/summary_name).write_text(json.dumps(result,indent=2));print(prefix,'COMPLETE',json.dumps(result),flush=True)
gpu_slot.release()
