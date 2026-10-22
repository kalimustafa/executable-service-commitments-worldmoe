"""Fit conditional tail first moments on 2021-2023; select using 2024 cash."""
import sys,json,time,subprocess,hashlib,shutil,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT
from src.models.tail_moment_r2 import TailMomentR2,MomentCompletionR2
from src.models.coupled_rollout import ArchitectedR2QPHead
import torch,numpy as np

p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=16);p.add_argument('--complete',action='store_true');args=p.parse_args()
prefix='MomentR2' if args.complete else 'TailR2';tag='moment' if args.complete else 'tail'
seed=20260920;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);torch.set_num_threads(3)
source=ROOT/'checkpoints/QPR2_seed20260920_epoch10.pt';source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
core=ArchitectedR2QPHead().cuda();core.load_state_dict(torch.load(source,map_location='cpu',weights_only=False)['state_dict']);core.eval().requires_grad_(False)
with np.load(ROOT/'cache/rollout_forecast_features.npz') as z:
    n=len(z['zone']);zone=np.tile(z['zone'],2);gate=np.repeat(np.arange(2),n);truth=np.tile(z['labels'][:,2],2)
    train_idx=np.r_[z['train_indices'],z['train_indices']+n];selection_idx=np.r_[z['selection_indices'],z['selection_indices']+n]
    assert set(str(d)[:4] for d in z['dates'][z['train_indices']])=={'2021','2022','2023'}
    adapted=[];qs=[]
    with torch.no_grad():
        for g in [0,1]:
            latent=torch.tensor(z[f'decision_latent_{g}'],device='cuda');a,shift=core.market_update(latent)
            adapted.append(a);qs.append(torch.tensor(z[f'rt_{g}'],device='cuda')+shift[:,None])
adapted=torch.cat(adapted);q=torch.cat(qs);zone=torch.tensor(zone,device='cuda');gate=torch.tensor(gate,device='cuda');truth=torch.tensor(truth,device='cuda')
train_idx=torch.tensor(train_idx,device='cuda');selection_idx=torch.tensor(selection_idx,device='cuda')
target=torch.stack([(truth-q[:,-1]).clamp_min(0),(q[:,0]-truth).clamp_min(0)],-1)
base=torch.zeros(2,4,2,device='cuda')
for g in [0,1]:
    for site in range(4):base[g,site]=target[train_idx[(gate[train_idx]==g)&(zone[train_idx]==site)]].mean(0).clamp_min(.02)
normalizer=target[train_idx].mean(0).clamp_min(.1);model=(MomentCompletionR2 if args.complete else TailMomentR2)(base.cpu()).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=2e-3,fused=True)
weights=torch.tensor([.075,.1,.2,.25,.2,.1,.075],device='cuda');mean=(q*weights).sum(-1);original_rmse=float((mean[selection_idx]-truth[selection_idx]).square().mean().sqrt())
clipped=torch.maximum(q[:,0],torch.minimum(truth,q[:,-1]));central_scale=(clipped[train_idx]-mean[train_idx]).square().mean().sqrt().clamp_min(1.)
del core
def loss_for(ids):
    out=model(adapted[ids],q[ids],zone[ids],gate[ids]);mu=out['excess']/normalizer;y=target[ids]/normalizer
    loss=(mu-y*mu.clamp_min(1e-7).log()).mean()+.001*out['log_ratio'].square().mean()
    if args.complete:loss=loss+((out['clipped_mean']-clipped[ids])/central_scale).square().mean()
    return loss,out
@torch.no_grad()
def validate():
    model.eval();parts=[];pred=[]
    for ids in selection_idx.split(4096):
        loss,out=loss_for(ids);parts.append(loss*len(ids));pred.append(mean[ids]+out['mean_correction'])
    pred=torch.cat(pred);err=pred-truth[selection_idx]
    return dict(tail_score=float(torch.stack(parts).sum()/len(selection_idx)),mean_rmse=float(err.square().mean().sqrt()),mean_bias=float(err.mean()))
def save(epoch,metrics):
    path=ROOT/'checkpoints'/f'{prefix}_seed{seed}_epoch{epoch:02d}.pt'
    torch.save(dict(state_dict=model.state_dict(),epoch=epoch,seed=seed,architecture=type(model).__name__,forecast_epoch=10,forecast_checkpoint_sha256=source_hash,training_years=[2021,2022,2023],selection_year=2024,quantile_heads_frozen=True,metrics=metrics),path);return path
initial=validate();save(0,initial);baseline=json.loads((ROOT/'results/RecourseR2_Eagle_2024_s20260920_eta0_k0.9_ep10.json').read_text());best=baseline['saving_usd'];anchor=best;best_epoch=None;stale=0;logs=[]
(ROOT/'configs'/f'{tag}_moment_protocol.json').write_text(json.dumps(dict(training_years=[2021,2022,2023],selection_year=2024,seven_quantiles_unchanged=True,baseline_selection_rmse=original_rmse,base_excess=base.cpu().tolist(),identity='mean = clipped central expectation + expected upper excess - expected lower excess',central_completion=args.complete,central_constraint='left/right quantile Riemann sums',objective='quasi-Poisson loss for nonnegative conditional tail means; squared loss for the central clipped mean when enabled',forecast_core_sha256=source_hash),indent=2))
print('TAIL INITIAL',json.dumps(dict(**initial,original_rmse=original_rmse,parameters=sum(p.numel() for p in model.parameters()))),flush=True)
for epoch in range(1,args.epochs+1):
    start=time.perf_counter();model.train();total=[]
    for ids in train_idx[torch.randperm(len(train_idx),device='cuda')].split(4096):
        optimizer.zero_grad(set_to_none=True);loss,_=loss_for(ids);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite tail-moment gradient')
        optimizer.step();total.append(loss.detach())
    met=validate();save(epoch,met);record=dict(epoch=epoch,train_loss=float(torch.stack(total).mean()),seconds=time.perf_counter()-start,**met)
    if epoch%2==0:
        subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/evaluate_scenario_recourse.py'),'--eta','0','--safety','0.9','--epoch','10',f'--{tag}-epoch',str(epoch)],check=True,cwd=ROOT)
        row=json.loads((ROOT/'results'/f'{prefix}_Eagle_2024_s{seed}_ep{epoch:02d}.json').read_text());accepted=row['late_IT_mwh']<1e-8 and row['closing_it_mwh']<1e-8 and met['mean_rmse']<=1.02*original_rmse
        if accepted and row['saving_usd']>best:best=row['saving_usd'];best_epoch=epoch
        if accepted and row['saving_usd']>anchor+50:anchor=row['saving_usd'];stale=0
        else:stale+=1
        record.update(saving_usd=row['saving_usd'],capture=row['capture_lower_fraction'],accepted=accepted,stale=stale)
    logs.append(record);(ROOT/'logs'/f'{tag}_training.json').write_text(json.dumps(logs,indent=2));print(prefix,'EPOCH',json.dumps(record),flush=True)
    if epoch>=6 and stale>=3:break
result=dict(seed=seed,best_epoch=best_epoch,epochs_completed=epoch,best_saving_usd=best,baseline_saving_usd=baseline['saving_usd'],improvement_usd=best-baseline['saving_usd'],training_years=[2021,2022,2023],selection_year=2024,other_seeds_started=False,quantile_heads_frozen=True,early_stopping='2024 annual savings; every two epochs; patience three; min_delta 50 USD; auxiliary mean RMSE guard 2 percent')
if best_epoch is not None:
    path=ROOT/'checkpoints'/f'{prefix}_seed{seed}_epoch{best_epoch:02d}.pt';shutil.copy2(path,ROOT/'checkpoints'/f'{prefix}_seed{seed}_best.pt');result['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
(ROOT/'results'/f'{tag}_primary_complete.json').write_text(json.dumps(result,indent=2));print(prefix,'COMPLETE',json.dumps(result),flush=True)
