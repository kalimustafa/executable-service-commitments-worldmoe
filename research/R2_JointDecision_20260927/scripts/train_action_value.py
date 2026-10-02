"""Train attention-based action values from realized costs of forecast-only plans."""
import sys,json,time,subprocess,hashlib,shutil,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT
from src.models.action_value_r2 import ActionValueDecoder
import numpy as np,torch
import torch.nn.functional as F

p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=30);args=p.parse_args()
seed=20260920;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);torch.set_num_threads(2)
def load(role):
    with np.load(ROOT/'cache'/f'action_qp10_{role}.npz') as z:
        if role=='train':assert all(str(t)[:4] in ['2022','2023'] for t in z['origins'])
        return {k:torch.as_tensor(z[k],device='cuda') for k in ['features','tokens','actions','base','adjustment','truth','mask']}
train=load('train');selection=load('selection')
truth=train['truth'].masked_fill(~train['mask'],float('inf'));regret=train['truth']-truth.amin(1)[:,None]
scale=max(1.,float(torch.quantile(regret[train['mask']],.75)))
errors=train['truth']-train['base'];errors=errors-errors[:,:1]
bound=max(3.,min(30.,float(torch.quantile(errors[train['mask']].abs(),.9))))
model=ActionValueDecoder(scale,bound).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=2e-3,fused=True)
keys=['features','tokens','actions','base','adjustment']
def objective(data,ids):
    predicted,correction=model(*[data[k][ids] for k in keys]);mask=data['mask'][ids];actual=data['truth'][ids]
    probability=torch.softmax((-predicted/scale).masked_fill(~mask,-float('inf')),1)
    actual=actual-actual.masked_fill(~mask,float('inf')).amin(1)[:,None]
    loss=(probability*actual/scale).sum(1).mean()
    target=(data['truth'][ids]-data['truth'][ids,:1])/scale
    relative=(predicted-predicted[:,:1])/scale
    loss=loss+.1*F.smooth_l1_loss(relative[mask],target[mask])+.002*(correction[mask]/scale).square().mean()
    return loss,correction
source=ROOT/'checkpoints/QPR2_seed20260920_epoch10.pt';source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
def save(epoch):
    path=ROOT/'checkpoints'/f'ActionR2_seed{seed}_epoch{epoch:02d}.pt'
    torch.save(dict(state_dict=model.state_dict(),epoch=epoch,seed=seed,architecture='ActionValueDecoder',dollar_scale=scale,bound=bound,forecast_epoch=10,forecast_checkpoint_sha256=source_hash,training_years=[2022,2023],selection_year=2024,targets='realized cost of forecast-selected future plans; no hindsight future allocation',price_core_frozen=True),path);return path
save(0);baseline=json.loads((ROOT/'results/RecourseR2_Eagle_2024_s20260920_eta0_k0.9_ep10.json').read_text());best=baseline['saving_usd'];anchor=best;best_epoch=0;stale=0;logs=[]
with torch.no_grad():
    _,correction=model(*[train[k][:4] for k in keys]);assert correction.abs().max()==0
print('ACTION INITIAL',json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()),train_states=len(train['features']),dollar_scale=scale,bound=bound,baseline_saving=best)),flush=True)
for epoch in range(1,args.epochs+1):
    start=time.perf_counter();model.train();totals=[]
    for ids in torch.randperm(len(train['features']),device='cuda').split(256):
        optimizer.zero_grad(set_to_none=True);loss,_=objective(train,ids);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite action-value gradient')
        optimizer.step();totals.append(loss.detach())
    model.eval()
    with torch.no_grad():validation,correction=objective(selection,torch.arange(len(selection['features']),device='cuda'))
    record=dict(epoch=epoch,train_loss=float(torch.stack(totals).mean()),selection_loss=float(validation),selection_mean_correction=float(correction[selection['mask']].mean()),seconds=time.perf_counter()-start)
    save(epoch)
    if epoch%3==0:
        subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/evaluate_scenario_recourse.py'),'--eta','0','--safety','0.9','--epoch','10','--action-epoch',str(epoch)],check=True,cwd=ROOT)
        row=json.loads((ROOT/'results'/f'ActionR2_Eagle_2024_s{seed}_ep{epoch:02d}.json').read_text());feasible=row['late_IT_mwh']<1e-8 and row['closing_it_mwh']<1e-8
        if feasible and row['saving_usd']>best:best=row['saving_usd'];best_epoch=epoch
        if feasible and row['saving_usd']>anchor+50:anchor=row['saving_usd'];stale=0
        else:stale+=1
        record.update(saving_usd=row['saving_usd'],capture=row['capture_lower_fraction'],stale=stale)
    logs.append(record);(ROOT/'logs/action_value_training.json').write_text(json.dumps(logs,indent=2));print('ACTION EPOCH',json.dumps(record),flush=True)
    if epoch>=9 and stale>=3:break
best_path=ROOT/'checkpoints'/f'ActionR2_seed{seed}_epoch{best_epoch:02d}.pt';shutil.copy2(best_path,ROOT/'checkpoints'/f'ActionR2_seed{seed}_best.pt')
result=dict(seed=seed,best_epoch=best_epoch,epochs_completed=epoch,best_saving_usd=best,baseline_saving_usd=baseline['saving_usd'],improvement_usd=best-baseline['saving_usd'],checkpoint_sha256=hashlib.sha256(best_path.read_bytes()).hexdigest(),early_stopping='actual 2024 economics, every three epochs, patience three, min_delta 50 USD',training_years=[2022,2023],selection_year=2024,other_seeds_started=False,price_core_frozen=True)
(ROOT/'results/action_primary_complete.json').write_text(json.dumps(result,indent=2));print('ACTION COMPLETE',json.dumps(result),flush=True)
