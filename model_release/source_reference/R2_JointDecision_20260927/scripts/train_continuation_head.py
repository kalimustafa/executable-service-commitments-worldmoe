"""Fit the R2 continuation head using a decision-regret surrogate and cash stopping."""
import sys,json,argparse,csv,time,subprocess,shutil,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT
import numpy as np,torch
from src.models.continuation_r2 import ContinuationR2Head

p=argparse.ArgumentParser();p.add_argument('--forecast-epoch',type=int,required=True);p.add_argument('--eta',type=float,default=0.);p.add_argument('--safety',type=float,default=.9);p.add_argument('--epochs',type=int,default=60);args=p.parse_args()
torch.manual_seed(20260920);np.random.seed(20260920);torch.set_num_threads(2)
source=ROOT/'checkpoints'/f'QPR2_seed20260920_epoch{args.forecast_epoch:02d}.pt'
model=ContinuationR2Head(args.eta);model.load_forecast(torch.load(source,map_location='cpu',weights_only=False)['state_dict']);model.requires_grad_(False);model.continuation.requires_grad_(True)
optimizer=torch.optim.AdamW(model.continuation.parameters(),lr=3e-4,weight_decay=1e-3)
def load(role):
    with np.load(ROOT/'cache'/f'continuation_qp{args.forecast_epoch:02d}_{role}.npz') as z:
        if role=='train':assert all(str(t)[:4] in ['2022','2023'] for t in z['origins'])
        return {k:torch.tensor(z[k]) for k in ['features','base','adjustment','truth','mask']}
train=load('train');validation=load('selection')
truth=train['truth'].masked_fill(~train['mask'],float('inf'));minimum=truth.amin(1)
regrets=train['truth']-minimum[:,None];scale=max(1.,float(torch.quantile(regrets[train['mask']],.75)))
def objective(data,ids):
    eta=model.continuation_adjustment(data['features'][ids]);mask=data['mask'][ids]
    predicted=(data['base'][ids]+eta[:,None]*data['adjustment'][ids])/scale;actual=data['truth'][ids]/scale
    oracle=actual.masked_fill(~mask,float('inf')).argmin(1)
    worst=(actual-2*predicted).masked_fill(~mask,-float('inf')).amax(1)
    loss=worst+2*predicted.gather(1,oracle[:,None])[:,0]-actual.gather(1,oracle[:,None])[:,0]
    return loss.mean()+.001*(eta-args.eta).square().mean(),eta
def save(epoch):
    path=ROOT/'checkpoints'/f'ContinuationR2_seed20260920_epoch{epoch:02d}.pt'
    torch.save(dict(state_dict=model.state_dict(),epoch=epoch,seed=20260920,base_eta=args.eta,safety=args.safety,forecast_epoch=args.forecast_epoch,forecast_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),training_years=[2022,2023],price_base_training_years=[2021,2022,2023],selection_year=2024,loss='SPO+ finite-action regret surrogate with a fixed training-only dollar scale',dollar_scale=scale),path);return path
def economics(epoch):
    subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/evaluate_scenario_recourse.py'),'--eta',str(args.eta),'--safety',str(args.safety),'--epoch',str(args.forecast_epoch),'--value-epoch',str(epoch)],check=True,cwd=ROOT)
    return json.loads((ROOT/'results'/f'ValueR2_Eagle_2024_s20260920_ep{epoch:02d}.json').read_text())
save(0)
baseline_path=ROOT/'results'/f'RecourseR2_Eagle_2024_s20260920_eta{args.eta:g}_k{args.safety:g}_ep{args.forecast_epoch:02d}.json'
baseline=json.loads(baseline_path.read_text());best=baseline['saving_usd'];anchor=best;best_epoch=0;stale=0;logs=[];start=time.perf_counter()
baseline.update(name='ValueR2_Eagle_2024_s20260920_ep00',value_head_epoch=0,baseline_reused_from=baseline_path.name,baseline_identity='identical frozen forecast/state-cost weights and zero residual continuation head')
(ROOT/'results/ValueR2_Eagle_2024_s20260920_ep00.json').write_text(json.dumps(baseline,indent=2))
for epoch in range(1,args.epochs+1):
    model.continuation.train();order=torch.randperm(len(train['features']));total=[]
    for ids in order.split(256):
        optimizer.zero_grad(set_to_none=True);loss,_=objective(train,ids)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite continuation loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.continuation.parameters(),1.);optimizer.step();total.append(float(loss.detach()))
    model.continuation.eval()
    with torch.no_grad():vl,eta=objective(validation,torch.arange(len(validation['features'])))
    met=dict(epoch=epoch,train_regret_surrogate=float(np.mean(total)),validation_regret_surrogate=float(vl),eta_mean=float(eta.mean()),eta_min=float(eta.min()),eta_max=float(eta.max()),saving_usd=None,best_epoch=best_epoch,stale=stale)
    if epoch%5==0:
        save(epoch);row=economics(epoch);feasible=row['late_IT_mwh']<1e-8 and row['closing_it_mwh']<1e-8
        if feasible and row['saving_usd']>best:best=row['saving_usd'];best_epoch=epoch
        if feasible and row['saving_usd']>anchor+50:anchor=row['saving_usd'];stale=0
        else:stale+=1
        met.update(saving_usd=row['saving_usd'],best_epoch=best_epoch,stale=stale)
    logs.append(met)
    with (ROOT/'logs/continuation_training_seed20260920.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(logs[0]));writer.writeheader();writer.writerows(logs)
    print('CONTINUATION',json.dumps(met),flush=True)
    if epoch>=15 and stale>=3:break
best_path=ROOT/'checkpoints'/f'ContinuationR2_seed20260920_epoch{best_epoch:02d}.pt';shutil.copyfile(best_path,ROOT/'checkpoints/ContinuationR2_seed20260920_best.pt')
result=dict(status='complete',seed=20260920,epochs=epoch,best_epoch=best_epoch,best_2024_savings_usd=best,best_restored=True,forecast_epoch=args.forecast_epoch,eta=args.eta,safety=args.safety,forecast_parameters_updated=False,price_quality_preserved_by_frozen_forecast_branch=True,other_seeds_active=False,checkpoint_sha256=hashlib.sha256(best_path.read_bytes()).hexdigest(),seconds=time.perf_counter()-start)
(ROOT/'results/continuation_primary_complete.json').write_text(json.dumps(result,indent=2));print('CONTINUATION COMPLETE',json.dumps(result),flush=True)
