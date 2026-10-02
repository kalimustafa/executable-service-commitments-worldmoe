"""Fast primary-seed observer training; annual 2024 cash controls early stop."""
import sys,json,time,random,subprocess,hashlib,shutil,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT
from src.models.causal_physical_encoder import CausalPhysicalEncoder
from src.data.physical_observer_data import training_data,annual_data
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=16);p.add_argument('--batch',type=int,default=2048);args=p.parse_args()
seed=20260920;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
torch.set_num_threads(3);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.fp32_precision='tf32'
train={k:torch.as_tensor(v,device='cuda') for k,v in training_data().items()}
selection=annual_data(2024)
model=CausalPhysicalEncoder().cuda()
optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-3,fused=True)
tau=torch.tensor([.05,.1,.25,.5,.75,.9,.95],device='cuda')
weights=torch.tensor([.075,.1,.2,.25,.2,.1,.075],device='cuda')
baseline=json.loads((ROOT/'results/RecourseR2_Eagle_2024_s20260920_eta0_k0.9_ep10.json').read_text())
best=baseline['saving_usd'];anchor=best;best_epoch=0;stale=0;log=[]
assert best>0

@torch.no_grad()
def predict(data):
    result=[];model.eval()
    for a in range(0,len(data['target']),4096):
        inputs=[torch.as_tensor(data[k][a:a+4096],device='cuda') for k in ['history','base_quantiles','upper','clock']]
        result.append(model(*inputs)['quantiles'].cpu().numpy())
    q=np.concatenate(result)
    error=data['target'][:,:,None]-q
    score=np.maximum(np.array([.05,.1,.25,.5,.75,.9,.95])*error,(np.array([.05,.1,.25,.5,.75,.9,.95])-1)*error).mean()
    return q,float(score)

def save(epoch,score):
    path=ROOT/'checkpoints'/f'PhysicalR2_seed{seed}_epoch{epoch:02d}.pt'
    torch.save(dict(state_dict=model.state_dict(),epoch=epoch,seed=seed,architecture='CausalPhysicalEncoder',training_years=[2022,2023],selection_year=2024,physical_observation_lag_quarters=1,price_core_frozen=True,forecast_epoch=10,selection_pinball=score),path)
    return path

initial,initial_score=predict(selection);save(0,initial_score)
assert np.max(np.abs(initial-selection['base_quantiles']))<2e-7
print('PHYSICAL INITIAL',json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()),train_origins=len(train['target']),selection_pinball=initial_score,baseline_saving=best)),flush=True)
for epoch in range(1,args.epochs+1):
    started=time.perf_counter();model.train();order=torch.randperm(len(train['target']),device='cuda');totals=[]
    for ids in order.split(args.batch):
        optimizer.zero_grad(set_to_none=True)
        out=model(*[train[k][ids] for k in ['history','base_quantiles','upper','clock']])
        err=train['target'][ids,:,None]-out['quantiles']
        loss=torch.maximum(tau*err,(tau-1)*err).mean()
        mean=(out['quantiles']*weights).sum(-1)
        loss=loss+.1*(mean-train['target'][ids]).square().mean()+.0001*(out['location'].square().mean()+(out['spread'].log().square()).mean())
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite observer gradient')
        optimizer.step();totals.append(loss.detach())
    q,score=predict(selection);path=save(epoch,score)
    record=dict(epoch=epoch,train_loss=float(torch.stack(totals).mean()),selection_pinball=score,seconds=time.perf_counter()-started)
    if epoch%2==0:
        cache=ROOT/'cache'/f'physical_observer_2024_s{seed}_ep{epoch:02d}.npz'
        np.savez_compressed(cache,quantiles=q,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),history_lag=1)
        subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/evaluate_scenario_recourse.py'),'--eta','0','--safety','0.9','--epoch','10','--physical-epoch',str(epoch)],check=True,cwd=ROOT)
        row=json.loads((ROOT/'results'/f'PhysicalR2_Eagle_2024_s{seed}_ep{epoch:02d}.json').read_text())
        feasible=row['late_IT_mwh']<1e-8 and row['closing_it_mwh']<1e-8
        accepted=feasible and score<=initial_score*1.02
        record.update(saving_usd=row['saving_usd'],capture=row['capture_lower_fraction'],accepted=accepted)
        if accepted and row['saving_usd']>best:best=row['saving_usd'];best_epoch=epoch
        if accepted and row['saving_usd']>anchor+50:anchor=row['saving_usd'];stale=0
        else:stale+=1
    log.append(record);(ROOT/'logs/physical_training.json').write_text(json.dumps(log,indent=2))
    print('PHYSICAL EPOCH',json.dumps(record),flush=True)
    if epoch>=6 and stale>=3:break
best_path=ROOT/'checkpoints'/f'PhysicalR2_seed{seed}_epoch{best_epoch:02d}.pt'
shutil.copy2(best_path,ROOT/'checkpoints'/f'PhysicalR2_seed{seed}_best.pt')
result=dict(seed=seed,best_epoch=best_epoch,epochs_completed=epoch,best_saving_usd=best,baseline_saving_usd=baseline['saving_usd'],improvement_usd=best-baseline['saving_usd'],best_restored=True,early_stopping='actual 2024 annual economics; every two epochs; patience three; minimum six; min_delta 50 USD; forecast pinball guard 2%',checkpoint_sha256=hashlib.sha256(best_path.read_bytes()).hexdigest(),training_years=[2022,2023],selection_year=2024,other_seeds_started=False,forecast_scope='queue-independent current-quarter PV quantiles from strictly lagged observations; existing price and decision network frozen')
(ROOT/'results/physical_primary_complete.json').write_text(json.dumps(result,indent=2));print('PHYSICAL COMPLETE',json.dumps(result),flush=True)
