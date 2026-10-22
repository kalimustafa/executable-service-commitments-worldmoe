"""Train an ordered multi-horizon physical forecast branch with cash early stop."""
import sys,json,time,random,subprocess,hashlib,shutil,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT
from src.models.causal_physical_encoder import HorizonPhysicalEncoder
from src.data.physical_horizon_data import build
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=16);p.add_argument('--batch',type=int,default=256);args=p.parse_args()
seed=20260920;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(3)
torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.fp32_precision='tf32'
raw=build('train');assert set(__import__('pandas').to_datetime(raw['origins'],utc=True).year)=={2022,2023}
train={k:torch.as_tensor(v,device='cuda') for k,v in raw.items() if k!='origins'};del raw
selection=build('2024');model=HorizonPhysicalEncoder().cuda()
optimizer=torch.optim.AdamW(model.parameters(),lr=6e-4,weight_decay=1e-3,fused=True)
tau=torch.tensor([.05,.1,.25,.5,.75,.9,.95],device='cuda');weights=torch.tensor([.075,.1,.2,.25,.2,.1,.075],device='cuda')
lead_weight=(.5+torch.exp(-torch.arange(96,device='cuda')/16))[None,None,:,None]
baseline=json.loads((ROOT/'results/RecourseR2_Eagle_2024_s20260920_eta0_k0.9_ep10.json').read_text())
best=baseline['saving_usd'];anchor=best;best_epoch=None;stale=0;log=[]

@torch.no_grad()
def predict(data):
    model.eval();q=np.empty((len(data['target']),4,96,7),np.float32)
    for a in range(0,len(q),512):
        q[a:a+512]=model(torch.as_tensor(data['history'][a:a+512],device='cuda'),torch.as_tensor(data['future'][a:a+512],device='cuda'))['quantiles'].cpu().numpy()
    # The last 24 h has no future target beyond year-end and cannot enter the score.
    error=data['target'][:-96,:,:,None]-q[:-96]
    quantiles=np.array([.05,.1,.25,.5,.75,.9,.95],np.float32)
    score=float(np.maximum(quantiles*error,(quantiles-1)*error).mean())
    return q,score

initial,initial_score=predict(selection);del initial
print('HORIZON INITIAL',json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()),training_origins=len(train['target']),selection_pinball=initial_score,baseline_saving=best)),flush=True)
for epoch in range(1,args.epochs+1):
    start=time.perf_counter();model.train();totals=[]
    for ids in torch.randperm(len(train['target']),device='cuda').split(args.batch):
        optimizer.zero_grad(set_to_none=True);out=model(train['history'][ids],train['future'][ids]);err=train['target'][ids,:,:,None]-out['quantiles']
        loss=(torch.maximum(tau*err,(tau-1)*err)*lead_weight).mean()+.05*((out['quantiles']*weights).sum(-1)-train['target'][ids]).square().mean()+1e-5*out['residual'].square().mean()
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite horizon forecast gradient')
        optimizer.step();totals.append(loss.detach())
    q,score=predict(selection)
    path=ROOT/'checkpoints'/f'HorizonR2_seed{seed}_epoch{epoch:02d}.pt'
    torch.save(dict(state_dict=model.state_dict(),epoch=epoch,seed=seed,architecture='HorizonPhysicalEncoder',training_years=[2022,2023],selection_year=2024,forecast_epoch=10,queue_independent=True,selection_pinball=score),path)
    record=dict(epoch=epoch,train_loss=float(torch.stack(totals).mean()),selection_pinball=score,seconds=time.perf_counter()-start)
    if epoch%2==0:
        cache=ROOT/'cache'/f'physical_horizon_2024_s{seed}_ep{epoch:02d}.npz'
        np.savez_compressed(cache,quantiles=q,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),history_lag=1)
        subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/evaluate_scenario_recourse.py'),'--eta','0','--safety','0.9','--epoch','10','--horizon-epoch',str(epoch)],check=True,cwd=ROOT)
        row=json.loads((ROOT/'results'/f'HorizonR2_Eagle_2024_s{seed}_ep{epoch:02d}.json').read_text())
        accepted=row['late_IT_mwh']<1e-8 and row['closing_it_mwh']<1e-8 and score<=initial_score
        record.update(saving_usd=row['saving_usd'],capture=row['capture_lower_fraction'],accepted=accepted)
        if accepted and row['saving_usd']>best:best=row['saving_usd'];best_epoch=epoch
        if accepted and row['saving_usd']>anchor+50:anchor=row['saving_usd'];stale=0
        else:stale+=1
    del q
    log.append(record);(ROOT/'logs/horizon_training.json').write_text(json.dumps(log,indent=2));print('HORIZON EPOCH',json.dumps(record),flush=True)
    if epoch>=6 and stale>=3:break
result=dict(seed=seed,best_epoch=best_epoch,epochs_completed=epoch,best_saving_usd=best,baseline_saving_usd=baseline['saving_usd'],improvement_usd=best-baseline['saving_usd'],price_core_frozen=True,training_years=[2022,2023],selection_year=2024,other_seeds_started=False)
if best_epoch is not None:
    path=ROOT/'checkpoints'/f'HorizonR2_seed{seed}_epoch{best_epoch:02d}.pt';shutil.copy2(path,ROOT/'checkpoints'/f'HorizonR2_seed{seed}_best.pt');result['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
(ROOT/'results/horizon_primary_complete.json').write_text(json.dumps(result,indent=2));print('HORIZON COMPLETE',json.dumps(result),flush=True)
