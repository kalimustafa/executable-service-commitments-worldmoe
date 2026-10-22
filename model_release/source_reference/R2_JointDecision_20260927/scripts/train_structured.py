"""Joint R2 training with economic early stopping and best-weight restoration."""
import sys,argparse,json,time,random,csv,subprocess,shutil,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import *
import numpy as np,pandas as pd,torch
from src.data.r2_data import R2Data,ZONES
from src.models.factory import make_model
from src.models.structured_decision import StructuredDecision,structured_allocation
from src.training.losses import training_loss,price_losses

p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=20260920);p.add_argument('--max-epochs',type=int,default=24);p.add_argument('--patience',type=int,default=3);args=p.parse_args()
seed=args.seed;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
torch.set_num_threads(3);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.fp32_precision='tf32'
data=R2Data(device='cuda',multiscale=True);train_ids=data.indices['train']
assert set(data.frame.iloc[train_ids.cpu().numpy()].date.str[:4])=={'2021','2022','2023'}
source=(PREVIOUS if seed==20260920 else ARCHIVE)/'checkpoints'/f'C1_ProfileResidual_seed{seed}_best.ckpt'
original=make_model('C1_ProfileResidual',seed,data);original.load_state_dict(torch.load(source,map_location='cpu',weights_only=False)['state_dict'])
def load(role):
    with np.load(ROOT/'cache'/f'structured_{role}.npz') as z:
        out={k:torch.as_tensor(z[k],device='cuda') for k in ['row_ids','states','publication','amounts','outcomes']};ref=torch.tensor(z['pue_reference'],device='cuda')
    return out,ref
groups,pue_ref=load('train');validation,_=load('selection')
model=StructuredDecision(original,data.target_scale,data.target_center[:,2],pue_ref).cuda()
model.r2.base.requires_grad_(False);model.calibration.requires_grad_(False)
model.r2.world_dropout=0.
optimizer=torch.optim.AdamW([dict(params=[x for x in model.r2.parameters() if x.requires_grad],lr=3e-5),dict(params=model.structured.parameters(),lr=3e-4)],weight_decay=1e-4,fused=True)
tau=torch.tensor([.05,.1,.25,.5,.75,.9,.95],device='cuda')
started=time.perf_counter();logs=[];gradient_checked=False;anchor_cache={}
model.eval()
with torch.inference_mode():
    for gate in [0,1]:
        values=[]
        for ids in torch.arange(len(data.frame),device='cuda').split(4096):
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(data.batch(ids),torch.full((len(ids),),float(gate),device='cuda'),data.labels[ids,0])
            values.append(torch.stack([out[k] for k in ['da','dart','rt']],1).detach())
        anchor_cache[gate]=torch.cat(values).clone()

def decision_loss(group,indices,training):
    ids=group['row_ids'][indices];state=group['states'][indices];publication=group['publication'][indices];flat=ids.reshape(-1)
    out=model(data.batch(flat),publication.reshape(-1),data.labels[flat,0],state)
    alloc=structured_allocation(out['segment_widths'],out['segment_scores'],group['amounts'][indices],straight_through=training)
    truth=group['outcomes'][indices];rt,pv,pue=truth[:,:,0],truth[:,:,1],truth[:,:,2]
    error=pv[:,:,None]-out['pv_quantiles'];physical=state[:,:,9,None]
    pv_loss=(torch.maximum(tau*error,(tau-1)*error)*physical).sum()/(physical.sum()*7).clamp_min(1)
    power=1.639059066772461;load=pue*(.23069976270198822+power*state[:,:,0]);increment=pue*power*alloc
    used=torch.minimum((pv-load).clamp_min(0),increment)
    fees=torch.ones_like(alloc)*2.;fees[:,0]=0.
    pue0=pue_ref[data.zone[flat].reshape(ids.shape)[:,0]]
    bill=((rt*(increment-used)-used)+fees*pue0[:,None]*power*alloc).sum(1)*.25
    with torch.no_grad():
        cap=(1-state[:,:,0]).clamp_min(0);solar=torch.minimum(((pv-load)/(pue*power)).clamp_min(0),cap);grid=cap-solar
        cs=-pue*power+fees*pue0[:,None]*power;cg=rt*pue*power+fees*pue0[:,None]*power
        avg=(cs*solar+cg*grid)/cap.clamp_min(1e-8);neg=cg<cs;cs=torch.where(neg,avg,cs);cg=torch.where(neg,avg,cg)
        costs=torch.cat([cs,cg],1);caps=torch.cat([solar,grid],1);order=costs.argsort(dim=1,stable=True);costs=costs.gather(1,order);caps=caps.gather(1,order)
        take=torch.minimum(caps,(group['amounts'][indices,None]-(caps.cumsum(1)-caps)).clamp_min(0));best=(costs*take).sum(1)*.25
    regret=((bill-best)/(group['amounts'][indices]*power*pue0*.25*50).clamp_min(1e-5)).mean()
    penalty=.002*out['opportunity'].square().mean()
    return pv_loss,regret,penalty

@torch.no_grad()
def validate(epoch):
    model.eval();prices=[];pv=[];regret=[]
    for gate in [0,1]:
        for ids in data.indices['selection'].split(4096):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=model(data.batch(ids),torch.full((len(ids),),float(gate),device='cuda'),data.labels[ids,0]);_,loss=price_losses(out,data.labels[ids],data.target_scale[data.zone[ids]])
            prices.append(loss)
    for gi in torch.arange(len(validation['row_ids']),device='cuda').split(64):
        with torch.autocast('cuda',dtype=torch.bfloat16):pl,rl,_=decision_loss(validation,gi,False)
        pv.append(pl*len(gi));regret.append(rl*len(gi))
    return dict(epoch=epoch,price_score=float(torch.cat(prices).mean()),pv_pinball=float(torch.stack(pv).sum()/len(validation['row_ids'])),hard_allocation_regret=float(torch.stack(regret).sum()/len(validation['row_ids'])))

def save(epoch,metrics):
    path=ROOT/'checkpoints'/f'StructuredR2_seed{seed}_epoch{epoch:02d}.pt'
    torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),epoch=epoch,seed=seed,metrics=metrics,training_years=[2021,2022,2023],physical_training_years=[2022,2023],selection_year=2024,world_trainable_parameters=0,source_checkpoint=str(source),normalization=data.normalization,architecture='StructuredDecision v2'),path)
    return path

with np.load(WORKSPACE/'external_validation/data/forcing_2024.npz') as z:times=pd.DatetimeIndex(pd.to_datetime(z['times'],utc=True))
hours=times.floor('h').unique();index=pd.MultiIndex.from_product([hours,ZONES],names=['timestamp_utc','zone'])
frame=data.frame.copy();frame.timestamp_utc=pd.to_datetime(frame.timestamp_utc,utc=True)
lookup=pd.Series(np.arange(len(frame)),index=pd.MultiIndex.from_frame(frame[['timestamp_utc','zone']]))
economic_ids=torch.tensor(lookup.reindex(index).to_numpy(np.int64),device='cuda')

@torch.no_grad()
def economic(epoch):
    model.eval();arrays=dict(hours=hours.asi8,pue_reference=pue_ref.cpu().numpy(),historical_rt=model.historical_rt.cpu().numpy());apic=0.
    for gate in [0,1]:
        values=[];latent=[]
        for ids in economic_ids.split(4096):
            y=data.labels[ids]
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(data.batch(ids),torch.full((len(ids),),float(gate),device='cuda'),y[:,0])
            if gate:apic=max(apic,float((out['rt']-(y[:,0,None]-out['dart'].flip(-1))).abs().max()))
            values.append(out['rt'].float().cpu().numpy());latent.append(out['decision_latent'].float().cpu().numpy())
        arrays[f'rt_{gate}']=np.concatenate(values).reshape(len(hours),4,7);arrays[f'latent_{gate}']=np.concatenate(latent).reshape(len(hours),4,128)
    np.savez_compressed(ROOT/'cache'/f'structured_market2024_s{seed}_ep{epoch:02d}.npz',**arrays)
    cmd=[sys.executable,'-B','-u',str(ROOT/'scripts/replay_development.py'),'--structured','--seed',str(seed),'--epochs',str(epoch)]
    subprocess.run(cmd,check=True,cwd=ROOT)
    row=json.loads((ROOT/'results'/f'StructuredR2_Eagle_2024_s{seed}_ep{epoch:02d}.json').read_text());row['apic_max_error']=apic
    return row

initial=validate(0);save(0,initial);base=economic(0);best_saving=base['saving_usd'];best_epoch=0;stale=0
print('INITIAL STRUCTURED',json.dumps(dict(**initial,saving_usd=best_saving)),flush=True)
decision_gradient=None
for epoch in range(1,args.max_epochs+1):
    model.train();epoch_started=time.perf_counter();order=torch.randperm(len(train_ids),device='cuda');totals=[]
    for step,positions in enumerate(order.split(2048)):
        ids=train_ids[positions];y,regime=data.row_targets(ids);publication=torch.randint(0,2,(len(ids),),device='cuda').float();gi=torch.randint(0,len(groups['row_ids']),(32,),device='cuda')
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(data.batch(ids),publication,y[:,0]);price=training_loss(out,y,regime,data.target_scale[data.zone[ids]],gated=True)
            prediction=torch.stack([out[k] for k in ['da','dart','rt']],1)
            anchor=torch.where(publication[:,None,None].bool(),anchor_cache[1][ids],anchor_cache[0][ids])
            distill=((prediction-anchor)/data.target_scale[data.zone[ids],:,None]).square().mean()
            pv,regret,penalty=decision_loss(groups,gi,True)
            loss=price+2*distill+2*pv+regret+penalty
        if step%32==0 and not torch.isfinite(loss):raise RuntimeError('Nonfinite structured objective')
        if epoch==1 and step==1:
            gradients=torch.autograd.grad(2*pv+regret,list(model.r2.adapter.parameters()),retain_graph=True,allow_unused=True)
            decision_gradient=sum(float(g.detach().norm()) for g in gradients if g is not None)
            if decision_gradient<=0:raise RuntimeError('Shared R2 receives no decision gradient')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True);optimizer.step()
        totals.append(torch.stack([price.detach(),pv.detach(),regret.detach(),distill.detach()]))
    means=torch.stack(totals).mean(0).cpu().tolist();metrics=validate(epoch)
    metrics.update(price_train=means[0],pv_train=means[1],decision_train=means[2],prediction_anchor_penalty=means[3],train_and_validation_seconds=time.perf_counter()-epoch_started,decision_adapter_gradient=decision_gradient,economic_saving_usd=None,late_it_mwh=None,closing_it_mwh=None,early_stop_stale=stale,best_epoch=best_epoch)
    save(epoch,metrics)
    if epoch%2==0:
        row=economic(epoch);feasible=row['late_IT_mwh']<1e-8 and row['closing_it_mwh']<1e-8 and metrics['price_score']<=initial['price_score']*1.02
        improvement=feasible and row['saving_usd']>best_saving+50
        if improvement:best_saving=row['saving_usd'];best_epoch=epoch;stale=0
        else:stale+=1
        metrics.update(economic_saving_usd=row['saving_usd'],late_it_mwh=row['late_IT_mwh'],closing_it_mwh=row['closing_it_mwh'],early_stop_stale=stale,best_epoch=best_epoch)
    logs.append(metrics)
    with (ROOT/'logs'/f'structured_training_seed{seed}.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(logs[0]));writer.writeheader();writer.writerows(logs)
    print('STRUCTURED EPOCH',json.dumps(metrics),flush=True)
    if epoch>=6 and stale>=args.patience:break
best=ROOT/'checkpoints'/f'StructuredR2_seed{seed}_epoch{best_epoch:02d}.pt'
restored=torch.load(best,map_location='cpu',weights_only=False);model.load_state_dict(restored['state_dict'])
best_path=ROOT/'checkpoints'/f'StructuredR2_seed{seed}_best.pt';shutil.copyfile(best,best_path)
result=dict(status='complete',seed=seed,completed_epochs=epoch,best_epoch=best_epoch,early_stopped=epoch<args.max_epochs,patience=args.patience,best_restored=True,baseline_2024_savings_usd=base['saving_usd'],best_2024_savings_usd=best_saving,improvement_usd=best_saving-base['saving_usd'],secondary_seed_effect_gate_passed=best_epoch>0,seconds=time.perf_counter()-started,checkpoint_sha256=hashlib.sha256(best_path.read_bytes()).hexdigest(),training_years=[2021,2022,2023],selection_year=2024,parameter_updates_on_selection=False,world_trainable_parameters=0,decision_adapter_gradient=decision_gradient)
(ROOT/'results'/f'structured_complete_s{seed}.json').write_text(json.dumps(result,indent=2));print('STRUCTURED COMPLETE',json.dumps(result),flush=True)
