import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import *
import torch,numpy as np,json,time,random,csv,argparse
import torch.nn.functional as F
from src.data.r2_data import R2Data
from src.models.factory import make_model
from src.models.joint_decision import JointDecision
from src.training.losses import training_loss,price_losses

parser=argparse.ArgumentParser()
parser.add_argument('--seed',type=int,default=20260920)
parser.add_argument('--resume',action='store_true')
parser.add_argument('--batch-size',type=int,default=2048)
parser.add_argument('--decision-batch',type=int,default=32)
parser.add_argument('--validation-batch',type=int,default=64)
parser.add_argument('--epochs',type=int,default=20)
args=parser.parse_args()
seed=args.seed;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
torch.set_float32_matmul_precision('high')
data=R2Data(device='cuda',multiscale=True)
train_ids=data.indices['train'];assert set(data.frame.iloc[train_ids.cpu().numpy()].date.str[:4])=={'2021','2022','2023'}
original=make_model('C1_ProfileResidual',seed,data)
source=(PREVIOUS if seed==20260920 else ARCHIVE)/'checkpoints'/f'C1_ProfileResidual_seed{seed}_best.ckpt'
original.load_state_dict(torch.load(source,map_location='cpu',weights_only=False)['state_dict'])
def load_groups(role):
    with np.load(ROOT/'cache'/f'decision_{role}.npz',allow_pickle=False) as z:
        result={k:torch.as_tensor(z[k],device='cuda') for k in ['row_ids','states','costs','publication','amounts','outcomes']}
        ref=torch.tensor(z['pue_reference'],device='cuda')
    return result,ref
groups,pue_ref=load_groups('train');validation,_=load_groups('selection')
model=JointDecision(original,data.target_scale,data.target_center[:,2],pue_ref).cuda()
for p in model.parameters():p.requires_grad_(True)
model.r2.base.slot_head.requires_grad_(False)
base_params=list(model.r2.base.parameters());base_ids={id(p) for p in base_params};r2_params=[p for p in model.r2.parameters() if id(p) not in base_ids]
r2_ids={id(p) for p in model.r2.parameters()};new_params=[p for p in model.parameters() if id(p) not in r2_ids]
optimizer=torch.optim.AdamW([dict(params=base_params,lr=1e-5),dict(params=r2_params,lr=1e-4),dict(params=new_params,lr=5e-4)],weight_decay=1e-4,fused=True)
started=time.perf_counter();logs=[];decision_only_gradient=None;start_epoch=0;prior_elapsed=0.
log_path=ROOT/'logs'/f'joint_training_seed{seed}.csv'
resume_files=sorted((ROOT/'checkpoints').glob(f'JointR2_seed{seed}_epoch*.pt')) if args.resume else []
if resume_files:
    state=torch.load(resume_files[-1],map_location='cpu',weights_only=False)
    model.load_state_dict(state['state_dict']);start_epoch=state['epoch']
    if 'optimizer' in state:optimizer.load_state_dict(state['optimizer'])
    if 'torch_rng' in state:
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    oldlog=log_path if log_path.exists() else ROOT/'logs/primary_joint_training.csv'
    if oldlog.exists():
        with oldlog.open(newline='',encoding='utf-8') as stream:
            logs=[dict((k,float(v) if v not in ('','None') else None) for k,v in row.items()) for row in csv.DictReader(stream)]
        logs=[row for row in logs if row['epoch']<=start_epoch]
        if logs:prior_elapsed=logs[-1]['elapsed_seconds'];decision_only_gradient=logs[-1]['decision_only_adapter_gradient_norm']
    (ROOT/'logs'/f'fast_resume_seed{seed}.json').write_text(json.dumps(dict(resume_checkpoint=str(resume_files[-1]),epoch=start_epoch,optimizer_state_restored='optimizer' in state,optimizer_restart_reason='Original checkpoint format contained weights only; Adam moments restarted for this efficiency change.' if 'optimizer' not in state else None,batch_size=args.batch_size,decision_batch=args.decision_batch,validation_batch=args.validation_batch,epochs=args.epochs,training_rows_per_epoch=len(train_ids),tf32=True,bfloat16=True),indent=2))
    print('RESUME',start_epoch,'batch',args.batch_size,'decision batch',args.decision_batch,flush=True)

def decision_pass(group,indices,temperature=5.):
    ids=group['row_ids'][indices];st=group['states'][indices];pub=group['publication'][indices]
    flat=ids.reshape(-1);y=data.labels[flat]
    output=model(data.batch(flat),pub.reshape(-1),y[:,0],st.reshape(-1,16))
    predicted=output['decision_cost'].reshape(ids.shape);truth=group['costs'][indices]
    valid=(1-st[:,:,0])>=st[:,:,1]-1e-8
    cost_loss=(F.smooth_l1_loss(predicted/50,truth/50,reduction='none')*valid).sum()/valid.sum().clamp_min(1)
    fees=torch.ones_like(predicted)*2;fees[:,0]=0
    scores=output['decision_score'].reshape(ids.shape)+fees
    cap=(1-st[:,:,0]).clamp_min(0);amount=group['amounts'][indices];allocation=torch.zeros_like(cap)
    for _ in range(8):
        left=(cap-allocation).clamp_min(0);need=(amount-allocation.sum(1)).clamp_min(0)
        logits=-scores/temperature+left.clamp_min(1e-12).log()
        probability=torch.softmax(logits.masked_fill(left<1e-9,-1e6),dim=1)
        allocation=allocation+torch.minimum(need[:,None]*probability,left)
    left=(cap-allocation).clamp_min(0);need=(amount-allocation.sum(1)).clamp_min(0)
    allocation=allocation+need[:,None]*left/left.sum(1,keepdim=True).clamp_min(1e-8)
    actual=group['outcomes'][indices];rt,pv,pue=actual[:,:,0],actual[:,:,1],actual[:,:,2]
    power=1.639059066772461;pue0=1.05
    load0=pue*(.23069976270198822+power*st[:,:,0]);increment=pue*power*allocation
    pv_used=torch.minimum((pv-load0).clamp_min(0),increment)
    bill=(rt*(increment-pv_used)-pv_used+fees*pue0*power*allocation).sum(1)*.25
    with torch.no_grad():
        solar=torch.minimum(((pv-load0)/(pue*power)).clamp_min(0),cap);grid=cap-solar
        cs=-pue*power+fees*pue0*power;cg=rt*pue*power+fees*pue0*power
        average=(cs*solar+cg*grid)/cap.clamp_min(1e-8);negative=cg<cs
        cs=torch.where(negative,average,cs);cg=torch.where(negative,average,cg)
        costs=torch.cat([cs,cg],1);caps=torch.cat([solar,grid],1);order=costs.argsort(1)
        costs=costs.gather(1,order);caps=caps.gather(1,order)
        takes=torch.minimum(caps,(amount[:,None]-(caps.cumsum(1)-caps)).clamp_min(0))
        best=(costs*takes).sum(1)*.25
    regret=((bill-best)/(amount*power*pue0*.25*50).clamp_min(1e-5)).mean()
    regret=regret+1e-5*output['opportunity_adjustment'].square().mean()
    return output,cost_loss,regret

@torch.no_grad()
def select_metrics(epoch):
    model.eval();losses=[]
    for gate in [0.,1.]:
        for ids in data.indices['selection'].split(4096):
            y=data.labels[ids]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=model(data.batch(ids),torch.full((len(ids),),gate,device='cuda'),y[:,0])
                _,loss=price_losses(out,y,data.target_scale[data.zone[ids]])
            losses.append(loss)
    price=float(torch.cat(losses).mean());cost=[];regret=[]
    for ids in torch.arange(len(validation['row_ids']),device='cuda').split(args.validation_batch):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,cl,rl=decision_pass(validation,ids,temperature=2.)
        cost.append(cl*len(ids));regret.append(rl*len(ids))
    return dict(epoch=epoch,price_score=price,cost_huber=float(torch.stack(cost).sum()/len(validation['row_ids'])),ranking_regret=float(torch.stack(regret).sum()/len(validation['row_ids'])))

def save(epoch,metrics):
    path=ROOT/'checkpoints'/f'JointR2_seed{seed}_epoch{epoch:02d}.pt'
    torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),epoch=epoch,seed=seed,metrics=metrics,training_years=[2021,2022,2023],physical_training_years=[2022,2023],selection_year=2024,world_trainable_parameters=0,source_checkpoint=str(source),normalization=data.normalization,training_batch=args.batch_size,decision_batch=args.decision_batch),path)
    return path

if start_epoch==0:
    initial=select_metrics(0);save(0,initial);print('INITIAL',initial,flush=True)
for epoch in range(start_epoch+1,args.epochs+1):
    epoch_started=time.perf_counter()
    model.train();order=torch.randperm(len(train_ids),device='cuda');totals=[];decision_gradient=0.
    for step,positions in enumerate(order.split(args.batch_size)):
        ids=train_ids[positions];y,regime=data.row_targets(ids);publication=torch.randint(0,2,(len(ids),),device='cuda').float()
        gi=torch.randint(0,len(groups['row_ids']),(args.decision_batch,),device='cuda')
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(data.batch(ids),publication,y[:,0]);pl=training_loss(out,y,regime,data.target_scale[data.zone[ids]],gated=True)
            dec,cl,rl=decision_pass(groups,gi,temperature=max(2.,8.-epoch*.3))
            loss=pl+cl+rl
        if step%32==0 and not torch.isfinite(loss):raise RuntimeError('Nonfinite joint objective')
        if decision_only_gradient is None and step==1:
            grads=torch.autograd.grad(cl+rl,list(model.r2.adapter.parameters()),retain_graph=True,allow_unused=True)
            decision_only_gradient=sum(float(g.detach().norm()) for g in grads if g is not None)
            if decision_only_gradient<=0:raise RuntimeError('Decision objective does not train the shared R2 adapter')
        loss.backward()
        if step==0:
            decision_gradient=sum(float(p.grad.detach().norm()) for p in model.r2.adapter.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True);optimizer.step();totals.append(torch.stack([pl.detach(),cl.detach(),rl.detach()]))
    means=torch.stack(totals).mean(0).cpu().tolist()
    metrics=select_metrics(epoch);metrics.update(price_train=means[0],cost_train=means[1],rank_train=means[2],adapter_gradient_norm=decision_gradient,decision_only_adapter_gradient_norm=decision_only_gradient,elapsed_seconds=prior_elapsed+time.perf_counter()-started,world_trainable_parameters=0)
    logs.append(metrics)
    # Every checkpoint is retained for 2024 economic selection; the surrogate
    # score alone never establishes success against the 80% financial target.
    save(epoch,metrics)
    with open(log_path,'w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(logs[0]));writer.writeheader();writer.writerows(logs)
    print(json.dumps(dict(**metrics,epoch_seconds=time.perf_counter()-epoch_started)),flush=True)
(ROOT/'logs'/f'training_complete_seed{seed}.json').write_text(json.dumps(dict(status='complete',epochs=args.epochs,seed=seed,training_years=[2021,2022,2023],selection_fitting=False,other_seeds_started=False,training_batch=args.batch_size,decision_batch=args.decision_batch),indent=2))
