from src.runtime import ROOT,FAILED
from src.data.access_guard import install_guard,assert_target_year
install_guard()
import argparse
import json
import random
import time
import csv
import numpy as np
import torch
from src.data.r2_data import R2Data
from src.models.factory import make_model
from src.training.losses import training_loss
from src.training.checkpoint_selection import selection_score
from src.training.repaired_optimizer import warmup_optimizer,restore_best_for_joint

def save_rows(path,rows):
    with open(path,'w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def run_one(name,data,seed=20260920,smoke=False):
    assert seed==20260920,'User requires primary-seed pilot before any other seed'
    done=ROOT/'logs'/f'{name}_seed{seed}_completed.json'
    if done.exists() and not smoke:return json.loads(done.read_text())
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
    model=make_model(name,seed,data,base_checkpoint=FAILED/'checkpoints'/f'A0_Base_seed{seed}_best.ckpt')
    optimizer=warmup_optimizer(model)
    model.eval()
    started=time.perf_counter();initial=selection_score(model,data)
    train=data.indices['train'];prefix='smoke_' if smoke else ''
    if smoke:train=torch.cat([train[data.world_available[train]==0][:256],train[data.world_available[train]>0][:768]])
    best_path=ROOT/'checkpoints'/f'{prefix}{name}_seed{seed}_best.ckpt'
    warm_path=ROOT/'checkpoints'/f'{prefix}{name}_seed{seed}_warmup_best.ckpt'
    logs=[];best=initial['price_score'];best_epoch=0;warm_best=best;stale=0
    total=sum(p.numel() for p in model.parameters());assert total<1500000
    def payload(epoch,stage,metrics):
        return dict(schema='r2_bpsr.pilot.v1',model=name,seed=seed,epoch=epoch,stage=stage,
            state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer_state=optimizer.state_dict(),
            validation=metrics,selection_precision='FP32',normalization=data.normalization,split_hash=data.split_hash,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),total_head_parameters=total,
            target_years=[2021,2022,2023,2024],w1_trainable=0,base_source=str(FAILED/'checkpoints'/f'A0_Base_seed{seed}_best.ckpt'))
    torch.save(payload(0,'equivalent_base',initial),best_path);torch.save(payload(0,'equivalent_base',initial),warm_path)
    print(json.dumps(dict(event='baseline_equivalent_epoch_zero',model=name,seed=seed,score=best,parameters=total)),flush=True)
    for epoch in range(1,3 if smoke else 41):
        switch=2 if smoke else 11
        if epoch==switch:
            warm=torch.load(warm_path,map_location=data.device,weights_only=False)
            transition=restore_best_for_joint(model,optimizer,warm)
            (ROOT/'logs'/f'{prefix}{name}_stage_transition.json').write_text(json.dumps(transition,indent=2))
            print(json.dumps(dict(event='restore_best_warmup',model=name,**transition)),flush=True)
            stale=0
        stage='warmup' if epoch<switch else 'joint'
        model.train();assert not model.base.training
        indices=train.repeat(2);pub=torch.cat([torch.zeros(len(train),device=data.device),torch.ones(len(train),device=data.device)])
        order=torch.randperm(len(indices),generator=torch.Generator(device=data.device).manual_seed(seed+epoch),device=data.device)
        total_loss=torch.zeros((),device=data.device);seen=0;nonzero_final_gradient=False
        for positions in order.split(4096):
            ix=indices[positions];y,regime=data.row_targets(ix);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(data.batch(ix),pub[positions],y[:,0])
            loss=training_loss(out,y,regime,data.target_scale[data.zone[ix]],gated=True)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite pilot loss')
            loss.backward()
            last=model.raw_projection.weight if hasattr(model,'raw_projection') else model.residual[-1].weight
            nonzero_final_gradient|=last.grad is not None and bool(last.grad.abs().sum()>0)
            if stage=='warmup':assert all(p.grad is None and not p.requires_grad for p in model.base.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total_loss+=loss.detach()*len(ix);seen+=len(ix)
        metrics=selection_score(model,data)
        row=dict(model=name,seed=seed,epoch=epoch,stage=stage,train_loss=float(total_loss/seen),**metrics,
            selection_precision='FP32',seconds=time.perf_counter()-started,nonzero_residual_gradient=nonzero_final_gradient,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),split_hash=data.split_hash)
        logs.append(row)
        if metrics['price_score']<best-1e-6:
            best=metrics['price_score'];best_epoch=epoch;stale=0;torch.save(payload(epoch,stage,row),best_path)
        elif stage=='joint':stale+=1
        if stage=='warmup' and metrics['price_score']<warm_best-1e-6:
            warm_best=metrics['price_score'];torch.save(payload(epoch,stage,row),warm_path)
        save_rows(ROOT/'logs'/f'{prefix}{name}_training.csv',logs)
        print(json.dumps(dict(**row,best_epoch=best_epoch,stale=stale)),flush=True)
        if stage=='joint' and stale>=12:break
    result=dict(model=name,seed=seed,best_epoch=best_epoch,best_price_score=best,initial_price_score=initial['price_score'],
        epochs_run=epoch,seconds=time.perf_counter()-started,checkpoint=str(best_path),total_parameters=total,
        selection_precision='FP32',split_hash=data.split_hash,status='complete')
    if not smoke:done.write_text(json.dumps(result,indent=2))
    del model,optimizer;torch.cuda.empty_cache();return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',default=['C0_Repaired','C1_ProfileResidual']);p.add_argument('--smoke',action='store_true');p.add_argument('--target-year',type=int,default=2024)
    args=p.parse_args();assert_target_year(args.target_year)
    assert set(args.models)<={'C0_Repaired','C1_ProfileResidual'}
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    if not args.smoke:assert json.loads((ROOT/'results/redesign_tests.json').read_text())['passed']
    data=R2Data(multiscale=True)
    for name in args.models:run_one(name,data,smoke=args.smoke)
    if not args.smoke:save_rows(ROOT/'results/checkpoint_selection_2024.csv',[json.loads(p.read_text()) for p in sorted((ROOT/'logs').glob('C*_completed.json'))])

if __name__=='__main__':main()
