from src.runtime import ROOT
from src.data.access_guard import install_guard, assert_target_year
install_guard()
import argparse
import csv
import json
import random
import time
import numpy as np
import torch
from src.data.r2_data import R2Data
from src.models.factory import make_model
from src.training.losses import training_loss
from src.training.optimizer_groups import optimizer_for
from src.training.checkpoint_selection import selection_score

MODELS=['A0_Base','A1_Shallow','A2_Deep_NoGate','A3_WGR']
SEEDS=[20260920,20260921,20260922]

def save_csv(path,rows):
    with open(path,'w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def train_one(name,seed,data,*,epochs=100,smoke=False):
    checkpoint=ROOT/'checkpoints'/f'{name}_seed{seed}_best.ckpt'
    complete=ROOT/'logs'/f'{name}_seed{seed}_completed.json'
    if complete.exists() and not smoke:
        print('Already complete',name,seed,flush=True);return json.loads(complete.read_text())
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    deep=name in MODELS[2:]
    base=ROOT/'checkpoints'/f'A0_Base_seed{seed}_best.ckpt' if deep else None
    model=make_model(name,seed,data,base_checkpoint=base)
    total=sum(p.numel() for p in model.parameters());full_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert full_trainable<1500000
    optimizer=optimizer_for(model,'warmup' if deep else 'joint')
    best=(float('inf'),float('inf'));best_epoch=0;stale=0;logs=[];started=time.perf_counter()
    train=data.indices['train'];max_batch=4096
    if smoke:
        train=torch.cat([train[data.world_available[train]==0][:512],train[data.world_available[train]>0][:512]])
    print(json.dumps(dict(event='start',model=name,seed=seed,parameters=total,trainable_parameters=full_trainable,frozen_W1=59156961,trainable_W1=0)),flush=True)
    for epoch in range(1,epochs+1):
        if deep and epoch==9:optimizer=optimizer_for(model,'joint');stale=0
        stage='warmup' if deep and epoch<=8 else 'joint'
        model.train();indices=train.repeat(2)
        publication=torch.cat([torch.zeros(len(train),device=data.device),torch.ones(len(train),device=data.device)])
        generator=torch.Generator(device=data.device).manual_seed(seed+epoch)
        permutation=torch.randperm(len(indices),generator=generator,device=data.device)
        total_loss=torch.zeros((),device=data.device);seen=0
        for positions in permutation.split(max_batch):
            ix=indices[positions];pub=publication[positions];y,regime=data.row_targets(ix)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(data.batch(ix),pub,y[:,0])
            loss=training_loss(out,y,regime,data.target_scale[data.zone[ix]],gated=name=='A3_WGR')
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss: {name}/{seed}/{epoch}')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total_loss+=loss.detach()*len(ix);seen+=len(ix)
        metrics=selection_score(model,data)
        row=dict(model=name,seed=seed,epoch=epoch,stage=stage,train_loss=float(total_loss/seen),**metrics,seconds=time.perf_counter()-started,
                 trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),frozen_W1_parameters=59156961,split_hash=data.split_hash)
        logs.append(row)
        improvement=metrics['price_score']<best[0]-1e-6
        tied=abs(metrics['price_score']-best[0])<=1e-12 and metrics['raw_weighted_pinball']<best[1]
        if improvement or tied:
            best=(metrics['price_score'],metrics['raw_weighted_pinball']);best_epoch=epoch;stale=0
            payload=dict(schema='r2_wgr.checkpoint.v1',model=name,seed=seed,epoch=epoch,stage=stage,
                         state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},validation=row,
                         normalization=data.normalization,split_hash=data.split_hash,trainable_parameters=full_trainable,
                         total_head_parameters=total,world_config='configs/selected_w1.json',w1_trainable=0,
                         candidate='fixed_small_regularization',target_years=[2021,2022,2023,2024])
            torch.save(payload,checkpoint if not smoke else ROOT/'checkpoints'/f'smoke_{name}.ckpt')
        elif stage!='warmup':stale+=1
        save_csv(ROOT/'logs'/f'{"smoke_" if smoke else ""}{name}_seed{seed}_training.csv',logs)
        print(json.dumps(dict(**row,best_epoch=best_epoch,stale=stale)),flush=True)
        if stale>=12:break
    summary=dict(model=name,seed=seed,best_epoch=best_epoch,best_price_score=best[0],epochs_run=epoch,seconds=time.perf_counter()-started,
                 total_parameters=total,trainable_parameters=full_trainable,checkpoint=str(checkpoint),split_hash=data.split_hash,status='complete')
    if not smoke:complete.write_text(json.dumps(summary,indent=2))
    del model,optimizer;torch.cuda.empty_cache()
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    parser.add_argument('--seeds',nargs='+',type=int,default=SEEDS);parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--target-year',type=int,default=2024);args=parser.parse_args()
    assert_target_year(args.target_year)
    assert set(args.seeds)<=set(SEEDS)
    assert (ROOT/'results/baseline_reproduction_passed.json').exists()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    data=R2Data(multiscale=any(name in MODELS[2:] for name in args.models))
    rows=[]
    for name in args.models:
        for seed in args.seeds:rows.append(train_one(name,seed,data,epochs=1 if args.smoke else args.epochs,smoke=args.smoke))
    if not args.smoke:
        all_rows=[json.loads(path.read_text()) for path in sorted((ROOT/'logs').glob('A*_completed.json'))]
        save_csv(ROOT/'results/checkpoint_selection_2024.csv',all_rows)
    print('TRAINING INVOCATION COMPLETE',flush=True)

if __name__=='__main__':main()
