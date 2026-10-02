from src.runtime import ROOT
from src.data.access_guard import install_guard
install_guard()
import json,time
import numpy as np
import pandas as pd
import torch
from src.data.r2_data import R2Data
from src.calibration.calibrate_and_predict import load_model
from src.models.causal_error_memory import R2CausalErrorMemory
from src.training.losses import training_loss
from src.training.checkpoint_selection import selection_score

def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.manual_seed(20260920)
    data=R2Data(multiscale=True,memory=True);source,_=load_model('C0_Repaired',20260920,data)
    n=data.error_memory.shape[-1];model=R2CausalErrorMemory(source,n).to(data.device).eval()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0005,weight_decay=1e-4)
    probe=data.indices['selection'][:32];pub=(torch.arange(32,device=data.device)%2).float();batch=data.batch(probe);anchor=data.labels[probe,0]
    with torch.no_grad():
        a=source(batch,pub,anchor);b=model(batch,pub,anchor)
        for key in ['da','dart','rt']:torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    source_ids={id(p) for p in source.parameters()}
    assert all(id(p) not in source_ids for group in optimizer.param_groups for p in group['params'])
    initial=selection_score(model,data);best=initial['price_score'];best_epoch=0;stale=0;logs=[];started=time.perf_counter()
    def save(epoch,metrics):
        torch.save(dict(schema='r2.causal_error_memory.v1',model='C6_CausalErrorMemory',seed=20260920,epoch=epoch,stage='frozen_source_memory',
            state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},memory_features=n,
            validation=metrics,split_hash=data.split_hash,normalization=data.normalization,selection_precision='FP32',
            total_head_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)),
            ROOT/'checkpoints/C6_CausalErrorMemory_seed20260920_best.ckpt')
    save(0,initial);train=data.indices['train']
    for epoch in range(1,41):
        model.train();indices=train.repeat(2);publication=torch.cat([torch.zeros(len(train),device=data.device),torch.ones(len(train),device=data.device)])
        order=torch.randperm(len(indices),generator=torch.Generator(device=data.device).manual_seed(20260920+epoch),device=data.device)
        total=0.;seen=0
        for positions in order.split(4096):
            ix=indices[positions];y,regime=data.row_targets(ix);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(data.batch(ix),publication[positions],y[:,0])
            loss=training_loss(out,y,regime,data.target_scale[data.zone[ix]],True)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite memory loss')
            loss.backward();assert all(p.grad is None for p in source.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=float(loss.detach())*len(ix);seen+=len(ix)
        metrics=selection_score(model,data);row=dict(model='C6_CausalErrorMemory',seed=20260920,epoch=epoch,train_loss=total/seen,**metrics,seconds=time.perf_counter()-started)
        if metrics['price_score']<best-1e-6:best=metrics['price_score'];best_epoch=epoch;stale=0;save(epoch,row)
        else:stale+=1
        logs.append(row);pd.DataFrame(logs).to_csv(ROOT/'logs/causal_error_memory_training.csv',index=False)
        print(json.dumps(dict(**row,best_epoch=best_epoch,stale=stale)),flush=True)
        if stale>=10:break
    (ROOT/'logs/C6_CausalErrorMemory_seed20260920_completed.json').write_text(json.dumps(dict(model='C6_CausalErrorMemory',seed=20260920,
        best_epoch=best_epoch,best_price_score=best,initial_price_score=initial['price_score'],epochs_run=epoch,
        source_frozen=True,new_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),status='complete'),indent=2))

if __name__=='__main__':main()
