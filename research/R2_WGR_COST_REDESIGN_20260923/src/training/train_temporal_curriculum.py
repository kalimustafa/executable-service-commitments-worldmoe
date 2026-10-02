from src.runtime import ROOT,FAILED
from src.data.access_guard import install_guard,assert_target_year
install_guard()
import argparse,json,time,random
import numpy as np
import pandas as pd
import torch
from src.data.r2_data import R2Data
from src.models.factory import make_model
from src.training.losses import training_loss
from src.training.checkpoint_selection import selection_score
from src.training.repaired_optimizer import warmup_optimizer,restore_best_for_joint

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['C3_TemporalWorld','C4_RecentReplay','C5_TemporalReplay']);p.add_argument('--target-year',type=int,default=2024)
    args=p.parse_args();assert_target_year(args.target_year)
    name=args.model;seed=20260920;recent=name in {'C4_RecentReplay','C5_TemporalReplay'};temporal=name in {'C3_TemporalWorld','C5_TemporalReplay'}
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
    data=R2Data(multiscale=True,temporal=temporal)
    model=make_model(name,seed,data,base_checkpoint=FAILED/'checkpoints/A0_Base_seed20260920_best.ckpt')
    optimizer=warmup_optimizer(model);model.eval()
    checkpoint=ROOT/'checkpoints'/f'{name}_seed{seed}_best.ckpt';warm_path=ROOT/'checkpoints'/f'{name}_seed{seed}_warmup_best.ckpt'
    complete=ROOT/'logs'/f'{name}_seed{seed}_completed.json'
    if complete.exists():print('Already complete',name,flush=True);return
    total_params=sum(p.numel() for p in model.parameters());assert total_params<1500000
    start=time.perf_counter();initial=selection_score(model,data);best=initial['price_score'];warm_best=best;best_epoch=0;logs=[];stale=0
    def payload(epoch,stage,metrics):
        return dict(schema='r2.temporal_curriculum.v1',model=name,seed=seed,epoch=epoch,stage=stage,
            state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer_state=optimizer.state_dict(),
            validation=metrics,selection_precision='FP32',split_hash=data.split_hash,normalization=data.normalization,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),total_head_parameters=total_params,
            temporal_world=temporal,recent_replay=recent,target_years=[2021,2022,2023,2024],w1_trainable=0)
    torch.save(payload(0,'equivalent_base',initial),checkpoint);torch.save(payload(0,'equivalent_base',initial),warm_path)
    start_epoch=1
    if recent:
        source='C0_Repaired' if name=='C4_RecentReplay' else 'C3_TemporalWorld'
        warm=torch.load(ROOT/'checkpoints'/f'{source}_seed{seed}_warmup_best.ckpt',map_location=data.device,weights_only=False)
        audit=restore_best_for_joint(model,optimizer,warm);audit['source_model']=source
        (ROOT/'logs'/f'{name}_stage_transition.json').write_text(json.dumps(audit,indent=2))
        current=selection_score(model,data)
        if current['price_score']<best:best=current['price_score'];best_epoch=int(warm['epoch']);torch.save(payload(best_epoch,'transferred_warmup',current),checkpoint)
        start_epoch=11
    train=data.indices['train'];year=torch.tensor(data.frame.date.str[:4].astype(int).to_numpy(),device=data.device)
    newest=train[year[train]==2023];older=train[year[train]<2023]
    print(json.dumps(dict(event='start',model=name,seed=seed,parameters=total_params,initial_score=initial['price_score'],start_epoch=start_epoch,recent_replay=recent)),flush=True)
    for epoch in range(start_epoch,41):
        if epoch==11 and not recent:
            warm=torch.load(warm_path,map_location=data.device,weights_only=False)
            audit=restore_best_for_joint(model,optimizer,warm)
            (ROOT/'logs'/f'{name}_stage_transition.json').write_text(json.dumps(audit,indent=2));stale=0
        stage='warmup' if epoch<=10 else 'joint';model.train()
        generator=torch.Generator(device=data.device).manual_seed(seed+epoch)
        if recent and stage=='joint':
            n=int(.8*len(train))
            sample=torch.cat([newest[torch.randint(len(newest),(n,),generator=generator,device=data.device)],
                              older[torch.randint(len(older),(len(train)-n,),generator=generator,device=data.device)]])
        else:sample=train
        indices=sample.repeat(2);pub=torch.cat([torch.zeros(len(sample),device=data.device),torch.ones(len(sample),device=data.device)])
        order=torch.randperm(len(indices),generator=generator,device=data.device);loss_sum=0.;seen=0
        for macro in order.split(4096):
            optimizer.zero_grad(set_to_none=True)
            for positions in macro.split(1024):
                ix=indices[positions];y,regime=data.row_targets(ix)
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(data.batch(ix),pub[positions],y[:,0])
                loss=training_loss(out,y,regime,data.target_scale[data.zone[ix]],gated=True)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite curriculum loss')
                (loss*(len(ix)/len(macro))).backward();loss_sum+=float(loss.detach())*len(ix);seen+=len(ix)
            if stage=='warmup':assert all(p.grad is None for p in model.base.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        metrics=selection_score(model,data)
        row=dict(model=name,seed=seed,epoch=epoch,stage=stage,train_loss=loss_sum/seen,**metrics,
            recent_2023_fraction=float((year[sample]==2023).float().mean()),selection_precision='FP32',seconds=time.perf_counter()-start,
            effective_batch=4096,microbatch=1024,split_hash=data.split_hash)
        logs.append(row)
        if metrics['price_score']<best-1e-6:
            best=metrics['price_score'];best_epoch=epoch;stale=0;torch.save(payload(epoch,stage,row),checkpoint)
        elif stage=='joint':stale+=1
        if stage=='warmup' and metrics['price_score']<warm_best-1e-6:
            warm_best=metrics['price_score'];torch.save(payload(epoch,stage,row),warm_path)
        pd.DataFrame(logs).to_csv(ROOT/'logs'/f'{name}_training.csv',index=False)
        print(json.dumps(dict(**row,best_epoch=best_epoch,stale=stale)),flush=True)
        if stage=='joint' and stale>=12:break
    complete.write_text(json.dumps(dict(model=name,seed=seed,best_epoch=best_epoch,best_price_score=best,
        initial_price_score=initial['price_score'],epochs_run=epoch,seconds=time.perf_counter()-start,split_hash=data.split_hash,status='complete',
        temporal_world=temporal,recent_replay=recent,source_warmup_reused=recent),indent=2))
    print('CANDIDATE TRAINING COMPLETE',name,flush=True)

if __name__=='__main__':main()
