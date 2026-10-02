"""Train only the compact R2 head from cached frozen features."""
from pathlib import Path
import os,sys,json,time,csv,random
os.environ.update(PYTHONUTF8='1',CUDA_LAUNCH_BLOCKING='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]/'power-dc-vpp';sys.path.insert(0,str(REPO/'src'))
import numpy as np,torch
import torch.nn.functional as F
from r2_data import R2Data
from dcvpp.vpp_world_moe.r2_market import R2MarketHead,R2Config,pinball,identity_errors

CONFIG=dict(seed=20260920,batch_size=4096,epochs=100,patience=12,learning_rate=.002,weight_decay=.0001,
    loss_weights=dict(da=.5,dart=1.,rt=.5,regime=.05),variants=['full','no_world'],
    training='2021-2023 price rows, explicitly masked unavailable 2021 world representations',
    early_stopping='2024 selection split, derived RT plus marginal DA/DART pinball; not MAE',
    calibration='2024 April-June only, performed after this stage',
    final_choice='2024 post-calibration economic comparison before 2025',
    future_2025_labels_allowed=False,backbone_backward=False,slot_ranking_stage='separate optional stage')

def score(model,data,indices):
    model.eval();sums=torch.zeros(4,device=data.device);count=0
    with torch.inference_mode():
        for gate_value in [0.,1.]:
            for ix in indices.split(CONFIG['batch_size']):
                labels,regime=data.row_targets(ix);zone=data.zone[ix];gate=torch.full((len(ix),),gate_value,device=data.device)
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(data.batch(ix),gate,labels[:,0])
                da=pinball(out['da'],labels[:,0])/data.target_scale[zone,0]
                dart=pinball(out['dart'],labels[:,1])/data.target_scale[zone,1]
                rt=pinball(out['rt'],labels[:,2])/data.target_scale[zone,2]
                sums+=torch.stack((da.sum(),dart.sum(),rt.sum(),(.5*da+dart+.5*rt).sum()));count+=len(ix)
    return (sums/count).cpu().tolist()

def main():
    started=time.perf_counter();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    data=R2Data();(ROOT/'training_config.json').write_text(json.dumps(CONFIG,indent=2))
    folder=ROOT/'checkpoints';folder.mkdir(exist_ok=True)
    train=data.indices['train'];validation=data.indices['selection'];results={}
    for variant in CONFIG['variants']:
        seed=CONFIG['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        config=R2Config(numerical_features=len(data.names),use_world=variant=='full',seed=seed)
        model=R2MarketHead(config,da_center=data.target_center[:,0],da_scale=data.target_scale[:,0],
            dart_center=data.target_center[:,1],dart_scale=data.target_scale[:,1]).to(data.device)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
        assert all(any(p is q for q in model.parameters()) for g in optimizer.param_groups for p in g['params'])
        param_count=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('START',variant,'R2 parameters',param_count,'trainable',trainable,'W0 trainable 0; W1 trainable 0',flush=True)
        probe=train[:32];y,_=data.row_targets(probe);g=torch.cat((torch.zeros(16,device='cuda'),torch.ones(16,device='cuda')))
        model.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            a=model(data.batch(probe),g,y[:,0]);b=model(data.batch(probe),g,y[:,0]+1000)
        assert torch.equal(a['rt'][g==0],b['rt'][g==0])
        identities=identity_errors(a,g,y[:,0]);assert max(identities.values())<1e-3
        best=float('inf');best_epoch=0;stale=0;logs=[];start=time.perf_counter()
        for epoch in range(1,CONFIG['epochs']+1):
            model.train();indices=train.repeat(2);gate=torch.cat((torch.zeros(len(train),device='cuda'),torch.ones(len(train),device='cuda')))
            order=torch.randperm(len(indices),device='cuda');total=torch.zeros((),device='cuda');seen=0
            for positions in order.split(CONFIG['batch_size']):
                ix=indices[positions];g=gate[positions];labels,regime=data.row_targets(ix);zone=data.zone[ix]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(data.batch(ix),g,labels[:,0])
                loss=(.5*pinball(out['da'],labels[:,0])/data.target_scale[zone,0]
                    +pinball(out['dart'],labels[:,1])/data.target_scale[zone,1]
                    +.5*pinball(out['rt'],labels[:,2])/data.target_scale[zone,2]
                    +.05*F.cross_entropy(out['regime_logits'],regime,reduction='none')).mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                total+=loss.detach()*len(ix);seen+=len(ix)
            metrics=score(model,data,validation);row=dict(variant=variant,epoch=epoch,train_loss=float(total/seen),
                val_da_pinball_normalized=metrics[0],val_dart_pinball_normalized=metrics[1],val_rt_pinball_normalized=metrics[2],
                val_price_score=metrics[3],seconds=time.perf_counter()-start)
            logs.append(row)
            if metrics[3]<best-1e-6:
                best=metrics[3];best_epoch=epoch;stale=0
                torch.save(dict(schema='r2.market-head.v1',model_config=model.configuration(),state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    variant=variant,epoch=epoch,validation=row,normalization=data.normalization,
                    selected_world=json.loads((ROOT/'selected_w0_w1.json').read_text(encoding='utf-8')),
                    trainable_W0=0,trainable_W1=0,trainable_R2=trainable),folder/f'r2_{variant}_best.pt')
            else:stale+=1
            with (ROOT/'results'/f'training_{variant}.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(logs)
            if epoch==1 or epoch%5==0:print(json.dumps(dict(**row,best_epoch=best_epoch,stale=stale)),flush=True)
            if stale>=CONFIG['patience']:break
        results[variant]=dict(best_epoch=best_epoch,best_price_score=best,epochs_run=epoch,parameters=param_count,trainable_R2=trainable,seconds=time.perf_counter()-start,identity_checks=identities)
        del model,optimizer;torch.cuda.empty_cache()
    freeze=json.loads((ROOT/'frozen_parameter_report.json').read_text());freeze['trainable_R2_parameters']=results['full']['trainable_R2'];freeze['R2_total_parameters']=results['full']['parameters']
    (ROOT/'frozen_parameter_report.json').write_text(json.dumps(freeze,indent=2))
    (ROOT/'results/training_completed.json').write_text(json.dumps(dict(status='complete',seconds=time.perf_counter()-started,variants=results,
        target_2025_opened=False,price_trunk_frozen_for_calibration=True),indent=2));print('TRAINING COMPLETE',results,flush=True)
if __name__=='__main__':main()
