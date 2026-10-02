from src.runtime import ROOT
from src.data.access_guard import install_guard,assert_target_year,assert_years
install_guard()
import argparse
import os
import json
import time
import numpy as np
import pandas as pd
import torch
from src.data.r2_data import R2Data
from src.models.factory import make_model
from src.calibration.native_affine import NativeCalibration
from src.training.losses import price_losses
from src.evaluation.forecast_metrics import evaluate_frame
from vendor.r2_market import QUANTILES,identity_errors

EVALUATION_PROTOCOL='fp32_v1'
INFERENCE_BATCH=int(os.environ.get('R2WGR_EVAL_BATCH','4096'))

def load_model(name,seed,data):
    path=ROOT/'checkpoints'/f'{name}_seed{seed}_best.ckpt'
    cp=torch.load(path,map_location='cpu',weights_only=False)
    assert cp['split_hash']==data.split_hash
    if name=='C6_CausalErrorMemory':
        from src.models.causal_error_memory import R2CausalErrorMemory
        source,_=load_model('C0_Repaired',seed,data)
        model=R2CausalErrorMemory(source,memory_features=cp['memory_features']).to(data.device)
    elif name=='C2_OrderedResidualTrees':
        from src.models.ordered_residual_trees import R2OrderedResidualTrees
        source,_=load_model('C0_Repaired',seed,data)
        model=R2OrderedResidualTrees(source).to(data.device)
        model.trees=cp['trees'];model.learning_rate=cp['learning_rate']
    else:model=make_model(name,seed,data)
    model.load_state_dict(cp['state_dict'],strict=True)
    model.eval().requires_grad_(False)
    return model,cp

def latent_rows(model,data,indices):
    keys=['latent','da','dart','zone','publication','anchor','indices','mean_gate','max_gate','gate_groups','expert_attention','router_entropy','regime_probs']
    store={key:[] for key in keys}
    with torch.inference_mode():
        for state in [0.,1.]:
            for ix in indices.split(INFERENCE_BATCH):
                y=data.labels[ix];publication=torch.full((len(ix),),state,device=data.device)
                with torch.autocast('cuda',enabled=False):out=model(data.batch(ix),publication,y[:,0])
                gate=out.get('world_gate',torch.zeros(len(ix),128,device=data.device))
                values=dict(latent=out['latent'].float(),da=out['da_raw'],dart=out['dart'],zone=data.zone[ix],publication=publication,
                    anchor=y[:,0],indices=ix,mean_gate=gate.mean(-1),max_gate=gate.max(-1).values,gate_groups=gate.reshape(-1,8,16).mean(-1),
                    expert_attention=out.get('expert_attention',torch.zeros(len(ix),7,device=data.device)),
                    router_entropy=out.get('router_entropy',torch.zeros(len(ix),device=data.device)),regime_probs=out['regime_logits'].softmax(-1))
                for k,v in values.items():store[k].append(v.detach().clone())
    return {k:torch.cat(v).clone() for k,v in store.items()}

def fit_calibration(model,data,pack):
    native=NativeCalibration().to(data.device);optimizer=torch.optim.Adam(native.parameters(),lr=.02)
    y=data.labels[pack['indices']];scales=data.target_scale[pack['zone']];logs=[]
    for epoch in range(250):
        optimizer.zero_grad(set_to_none=True)
        da,dart=native(pack['da'],pack['dart'],pack['zone'],pack['publication'],data.target_scale)
        with torch.autocast('cuda',enabled=False):
            out=model.from_latent(pack['latent'],pack['zone'],pack['publication'],pack['anchor'],da_quantiles=da,dart_quantiles=dart)
        _,price=price_losses(out,y,scales);loss=price.mean();loss.backward();optimizer.step()
        if epoch%25==0:logs.append(dict(epoch=epoch,loss=float(loss.detach())))
    native.eval().requires_grad_(False)
    return native,logs

def predict_pack(model,data,pack,calibrator=None):
    with torch.no_grad():
        if calibrator is None:qa,qd=pack['da'],pack['dart']
        else:qa,qd=calibrator(pack['da'],pack['dart'],pack['zone'],pack['publication'],data.target_scale)
    frames=[];max_error=0.
    with torch.inference_mode():
        for start in range(0,len(qa),4096):
            sl=slice(start,start+4096)
            with torch.autocast('cuda',enabled=False):out=model.from_latent(pack['latent'][sl],pack['zone'][sl],pack['publication'][sl],pack['anchor'][sl],da_quantiles=qa[sl],dart_quantiles=qd[sl])
            max_error=max(max_error,max(identity_errors(out,pack['publication'][sl],pack['anchor'][sl]).values()))
            ids=pack['indices'][sl].cpu().numpy()
            f=data.frame.iloc[ids][['timestamp_utc','date','hour','zone','split','da_lbmp','spread','rt_integrated','lead_hours','day_std_mean7','feature_last_interval_end_utc']].copy()
            f['gate']=np.where(pack['publication'][sl].cpu().numpy()>0,'post_DA','pre_DA')
            f['world_available']=data.world_available[pack['indices'][sl]].cpu().numpy()
            f['market_regime']=data.regime_target[pack['indices'][sl]].cpu().numpy();f['tail_event']=f.market_regime==2
            for head in ['da','dart','rt']:
                q=out[head].cpu().numpy()
                for j,tau in enumerate(QUANTILES):f[f'{head}_q{tau:.2f}']=q[:,j]
                f[head+'_point_mean']=out[head+'_mean'].cpu().numpy()
            for j in range(3):f[f'regime_p{j}']=pack['regime_probs'][sl,j].cpu().numpy()
            f['mean_gate']=pack['mean_gate'][sl].cpu().numpy();f['max_gate']=pack['max_gate'][sl].cpu().numpy()
            f['router_entropy']=pack['router_entropy'][sl].cpu().numpy();f['rho']=out['rho'].cpu().numpy()
            for j in range(8):f[f'gate_group_{j}']=pack['gate_groups'][sl,j].cpu().numpy()
            for j in range(7):f[f'expert_attention_{j}']=pack['expert_attention'][sl,j].cpu().numpy()
            frames.append(f)
    assert max_error<1e-3
    frame=pd.concat(frames,ignore_index=True);assert_years(frame.date,'predictions')
    return frame,max_error

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',default=['C0_Repaired','C1_ProfileResidual'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[20260920]);parser.add_argument('--target-year',type=int,default=2024)
    args=parser.parse_args();assert_target_year(args.target_year)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assert args.seeds==[20260920],'Primary pilot only'
    data=R2Data(multiscale=any(x.startswith('C') for x in args.models),temporal=any(x in {'C3_TemporalWorld','C5_TemporalReplay'} for x in args.models),memory='C6_CausalErrorMemory' in args.models)
    output=ROOT/'results/predictions_2024';output.mkdir(exist_ok=True)
    evalindices=torch.cat([data.indices[s] for s in ['selection','calibration','exploratory_test']])
    for name in args.models:
        for seed in args.seeds:
            done=ROOT/'logs'/f'{name}_seed{seed}_calibration_complete.json'
            if done.exists() and json.loads(done.read_text()).get('evaluation_protocol')==EVALUATION_PROTOCOL:continue
            started=time.perf_counter();model,cp=load_model(name,seed,data)
            calpack=latent_rows(model,data,data.indices['calibration']);native,logs=fit_calibration(model,data,calpack)
            calstate={k:v.cpu().tolist() for k,v in native.state_dict().items()}
            (ROOT/'checkpoints'/f'{name}_seed{seed}_calibration.json').write_text(json.dumps(dict(parameters=calstate,epochs=250,fit_rows=len(calpack['indices']),fit_dates=['2024-04-01','2024-06-29'],logs=logs),indent=2))
            torch.save(native.state_dict(),ROOT/'checkpoints'/f'{name}_seed{seed}_calibration.pt')
            pack=latent_rows(model,data,evalindices);metrics=[]
            for kind,cal in [('raw',None),('native',native)]:
                frame,error=predict_pack(model,data,pack,cal)
                frame.to_parquet(output/f'{name}_seed{seed}_{kind}.parquet',index=False)
                metrics+=evaluate_frame(frame,name,seed,kind)
            pd.DataFrame(metrics).to_csv(ROOT/'results'/f'metrics_{name}_seed{seed}.csv',index=False)
            done.write_text(json.dumps(dict(model=name,seed=seed,selected_epoch=cp['epoch'],status='complete',seconds=time.perf_counter()-started,identity_max_error=error,calibration_parameters=24,evaluation_protocol=EVALUATION_PROTOCOL),indent=2))
            print(name,seed,'CALIBRATION COMPLETE',round(time.perf_counter()-started,2),flush=True)
            del model,native,calpack,pack;torch.cuda.empty_cache()
    tables=[pd.read_csv(p) for p in sorted((ROOT/'results').glob('metrics_C*_seed*.csv'))]
    metrics=pd.concat(tables,ignore_index=True);metrics.to_csv(ROOT/'results/forecast_metrics_2024.csv',index=False)
    metrics[metrics.split.eq('calibration')].to_csv(ROOT/'results/calibration_metrics_2024.csv',index=False)

if __name__=='__main__':main()
