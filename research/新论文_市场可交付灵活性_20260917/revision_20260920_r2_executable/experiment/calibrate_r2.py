"""Compare 2024-only residual and native affine quantile calibration."""
from pathlib import Path
import os,sys,json,time
os.environ.update(PYTHONUTF8='1',CUDA_LAUNCH_BLOCKING='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT.parent;REPO=ROOT.parents[1]/'power-dc-vpp';sys.path.insert(0,str(REPO/'src'))
import numpy as np,pandas as pd,torch
from torch import nn
from r2_data import R2Data,ZONES
from dcvpp.vpp_world_moe.r2_market import R2MarketHead,R2Config,QUANTILES,pinball,identity_errors

class NativeCalibration(nn.Module):
    def __init__(self):
        super().__init__();self.da_shift=nn.Parameter(torch.zeros(4));self.da_width=nn.Parameter(torch.zeros(4))
        self.dart_shift=nn.Parameter(torch.zeros(2,4));self.dart_width=nn.Parameter(torch.zeros(2,4))
    def forward(self,da,dart,zone,gate,scales):
        z=zone.long();g=gate.long()
        a=da[:,3:4]+self.da_shift[z,None]*scales[z,0,None]+torch.exp(self.da_width[z,None].clamp(-2,2))*(da-da[:,3:4])
        d=dart[:,3:4]+self.dart_shift[g,z,None]*scales[z,1,None]+torch.exp(self.dart_width[g,z,None].clamp(-2,2))*(dart-dart[:,3:4])
        return a,d

def load_model(variant,data):
    p=ROOT/'checkpoints'/f'r2_{variant}_best.pt';x=torch.load(p,map_location='cpu',weights_only=False)
    m=R2MarketHead(R2Config(**x['model_config']),da_center=data.target_center[:,0],da_scale=data.target_scale[:,0],
        dart_center=data.target_center[:,1],dart_scale=data.target_scale[:,1]).to(data.device)
    m.load_state_dict(x['state_dict'],strict=True);m.eval().requires_grad_(False)
    return m,x

def latent_rows(model,data,indices):
    store={k:[] for k in ['latent','da','dart','zone','gate','anchor','indices']}
    with torch.inference_mode():
        for g in [0,1]:
            for ix in indices.split(4096):
                labels=data.labels[ix];gate=torch.full((len(ix),),g,device=data.device)
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(data.batch(ix),gate,labels[:,0])
                values=dict(latent=out['latent'].float(),da=out['da_raw'],dart=out['dart'],zone=data.zone[ix],gate=gate,anchor=labels[:,0],indices=ix)
                for k,v in values.items():store[k].append(v.detach().clone())
    # Clone outside inference_mode so calibrator autograd can save these inputs.
    return {k:torch.cat(v).clone() for k,v in store.items()}

def external_fit(pack,data):
    indices=pack['indices'].cpu().numpy();f=data.frame.iloc[indices]
    regime_cut=float(data.frame[data.frame.split.eq('train')].day_abs_mean7.median())
    reg=(f.day_abs_mean7.to_numpy()>regime_cut).astype(int);block=(f.hour.to_numpy()//6).astype(int)
    gates=pack['gate'].cpu().numpy().astype(int);zones=pack['zone'].cpu().numpy();y=data.labels[pack['indices']].cpu().numpy()
    result=dict(schema='r2.external_regime_residual.v1',regime_cut=regime_cut,shrinkage=84.,heads={})
    for head,column in [('da',0),('dart',1)]:
        pred=pack[head].cpu().numpy();state={}
        for gate in [0,1]:
            keep=gates==gate;res=y[keep,column,None]-pred[keep]
            global_q=np.array([np.quantile(res[:,j],tau) for j,tau in enumerate(QUANTILES)])
            groups={}
            for z in range(4):
                for regime in [0,1]:
                    for b in range(4):
                        mask=keep&(zones==z)&(reg==regime)&(block==b);n=int(mask.sum())
                        if n:
                            resid=y[mask,column,None]-pred[mask]
                            q=np.array([np.quantile(resid[:,j],tau) for j,tau in enumerate(QUANTILES)])
                            groups[f'{z}:{regime}:{b}']=(n/(n+84)*q+84/(n+84)*global_q).tolist()
            state[str(gate)]=dict(global_delta=global_q.tolist(),groups=groups)
        result['heads'][head]=state
    return result

def external_apply(pack,data,cal):
    f=data.frame.iloc[pack['indices'].cpu().numpy()];reg=(f.day_abs_mean7.to_numpy()>cal['regime_cut']).astype(int)
    block=(f.hour.to_numpy()//6).astype(int);z=pack['zone'].cpu().numpy();g=pack['gate'].cpu().numpy().astype(int)
    outputs=[]
    for head in ['da','dart']:
        delta=[]
        for i in range(len(f)):
            entry=cal['heads'][head][str(g[i])];delta.append(entry['groups'].get(f'{z[i]}:{reg[i]}:{block[i]}',entry['global_delta']))
        q=pack[head]+torch.tensor(np.asarray(delta),dtype=torch.float32,device=data.device)
        outputs.append(torch.sort(q,dim=-1).values)
    return outputs

def predict_calibrated(model,data,pack,kind,calibrator):
    frames=[];identity=[]
    if kind=='external':qa,qd=external_apply(pack,data,calibrator)
    elif kind=='native':
        with torch.no_grad():qa,qd=calibrator(pack['da'],pack['dart'],pack['zone'],pack['gate'],data.target_scale)
    else:qa,qd=pack['da'],pack['dart']
    with torch.inference_mode():
        for start in range(0,len(qa),4096):
            ix=slice(start,start+4096)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=model.from_latent(pack['latent'][ix],pack['zone'][ix],pack['gate'][ix],pack['anchor'][ix],da_quantiles=qa[ix],dart_quantiles=qd[ix])
            identity.append(identity_errors(out,pack['gate'][ix],pack['anchor'][ix]))
            ids=pack['indices'][ix].cpu().numpy()
            f=data.frame.iloc[ids][['timestamp_utc','date','hour','zone','split','da_lbmp','spread','rt_integrated','feature_last_interval_end_utc']].copy()
            f['gate']=np.where(pack['gate'][ix].cpu().numpy()>0,'post_DA','pre_DA')
            f['native_generation_utc']=(pd.to_datetime(f.date)-pd.Timedelta(days=1)+pd.Timedelta(hours=4,minutes=55)).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
            for head in ['da','dart','rt']:
                q=out[head].cpu().numpy()
                for j,tau in enumerate(QUANTILES):f[f'{head}_q{tau:.2f}']=q[:,j]
                f[head+'_point_mean']=out[head+'_mean'].cpu().numpy()
            f['tail_probability']=out['regime_logits'].softmax(-1)[:,2].cpu().numpy();f['rho']=out['rho'].cpu().numpy()
            f['slot_score']=out['slot_score'].cpu().numpy();f['tail_threshold']=np.array(data.regime_edges)[pack['zone'][ix].cpu().numpy(),1]
            frames.append(f)
    maximum={k:max(row[k] for row in identity) for k in identity[0]}
    assert max(maximum.values())<1e-3
    return pd.concat(frames,ignore_index=True),maximum

def forecast_metrics(frame,model):
    rows=[]
    for (split,gate),f in frame.groupby(['split','gate']):
        for head,target in [('da','da_lbmp'),('dart','spread'),('rt','rt_integrated')]:
            q=f[[f'{head}_q{t:.2f}' for t in QUANTILES]].to_numpy();y=f[target].to_numpy();e=y[:,None]-q
            loss=np.maximum(np.asarray(QUANTILES)*e,(np.asarray(QUANTILES)-1)*e).mean()
            row=dict(model=model,split=split,gate=gate,target=head,rows=len(f),mae=float(np.abs(y-q[:,3]).mean()),rmse=float(np.sqrt(np.square(y-q[:,3]).mean())),pinball=float(loss))
            tail=np.abs(f.spread.to_numpy())>f.tail_threshold.to_numpy()
            for nominal,lo,hi in [(50,2,4),(80,1,5),(90,0,6)]:
                inside=(y>=q[:,lo])&(y<=q[:,hi]);row[f'coverage{nominal}']=float(inside.mean());row[f'width{nominal}']=float((q[:,hi]-q[:,lo]).mean());row[f'tail_coverage{nominal}']=float(inside[tail].mean())
            row['DART_sign_accuracy']=float((np.sign(q[:,3])==np.sign(y)).mean()) if head=='dart' else None
            row['tail_brier']=float(np.square(f.tail_probability.to_numpy()-tail.astype(float)).mean())
            rows.append(row)
    return rows

def main():
    started=time.perf_counter();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);data=R2Data();allmetrics=[];report={}
    output=ROOT/'forecast_development';output.mkdir(exist_ok=True)
    protocol=dict(native_affine_parameters=24,native_epochs=250,native_learning_rate=.02,native_width_factor_bounds=[float(np.exp(-2)),float(np.exp(2))],
        fitting_split='2024-04-01 to 2024-06-29',evaluation_split='2024-07-01 to 2024-12-31',external_shrinkage=84,post_DA_complementary_quantile_identity=True,
        target_2025_opened=False)
    (ROOT/'calibration_config.json').write_text(json.dumps(protocol,indent=2))
    for variant in ['full','no_world']:
        model,checkpoint=load_model(variant,data);calpack=latent_rows(model,data,data.indices['calibration'])
        native=NativeCalibration().to(data.device);optimizer=torch.optim.Adam(native.parameters(),lr=.02)
        y=data.labels[calpack['indices']];scale=data.target_scale[calpack['zone']];logs=[]
        for epoch in range(250):
            optimizer.zero_grad(set_to_none=True);da,dart=native(calpack['da'],calpack['dart'],calpack['zone'],calpack['gate'],data.target_scale)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model.from_latent(calpack['latent'],calpack['zone'],calpack['gate'],calpack['anchor'],da_quantiles=da,dart_quantiles=dart)
            loss=(.5*pinball(out['da'],y[:,0])/scale[:,0]+pinball(out['dart'],y[:,1])/scale[:,1]+.5*pinball(out['rt'],y[:,2])/scale[:,2]).mean()
            loss.backward();optimizer.step()
            if epoch%25==0:logs.append(dict(epoch=epoch,loss=float(loss.detach())))
        native.eval().requires_grad_(False);torch.save(native.state_dict(),ROOT/'checkpoints'/f'r2_{variant}_native_calibration.pt')
        external=external_fit(calpack,data);(ROOT/'checkpoints'/f'r2_{variant}_external_calibration.json').write_text(json.dumps(external,indent=2))
        evalindices=torch.cat((data.indices['calibration'],data.indices['exploratory_test']))
        pack=latent_rows(model,data,evalindices);report[variant]=dict(native_training=logs,identity={})
        for kind,calibrator in [('raw',None),('native',native),('external',external)]:
            f,identity=predict_calibrated(model,data,pack,kind,calibrator);name=f'R2_{variant}_{kind}';f['model']=name
            f.to_parquet(output/(name+'.parquet'),index=False);allmetrics+=forecast_metrics(f,name);report[variant]['identity'][kind]=identity
            print(name,'forecasts',len(f),'identity',identity,flush=True)
        del model,pack,calpack,native,optimizer;torch.cuda.empty_cache()
    metrics=pd.DataFrame(allmetrics);metrics.to_csv(ROOT/'results/r2_forecast_metrics_2024.csv',index=False)
    metrics[metrics.split.eq('calibration')].to_csv(ROOT/'results/r2_calibration_metrics.csv',index=False)
    (ROOT/'results/calibration_completed.json').write_text(json.dumps(dict(status='complete',seconds=time.perf_counter()-started,details=report,target_2025_opened=False),indent=2))
    print('CALIBRATION COMPLETE',time.perf_counter()-started,flush=True)
if __name__=='__main__':main()
