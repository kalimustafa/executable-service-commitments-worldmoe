"""Read-only frozen R2 price forecasts for the full local 2025 calendar."""
from pathlib import Path
import os,sys,json,time,hashlib
os.environ.update(PYTHONUTF8='1',CUDA_LAUNCH_BLOCKING='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT.parent;REPO=ROOT.parents[1]/'power-dc-vpp';sys.path.insert(0,str(REPO/'src'))
import numpy as np,pandas as pd,torch
from r2_data import R2Data,WORLD_FIELDS,ZONES
from calibrate_r2 import load_model,latent_rows,predict_calibrated,forecast_metrics,NativeCalibration

class R2EvaluationData(R2Data):
    def __init__(self,device='cuda'):
        freeze=json.loads((ROOT/'EVALUATION_FREEZE.json').read_text(encoding='utf-8'))
        assert freeze['post_evaluation_parameter_changes_allowed'] is False
        self.device=torch.device(device);normalization=json.loads((ROOT/'cache/r2_normalization.json').read_text());self.normalization=normalization
        self.frame=pd.read_parquet(PAPER/'revision_20260918/forecast_2025/gate_hourly_2025_with_context.parquet')
        self.frame=self.frame[self.frame.date.between('2025-01-01','2025-12-31')].sort_values(['timestamp_utc','zone']).reset_index(drop=True)
        assert len(self.frame)==35040 and not self.frame.duplicated(['timestamp_utc','zone']).any()
        self.frame['split']='evaluation_2025'
        for zone in ZONES:self.frame['zone_'+zone]=self.frame.zone.eq(zone).astype(float)
        naive=pd.to_datetime(self.frame.date)-pd.Timedelta(days=1)+pd.Timedelta(hours=4,minutes=55)
        issue=naive.dt.tz_localize('America/New_York').dt.tz_convert('UTC')
        assert (pd.to_datetime(self.frame.feature_last_interval_end_utc,utc=True)<issue).all()
        self.frame['lead_hours']=(pd.to_datetime(self.frame.timestamp_utc,utc=True)-issue).dt.total_seconds()/3600
        self.names=normalization['feature_names'];x=self.frame[self.names].to_numpy(np.float32)
        center=np.asarray(normalization['feature_center']);scale=np.asarray(normalization['feature_scale'])
        x=np.clip((np.where(np.isfinite(x),x,center)-center)/scale,-normalization['feature_clip'],normalization['feature_clip'])
        self.numerical=torch.tensor(x,dtype=torch.float32,device=self.device)
        self.zone=torch.tensor(pd.Categorical(self.frame.zone,categories=ZONES).codes,dtype=torch.long,device=self.device)
        self.labels=torch.tensor(self.frame[['da_lbmp','spread','rt_integrated']].to_numpy(),dtype=torch.float32,device=self.device)
        self.target_center=torch.tensor(normalization['target_center'],dtype=torch.float32,device=self.device)
        self.target_scale=torch.tensor(normalization['target_scale'],dtype=torch.float32,device=self.device)
        self.regime_edges=np.asarray(normalization['regime_edges'])
        manifest=json.loads((ROOT/'cache/evaluation_cache_manifest.json').read_text());assert manifest['status']=='complete'
        shards=[np.load(ROOT/'cache'/name,allow_pickle=False) for name in manifest['shards']]
        dates=np.concatenate([s['dates'] for s in shards]);lookup={str(v):i+1 for i,v in enumerate(dates)}
        self.world_index=torch.tensor([lookup.get(str(d),0) for d in self.frame.date],dtype=torch.long,device=self.device)
        self.world_available=(self.world_index>0).float();self.world={}
        for key in WORLD_FIELDS:
            values=np.concatenate([s[key] for s in shards],axis=0);assert np.isfinite(values).all()
            values=np.concatenate([np.zeros_like(values[:1]),values],axis=0)
            self.world[key]=torch.tensor(values,dtype=torch.float16,device=self.device)
        self.indices={'evaluation_2025':torch.arange(len(self.frame),device=self.device)}

def baseline_metrics(data):
    f=pd.read_parquet(PAPER/'revision_20260918/forecast_2025/predictions_2025.parquet')
    f=f[f.model.isin(['Seasonal_quantiles','Ridge_residual_quantiles','RASC'])&f.date.between('2025-01-01','2025-12-31')].copy()
    rows=[]
    for model,g in f.groupby('model'):
        g['split']='evaluation_2025';g['gate']='post_DA'
        q=g[[f'q{x:.2f}' for x in [.05,.1,.25,.5,.75,.9,.95]]].to_numpy()
        for j,tau in enumerate([.05,.1,.25,.5,.75,.9,.95]):
            g[f'dart_q{tau:.2f}']=q[:,j];g[f'rt_q{tau:.2f}']=g.da_lbmp.to_numpy()-q[:,6-j];g[f'da_q{tau:.2f}']=g.da_lbmp
        zones=pd.Categorical(g.zone,categories=ZONES).codes
        g['tail_threshold']=data.regime_edges[zones,1];g['tail_probability']=np.nan
        rows+=forecast_metrics(g,model)
    return rows

def main():
    start=time.perf_counter();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    freeze=json.loads((ROOT/'EVALUATION_FREEZE.json').read_text(encoding='utf-8'))
    # Hash only the small selected price artifacts, not the large backbone files.
    for name,item in freeze['files'].items():assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
    data=R2EvaluationData();outdir=ROOT/'forecast_2025';outdir.mkdir(exist_ok=True);metrics=[];checks={}
    for variant,kinds in [('full',['raw','native','external']),('no_world',['native'])]:
        model,_=load_model(variant,data);pack=latent_rows(model,data,data.indices['evaluation_2025'])
        native=NativeCalibration().to(data.device);native.load_state_dict(torch.load(ROOT/'checkpoints'/f'r2_{variant}_native_calibration.pt',map_location=data.device,weights_only=True));native.eval().requires_grad_(False)
        external=json.loads((ROOT/'checkpoints'/f'r2_{variant}_external_calibration.json').read_text())
        for kind in kinds:
            cal={'raw':None,'native':native,'external':external}[kind]
            frame,identity=predict_calibrated(model,data,pack,kind,cal);name=f'R2_{variant}_{kind}';frame['model']=name
            frame['slot_score_source']='RT_median_no_separate_rank_head';frame.to_parquet(outdir/(name+'.parquet'),index=False)
            metrics+=forecast_metrics(frame,name);checks[name]=identity;print(name,len(frame),'identity',identity,flush=True)
        del model,pack,native;torch.cuda.empty_cache()
    metrics+=baseline_metrics(data);pd.DataFrame(metrics).to_csv(ROOT/'results/R2_forecast_metrics.csv',index=False)
    result=dict(status='complete',forecast_hour_zone_rows=35040,gates=2,models=list(checks),identity_checks=checks,
        missing_world_context_rows=int((data.world_available==0).sum()),no_training_or_calibration_updates=True,seconds=time.perf_counter()-start)
    (outdir/'completed.json').write_text(json.dumps(result,indent=2));print('FORECAST COMPLETE',result,flush=True)
if __name__=='__main__':main()
