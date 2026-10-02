"""Price rows and frozen context for the small R2 model, with explicit splits."""
from pathlib import Path
import json
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT.parent
ZONES=['WEST','CAPITL','N.Y.C.','LONGIL']
WORLD_FIELDS=['w0_local','w1_local','w1_global','expert_bank','expert_prior','tail_context','tail_probability','regime','log_scale']

class R2Data:
    def __init__(self,device='cuda'):
        self.device=torch.device(device)
        self.frame=pd.read_parquet(PAPER/'experiments/gate_hourly_panel.parquet').sort_values(['timestamp_utc','zone']).reset_index(drop=True)
        assert self.frame.date.max()<='2024-12-31'
        names=json.loads((PAPER/'revision_20260918/forecast_2025/feature_reconstruction_check.json').read_text())['frozen_feature_names']
        for zone in ZONES:self.frame['zone_'+zone]=self.frame.zone.eq(zone).astype(float)
        self.names=names+['lead_hours']
        naive=pd.to_datetime(self.frame.date)-pd.Timedelta(days=1)+pd.Timedelta(hours=4,minutes=55)
        issue=naive.dt.tz_localize('America/New_York').dt.tz_convert('UTC')
        assert (pd.to_datetime(self.frame.feature_last_interval_end_utc,utc=True)<issue).all()
        self.frame['lead_hours']=(pd.to_datetime(self.frame.timestamp_utc,utc=True)-issue).dt.total_seconds()/3600
        train=self.frame.split.eq('train').to_numpy();features=self.frame[self.names].to_numpy(np.float32)
        center=np.nanmedian(features[train],axis=0);scale=np.nanquantile(features[train],.75,axis=0)-np.nanquantile(features[train],.25,axis=0)
        scale=np.where(scale>1e-3,scale,np.nanstd(features[train],axis=0));scale=np.maximum(scale,1e-3)
        features=np.clip((np.where(np.isfinite(features),features,center)-center)/scale,-20,20)
        self.numerical=torch.as_tensor(features,dtype=torch.float32,device=self.device)
        self.zone=torch.tensor(pd.Categorical(self.frame.zone,categories=ZONES).codes,dtype=torch.long,device=self.device)
        self.labels=torch.tensor(self.frame[['da_lbmp','spread','rt_integrated']].to_numpy(),dtype=torch.float32,device=self.device)
        self.indices={split:torch.tensor(np.flatnonzero(self.frame.split.eq(split)),dtype=torch.long,device=self.device) for split in self.frame.split.unique()}
        self.target_center=[];self.target_scale=[];self.regime_edges=[]
        for z in ZONES:
            y=self.frame[self.frame.split.eq('train')&self.frame.zone.eq(z)][['da_lbmp','spread','rt_integrated']].to_numpy()
            self.target_center.append(np.median(y,axis=0));self.target_scale.append(np.maximum((np.quantile(y,.75,axis=0)-np.quantile(y,.25,axis=0))/1.349,5.))
            self.regime_edges.append(np.quantile(np.abs(y[:,1]),[.5,.9]))
        self.target_center=torch.tensor(np.array(self.target_center),dtype=torch.float32,device=self.device)
        self.target_scale=torch.tensor(np.array(self.target_scale),dtype=torch.float32,device=self.device)
        edges=torch.tensor(np.array(self.regime_edges),dtype=torch.float32,device=self.device)[self.zone]
        self.regime_target=(self.labels[:,1,None].abs()>edges).sum(-1)
        manifest=json.loads((ROOT/'cache/development_cache_manifest.json').read_text(encoding='utf-8'))
        assert manifest['status']=='complete' and manifest['last_date']<'2025-01-01'
        shards=[np.load(ROOT/'cache'/name,allow_pickle=False) for name in manifest['shards']]
        dates=np.concatenate([s['dates'] for s in shards]);lookup={str(v):i+1 for i,v in enumerate(dates)}
        self.world_index=torch.tensor([lookup.get(str(d),0) for d in self.frame.date],dtype=torch.long,device=self.device)
        self.world_available=(self.world_index>0).float();self.world={}
        for key in WORLD_FIELDS:
            values=np.concatenate([s[key] for s in shards],axis=0)
            assert np.isfinite(values).all()
            values=np.concatenate([np.zeros_like(values[:1]),values],axis=0)
            self.world[key]=torch.tensor(values,dtype=torch.float16,device=self.device)
        self.normalization=dict(feature_names=self.names,feature_center=center.tolist(),feature_scale=scale.tolist(),feature_clip=20.,
            target_center=self.target_center.cpu().tolist(),target_scale=self.target_scale.cpu().tolist(),regime_edges=np.array(self.regime_edges).tolist(),
            zones=ZONES,fit_split='2021-2023 train',representation_years=[2022,2023,2024],missing_world_policy='explicit zero context and unavailable-world mask',
            representation_summary='daily issued context; forecast target calendar and lead are supplied separately')
        (ROOT/'cache/r2_normalization.json').write_text(json.dumps(self.normalization,indent=2))
        rows=self.frame[['date','split']].copy();rows['available']=self.world_available.cpu().numpy();rows['year']=rows.date.str[:4]
        (ROOT/'cache/r2_data_manifest.json').write_text(json.dumps(dict(rows=len(self.frame),price_years=[2021,2022,2023,2024],
            split_rows={k:len(v) for k,v in self.indices.items()},world_availability=rows.groupby('year').available.agg(['sum','count']).reset_index().to_dict('records'),
            train_label_max=self.frame.loc[train,'date'].max(),target_2025_opened=False),indent=2))

    def batch(self,indices):
        world_index=self.world_index[indices];zone=self.zone[indices]
        out={key:(value[world_index,zone] if key in ['w0_local','w1_local'] else value[world_index]) for key,value in self.world.items()}
        out.update(numerical=self.numerical[indices],zone=zone,world_available=self.world_available[indices])
        return out

    def row_targets(self,indices):return self.labels[indices],self.regime_target[indices]
