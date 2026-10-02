"""Final-year rows transformed exclusively with frozen development statistics."""
from src.runtime import ROOT,PAPER
from src.data.r2_data import R2Data,WORLD_FIELDS,ZONES
from src.evaluation.final_freeze import verify_freeze,digest
import json
import numpy as np
import pandas as pd
import torch

class R2FinalData(R2Data):
    def __init__(self,device='cuda'):
        verify_freeze();self.device=torch.device(device)
        self.normalization=json.loads((ROOT/'cache/r2_normalization.json').read_text())
        normal=self.normalization;self.names=normal['feature_names']
        path=PAPER/'revision_20260918/forecast_2025/gate_hourly_2025_with_context.parquet'
        frame=pd.read_parquet(path)
        self.frame=frame[frame.date.between('2025-01-01','2025-12-31')].sort_values(['timestamp_utc','zone']).reset_index(drop=True)
        assert len(self.frame)==35040 and not self.frame.duplicated(['timestamp_utc','zone']).any()
        self.frame['split']='evaluation_2025'
        for z in ZONES:self.frame['zone_'+z]=self.frame.zone.eq(z).astype(float)
        naive=pd.to_datetime(self.frame.date)-pd.Timedelta(days=1)+pd.Timedelta(hours=4,minutes=55)
        issue=naive.dt.tz_localize('America/New_York').dt.tz_convert('UTC')
        assert (pd.to_datetime(self.frame.feature_last_interval_end_utc,utc=True)<issue).all()
        self.frame['lead_hours']=(pd.to_datetime(self.frame.timestamp_utc,utc=True)-issue).dt.total_seconds()/3600
        x=self.frame[self.names].to_numpy(np.float32)
        center=np.asarray(normal['feature_center']);scale=np.asarray(normal['feature_scale'])
        x=np.clip((np.where(np.isfinite(x),x,center)-center)/scale,-normal['feature_clip'],normal['feature_clip'])
        self.numerical=torch.tensor(x,dtype=torch.float32,device=self.device)
        self.zone=torch.tensor(pd.Categorical(self.frame.zone,categories=ZONES).codes,dtype=torch.long,device=self.device)
        self.labels=torch.tensor(self.frame[['da_lbmp','spread','rt_integrated']].to_numpy(),dtype=torch.float32,device=self.device)
        self.lead=torch.tensor(self.frame.lead_hours.to_numpy(),dtype=torch.float32,device=self.device)
        self.utc_offset=torch.tensor(((pd.to_datetime(self.frame.timestamp_utc,utc=True)-pd.to_datetime(self.frame.date,utc=True)).dt.total_seconds()/3600).to_numpy(),dtype=torch.float32,device=self.device)
        self.target_center=torch.tensor(normal['target_center'],dtype=torch.float32,device=self.device)
        self.target_scale=torch.tensor(normal['target_scale'],dtype=torch.float32,device=self.device)
        self.regime_edges=np.asarray(normal['regime_edges'])
        edges=torch.tensor(self.regime_edges,dtype=torch.float32,device=self.device)[self.zone]
        self.regime_target=(self.labels[:,1,None].abs()>edges).sum(-1)
        groups=self.frame.groupby(['date','zone'],sort=False).indices
        index=np.zeros((len(groups),25),np.int64);mask=np.zeros((len(groups),25),bool)
        row_group=np.empty(len(self.frame),np.int64);row_position=np.empty(len(self.frame),np.int64)
        for gi,ids in enumerate(groups.values()):
            assert 1<=len(ids)<=25
            index[gi,:len(ids)]=ids;mask[gi,:len(ids)]=True
            row_group[ids]=gi;row_position[ids]=np.arange(len(ids))
        for name,array in [('profile_index',index),('profile_mask',mask),('row_group',row_group),('row_position',row_position)]:
            setattr(self,name,torch.tensor(array,device=self.device))
        folder=ROOT/'cache/final_world';manifest=json.loads((folder/'manifest.json').read_text());assert manifest['status']=='complete'
        keys=WORLD_FIELDS+['early','middle','late','site'];pieces={key:[] for key in keys};dates=[]
        for filename in manifest['shards']:
            with np.load(folder/filename,allow_pickle=False) as shard:
                dates.extend(shard['dates'].tolist())
                for key in keys:pieces[key].append(shard[key])
        assert len(dates)==len(set(dates))==362
        self.cache_dates=dates;lookup={d:i+1 for i,d in enumerate(dates)}
        self.world_index=torch.tensor([lookup.get(d,0) for d in self.frame.date],device=self.device)
        self.world_available=(self.world_index>0).float();self.world={}
        for key,parts in pieces.items():
            array=np.concatenate(parts);assert np.isfinite(array).all()
            self.world[key]=torch.tensor(np.concatenate([np.zeros_like(array[:1]),array]),dtype=torch.float16,device=self.device)
        self.error_memory=None;self.indices={'evaluation_2025':torch.arange(len(self.frame),device=self.device)}
        self.split_hash=digest(ROOT/'configs/splits_2024_frozen.json')
        (ROOT/'cache/final_data_manifest.json').write_text(json.dumps(dict(rows=len(self.frame),world_days=len(dates),
            missing_world_dates=manifest['missing_dates'],source=str(path),normalization_fitted=False,
            exact_previous_feature_order=True,all_lag_features_precede_issue=True),indent=2))
