from src.runtime import ROOT, PAPER, LEGACY
from src.data.access_guard import install_guard, assert_years, read_development_parquet
import hashlib
import json
import numpy as np
import pandas as pd
import torch
install_guard()

ZONES = ['WEST', 'CAPITL', 'N.Y.C.', 'LONGIL']
WORLD_FIELDS = ['w0_local','w1_local','w1_global','expert_bank','expert_prior','tail_context','tail_probability','regime','log_scale']

class R2Data:
    def __init__(self, device='cuda', multiscale=False):
        self.device = torch.device(device)
        self.frame = read_development_parquet(PAPER / 'experiments/gate_hourly_panel.parquet').sort_values(['timestamp_utc','zone']).reset_index(drop=True)
        historical = json.loads((LEGACY / 'cache/r2_normalization.json').read_text(encoding='utf-8'))
        self.names = historical['feature_names']
        for zone in ZONES:
            self.frame['zone_'+zone] = self.frame.zone.eq(zone).astype(float)
        naive = pd.to_datetime(self.frame.date)-pd.Timedelta(days=1)+pd.Timedelta(hours=4,minutes=55)
        issue = naive.dt.tz_localize('America/New_York').dt.tz_convert('UTC')
        assert (pd.to_datetime(self.frame.feature_last_interval_end_utc,utc=True)<issue).all()
        self.frame['lead_hours'] = (pd.to_datetime(self.frame.timestamp_utc,utc=True)-issue).dt.total_seconds()/3600
        train = self.frame.split.eq('train').to_numpy()
        features = self.frame[self.names].to_numpy(np.float32)
        center = np.nanmedian(features[train],axis=0)
        scale = np.nanquantile(features[train],.75,axis=0)-np.nanquantile(features[train],.25,axis=0)
        scale = np.maximum(np.where(scale>1e-3,scale,np.nanstd(features[train],axis=0)),1e-3)
        features = np.clip((np.where(np.isfinite(features),features,center)-center)/scale,-20,20)
        np.testing.assert_allclose(center,historical['feature_center'],atol=1e-6)
        np.testing.assert_allclose(scale,historical['feature_scale'],atol=1e-6)
        self.numerical = torch.as_tensor(features,dtype=torch.float32,device=self.device)
        self.zone = torch.tensor(pd.Categorical(self.frame.zone,categories=ZONES).codes,dtype=torch.long,device=self.device)
        self.lead = torch.tensor(self.frame.lead_hours.to_numpy(),dtype=torch.float32,device=self.device)
        self.labels = torch.tensor(self.frame[['da_lbmp','spread','rt_integrated']].to_numpy(),dtype=torch.float32,device=self.device)
        self.indices = {s:torch.tensor(np.flatnonzero(self.frame.split.eq(s)),device=self.device) for s in self.frame.split.unique()}
        self.target_center=[]; self.target_scale=[]; self.regime_edges=[]
        for z in ZONES:
            y = self.frame[self.frame.split.eq('train')&self.frame.zone.eq(z)][['da_lbmp','spread','rt_integrated']].to_numpy()
            self.target_center.append(np.median(y,axis=0))
            self.target_scale.append(np.maximum((np.quantile(y,.75,axis=0)-np.quantile(y,.25,axis=0))/1.349,5.))
            self.regime_edges.append(np.quantile(np.abs(y[:,1]),[.5,.9]))
        self.target_center=torch.tensor(np.array(self.target_center),dtype=torch.float32,device=self.device)
        self.target_scale=torch.tensor(np.array(self.target_scale),dtype=torch.float32,device=self.device)
        edges=torch.tensor(np.array(self.regime_edges),dtype=torch.float32,device=self.device)[self.zone]
        self.regime_target=(self.labels[:,1,None].abs()>edges).sum(-1)
        manifest=json.loads((LEGACY/'cache/development_cache_manifest.json').read_text(encoding='utf-8'))
        assert manifest['status']=='complete'
        assert_years([manifest['first_date'],manifest['last_date']],'shallow cache manifest')
        pieces={key:[] for key in WORLD_FIELDS}; all_dates=[]
        for name in manifest['shards']:
            assert name.startswith('development_') and '/' not in name and '\\' not in name
            with np.load(LEGACY/'cache'/name,allow_pickle=False) as shard:
                assert_years(shard['dates'],'shallow cache dates')
                all_dates.extend(shard['dates'].tolist())
                for key in WORLD_FIELDS: pieces[key].append(shard[key])
        assert len(set(all_dates))==len(all_dates)
        self.cache_dates=all_dates
        lookup={str(v):i+1 for i,v in enumerate(all_dates)}
        self.world_index=torch.tensor([lookup.get(str(d),0) for d in self.frame.date],dtype=torch.long,device=self.device)
        self.world_available=(self.world_index>0).float()
        assert not self.world_available[self.frame.date.str.startswith('2021').to_numpy()].any()
        self.world={}
        for key,parts in pieces.items():
            values=np.concatenate(parts,axis=0)
            assert np.isfinite(values).all()
            self.world[key]=torch.tensor(np.concatenate([np.zeros_like(values[:1]),values],axis=0),dtype=torch.float16,device=self.device)
        if multiscale:
            m=json.loads((ROOT/'cache/multiscale_manifest.json').read_text())
            assert m['status']=='complete'
            assert m['selected_w1_id']==manifest['selected_w1_id']
            values={k:[] for k in ['early','middle','late','site']}; dates=[]
            for filename in m['shards']:
                with np.load(ROOT/'cache'/filename,allow_pickle=False) as shard:
                    assert_years(shard['dates'],'multiscale cache')
                    dates.extend(shard['dates'].tolist())
                    for key in values:values[key].append(shard[key])
            assert dates==all_dates, 'Multiscale cache dates must exactly match legacy development cache'
            for key,parts in values.items():
                array=np.concatenate(parts)
                assert np.isfinite(array).all()
                self.world[key]=torch.tensor(np.concatenate([np.zeros_like(array[:1]),array]),dtype=torch.float16,device=self.device)
        self.normalization=historical
        self.split_hash=hashlib.sha256((ROOT/'configs/splits_2024_frozen.json').read_bytes()).hexdigest()
        (ROOT/'cache/r2_normalization.json').write_text(json.dumps(historical,indent=2))
        (ROOT/'cache/r2_data_manifest.json').write_text(json.dumps(dict(rows=len(self.frame),target_years=[2021,2022,2023,2024],
            split_rows={k:len(v) for k,v in self.indices.items()},split_hash=self.split_hash,
            source_feature_names=str(LEGACY/'cache/r2_normalization.json'),world_days=len(all_dates),locked_targets_accessed=False),indent=2))

    def batch(self,indices):
        w=self.world_index[indices];z=self.zone[indices]
        out={k:(v[w,z] if k in ['w0_local','w1_local'] else v[w]) for k,v in self.world.items()}
        out.update(numerical=self.numerical[indices],zone=z,lead_hours=self.lead[indices],world_available=self.world_available[indices])
        return out

    def row_targets(self,indices):
        return self.labels[indices],self.regime_target[indices]
