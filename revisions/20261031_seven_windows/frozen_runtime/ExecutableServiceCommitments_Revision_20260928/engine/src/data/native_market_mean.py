"""Integrate native DA/DART inverse CDFs instead of compressing RT to seven atoms."""
import numpy as np,pandas as pd,torch
from src.runtime import ROOT,WORKSPACE

# Exact integration of the native piecewise-linear inverse CDF, including its
# linear continuations from tau=.05 to 0 and from tau=.95 to 1.
MEAN_WEIGHTS=np.array([.1,.075,.2,.25,.2,.075,.1],np.float64)
ATOMIC_WEIGHTS=np.array([.075,.1,.2,.25,.2,.1,.075],np.float64)


def prepare_native_mean(year):
    path=ROOT/'cache'/f'native_apic_mean_delta_{year}.npz'
    if path.exists():return path
    with np.load(WORKSPACE/f'external_validation/data/forcing_{year}.npz') as z:times=pd.DatetimeIndex(pd.to_datetime(z['times'],utc=True))
    output={}
    if year==2024:
        with np.load(ROOT/'cache/rollout_forecast_features.npz') as z:
            source=pd.MultiIndex.from_arrays([pd.DatetimeIndex(pd.to_datetime(z['timestamps'],utc=True)).asi8,z['zone']])
            target=pd.MultiIndex.from_arrays([np.repeat(times.floor('h').asi8,4),np.tile(np.arange(4),len(times))]);index=source.get_indexer(target);assert (index>=0).all()
            for gate in [0,1]:
                mean=z[f'da_{gate}']@MEAN_WEIGHTS-z[f'dart_{gate}']@MEAN_WEIGHTS
                difference=mean-z[f'rt_{gate}']@ATOMIC_WEIGHTS
                output[f'delta_{gate}']=np.r_[difference[index].reshape(len(times),4),np.zeros((96,4))].astype(np.float32)
    else:
        assert year==2025
        from src.data.structured_final_data import StructuredFinalData
        from src.data.r2_data import ZONES
        from src.models.factory import make_model
        from src.models.structured_decision import StructuredDecision
        data=StructuredFinalData(20260920)
        model=StructuredDecision(make_model('C1_ProfileResidual',20260920,data),data.target_scale,data.target_center[:,2],torch.ones(4,device='cuda')).cuda()
        model.load_state_dict(torch.load(ROOT/'checkpoints/StructuredR2_seed20260920_epoch10.pt',map_location='cpu',weights_only=False)['state_dict']);model.eval().requires_grad_(False)
        frame=data.frame.copy();frame.timestamp_utc=pd.to_datetime(frame.timestamp_utc,utc=True)
        hours=times.floor('h').unique();keys=pd.MultiIndex.from_product([hours,ZONES],names=['timestamp_utc','zone'])
        lookup=pd.Series(np.arange(len(frame)),index=pd.MultiIndex.from_frame(frame[['timestamp_utc','zone']]));raw_indices=lookup.reindex(keys).to_numpy();assert np.isfinite(raw_indices).all()
        indices=torch.tensor(raw_indices.astype(np.int64),device='cuda');quarters=np.searchsorted(hours.asi8,times.floor('h').asi8)
        with torch.inference_mode():
            for gate in [0,1]:
                pieces=[]
                for ids in indices.split(4096):
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(data.batch(ids),torch.full((len(ids),),float(gate),device='cuda'),data.labels[ids,0])
                    da=out['da'].float().cpu().numpy();dart=out['dart'].float().cpu().numpy();rt=out['rt'].float().cpu().numpy()
                    pieces.append(da@MEAN_WEIGHTS-dart@MEAN_WEIGHTS-rt@ATOMIC_WEIGHTS)
                values=np.concatenate(pieces).reshape(len(hours),4)[quarters]
                output[f'delta_{gate}']=np.r_[values,np.zeros((96,4))].astype(np.float32)
        del data,model
        torch.cuda.empty_cache()
    np.savez_compressed(path,**output,rule='E[RT]=integral Q_DA - integral Q_DART; native inverse-CDF tail continuation; no fitting')
    return path
