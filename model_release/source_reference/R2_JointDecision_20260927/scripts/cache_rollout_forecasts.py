"""Frozen forecast evidence for a light R2 decision adapter and temporal training."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT
import numpy as np,torch
from src.data.r2_data import R2Data
from src.models.factory import make_model
from src.models.structured_decision import StructuredDecision
torch.set_num_threads(3)
data=R2Data(device='cuda',multiscale=True);seed=20260920
model=StructuredDecision(make_model('C1_ProfileResidual',seed,data),data.target_scale,data.target_center[:,2],torch.ones(4,device='cuda')).cuda()
source=ROOT/'checkpoints/StructuredR2_seed20260920_best.pt'
model.load_state_dict(torch.load(source,map_location='cpu',weights_only=False)['state_dict']);model.eval().requires_grad_(False)
arrays=dict(zone=data.zone.cpu().numpy(),labels=data.labels.cpu().numpy(),target_scale=data.target_scale.cpu().numpy(),historical_rt=model.historical_rt.cpu().numpy(),pue_reference=model.pue_reference.cpu().numpy(),dates=data.frame.date.to_numpy(dtype='U10'),timestamps=data.frame.timestamp_utc.to_numpy(dtype='datetime64[ns]'),train_indices=data.indices['train'].cpu().numpy(),selection_indices=data.indices['selection'].cpu().numpy())
with torch.inference_mode():
    for gate in [0,1]:
        pieces={k:[] for k in ['da','dart','rt','decision_latent']}
        for ids in torch.arange(len(data.frame),device='cuda').split(4096):
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(data.batch(ids),torch.full((len(ids),),float(gate),device='cuda'),data.labels[ids,0])
            for key in pieces:pieces[key].append(out[key].float().cpu().numpy())
        for key,values in pieces.items():arrays[f'{key}_{gate}']=np.concatenate(values)
        print('CACHED FORECAST GATE',gate,flush=True)
np.savez_compressed(ROOT/'cache/rollout_forecast_features.npz',**arrays)
(ROOT/'cache/rollout_forecast_features.json').write_text(json.dumps(dict(source=str(source),training_years=[2021,2022,2023],selection_only_year=2024,world_and_original_market_encoder_frozen=True,new_trainable='light shared R2 residual adapter, APIC-preserving price shift, joint price-PV cross moment and temporal cost head',rows=len(data.frame)),indent=2))
