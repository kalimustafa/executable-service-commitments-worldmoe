from src.runtime import ROOT,REPO
from src.data.access_guard import install_guard,register_historical,assert_years
install_guard()
from pathlib import Path
import json,time,types
import numpy as np
import pandas as pd
import torch
from dcvpp.vpp_world_moe.config import _read_config_with_base,WorldMoEConfig
from dcvpp.vpp_world_moe.model import VPPWorldMoE,load_plannerless_released_w1_state
from dcvpp.vpp_world_moe.data import WorldMoECorpus
from dcvpp.vpp_world_moe.r_ecmoe_runtime import _to_device
from vendor.causal_world import MarketContext,causal_weather,cutoff,generation
from vendor import gpu_recurrence

def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    selected=json.loads((ROOT/'configs/selected_w1.json').read_text(encoding='utf-8'));register_historical(selected['checkpoint'])
    cfg=_read_config_with_base(Path(selected['config']));cfg['data']['migration']['enabled']=False;cfg['model']['gradient_checkpointing']=False
    model=VPPWorldMoE(WorldMoEConfig.from_mapping(cfg['model']),include_legacy_planner=False)
    cp=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    load_plannerless_released_w1_state(model,cp['model_state']);del cp
    model.eval().requires_grad_(False).to('cuda');gpu_recurrence.install(model)
    started=time.perf_counter();print('Preparing causal physical forecast extraction',flush=True)
    corpus=WorldMoECorpus(cfg,REPO,materialize=False);corpus._weather_context=types.MethodType(causal_weather,corpus)
    records=sorted(corpus.records('trainval',require_h96=False),key=lambda r:str(r['operating_date']));assert_years([r['operating_date'] for r in records])
    market=MarketContext();pieces={k:[] for k in ['world_pv','world_pue','issue_pv','issue_pue','issue_arrivals']};dates=[];issues=[]
    for start in range(0,len(records),12):
        batch=records[start:start+12];d=[str(r['operating_date']) for r in batch]
        for r in batch:assert pd.Timestamp(r['issue_time_utc'])<=cutoff(str(r['operating_date']))
        packet=corpus._packet(batch,origin=0,device=torch.device('cpu'));packet,_=market.overlay(packet,batch,corpus.statistics,corpus.graph)
        center=corpus.statistics.fast_center;scale=corpus.statistics.fast_scale
        if center.ndim==1:center=center[None].expand(40,-1)
        if scale.ndim==1:scale=scale[None].expand(40,-1)
        raw=packet.fast_values*scale+center
        pieces['issue_pv'].append(raw[:,:,[2,10,18,26],0].numpy())
        pieces['issue_pue'].append(raw[:,:,[3,11,19,27],0].numpy())
        pieces['issue_arrivals'].append(raw[:,:,[4,12,20,28],0].numpy())
        packet=_to_device(packet,torch.device('cuda'))
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=model.encode(packet,sample_stochastic=False,phase='w1',capture_expert_context=False)
            prediction=model.query_observations(state,packet.constraints)
        pieces['world_pv'].append(prediction.pv.float().cpu().numpy());pieces['world_pue'].append(prediction.pue.float().cpu().numpy())
        dates.extend(d);issues.extend([generation(day).isoformat() for day in d]);del packet,state,prediction
        if start%240==0:print(json.dumps(dict(days=start+len(batch),seconds=time.perf_counter()-started)),flush=True)
    arrays={k:np.concatenate(v).astype(np.float32) for k,v in pieces.items()}
    assert all(np.isfinite(v).all() for v in arrays.values())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    np.savez_compressed(ROOT/'cache/world_physical_forecasts.npz',dates=np.asarray(dates),issue_utc=np.asarray(issues),**arrays)
    (ROOT/'cache/world_physical_forecasts_manifest.json').write_text(json.dumps(dict(status='complete',days=len(dates),first_date=dates[0],last_date=dates[-1],
        shape=list(arrays['world_pv'].shape),selected_w1=selected['recorded_sha256'],prediction_interface='query_observations(state, packet.constraints)',
        issue_priors='denormalized causal input slots 2/3/4 per site',generation='D-1 04:55 America/New_York; enforced at decision time',
        future_realized_physical_values_used_as_predictions=False,new_fitted_parameters=0,seconds=time.perf_counter()-started,locked_targets_accessed=False),indent=2))
    print('WORLD PHYSICAL FORECASTS COMPLETE',flush=True)

if __name__=='__main__':main()
