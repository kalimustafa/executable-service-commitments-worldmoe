"""Frozen 2025 W1 context and issue-prior extraction; no learned update."""
from src.runtime import ROOT,REPO,LEGACY,PAPER
from src.data.access_guard import enable_final_evaluation,install_guard,register_historical
enable_final_evaluation();install_guard()
from pathlib import Path
import json,time,types
import numpy as np
import pandas as pd
import torch
from dcvpp.vpp_world_moe.config import _read_config_with_base,WorldMoEConfig
from dcvpp.vpp_world_moe.model import VPPWorldMoE,load_plannerless_released_w1_state
from dcvpp.vpp_world_moe.r_ecmoe_sealed_corpus import build_sealed_final_corpus
from dcvpp.vpp_world_moe.r_ecmoe_runtime import _to_device
from vendor.causal_world import MarketContext,causal_weather,cutoff,generation,reduce_features
from vendor import gpu_recurrence

def issue_weather_timestamps(decision_origin,valid_times):
    origin=pd.Timestamp(decision_origin);targets=tuple(pd.Timestamp(v) for v in valid_times)
    assert origin.tzinfo is not None and targets and all(t.tzinfo is not None for t in targets)
    origin=origin.tz_convert('UTC');targets=tuple(t.tz_convert('UTC') for t in targets)
    days={t.date() for t in targets};assert len(days)==1
    day=next(iter(days));assert day.year==2025 and origin==cutoff(str(day))
    return origin,targets,day

def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    out=ROOT/'cache/final_world';out.mkdir(exist_ok=True)
    selected=json.loads((ROOT/'configs/selected_w1.json').read_text());register_historical(selected['checkpoint'])
    cfg=_read_config_with_base(Path(selected['config']));cfg['data']['migration']['enabled']=False
    cfg['model']['gradient_checkpointing']=False;cfg['project']['output_root']=str(ROOT/'cache/world_internal_final')
    model=VPPWorldMoE(WorldMoEConfig.from_mapping(cfg['model']),include_legacy_planner=False)
    payload=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    load_plannerless_released_w1_state(model,payload['model_state']);del payload
    model.eval().requires_grad_(False).to('cuda');gpu_recurrence.install(model)
    marker=json.loads((LEGACY/'cache/frozen_evaluation_marker.json').read_text())
    corpus=build_sealed_final_corpus(config=cfg,repository_root=REPO,frozen_evaluation_marker=marker)
    corpus.weather_store._timestamps=issue_weather_timestamps
    corpus._weather_context=types.MethodType(causal_weather,corpus)
    records=sorted(corpus.records('locked_test',require_h96=False),key=lambda r:str(r['operating_date']))
    market=MarketContext(PAPER/'revision_20260918/forecast_2025/gate_hourly_2025_with_context.parquet')
    captures={};hooks=[]
    for key,index in zip(['early','middle','late'],selected['extraction_blocks_zero_based']):
        def capture(module,inputs,output,key=key):captures[key]=output[0].float().mean(1).detach()
        hooks.append(model.backbone.blocks[index].register_forward_hook(capture))
    hooks.append(model.tokenizer.register_forward_hook(lambda m,i,o:captures.update(tokenized=o[0])))
    started=time.perf_counter();names=[];dates=[];physical=[];prior_records=[]
    for offset in range(0,len(records),12):
        batch=records[offset:offset+12];days=[str(r['operating_date']) for r in batch]
        path=out/f'final_{offset:05d}_{offset+len(batch):05d}.npz'
        assert all(pd.Timestamp(r['issue_time_utc'])<=cutoff(str(r['operating_date'])) for r in batch)
        if not path.exists():
            packet=corpus._packet(batch,origin=0,device=torch.device('cpu'))
            packet,_=market.overlay(packet,batch,corpus.statistics,corpus.graph)
            center=corpus.statistics.fast_center;scale=corpus.statistics.fast_scale
            if center.ndim==1:center=center[None].expand(40,-1)
            if scale.ndim==1:scale=scale[None].expand(40,-1)
            raw=packet.fast_values*scale+center
            issues=dict(issue_pv=raw[:,:,[2,10,18,26],0].numpy(),issue_pue=raw[:,:,[3,11,19,27],0].numpy())
            packet=_to_device(packet,torch.device('cuda'))
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                state=model.encode(packet,sample_stochastic=False,phase='w1',capture_expert_context=True)
                data=reduce_features(model,state,captures.pop('tokenized'))
                data.update({k:v.cpu().numpy().astype(np.float16) for k,v in captures.items()});captures.clear()
                data['site']=state.slots[:,:,[0,8,16,24]].float().mean(1).cpu().numpy().astype(np.float16)
            data.update(dates=np.asarray(days),issue_utc=np.asarray([generation(d).isoformat() for d in days]),**issues)
            np.savez_compressed(path,**data);del packet,state,data
        with np.load(path,allow_pickle=False) as cache:
            assert cache['dates'].tolist()==days
            physical.append({k:cache[k].copy() for k in ['dates','issue_utc','issue_pv','issue_pue']})
        prior_records.extend(batch);names.append(path.name);dates+=days
        manifest=dict(status='running',shards=names,days=len(dates),selected_w1_id=selected['recorded_sha256'],
            W1_trainable_parameters=0,precision='BF16 world; FP16 cache; FP32 issue priors',seconds=time.perf_counter()-started)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        print('FINAL WORLD',len(dates),'/',len(records),round(time.perf_counter()-started,1),flush=True)
    for h in hooks:h.remove()
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    manifest.update(status='complete',first_date=dates[0],last_date=dates[-1],
        missing_dates=sorted(set(pd.date_range('2025-01-01','2025-12-31').strftime('%Y-%m-%d'))-set(dates)))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    np.savez_compressed(ROOT/'cache/final_physical_priors.npz',**{k:np.concatenate([p[k] for p in physical]) for k in physical[0]})
    (out/'source_records.json').write_text(json.dumps(prior_records,indent=2,default=str))
    print('FINAL WORLD COMPLETE',manifest['missing_dates'],flush=True)

if __name__=='__main__':main()
