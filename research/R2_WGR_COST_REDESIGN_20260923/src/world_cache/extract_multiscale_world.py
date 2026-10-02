from src.runtime import ROOT, REPO, LEGACY
from src.data.access_guard import install_guard, register_historical, assert_years
install_guard()
import argparse
import json
import time
import types
from pathlib import Path
import numpy as np
import torch
from dcvpp.vpp_world_moe.config import _read_config_with_base, WorldMoEConfig
from dcvpp.vpp_world_moe.model import VPPWorldMoE, load_plannerless_released_w1_state
from dcvpp.vpp_world_moe.data import WorldMoECorpus
from dcvpp.vpp_world_moe.r_ecmoe_runtime import _to_device
from vendor.causal_world import MarketContext, causal_weather, cutoff, generation, reduce_features
from vendor import gpu_recurrence

def save_manifest(value):
    p=ROOT/'cache/multiscale_manifest.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=int,default=12);args=parser.parse_args()
    assert (ROOT/'results/baseline_reproduction_passed.json').exists()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    selected=json.loads((ROOT/'configs/selected_w1.json').read_text(encoding='utf-8'))
    register_historical(selected['checkpoint'])
    cfg=_read_config_with_base(Path(selected['config']))
    cfg['data']['migration']['enabled']=False
    cfg['model']['gradient_checkpointing']=False
    cfg['project']['output_root']=str(ROOT/'cache/world_internal')
    cfg['data']['normalization']['artifact_path']=str(ROOT/'cache/world_normalization.json')
    cfg['data']['lineage_path']=str(ROOT/'cache/world_lineage.json')
    model=VPPWorldMoE(WorldMoEConfig.from_mapping(cfg['model']),include_legacy_planner=False)
    payload=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    load_plannerless_released_w1_state(model,payload['model_state']);del payload
    model.eval().requires_grad_(False).to('cuda')
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    gpu_recurrence.install(model)
    started=time.perf_counter()
    print('Preparing guarded 2022-2024 world corpus',flush=True)
    corpus=WorldMoECorpus(cfg,REPO,materialize=False)
    corpus._weather_context=types.MethodType(causal_weather,corpus)
    records=sorted(corpus.records('trainval',require_h96=False),key=lambda r:str(r['operating_date']))
    assert_years([r['operating_date'] for r in records],'world extraction targets')
    market=MarketContext();captures={};hooks=[]
    blocks=selected['extraction_blocks_zero_based']
    for key,index in zip(['early','middle','late'],blocks):
        assert model.backbone.blocks[index].block_index==index
        def capture(module,inputs,output,key=key):
            captures[key]=output[0].float().mean(1).detach()
        hooks.append(model.backbone.blocks[index].register_forward_hook(capture))
    hooks.append(model.tokenizer.register_forward_hook(lambda m,i,o:captures.update(tokenized=o[0])))
    manifest=dict(status='running',schema='r2_wgr.multiscale.v1',selected_w1_id=selected['recorded_sha256'],
                  expected_days=len(records),shards=[],block_indices=blocks,world_width=model.config.d_model,
                  slots_per_scale=40,source_context='exact formal causal_issue_reference_context.v1',
                  inference_precision='BF16',storage_precision='FP16',locked_targets_accessed=False,
                  frozen_parameters=sum(p.numel() for p in model.parameters()),trainable_W1=0)
    save_manifest(manifest)
    shallow=json.loads((LEGACY/'cache/development_cache_manifest.json').read_text())
    dates_seen=[];max_delta=0.
    for offset in range(0,len(records),args.batch):
        batch=records[offset:offset+args.batch];dates=[str(r['operating_date']) for r in batch]
        assert_years(dates,'world batch')
        path=ROOT/'cache'/f'multiscale_{offset:05d}_{offset+len(batch):05d}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as cached:assert cached['dates'].tolist()==dates
        else:
            for r in batch:
                import pandas as pd
                assert pd.Timestamp(r['issue_time_utc'])<=cutoff(str(r['operating_date']))
                for d in r['medium_source_dates']+r['slow_source_dates']:
                    if d is not None:assert pd.Timestamp(str(d),tz='UTC')+pd.Timedelta(days=1)<=cutoff(str(r['operating_date']))
            packet=corpus._packet(batch,origin=0,device=torch.device('cpu'))
            packet,_=market.overlay(packet,batch,corpus.statistics,corpus.graph)
            packet=_to_device(packet,torch.device('cuda'))
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                state=model.encode(packet,sample_stochastic=False,phase='w1',capture_expert_context=True)
                data=reduce_features(model,state,captures.pop('tokenized'))
                data.update({k:v.cpu().numpy().astype(np.float16) for k,v in captures.items()});captures.clear()
                # Keep four market-site slots as distinct tokens, plus all
                # 40 tokens in each of the three interaction states.
                data['site']=state.slots[:,:,[0,8,16,24]].float().mean(1).cpu().numpy().astype(np.float16)
            data.update(dates=np.asarray(dates),source_issue=np.asarray([str(r['issue_time_utc']) for r in batch]),
                        generation=np.asarray([generation(d).isoformat() for d in dates]),
                        price_cutoff=np.asarray([cutoff(d).isoformat() for d in dates]),zones=np.asarray(['WEST','CAPITL','N.Y.C.','LONGIL']),
                        availability=np.ones(len(dates),dtype=bool))
            if offset==0:
                with np.load(LEGACY/'cache'/shallow['shards'][0],allow_pickle=False) as old:
                    assert old['dates'].tolist()==dates
                    max_delta=float(np.max(np.abs(data['w1_local'].astype(np.float32)-old['w1_local'].astype(np.float32))))
            np.savez_compressed(path,**data)
            del state,packet,data
        manifest['shards'].append(path.name);dates_seen.extend(dates)
        manifest.update(completed_days=len(dates_seen),elapsed_seconds=time.perf_counter()-started,
                        first_date=dates_seen[0],last_date=dates_seen[-1],legacy_first_batch_local_max_diff=max_delta)
        save_manifest(manifest)
        if offset==0 or (offset//args.batch)%5==0:
            print(json.dumps(dict(days=len(dates_seen),total=len(records),seconds=round(manifest['elapsed_seconds'],1),gpu_mib=round(torch.cuda.max_memory_allocated()/1048576))),flush=True)
    for hook in hooks:hook.remove()
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    manifest.update(status='complete',cache_bytes=sum((ROOT/'cache'/name).stat().st_size for name in manifest['shards']),w1_gradients_none=True)
    save_manifest(manifest);print('MULTISCALE CACHE COMPLETE',flush=True)

if __name__=='__main__':main()
