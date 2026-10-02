from src.runtime import ROOT,REPO,FAILED
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
    folder=ROOT/'cache/hourly_world';folder.mkdir(exist_ok=True)
    selected=json.loads((ROOT/'configs/selected_w1.json').read_text(encoding='utf-8'))
    register_historical(selected['checkpoint']);cfg=_read_config_with_base(Path(selected['config']))
    cfg['data']['migration']['enabled']=False;cfg['model']['gradient_checkpointing']=False
    model=VPPWorldMoE(WorldMoEConfig.from_mapping(cfg['model']),include_legacy_planner=False)
    cp=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    load_plannerless_released_w1_state(model,cp['model_state']);del cp
    model.eval().requires_grad_(False).to('cuda');gpu_recurrence.install(model)
    print('Preparing 2022-2024 causal corpus for hourly states',flush=True)
    started=time.perf_counter();corpus=WorldMoECorpus(cfg,REPO,materialize=False)
    corpus._weather_context=types.MethodType(causal_weather,corpus)
    records=sorted(corpus.records('trainval',require_h96=False),key=lambda r:str(r['operating_date']))
    assert_years([r['operating_date'] for r in records]);market=MarketContext()
    slot_ids=[[8*z,8*z+2,8*z+4,34] for z in range(4)]
    captures={};hooks=[]
    for key,block in zip(['early','middle','late'],[2,5,9]):
        def capture(module,args,output,key=key):
            values=output[0].float()
            # B,96,zone,entity,width -> B,zone,24,entity,width.
            selected_values=values[:,:,slot_ids]
            captures[key]=selected_values.reshape(len(values),24,4,4,4,256).mean(2).permute(0,2,1,3,4).contiguous().cpu().numpy().astype(np.float16)
        hooks.append(model.backbone.blocks[block].register_forward_hook(capture))
    manifest=dict(status='running',schema='r2.hourly_world.v1',expected_days=len(records),shards=[],selected_w1_id=selected['recorded_sha256'],
        block_indices=[2,5,9],slot_ids_by_zone=slot_ids,shape='days,zone,24_UTC_hours,4_entities,256',
        token_time='operating_date 00:00 UTC + hour; encode target-relative offset without modulo',
        frozen_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=0,locked_targets_accessed=False)
    for start in range(0,len(records),12):
        batch=records[start:start+12];dates=[str(r['operating_date']) for r in batch];assert_years(dates)
        file=folder/f'hourly_{start:05d}_{start+len(batch):05d}.npz'
        if file.exists():
            with np.load(file,allow_pickle=False) as cached:assert cached['dates'].tolist()==dates
        else:
            for r in batch:assert pd.Timestamp(r['issue_time_utc'])<=cutoff(str(r['operating_date']))
            packet=corpus._packet(batch,origin=0,device=torch.device('cpu'))
            packet,_=market.overlay(packet,batch,corpus.statistics,corpus.graph)
            mask=packet.fast_mask[:,:,slot_ids].reshape(len(batch),24,4,4,4).any(2).permute(0,2,1,3).numpy()
            packet=_to_device(packet,torch.device('cuda'))
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                state=model.encode(packet,sample_stochastic=False,phase='w1',capture_expert_context=False)
            np.savez_compressed(file,dates=np.asarray(dates),generation=np.asarray([generation(d).isoformat() for d in dates]),
                source_issue=np.asarray([str(r['issue_time_utc']) for r in batch]),mask=mask,**captures)
            captures.clear();del packet,state
        manifest['shards'].append(file.name);manifest.update(completed_days=start+len(batch),seconds=time.perf_counter()-started)
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
        if start%120==0:print(json.dumps(dict(days=start+len(batch),total=len(records),seconds=manifest['seconds'])),flush=True)
    for hook in hooks:hook.remove()
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    manifest.update(status='complete',cache_bytes=sum((folder/name).stat().st_size for name in manifest['shards']),first_date='2022-01-01',last_date='2024-12-31')
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2));print('HOURLY WORLD CACHE COMPLETE',flush=True)

if __name__=='__main__':main()
