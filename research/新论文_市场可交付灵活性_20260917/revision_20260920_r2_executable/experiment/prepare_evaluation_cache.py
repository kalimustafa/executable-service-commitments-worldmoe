"""Generate 2025 frozen-world contexts only after the R2 evaluation freeze."""
from pathlib import Path
import json,os,sys,time,types,importlib.util
os.environ.update(PYTHONUTF8='1',CUDA_LAUNCH_BLOCKING='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT.parent;REPO=ROOT.parents[1]/'power-dc-vpp';sys.path.insert(0,str(REPO/'src'))
import numpy as np,pandas as pd,torch
from prepare_frozen_cache import load_world,causal_weather,MarketContext,reduce_features,save,cutoff,generation,INACTIVE
from dcvpp.vpp_world_moe.r_ecmoe_sealed_corpus import build_sealed_final_corpus
from dcvpp.vpp_world_moe.r_ecmoe_runtime import _to_device

def issue_weather_timestamps(decision_origin,valid_times):
    origin=pd.Timestamp(decision_origin)
    targets=tuple(pd.Timestamp(v) for v in valid_times)
    if origin.tzinfo is None or not targets or any(t.tzinfo is None for t in targets):raise ValueError('Explicit UTC-aware weather times required')
    origin=origin.tz_convert('UTC');targets=tuple(t.tz_convert('UTC') for t in targets)
    days={t.date() for t in targets}
    if len(days)!=1 or next(iter(days)).year!=2025:raise ValueError('R2 evaluation weather targets must be one 2025 UTC day')
    day=next(iter(days))
    if origin!=cutoff(str(day)):raise ValueError('Weather query differs from the frozen pre-DA information cutoff')
    # Jan 1 forecasts legitimately use Dec 31 information. Preserve the source
    # store's deployable-availability checks; expand only this boundary origin.
    return origin,targets,day

def main():
    start=time.perf_counter();freeze=json.loads((ROOT/'EVALUATION_FREEZE.json').read_text(encoding='utf-8'))
    assert freeze['post_evaluation_parameter_changes_allowed'] is False
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    model,cfg,selection=load_world();marker=json.loads((ROOT/'cache/frozen_evaluation_marker.json').read_text())
    corpus=build_sealed_final_corpus(config=cfg,repository_root=REPO,frozen_evaluation_marker=marker)
    corpus.weather_store._timestamps=issue_weather_timestamps
    corpus._weather_context=types.MethodType(causal_weather,corpus)
    records=sorted(corpus.records('locked_test',require_h96=False),key=lambda r:str(r['operating_date']))
    market=MarketContext(PAPER/'revision_20260918/forecast_2025/gate_hourly_2025_with_context.parquet')
    path=PAPER/'revision_20260918/annual_native_2025/performance_v2/gpu_recurrence.py'
    spec=importlib.util.spec_from_file_location('r2_eval_recurrence',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.install(model)
    captures={};hook=model.tokenizer.register_forward_hook(lambda module,inputs,output:captures.update(tokens=output[0]))
    names=[];dates=[];inference_seconds=0.
    for offset in range(0,len(records),12):
        batch=records[offset:offset+12];file=ROOT/'cache'/f'evaluation_2025_{offset:05d}_{offset+len(batch):05d}.npz'
        days=[str(r['operating_date']) for r in batch]
        for r in batch:assert pd.Timestamp(r['issue_time_utc'])<=cutoff(str(r['operating_date']))
        if file.exists():
            with np.load(file) as f:assert f['dates'].tolist()==days
        else:
            packet=corpus._packet(batch,origin=0,device=torch.device('cpu'))
            packet,_=market.overlay(packet,batch,corpus.statistics,corpus.graph)
            assert not packet.fast_mask[...,list(INACTIVE)].any()
            packet=_to_device(packet,torch.device('cuda'));t=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                state=model.encode(packet,sample_stochastic=False,phase='w1',capture_expert_context=True)
                features=reduce_features(model,state,captures.pop('tokens'))
            torch.cuda.synchronize();inference_seconds+=time.perf_counter()-t
            features.update(dates=np.asarray(days),source_issue=np.asarray([str(r['issue_time_utc']) for r in batch]),
                price_cutoff=np.asarray([cutoff(d).isoformat() for d in days]),generation=np.asarray([generation(d).isoformat() for d in days]))
            np.savez_compressed(file,**features);del packet,state
        names.append(file.name);dates+=days
        if offset%120==0:print('Evaluation context',offset+len(batch),'/',len(records),'seconds',round(time.perf_counter()-start,1),flush=True)
    hook.remove()
    all_dates=pd.date_range('2025-01-01','2025-12-31').strftime('%Y-%m-%d').tolist();missing=sorted(set(all_dates)-set(dates))
    save(ROOT/'cache/evaluation_cache_manifest.json',dict(status='complete',shards=names,days=len(dates),missing_world_dates=missing,
        missing_policy='price head receives explicit zero world context with unavailable-world mask; no physical interval is dropped',
        frozen_world_id=selection['w1']['recorded_checkpoint_id'],frozen_R2_primary=freeze['primary_model'],
        total_seconds=time.perf_counter()-start,inference_seconds=inference_seconds,price_context_imputed_cells=market.missing_cells))
    print('EVALUATION CACHE COMPLETE',time.perf_counter()-start,missing,flush=True)
if __name__=='__main__':main()
