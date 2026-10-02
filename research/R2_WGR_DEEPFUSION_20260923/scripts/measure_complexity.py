from src.runtime import ROOT
from src.data.access_guard import install_guard
install_guard()
import json
import time
import numpy as np
import pandas as pd
import torch
from src.data.r2_data import R2Data
from src.calibration.calibrate_and_predict import load_model

def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    data=R2Data(multiscale=True);rows=[];stability=[]
    ix=data.indices['selection'][:512];y=data.labels[ix];batch=data.batch(ix)
    pub=(torch.arange(len(ix),device=data.device)%2).float()
    cache=json.loads((ROOT/'cache/multiscale_manifest.json').read_text())
    for name in ['A0_Base','A1_Shallow','A2_Deep_NoGate','A3_WGR']:
        model,cp=load_model(name,20260920,data)
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            for _ in range(3):model(batch,pub,y[:,0])
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(20):out=model(batch,pub,y[:,0])
            torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000/20
            parts=[]
            for start in range(0,len(ix),64):
                b={k:v[start:start+64] for k,v in batch.items()}
                parts.append(model(b,pub[start:start+64],y[start:start+64,0])['rt'])
            delta=(out['rt']-torch.cat(parts)).abs()
            stability.append(dict(model=name,precision='FP32',batch_a=512,batch_b=64,max_rt_quantile_delta=float(delta.max()),
                                  mean_rt_quantile_delta=float(delta.mean()),rms_rt_quantile_delta=float(delta.square().mean().sqrt())))
        frozen_world=0 if name=='A0_Base' else cache['frozen_parameters']
        total_head=cp['total_head_parameters'];trainable=cp['trainable_parameters']
        rows.append(dict(model=name,total_parameters=total_head+frozen_world,head_total_parameters=total_head,
                         trainable_parameters=trainable,frozen_parameters=total_head-trainable+frozen_world,
                         frozen_world_parameters=frozen_world,selected_checkpoint_stage=cp['stage'],
                         selected_checkpoint_stage_trainable_parameters=cp['validation']['trainable_parameters'],
                         inference_ms_per_batch=ms,batch_size=512,cache_size_gb=0 if name=='A0_Base' else cache['cache_bytes']/1e9,
                         latency_scope='FP32 cached R2 head only, 20 timed repetitions; no W1 forward or disk I/O'))
        del model;torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(ROOT/'results/model_complexity.csv',index=False)
    pd.DataFrame(stability).to_csv(ROOT/'results/batch_stability_fp32.csv',index=False)
    assert max(row['max_rt_quantile_delta'] for row in stability)<.001
    print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':main()
