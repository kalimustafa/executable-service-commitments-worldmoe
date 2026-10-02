from src.runtime import ROOT,PREVIOUS
from src.data.access_guard import enable_final_evaluation,install_guard
enable_final_evaluation();install_guard()
import json,time
import pandas as pd
import torch
from src.data.final_data import R2FinalData
from src.calibration.calibrate_and_predict import load_model,latent_rows,predict_pack
from src.calibration.native_affine import NativeCalibration
from src.evaluation.forecast_metrics import metrics_for
from src.evaluation.final_freeze import verify_freeze

def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    data=R2FinalData();output=ROOT/'results/predictions_2025';output.mkdir(exist_ok=True)
    rows=[];checks=[];started=time.perf_counter()
    for seed in [20260920,20260921,20260922]:
        source=PREVIOUS if seed==20260920 else ROOT
        model,checkpoint=load_model('C1_ProfileResidual',seed,data)
        cal=NativeCalibration().to(data.device)
        cal.load_state_dict(torch.load(source/'checkpoints'/f'C1_ProfileResidual_seed{seed}_calibration.pt',map_location=data.device,weights_only=True))
        cal.eval().requires_grad_(False)
        for ablation in (['normal','force_gate_0'] if seed==20260920 else ['normal']):
            model.ablation=ablation
            pack=latent_rows(model,data,data.indices['evaluation_2025'])
            for kind,transform in [('raw',None),('native',cal)]:
                frame,error=predict_pack(model,data,pack,transform)
                frame.to_parquet(output/f'R2_seed{seed}_{ablation}_{kind}.parquet',index=False)
                rows.append(metrics_for(frame,'R2_'+ablation,seed,'evaluation_2025',kind))
                for gate,f in frame.groupby('gate'):rows.append(metrics_for(f,'R2_'+ablation,seed,'evaluation_2025',kind,gate))
                checks.append(dict(seed=seed,ablation=ablation,calibration=kind,identity_max_error=error,rows=len(frame)))
                print(seed,ablation,kind,'FORECAST COMPLETE',error,flush=True)
            del pack
        del model,cal;torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(ROOT/'results/forecast_metrics_2025.csv',index=False)
    verify_freeze()
    (ROOT/'results/final_forecast_complete.json').write_text(json.dumps(dict(status='complete',checks=checks,
        frozen_artifacts_unchanged=True,parameters_updated=False,seconds=time.perf_counter()-started),indent=2))

if __name__=='__main__':main()
