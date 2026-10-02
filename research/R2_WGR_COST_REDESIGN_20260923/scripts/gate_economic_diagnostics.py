from src.runtime import ROOT
from src.data.access_guard import install_guard
install_guard()
import json
import pandas as pd
import torch
from src.data.r2_data import R2Data
from src.calibration.calibrate_and_predict import load_model,latent_rows,predict_pack
from src.calibration.native_affine import NativeCalibration
from src.evaluation.forecast_metrics import evaluate_frame
from src.evaluation.economic_dev_eval import configure_development,run_one

torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
data=R2Data(multiscale=True);arr,reference=configure_development()
indices=data.indices['exploratory_test'];rows=[]
for name in ['C0_Repaired','C1_ProfileResidual']:
    model,cp=load_model(name,20260920,data)
    cal=NativeCalibration().to(data.device)
    cal.load_state_dict(torch.load(ROOT/'checkpoints'/f'{name}_seed20260920_calibration.pt',map_location=data.device,weights_only=True));cal.eval().requires_grad_(False)
    for variant in ['normal','force_gate_0','force_gate_1']:
        model.ablation=variant
        pack=latent_rows(model,data,indices);frame,error=predict_pack(model,data,pack,cal)
        forecast=evaluate_frame(frame,name,20260920,'native_fixed')
        metric=next(r for r in forecast if r['split']=='development_h2' and r['publication']=='pooled')
        economic=run_one(name,20260920,frame,arr,reference,windows=(12,24),ablation=variant)
        for row in economic:
            rows.append(dict(**row,pinball=metric['pinball'],tail_coverage=metric['tail_coverage'],
                savings_gain_vs_fixed_reference_pct=100*(row['value_usd']/3343.4716892558006-1)))
        pd.DataFrame(rows).to_csv(ROOT/'results/gate_economic_diagnostics_2024.csv',index=False)
        print(name,variant,'RT pinball',metric['pinball'],'tail',metric['tail_coverage'],flush=True)
        del pack
    del model,cal;torch.cuda.empty_cache()
print('GATE ECONOMIC DIAGNOSTICS COMPLETE',flush=True)
