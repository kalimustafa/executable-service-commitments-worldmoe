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

def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    data=R2Data(multiscale=True);rows=[];gates=[]
    indices=torch.cat([data.indices['selection'],data.indices['exploratory_test']])
    for name in ['C0_Repaired','C1_ProfileResidual']:
        model,cp=load_model(name,20260920,data)
        cal=NativeCalibration().to(data.device)
        cal.load_state_dict(torch.load(ROOT/'checkpoints'/f'{name}_seed20260920_calibration.pt',map_location=data.device,weights_only=True));cal.eval().requires_grad_(False)
        for variant in ['normal','force_gate_0','force_gate_1']:
            model.ablation=variant
            pack=latent_rows(model,data,indices);frame,error=predict_pack(model,data,pack,cal)
            for metric in evaluate_frame(frame,name,20260920,'native_fixed'):
                if metric['publication']=='pooled':rows.append(dict(ablation=variant,**metric))
            if variant=='normal':
                for split,f in frame.groupby('split'):
                    for group,g in [('normal',f[f.market_regime.eq(0)]),('tail',f[f.tail_event]),('pre_DA',f[f.gate.eq('pre_DA')]),('post_DA',f[f.gate.eq('post_DA')])]:
                        gates.append(dict(model=name,seed=20260920,split=split,group=group,mean_gate=g.mean_gate.mean(),min_gate=g.mean_gate.min(),max_gate=g.mean_gate.max(),rows=len(g)))
            del pack
        del model,cal;torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(ROOT/'results/pilot_gate_ablation_2024.csv',index=False)
    pd.DataFrame(gates).to_csv(ROOT/'results/pilot_gate_summary_2024.csv',index=False)

if __name__=='__main__':main()
