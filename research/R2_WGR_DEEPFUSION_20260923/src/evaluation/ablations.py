from src.runtime import ROOT
from src.data.access_guard import install_guard,assert_target_year
install_guard()
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from src.data.r2_data import R2Data
from src.calibration.calibrate_and_predict import load_model,latent_rows,predict_pack
from src.calibration.native_affine import NativeCalibration
from src.evaluation.forecast_metrics import evaluate_frame
from src.evaluation.economic_dev_eval import configure_development,run_one,aggregate_results,choose_policy

VARIANTS=['normal','force_gate_0','force_gate_1','no_expert_tokens','late_scale_only']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--target-year',type=int,default=2024);args=parser.parse_args()
    assert_target_year(args.target_year);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    seed=20260920;data=R2Data(multiscale=True);model,cp=load_model('A3_WGR',seed,data)
    native=NativeCalibration().to(data.device)
    native.load_state_dict(torch.load(ROOT/'checkpoints'/f'A3_WGR_seed{seed}_calibration.pt',weights_only=True,map_location=data.device))
    native.eval().requires_grad_(False)
    indices=torch.cat([data.indices[s] for s in ['selection','calibration','exploratory_test']])
    arr,reference=configure_development();allrows=[]
    for variant in VARIANTS:
        model.ablation=variant
        prediction=ROOT/'results/predictions_2024'/f'A3_WGR_seed{seed}_ablation_{variant}_fp32.parquet'
        if prediction.exists():frame=pd.read_parquet(prediction)
        else:
            pack=latent_rows(model,data,indices);frame,error=predict_pack(model,data,pack,native)
            frame.to_parquet(prediction,index=False);del pack
        forecast=evaluate_frame(frame,'A3_WGR',seed,'native_frozen')
        economics=run_one('A3_WGR',seed,frame,arr,reference,ablation=variant)
        for met in forecast:
            if met['publication']!='pooled':continue
            for window in [12,24]:
                erows=[r for r in economics if r['window_h']==window]
                # Also provide a fair fixed-policy metric using the normal
                # model's selected policy, so a policy switch is visible.
                chosen=choose_policy(erows)
                allrows.append(dict(ablation=variant,**met,window_h=window,economic_policy=chosen['policy'],
                    economic_selection_metric=chosen['selection_score'],value_usd=chosen['value_usd'],late_it_mwh=chosen['late_it_mwh'],
                    calibration_refitted=False,checkpoint_epoch=cp['epoch']))
        print('ABLATION COMPLETE',variant,flush=True)
        aggregate_results()
        pd.DataFrame(allrows).to_csv(ROOT/'results/world_ablation_2024.csv',index=False)
    table=pd.DataFrame(allrows)
    table['fixed_normal_policy']='';table['fixed_normal_policy_selection_metric']=np.nan;table['fixed_normal_policy_value_usd']=np.nan
    for window in [12,24]:
        normal=choose_policy([json.loads(p.read_text()) for p in (ROOT/'results/economic_trajectories').glob(f'A3_WGR_seed{seed}_normal_{window}h_*_fp32v1_metrics.json')])
        policy='robust' if normal['policy']=='interval' else 'point'
        for variant in VARIANTS:
            econ=json.loads((ROOT/'results/economic_trajectories'/f'A3_WGR_seed{seed}_{variant}_{window}h_{policy}_fp32v1_metrics.json').read_text())
            mask=table.window_h.eq(window)&table.ablation.eq(variant)
            table.loc[mask,'fixed_normal_policy']=normal['policy']
            table.loc[mask,'fixed_normal_policy_selection_metric']=econ['selection_score']
            table.loc[mask,'fixed_normal_policy_value_usd']=econ['value_usd']
    table.to_csv(ROOT/'results/world_ablation_2024.csv',index=False)
    (ROOT/'logs/ablations_completed.json').write_text(json.dumps(dict(status='complete',seed=seed,variants=VARIANTS,
        calibration='same frozen primary-seed native calibrator, never refitted',
        late_scale_only='remove early and middle interaction token banks; retain final-scale site/global/tail and expert bank',
        force_gate_zero='internal jointly trained base path; not a replacement for independent A0 comparison',locked_targets_accessed=False),indent=2))

if __name__=='__main__':main()
