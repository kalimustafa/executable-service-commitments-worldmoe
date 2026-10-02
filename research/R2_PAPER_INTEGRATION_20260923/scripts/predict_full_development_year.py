"""Full-calendar 2024 predictions for declared development sensitivities."""
from src.runtime import ROOT,PREVIOUS
from src.data.access_guard import install_guard
install_guard()
import torch
from src.data.r2_data import R2Data
from src.calibration.calibrate_and_predict import load_model,latent_rows,predict_pack
from src.calibration.native_affine import NativeCalibration
from src.evaluation.final_freeze import verify_freeze

verify_freeze();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
data=R2Data(multiscale=True)
indices=torch.tensor(data.frame.index[data.frame.date.between('2024-01-01','2024-12-31')].to_numpy(),device=data.device)
assert len(indices)==35136
for seed in [20260920,20260921,20260922]:
    source=PREVIOUS if seed==20260920 else ROOT
    model,_=load_model('C1_ProfileResidual',seed,data)
    cal=NativeCalibration().to(data.device)
    cal.load_state_dict(torch.load(source/'checkpoints'/f'C1_ProfileResidual_seed{seed}_calibration.pt',map_location=data.device,weights_only=True))
    cal.eval().requires_grad_(False)
    pack=latent_rows(model,data,indices);frame,error=predict_pack(model,data,pack,cal)
    frame['evaluation_role']='2024 development sensitivity; overlaps selection/calibration'
    frame.to_parquet(ROOT/'results/predictions_2024'/f'R2_seed{seed}_normal_native.parquet',index=False)
    print('FULL 2024',seed,len(frame),error,flush=True)
    del model,cal,pack;torch.cuda.empty_cache()
verify_freeze()
