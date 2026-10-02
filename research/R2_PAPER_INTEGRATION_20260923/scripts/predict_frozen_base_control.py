from src.runtime import ROOT,FAILED
from src.data.access_guard import enable_final_evaluation,install_guard
enable_final_evaluation();install_guard()
import json
import torch
from src.data.final_data import R2FinalData
from src.models.factory import make_model
from src.calibration.calibrate_and_predict import latent_rows,predict_pack
from src.calibration.native_affine import NativeCalibration
from src.evaluation.final_freeze import digest,verify_freeze

torch.set_num_threads(4);torch.use_deterministic_algorithms(True);data=R2FinalData()
path=FAILED/'checkpoints/A0_Base_seed20260920_best.ckpt';calpath=FAILED/'checkpoints/A0_Base_seed20260920_calibration.pt'
model=make_model('A0_Base',20260920,data)
cp=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(cp['state_dict'])
model.eval().requires_grad_(False)
cal=NativeCalibration().to(data.device);cal.load_state_dict(torch.load(calpath,map_location=data.device,weights_only=True));cal.eval().requires_grad_(False)
pack=latent_rows(model,data,data.indices['evaluation_2025']);frame,error=predict_pack(model,data,pack,cal)
frame.to_parquet(ROOT/'results/predictions_2025/A0_Base_seed20260920_native.parquet',index=False)
(ROOT/'results/base_control_provenance.json').write_text(json.dumps(dict(checkpoint=str(path),checkpoint_sha256=digest(path),
    calibration=str(calpath),calibration_sha256=digest(calpath),identity_max_error=error,parameters_updated=False,
    rule='Previously trained primary-seed no-world control; no final-year selection'),indent=2))
verify_freeze();print('FROZEN BASE CONTROL PREDICTIONS COMPLETE',error,flush=True)
