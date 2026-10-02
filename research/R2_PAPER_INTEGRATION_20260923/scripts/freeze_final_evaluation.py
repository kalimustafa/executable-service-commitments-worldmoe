from src.runtime import ROOT,PREVIOUS,COST_STUDY
from src.evaluation.final_freeze import digest,verify_freeze
from datetime import datetime,timezone
import json

path=ROOT/'frozen_candidate/EVALUATION_FREEZE.json'
if path.exists():
    verify_freeze();print('Existing immutable freeze verified');raise SystemExit(0)
protocol=json.loads((ROOT/'configs/locked_protocol.json').read_text())
files=[];models=[]
def add(p):
    assert p.exists(),str(p)
    files.append(dict(path=str(p.resolve()),sha256=digest(p),bytes=p.stat().st_size))
for seed in [20260920,20260921,20260922]:
    source=PREVIOUS if seed==20260920 else ROOT
    trained=json.loads((source/'logs'/f'C1_ProfileResidual_seed{seed}_completed.json').read_text())
    calibrated=json.loads((source/'logs'/f'C1_ProfileResidual_seed{seed}_calibration_complete.json').read_text())
    assert trained['status']==calibrated['status']=='complete'
    assert trained['split_hash']==digest(ROOT/'configs/splits_2024_frozen.json')
    assert calibrated['evaluation_protocol']=='fp32_v1'
    for suffix in ['best.ckpt','calibration.pt','calibration.json']:
        add(source/'checkpoints'/f'C1_ProfileResidual_seed{seed}_{suffix}')
    models.append(dict(seed=seed,selected_epoch=trained['best_epoch'],selection_score=trained['best_price_score'],source=str(source)))
for filename in ['locked_protocol.json','paper_support_protocol.json','pv_particle_state_fitted.json','splits_2024_frozen.json','selected_w1.json']:
    add(ROOT/'configs'/filename)
add(ROOT/'cache/r2_normalization.json')
for p in sorted((ROOT/'src/models').glob('*.py')):
    assert digest(p)==digest(PREVIOUS/'src/models'/p.name),'Market architecture differs from evaluated pilot'
    add(p)
for p in [ROOT/'src/reservation_state_cost.py',ROOT/'vendor/reservation.py',ROOT/'vendor/economic_core.py',ROOT/'vendor/forecast_provider.py']:
    reference=COST_STUDY/('src' if p.parent.name=='src' else 'vendor')/p.name
    assert digest(p)==digest(reference),str(p)
    add(p)
manifest=dict(status='frozen',frozen_utc=datetime.now(timezone.utc).isoformat(),seeds=[20260920,20260921,20260922],
    primary_seed=20260920,models=models,files=files,selection_source='2024 development only',
    user_authorized_final_year_test=True,authorized_scope='2025 full year plus necessary early-2026 settlement tail; original external replications',
    post_evaluation_parameter_changes_allowed=False,primary_policy=dict(window_h=24,policy='interval',physical='particle_trailing4'),
    original_paper_inputs_read_only=True,W1_trainable_parameters=0)
path.write_text(json.dumps(manifest,indent=2));verify_freeze()
print(json.dumps({k:manifest[k] for k in ['status','frozen_utc','seeds','models']},indent=2))
