from src.runtime import ROOT,PREVIOUS
from src.evaluation.final_freeze import verify_freeze
import json,hashlib
import torch
verify_freeze();rows=[]
for seed in [20260920,20260921,20260922]:
    source=PREVIOUS if seed==20260920 else ROOT
    cp=torch.load(source/'checkpoints'/f'C1_ProfileResidual_seed{seed}_best.ckpt',map_location='cpu',weights_only=False)
    state=cp['state_dict'];draws=state['base.joint_normal_draws'].numpy()
    completed=json.loads((source/'logs'/f'C1_ProfileResidual_seed{seed}_completed.json').read_text())
    rows.append(dict(seed=seed,selected_epoch=cp['epoch'],phase=cp['stage'],price_selection_score=completed['best_price_score'],
        total_head_parameters=completed['total_parameters'],raw_residual_projection_norm=float(state['raw_projection.weight'].norm()),
        joint_samples=len(draws),joint_quadrature_sha256=hashlib.sha256(draws.tobytes()).hexdigest(),normalization_fitted_on_test=False))
result=dict(seeds=rows,quadrature_identical_across_seeds=len({r['joint_quadrature_sha256'] for r in rows})==1,
    torch_version=torch.__version__,post_final_test_parameters_changed=False)
(ROOT/'results/model_training_provenance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
