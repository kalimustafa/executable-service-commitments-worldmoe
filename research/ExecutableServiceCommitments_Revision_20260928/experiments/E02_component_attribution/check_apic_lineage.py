"""Measure retained R2 price identity and relabel the final-shift intervention."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
WORKSPACE=BASE.parent
SOURCE=BASE/'source_data'
path=WORKSPACE/'R2_PAPER_INTEGRATION_20260923/results/predictions_2025/R2_seed20260920_normal_native.parquet'
tau=(.05,.10,.25,.50,.75,.90,.95)
cols=['gate','da_lbmp']+[f'rt_q{x:.2f}' for x in tau]+[f'dart_q{x:.2f}' for x in tau]
frame=pd.read_parquet(path,columns=cols)
post=frame[frame.gate.eq('post_DA')]
rt=post[[f'rt_q{x:.2f}' for x in tau]].to_numpy(float)
dart=post[[f'dart_q{x:.2f}' for x in tau]].to_numpy(float)[:,::-1]
residual=rt-(post.da_lbmp.to_numpy(float)[:,None]-dart)
result=dict(stage='R2 post-DA',seed=20260920,rows=len(post),quantiles=len(tau),
    max_abs_identity_residual_usd_mwh=float(np.max(np.abs(residual))),
    mean_abs_identity_residual_usd_mwh=float(np.mean(np.abs(residual))),
    q99_abs_identity_residual_usd_mwh=float(np.quantile(np.abs(residual),.99)),
    formula='q_RT(tau|DA)=DA-q_DART(1-tau|DA)',source_parquet=str(path))
pd.DataFrame([result]).to_csv(SOURCE/'apic_r2_postda_identity.csv',index=False)

raw_source=HERE/'terminal_market_shift_raw.csv'
if not raw_source.exists():raw_source=HERE/'apic_diagnostic.csv'
raw=pd.read_csv(raw_source)
raw=raw.rename(columns={'apic_residual':'final_market_shift'})
raw['upstream_w0w1_r2_apic_unchanged']=True
raw['interpretation']='Only the final learned market residual shift is zeroed; W0/W1 and R2 APIC representations remain active. This is not an APIC on/off ablation.'
raw.to_csv(SOURCE/'final_market_shift_diagnostic.csv',index=False)

lineage=dict(w0w1_identity_source='power-dc-vpp/src/dcvpp/vpp_world_moe/losses.py:_assert_price_identity and _assert_joint_price_paths',
    r2_identity_source='R2_PAPER_INTEGRATION_20260923/vendor/r2_market.py:from_latent and identity_errors',
    r2_postda_result=result,
    final_shift_scope='Final CdfR2 learned residual shift only; upstream APIC is retained in both branches',
    forbidden_inference='The shift-zero diagnostic cannot establish APIC contribution or remove the APIC mechanism.')
(SOURCE/'apic_lineage_check.json').write_text(json.dumps(lineage,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
