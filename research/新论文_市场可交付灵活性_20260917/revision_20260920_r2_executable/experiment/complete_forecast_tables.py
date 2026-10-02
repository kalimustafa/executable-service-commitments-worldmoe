"""Complete reporting, without changing forecasts or selected parameters."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT.parent
from calibrate_r2 import forecast_metrics
f=pd.read_csv(ROOT/'results/R2_forecast_metrics.csv')
if not f.model.eq('Published_DA_point').any():
    g=pd.read_parquet(ROOT/'forecast_2025/R2_full_native.parquet');g=g[g.gate.eq('post_DA')].copy()
    for tau in [.05,.1,.25,.5,.75,.9,.95]:
        g[f'dart_q{tau:.2f}']=0.;g[f'rt_q{tau:.2f}']=g.da_lbmp.to_numpy();g[f'da_q{tau:.2f}']=g.da_lbmp.to_numpy()
    g['tail_probability']=np.nan
    f=pd.concat([f,pd.DataFrame(forecast_metrics(g,'Published_DA_point'))],ignore_index=True)
f['role']='forecast'
known=f.target.eq('da')&f.gate.eq('post_DA')
f.loc[known,'role']='published_anchor_not_a_forecast'
# Degenerate known DA anchors in float32 are not an uncertainty forecast.
for col in f.columns:
    if 'coverage' in col or 'width' in col or col=='tail_brier':f.loc[known,col]=np.nan
f.to_csv(ROOT/'results/R2_forecast_metrics.csv',index=False)
cols=['model','split','gate','target','rows','coverage50','width50','tail_coverage50','coverage80','width80','tail_coverage80','coverage90','width90','tail_coverage90','tail_brier','role']
f[cols].to_csv(ROOT/'results/R2_calibration_metrics_2025.csv',index=False)
dev=pd.read_csv(ROOT/'results/r2_calibration_metrics.csv')
pd.concat([dev,f[cols]],ignore_index=True).to_csv(ROOT/'results/R2_calibration_metrics.csv',index=False)
metadata=dict(tail_definition='absolute realized DART above the corresponding zone pre-2024 training 90th percentile; same mask for all models',
    quantile_levels=[.05,.1,.25,.5,.75,.9,.95],forecast_rows_per_gate=35040,post_DA_prices_known=True,
    known_DA_anchor_metrics_not_forecast_skill=True,model_choice_uses_2024_only=True,
    published_DA_point='RT equals published DA, DART equals zero; degenerate point distribution; post-DA only')
(ROOT/'results/forecast_metric_definitions.json').write_text(json.dumps(metadata,indent=2))
print(f[f.target.eq('dart')&f.gate.eq('post_DA')][['model','pinball','coverage90','tail_coverage90']].to_string(index=False))
