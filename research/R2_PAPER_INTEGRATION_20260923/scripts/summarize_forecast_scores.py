from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT.parent/'新论文_市场可交付灵活性_20260917'
tau=np.array([.05,.1,.25,.5,.75,.9,.95]);rows=[]
specs=[(f'R2 (seed {seed})',ROOT/'results/predictions_2025'/f'R2_seed{seed}_normal_native.parquet') for seed in [20260920,20260921,20260922]]
specs += [('R2 without residual correction',ROOT/'results/predictions_2025/R2_seed20260920_force_gate_0_native.parquet'),('Base price model',ROOT/'results/predictions_2025/A0_Base_seed20260920_native.parquet')]
def add(name,z,q):
    y=z.spread.to_numpy();e=y[:,None]-q;inside=(y>=q[:,0])&(y<=q[:,-1]);tail=z.tail_event.to_numpy(bool)
    rows.append(dict(Model=name,Pinball=f'{np.maximum(tau*e,(tau-1)*e).mean():.3f}',MAE=f'{np.abs(y-q[:,3]).mean():.3f}',
        Coverage50=f'{100*((y>=q[:,2])&(y<=q[:,4])).mean():.2f}',Coverage90=f'{100*inside.mean():.2f}',Tail90=f'{100*inside[tail].mean():.2f}'))
for name,path in specs:
    z=pd.read_parquet(path);z=z[z.gate.eq('post_DA')];truth=z[['timestamp_utc','zone','spread','tail_event']]
    add(name,z,z[[f'dart_q{t:.2f}' for t in tau]].to_numpy())
z=pd.read_parquet(PAPER/'revision_20260918/forecast_2025/bridge_forecast_only_hourly.parquet')
z=z[z.model.eq('RASC')].drop(columns=['spread','tail_event'],errors='ignore').merge(truth,on=['timestamp_utc','zone'],validate='one_to_one')
add('RASC',z,z[[f'q{t:.2f}' for t in tau]].to_numpy())
pd.DataFrame(rows).to_csv(ROOT/'results/paper_forecast_summary.csv',index=False)
print('Forecast summary exported',len(rows))
