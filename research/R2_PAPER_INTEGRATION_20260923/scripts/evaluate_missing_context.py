from src.evaluation.paper_economics import *
missing=json.loads((ROOT/'cache/final_world/manifest.json').read_text())['missing_dates']
normal=pd.read_parquet(forecast_file(2025))
base=pd.read_parquet(ROOT/'results/predictions_2025/A0_Base_seed20260920_native.parquet')
keys=['timestamp_utc','zone','gate'];assert normal[keys].equals(base[keys])
mask=normal.date.isin(missing);assert mask.sum()==3*24*4*2
replaced=normal.copy();replaced.loc[mask,:]=base.loc[mask,:]
replaced.to_parquet(forecast_file(2025,20260920,'missing_base'),index=False)
arr,reference,shared=configure(2025);rows=[]
for window in [12,24]:
    for policy in ['point','interval']:
        rows.append(run_homogeneous(arr,reference,shared,year=2025,trace='Eagle',proxy='registered',seed=20260920,
            window=window,policy=policy,ablation='missing_base'))
pd.DataFrame(rows).to_csv(ROOT/'results/missing_context_replacement.csv',index=False)
original=pd.read_csv(ROOT/'results/multiseed_economics_2025.csv');baseline=pd.read_csv(ROOT/'results/final_baseline_window_sweep.csv')
subtotals=[]
for row in original.to_dict('records'):
    other=baseline[(baseline.model=='RASC')&(baseline.window_h==row['window_h'])&(baseline.policy==row['policy'])].iloc[0]
    directory=ROOT/'results/economic_trajectories'
    a=pd.read_csv(directory/f'{row["name"]}_daily.csv').set_index('date')
    b=pd.read_csv(directory/f'{other["name"]}_daily.csv').set_index('date');assert a.index.equals(b.index)
    difference=a.mtm_value_usd-b.mtm_value_usd
    subtotals.append(dict(seed=row['seed'],window_h=row['window_h'],policy=row['policy'],full_difference_usd=float(difference.sum()),
        retained_days=int((~difference.index.isin(missing)).sum()),retained_difference_usd=float(difference[~difference.index.isin(missing)].sum()),
        trajectories_truncated=False,annualized=False))
pd.DataFrame(subtotals).to_csv(ROOT/'results/missing_context_day_exclusion.csv',index=False)
print('MISSING CONTEXT SENSITIVITIES COMPLETE',flush=True)
