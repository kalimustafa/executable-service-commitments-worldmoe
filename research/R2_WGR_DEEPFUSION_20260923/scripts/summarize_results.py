from src.runtime import ROOT
from src.data.access_guard import install_guard
install_guard()
import json
import numpy as np
import pandas as pd

def main():
    forecast=pd.read_csv(ROOT/'results/forecast_metrics_2024.csv')
    economic=pd.read_csv(ROOT/'results/economic_dev_metrics_2024.csv')
    summary=[]
    base=forecast[forecast.calibration.eq('native')&forecast.publication.eq('pooled')]
    for (name,split),f in base.groupby(['model','split']):
        assert f.seed.nunique()==3
        for key in ['pinball','da_pinball','dart_pinball','coverage50','coverage90','tail_coverage','tail_pinball','tail_interval_score90','interval_width','regime_accuracy','regime_log_loss','mean_gate']:
            summary.append(dict(domain='forecast',model=name,split=split,window_h=None,policy=None,metric=key,n_seeds=len(f),mean=f[key].mean(),std=f[key].std(ddof=1),minimum=f[key].min(),maximum=f[key].max()))
    for (name,window,policy),f in economic[economic.window_h.isin([12,24])].groupby(['model','window_h','policy']):
        for key in ['value_usd','cvar95','shifted_it_mwh','late_it_mwh','closing_work_mwh','selection_score']:
            summary.append(dict(domain='economics',model=name,split='economic_selection',window_h=window,policy=policy,metric=key,n_seeds=len(f),mean=f[key].mean(),std=f[key].std(ddof=1) if len(f)>1 else 0.,minimum=f[key].min(),maximum=f[key].max()))
    pd.DataFrame(summary).to_csv(ROOT/'results/seed_summary_2024.csv',index=False)
    rows=[];normalization=json.loads((ROOT/'cache/r2_normalization.json').read_text());scales=np.array(normalization['target_scale'])
    for path in sorted((ROOT/'checkpoints').glob('A*_seed*_calibration.json')):
        name,seedtext=path.stem.removesuffix('_calibration').rsplit('_seed',1);seed=int(seedtext)
        p=json.loads(path.read_text())['parameters']
        for target,j in [('da',0),('dart',1)]:
            for state in ([0] if target=='da' else [0,1]):
                shifts=np.array(p[target+'_shift']);width=np.array(p[target+'_width'])
                if target=='dart':shifts=shifts[state];width=width[state]
                for zone,z in enumerate(normalization['zones']):
                    rows.append(dict(model=name,seed=seed,zone=z,target=target,publication='pre_DA' if state==0 else 'post_DA',
                        shift_normalized=shifts[zone],shift_usd_per_mwh=shifts[zone]*scales[zone,j],raw_log_width=width[zone],
                        width_factor=np.exp(np.clip(width[zone],-2,2)),calibration_parameters=24,epochs=250))
    params=pd.DataFrame(rows);params.to_csv(ROOT/'results/calibration_parameters_2024.csv',index=False)
    stats=params.groupby(['model','seed']).agg(min_shift_usd=('shift_usd_per_mwh','min'),max_shift_usd=('shift_usd_per_mwh','max'),
        min_width_factor=('width_factor','min'),max_width_factor=('width_factor','max')).reset_index()
    forecast[forecast.split.eq('calibration')].merge(stats,on=['model','seed'],validate='many_to_one').to_csv(ROOT/'results/calibration_metrics_2024.csv',index=False)
    h2=base[base.split.eq('development_h2')].set_index(['model','seed']);comparisons=[]
    for seed in [20260920,20260921,20260922]:
        wgr=h2.loc[('A3_WGR',seed)]
        for name in ['A0_Base','A1_Shallow','A2_Deep_NoGate']:
            other=h2.loc[(name,seed)]
            comparisons.append(dict(seed=seed,comparison='A3_WGR minus '+name,pinball_delta=wgr.pinball-other.pinball,
                tail_coverage_pp_delta=100*(wgr.tail_coverage-other.tail_coverage),tail_pinball_delta=wgr.tail_pinball-other.tail_pinball))
    pd.DataFrame(comparisons).to_csv(ROOT/'results/matched_seed_forecast_deltas.csv',index=False)
    edelta=[]
    e=economic[economic.window_h.isin([12,24])].set_index(['model','seed','window_h','policy'])
    for seed in [20260920,20260921,20260922]:
        for window in [12,24]:
            for policy in ['point','interval']:
                w=e.loc[('A3_WGR',seed,window,policy)]
                for name in ['A0_Base','A1_Shallow']:
                    other=e.loc[(name,seed,window,policy)]
                    edelta.append(dict(seed=seed,window_h=window,policy=policy,comparison='A3_WGR minus '+name,
                        value_delta_usd=w.value_usd-other.value_usd,selection_score_delta=w.selection_score-other.selection_score,
                        late_delta_mwh=w.late_it_mwh-other.late_it_mwh))
    pd.DataFrame(edelta).to_csv(ROOT/'results/matched_seed_economic_deltas.csv',index=False)
    print(base[base.split.eq('development_h2')][['model','seed','pinball','tail_coverage','mean_gate']].to_string(index=False),flush=True)

if __name__=='__main__':main()
