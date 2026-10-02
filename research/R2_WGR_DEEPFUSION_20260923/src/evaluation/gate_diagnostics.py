from src.runtime import ROOT
from src.data.access_guard import install_guard,read_development_parquet
install_guard()
import json
import numpy as np
import pandas as pd

def main():
    frames=[];summaries=[]
    for seed in [20260920,20260921,20260922]:
        frame=read_development_parquet(ROOT/'results/predictions_2024'/f'A3_WGR_seed{seed}_native.parquet')
        frame=frame.copy();frame['seed']=seed
        columns=['timestamp_utc','date','zone','seed','split','lead_hours','gate','market_regime','tail_event','world_available','mean_gate','max_gate','router_entropy','day_std_mean7']
        columns += [f'gate_group_{j}' for j in range(8)]+[f'expert_attention_{j}' for j in range(7)]
        diagnostics=frame[columns].rename(columns={'lead_hours':'target_lead','gate':'publication_state'})
        groups=diagnostics[[f'gate_group_{j}' for j in range(8)]].to_numpy()
        alpha=diagnostics[[f'expert_attention_{j}' for j in range(7)]].to_numpy()
        diagnostics['gate_by_latent_group']=[json.dumps([round(float(x),6) for x in v]) for v in groups]
        diagnostics['selected_experts']=[json.dumps(np.argsort(-v)[:2].tolist()) if v.sum()>0 else '[]' for v in alpha]
        frames.append(diagnostics)
        for split,sub in diagnostics.groupby('split'):
            # Volatility threshold is from training data, not dev outcomes.
            from src.runtime import PAPER
            if 'threshold' not in locals():
                train=read_development_parquet(PAPER/'experiments/gate_hourly_panel.parquet',columns=['date','split','day_std_mean7'])
                threshold=float(train[train.split.eq('train')].day_std_mean7.median())
            groups={'normal_regime':sub.market_regime.eq(0),'moderate_regime':sub.market_regime.eq(1),'tail_regime':sub.tail_event,
                    'low_volatility':sub.day_std_mean7<=threshold,'high_volatility':sub.day_std_mean7>threshold,
                    'pre_DA':sub.publication_state.eq('pre_DA'),'post_DA':sub.publication_state.eq('post_DA')}
            for label,mask in groups.items():
                part=sub[mask]
                summaries.append(dict(seed=seed,split=split,group=label,rows=len(part),mean_gate=part.mean_gate.mean(),
                    median_gate=part.mean_gate.median(),q05_gate=part.mean_gate.quantile(.05),q95_gate=part.mean_gate.quantile(.95),
                    near_zero_fraction=(part.mean_gate<.01).mean(),near_one_fraction=(part.mean_gate>.99).mean(),
                    world_available_rate=part.world_available.mean(),volatility_threshold_train_median=threshold))
            for lead,part in sub.groupby(pd.cut(sub.target_lead,[0,24,30,36,42,49],right=False,include_lowest=True),observed=True):
                summaries.append(dict(seed=seed,split=split,group='lead_'+str(lead),rows=len(part),mean_gate=part.mean_gate.mean(),median_gate=part.mean_gate.median(),
                    q05_gate=part.mean_gate.quantile(.05),q95_gate=part.mean_gate.quantile(.95),near_zero_fraction=(part.mean_gate<.01).mean(),
                    near_one_fraction=(part.mean_gate>.99).mean(),world_available_rate=part.world_available.mean(),volatility_threshold_train_median=threshold))
    pd.concat(frames,ignore_index=True).to_csv(ROOT/'results/gate_diagnostics.csv',index=False)
    pd.DataFrame(summaries).to_csv(ROOT/'results/gate_summary_2024.csv',index=False)
    primary=frames[0];h2=primary[primary.split.eq('exploratory_test')]
    expert=h2.groupby('market_regime')[[f'expert_attention_{j}' for j in range(7)]].mean()
    expert.to_csv(ROOT/'results/expert_usage_2024.csv')
    print('GATE DIAGNOSTICS COMPLETE',sum(len(f) for f in frames),flush=True)

if __name__=='__main__':main()
