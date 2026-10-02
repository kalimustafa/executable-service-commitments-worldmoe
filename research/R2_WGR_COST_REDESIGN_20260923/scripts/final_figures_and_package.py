from src.runtime import ROOT,FAILED
from src.data.access_guard import install_guard,read_development_parquet
install_guard()
import json,hashlib,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.evaluate_world_site_cost import physical_forecasts

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    source=read_development_parquet(ROOT/'results/predictions_2024/C0_Repaired_seed20260920_native.parquet')
    old=pd.read_csv(FAILED/'results/forecast_metrics_2024.csv');new=pd.read_csv(ROOT/'results/forecast_metrics_2024.csv')
    f=pd.concat([old[old.model.isin(['A0_Base','A1_Shallow'])&old.seed.eq(20260920)],new],ignore_index=True)
    f=f[(f.split=='development_h2')&(f.calibration=='native')&(f.publication=='pooled')]
    order=['A0_Base','A1_Shallow','C0_Repaired','C1_ProfileResidual','C3_TemporalWorld','C4_RecentReplay']
    labels=['Base','Shallow','Repaired WGR','Profile residual','Hourly world','Recent replay']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,key,title,mult in [(axes[0],'pinball','RT pinball (USD/MWh)',1),(axes[1],'tail_coverage','Tail coverage of 90% RT interval (%)',100)]:
        for i,name in enumerate(order):
            row=f[f.model.eq(name)].iloc[0];ax.scatter(row[key]*mult,i,color='#3a6e9e',s=45)
        ax.set_yticks(range(len(order)),labels);ax.invert_yaxis();ax.set_xlabel(title);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Market forecasting controls | seed 20260920 | 2024 development H2')
    fig.text(.5,.01,'Selected SiteCost uses the unchanged Repaired WGR market forecasts; the cost gain is not a new RT-accuracy gain.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(folder/'Fig_A_model_comparison.png',dpi=180);plt.close(fig)
    with np.load(ROOT/'results/C7_IssueSiteCost_cost_head_2024.npz',allow_pickle=False) as cost:
        times=pd.DatetimeIndex(pd.to_datetime(cost['timestamps_utc'],utc=True));a=cost['a'];b=cost['b'];valid=cost['pre_available']
    pv,pue,_,_,_=physical_forecasts(times,'issue')
    config=json.loads((ROOT/'results/world_site_cost_audit.json').read_text())['models'][1]
    ref=np.asarray(config['pue_reference']);gate=(-b*ref[None]/pue).clip(0,1);gate[~valid]=0
    g=pd.DataFrame({'UTC_hour':times.hour,'PV_gate':gate.mean(1),'cost_scale':a.mean(1)})
    hour=g.groupby('UTC_hour').mean()
    fig,ax=plt.subplots(figsize=(8.8,4));ax.plot(hour.index,hour.PV_gate,'o-',color='#2c8974')
    ax.set_ylim(0,1);ax.set_xlabel('UTC hour');ax.set_ylabel('Mean fraction of added load covered by forecast PV')
    ax.set_title('Physical cost gate: PV matters when it can cover additional work');ax.grid(alpha=.2)
    fig.text(.5,.01,'DEVELOPMENT YEAR ONLY | derived from registered issued priors, not from realized future PV',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(folder/'Fig_B_gate_behavior.png',dpi=180);plt.close(fig)
    rows=[]
    for name,label,base in [('RASC','RASC',FAILED),('A0_Base','Base',FAILED),('A1_Shallow','Shallow',FAILED),
                            ('C0_Repaired','Repaired WGR',ROOT),('C7_WorldSiteCost','W1 decoded cost',ROOT),('C7_IssueSiteCost','Issued-prior SiteCost',ROOT)]:
        seed=0 if name=='RASC' else 20260920
        p=base/'results/economic_trajectories'/f'{name}_seed{seed}_normal_24h_robust_fp32v1_metrics.json'
        row=json.loads(p.read_text());rows.append(dict(label=label,**row))
    table=pd.DataFrame(rows);full=json.loads((ROOT/'results/cost_scope.json').read_text())['reference_operating_cost_usd']
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8));colors=['#95a1aa']*5+['#22836b']
    axes[0].barh(table.label,table.value_usd,color=colors);axes[0].invert_yaxis();axes[0].axvline(4346.513196032541,color='#a64b59',ls='--',label='Fixed +30% savings target')
    axes[0].set_xlabel('Savings versus reference operation (USD)');axes[0].legend(fontsize=8)
    net=full-table.value_usd;axes[1].scatter(net,np.arange(len(table)),c=colors,s=45)
    axes[1].set_yticks(np.arange(len(table)),table.label);axes[1].invert_yaxis();axes[1].set_xlabel('Total net operating cost (USD)');axes[1].grid(axis='x',alpha=.2)
    fig.suptitle('24h interval | same executor and service contract | seed 20260920')
    fig.text(.5,.012,'DEVELOPMENT YEAR ONLY | savings +54.30%; total operating cost -1.60%. These are different denominators.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(folder/'Fig_C_economic_dev.png',dpi=180);plt.close(fig)
    h2=source[(source.split=='exploratory_test')&(source.world_available>0)]
    experts=h2.groupby('market_regime')[[f'expert_attention_{j}' for j in range(7)]].mean()
    fig,ax=plt.subplots(figsize=(8.8,4))
    for regime,label in [(0,'Normal'),(1,'Moderate'),(2,'Tail')]:ax.plot(range(7),experts.loc[regime],'o-',label=label)
    ax.set_xticks(range(7),[f'Expert {j}' for j in range(7)]);ax.set_ylim(0,1);ax.legend();ax.set_ylabel('Mean expert attention')
    ax.set_title('Inherited market-branch expert usage; not the source of the cost-head gain');ax.grid(axis='y',alpha=.2)
    fig.text(.5,.01,'DEVELOPMENT YEAR ONLY | no physical or economic role is asserted for a numbered expert',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(folder/'Fig_D_expert_usage.png',dpi=180);plt.close(fig)
    (folder/'figure_sources.json').write_text(json.dumps(dict(A='forecast_metrics_2024.csv, primary seed, FP32 native, development H2',
        B='C7_IssueSiteCost_cost_head_2024.npz; gate = forecast additional PV / incremental load',
        C='original and revised 24h interval metric JSON files, same fixed block',D='unchanged C0 market predictions, observed regime diagnostic'),indent=2))
    # Minimal executable source snapshot; large W1 weights/caches remain
    # read-only references rather than copies in the frozen package.
    package=ROOT/'frozen_candidate';code=package/'source_snapshot';code.mkdir(exist_ok=True)
    sources=['src/models/world_site_cost.py','src/models/r2_repaired.py','src/models/market_query.py',
             'src/models/world_multiscale_adapter.py','src/models/cross_attention.py','src/models/world_gate.py',
             'vendor/r2_market.py','vendor/forecast_provider.py','vendor/reservation.py','vendor/economic_core.py',
             'scripts/evaluate_world_site_cost.py']
    manifest={}
    for relative in sources:
        p=ROOT/relative;target=code/relative.replace('/','__');shutil.copy2(p,target);manifest[relative]=dict(sha256=sha(p),snapshot=target.name)
    (package/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    (package/'REPRODUCE.md').write_text('# Reproduce the selected development candidate\n\nFrom the experiment root, run:\n\n```powershell\n.\\scripts\\run.ps1 scripts/evaluate_world_site_cost.py --recompute\n```\n\nThis uses the fixed 2024 inputs and reruns the unchanged executor, overriding its local result cache. The frozen price checkpoint/calibrator and deterministic cost configuration are here; registered W1 and development caches remain read-only references in the experiment workspace. The source snapshot is inspectable code evidence, not a standalone replacement environment. No holdout year is enabled.\n',encoding='utf-8')
    hashes={str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    (package/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2))
    report=ROOT/'reports/SAVINGS_30PCT_RESULT.md'
    with report.open('a',encoding='utf-8') as stream:
        stream.write('\n## 诊断图\n\n- [预测对照](../figures/Fig_A_model_comparison.png)\n- [物理成本 gate](../figures/Fig_B_gate_behavior.png)\n- [开发期经济对照](../figures/Fig_C_economic_dev.png)\n- [市场分支 expert 使用](../figures/Fig_D_expert_usage.png)\n')
    (ROOT/'README.md').write_text('# R2-SiteCost: primary-seed savings goal met\n\nThe user-defined target was at least 30% greater savings than the fixed corrected WGR comparator. Measured 24h interval savings: USD 5158.850885 versus USD 3343.471689, +54.2962%. Total net operating cost declined 1.60115%.\n\nSelected variant C7_IssueSiteCost combines the frozen repaired market branch with an explicitly separate physical marginal-cost head using registered issued priors. It does not claim a +54% market forecasting or learned-W1-decoder gain.\n\nRead reports/SAVINGS_30PCT_RESULT.md and results/SAVINGS_TARGET_MET.json. frozen_candidate contains the fixed primary-seed package. All evidence is 2024 development/counterfactual replay; other seeds and locked 2025/2026 years were not run. Search stopped once the goal was met.\n',encoding='utf-8')
    print('FOUR FIGURES AND FROZEN PACKAGE COMPLETE',flush=True)

if __name__=='__main__':main()
