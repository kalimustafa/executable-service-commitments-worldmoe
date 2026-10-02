from src.runtime import ROOT
from src.data.access_guard import install_guard
install_guard()
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORDER=['A0_Base','A1_Shallow','A2_Deep_NoGate','A3_WGR']
LABELS={'RASC':'RASC','A0_Base':'Base','A1_Shallow':'Shallow','A2_Deep_NoGate':'Deep-NoGate','A3_WGR':'WGR'}
COLORS={'RASC':'#878787','A0_Base':'#3A6EA5','A1_Shallow':'#36917F','A2_Deep_NoGate':'#AF8A3E','A3_WGR':'#AA4F64'}

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,
                         'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})

def seed_points(ax,table,models,metric,multiplier=1):
    for x,name in enumerate(models):
        values=table[table.model.eq(name)].sort_values('seed')[metric].to_numpy()*multiplier
        ax.vlines(x,values.min(),values.max(),color=COLORS[name],alpha=.5,lw=2)
        ax.scatter(x+np.linspace(-.08,.08,len(values)),values,color=COLORS[name],s=35,zorder=3)
        ax.plot([x-.19,x+.19],[values.mean()]*2,color=COLORS[name],lw=2.5)
    ax.set_xticks(range(len(models)),[LABELS[m] for m in models]);ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)

def main():
    style();out=ROOT/'figures'
    f=pd.read_csv(ROOT/'results/forecast_metrics_2024.csv')
    f=f[f.split.eq('development_h2')&f.calibration.eq('native')&f.publication.eq('pooled')]
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.2))
    seed_points(axes[0],f,ORDER,'pinball');axes[0].set_ylabel('RT seven-quantile pinball (USD/MWh)');axes[0].set_title('Lower forecast loss is better')
    seed_points(axes[1],f,ORDER,'tail_coverage',100);axes[1].set_ylabel('90% RT interval coverage on tail rows (%)');axes[1].set_title('Tail coverage varies by seed')
    fig.suptitle('2024 development H2 | unified native affine calibration',fontsize=13)
    fig.text(.5,.01,'DEVELOPMENT YEAR ONLY  |  dots: all three seeds; line: mean; span: min-max (not a confidence interval)',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.055,1,.94]);fig.savefig(out/'Fig_A_model_comparison.png',dpi=200);plt.close(fig)
    g=pd.read_csv(ROOT/'results/gate_summary_2024.csv');g=g[g.split.eq('exploratory_test')]
    pairs=[('normal_regime','tail_regime','Observed regime'),('low_volatility','high_volatility','Lagged volatility'),('pre_DA','post_DA','Publication state')]
    fig,axes=plt.subplots(1,3,figsize=(11,4.1))
    for ax,(a,b,title) in zip(axes,pairs):
        for seed,color in zip([20260920,20260921,20260922],['#AA4F64','#366EAA','#3F8C76']):
            sub=g[g.seed.eq(seed)].set_index('group');vals=[sub.loc[a,'mean_gate'],sub.loc[b,'mean_gate']]
            ax.plot([0,1],vals,'o-',label=str(seed),color=color,lw=1.8)
        ax.set_xticks([0,1],[a.replace('_',' '),b.replace('_',' ')]);ax.set_ylim(0,1);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Mean vector gate (all development H2 rows)');axes[2].legend(title='Fixed seed',fontsize=8)
    fig.suptitle('Gate responds to context; stronger opening does not establish value',fontsize=13)
    fig.text(.5,.01,'DEVELOPMENT YEAR ONLY  |  tail labels are retrospective diagnostics; volatility split uses training median',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(out/'Fig_B_gate_behavior.png',dpi=200);plt.close(fig)
    economic=pd.read_csv(ROOT/'results/economic_dev_metrics_2024.csv')
    models=['RASC']+ORDER
    fig,axes=plt.subplots(2,2,figsize=(11.5,7))
    for i,window in enumerate([12,24]):
        for j,policy in enumerate(['point','interval']):
            ax=axes[i,j];table=economic[economic.window_h.eq(window)&economic.policy.eq(policy)]
            seed_points(ax,table,models,'value_usd');ax.axhline(0,color='#666',lw=.7)
            ax.set_title(f'{window} h | {policy} policy');ax.set_ylabel('2024 block value (USD)');ax.tick_params(axis='x',labelsize=9)
    fig.suptitle('2024-08-21 to 2024-12-31 UTC | unchanged execution rules',fontsize=13)
    fig.text(.5,.012,'DEVELOPMENT YEAR ONLY  |  RASC is one fixed comparator; model dots show the three retained seeds',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(out/'Fig_C_economic_dev.png',dpi=200);plt.close(fig)
    expert=pd.read_csv(ROOT/'results/expert_usage_2024.csv').set_index('market_regime')
    fig,ax=plt.subplots(figsize=(9,4.2))
    for regime,label,color in [(0,'Normal','#3A6EA5'),(1,'Moderate','#36917F'),(2,'Tail','#AA4F64')]:
        if regime in expert.index:ax.plot(np.arange(7),expert.loc[regime].to_numpy(),'o-',label=label,color=color)
    ax.set_xticks(np.arange(7),[f'Expert {j}' for j in range(7)]);ax.set_ylim(0,1);ax.set_ylabel('Mean market-query expert attention')
    ax.set_title('Primary seed: descriptive expert profiles by observed regime');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.text(.5,.01,'DEVELOPMENT YEAR ONLY  |  numbered experts have no asserted economic or physical meaning',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(out/'Fig_D_expert_usage.png',dpi=200);plt.close(fig)
    (out/'figure_sources.json').write_text(json.dumps(dict(A='results/forecast_metrics_2024.csv: native, pooled, development_h2',
        B='results/gate_summary_2024.csv: exploratory_test (development H2)',C='results/economic_dev_metrics_2024.csv: normal, 12/24h',
        D='results/expert_usage_2024.csv: primary seed 20260920, development H2',
        uncertainty='three observed seeds, min-max and mean; no population confidence interval claimed'),indent=2))
    print('Four requested figures generated',flush=True)

if __name__=='__main__':main()
