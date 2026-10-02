from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'source_data';OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})
P='#7868A2';R='#D46672';B='#597C96';G='#8E9C7A';GRAY='#A9A9AF'
def panel(ax,letter,title):
    ax.set_title(title,loc='left',pad=12)
    ax.text(-.15,1.11,letter,transform=ax.transAxes,fontweight='bold',fontsize=11)
    ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=320,bbox_inches='tight');fig.savefig(OUT/f'{name}.svg',bbox_inches='tight');plt.close(fig)

bounds=pd.read_csv(D/'certified_value_bounds.csv');info=pd.read_csv(D/'information_value_decomposition.csv');ext=pd.read_csv(D/'extreme_state_results.csv')
fig,axs=plt.subplots(2,2,figsize=(7.1,5.5));fig.subplots_adjust(wspace=.42,hspace=.70,left=.09,right=.97,bottom=.10,top=.90)
ax=axs[0,0];b=bounds.query("year==2025 and trace=='Eagle'")
ax.plot(b.window_h,b.maximum_saving_upper_usd/1000,'D--',color=GRAY,label='Certified savings ceiling')
ax.plot(b.window_h,b.policy_saving_usd/1000,'o-',color=P,label='Frozen interval policy')
ax.set(xticks=[12,16,24],xlabel='Service window (h)',ylabel='Annual savings (US$k)',ylim=(0,145));ax.legend(frameon=False,loc='lower right');panel(ax,'a','Attainable value')
ax=axs[0,1]
for col,label,color,marker in [('full_saving_usd','Full information',P,'o'),('post_only_saving_usd','Matched publication gate',R,'s'),('rasc_saving_usd','RASC-ME',B,'^'),('seasonal_saving_usd','Seasonal-ME',G,'d')]:
    ax.plot(info.window_h,info[col]/1000,marker+'-',color=color,label=label,markersize=4)
ax.set(xticks=[12,16,24],xlabel='Service window (h)',ylabel='Annual savings (US$k)');ax.legend(frameon=False,fontsize=6.5,loc='upper left');panel(ax,'b','Information access changes value')
ax=axs[1,0];x=np.arange(3)
ax.bar(x,info.early_path_value_usd/1000,color=P,width=.56)
lo=info.early_path_lower95.to_numpy()/1000;hi=info.early_path_upper95.to_numpy()/1000;y=info.early_path_value_usd.to_numpy()/1000
ax.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]),fmt='none',ecolor='#333333',capsize=3,linewidth=1)
ax.axhline(0,color='black',lw=.6);ax.set(xticks=x,xticklabels=['12','16','24'],xlabel='Service window (h)',ylabel='Early-path increment (US$k)');panel(ax,'c','Paired calendar uncertainty')
ax=axs[1,1];confirmation=json.loads((ROOT/'results/extreme_confirmation.json').read_text())
x=np.arange(3);totals=[confirmation['total_increment_usd'],confirmation['remaining_total_without_top_date_usd'],confirmation['worst_week']['remaining_total_usd']];extremes=[confirmation['extreme_increment_usd'],confirmation['remaining_extreme_without_top_date_usd'],confirmation['worst_week']['remaining_extreme_usd']]
ax.bar(x-.17,np.array(totals)/1000,.34,color=B,label='All remaining dates');ax.bar(x+.17,np.array(extremes)/1000,.34,color=R,label='Remaining extreme dates')
ax.axhline(0,color='black',lw=.6);ax.set(xticks=x,xticklabels=['Full calendar','Exclude 25 Jul','Exclude 22–28 Jul'],ylabel='World minus RASC-ME (US$k)',ylim=(-2,13));ax.tick_params(axis='x',labelsize=6.1);ax.legend(frameon=False,loc='upper right',fontsize=6.3);panel(ax,'d','One event dominates concentration')
save(fig,'Fig4_value_and_information')

grid=pd.read_csv(D/'grid_execution_summary.csv');ad=pd.read_csv(D/'adaptive_execution_summary.csv')
g=grid.query("year==2025 and trace=='Eagle' and window_h==24")
order=['cap75','cap50','ramp25','ramp125','cap75_ramp25'];labels=['75% cap','50% cap','25% ramp','12.5% ramp','Cap + ramp']
fig,axs=plt.subplots(2,2,figsize=(7.1,5.5));fig.subplots_adjust(wspace=.42,hspace=.75,left=.09,right=.97,bottom=.12,top=.90)
ax=axs[0,0];x=np.arange(5)
for exp,offset,color,label in [('I3',-.18,GRAY,'Projection only'),('I3G',.18,P,'Deadline-aware projection')]:
    a=g[g.experiment.eq(exp)].set_index('scenario').loc[order];ax.bar(x+offset,a.saving_usd/1000,.36,color=color,label=label)
ax.axhline(info.loc[info.window_h.eq(24),'full_saving_usd'].iloc[0]/1000,color=R,ls='--',lw=1,label='Unconstrained')
ax.set(xticks=x,xticklabels=labels,ylabel='Matched-reference savings (US$k)',ylim=(0,115));ax.tick_params(axis='x',rotation=30);ax.legend(frameon=False,fontsize=6.1,loc='upper right');panel(ax,'a','Value under electrical constraints')
ax=axs[0,1]
for exp,offset,color in [('I3',-.18,GRAY),('I3G',.18,P)]:
    a=g[g.experiment.eq(exp)].set_index('scenario').loc[order];ax.bar(x+offset,a.late_it_mwh,.36,color=color)
ax.set(xticks=x,xticklabels=labels,ylabel='Late fluid work (MWh)',ylim=(0,23));ax.tick_params(axis='x',rotation=30);ax.text(.98,.96,'Deadline-aware: zero in all five cases',ha='right',va='top',transform=ax.transAxes,fontsize=6.5);panel(ax,'b','Savings alone do not establish delivery')
ax=axs[1,0];a=g[(g.experiment.eq('I3G')&g.scenario.isin(['ramp125','ramp25']))|(g.experiment.eq('I3')&g.scenario.eq('unconstrained'))].sort_values('upward_ramp_mw')
ax.plot(a.upward_ramp_mw,a.saving_usd/1000,'o-',color=P,markersize=5)
for _,row in a.iterrows():
    label={'ramp125':'12.5% ramp','ramp25':'25% ramp','unconstrained':'Unconstrained'}[row.scenario]
    ax.annotate(label,(row.upward_ramp_mw,row.saving_usd/1000),xytext=(5,-14 if row.scenario=='unconstrained' else 8),textcoords='offset points',fontsize=6.3,ha='right' if row.scenario=='unconstrained' else 'left')
ax.set(xlabel='Maximum upward site ramp (MW / 15 min)',ylabel='Matched-reference savings (US$k)',ylim=(0,90));panel(ax,'c','Measured zero-late ramp–value trade-off')
ax=axs[1,1];xx=np.arange(3);traceyears=[(2024,'Eagle'),(2025,'Eagle'),(2025,'Frontier')]
for scenario,offset,color,label in [('cap50',-.24,GRAY,'50% cap'),('ramp25',0,P,'25% ramp'),('ramp125',.24,B,'12.5% ramp')]:
    values=[]
    for year,trace in traceyears:
        original=grid.query("experiment=='I3' and window_h==24 and scenario=='unconstrained'");original=original[original.year.eq(year)&original.trace.eq(trace)].saving_usd.iloc[0]
        constrained=grid.query("experiment=='I3G' and window_h==24");constrained=constrained[constrained.year.eq(year)&constrained.trace.eq(trace)&constrained.scenario.eq(scenario)].saving_usd.iloc[0]
        values.append(100*constrained/original)
    ax.bar(xx+offset,values,.24,color=color,label=label)
ax.set(xticks=xx,xticklabels=['Eagle 2024','Eagle 2025','Frontier 2025'],ylabel='Unconstrained savings retained (%)',ylim=(0,140));ax.tick_params(axis='x',labelsize=6.5);ax.legend(frameon=False,fontsize=6.2,loc='upper left',ncol=2);panel(ax,'d','Guarded recovery across workloads')
save(fig,'Fig5_execution_constraints')
print('Two four-panel figures written')
