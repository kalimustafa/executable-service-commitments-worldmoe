"""Editable vector manuscript figures, drawn directly from exported results."""
from pathlib import Path
import json,shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle,Circle,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'figures';F.mkdir(exist_ok=True)
S=F/'source_data';S.mkdir(exist_ok=True)
TEAL='#008F95';NAVY='#243E58';GOLD='#E5A33D';RED='#CB6158';PURPLE='#7A67A5';GREY='#7D8A93';LIGHT='#EAF4F3'
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#93A2AA','xtick.color':NAVY,'ytick.color':NAVY,'text.color':NAVY,'axes.labelcolor':NAVY,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
def export(fig,name):
    for ext in ['png','svg','pdf']:fig.savefig(F/(name+'.'+ext),dpi=360,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)
def label(ax,s):ax.text(0.,1.065,s,transform=ax.transAxes,weight='bold',size=12,va='bottom',clip_on=False)
def architecture():
    fig,ax=plt.subplots(figsize=(7.1,6.1));ax.set(xlim=(0,100),ylim=(0,100));ax.axis('off')
    def box(x,y,w,h,text='',fc=LIGHT,ec=TEAL,size=9,bold=False):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.35,rounding_size=1.4',facecolor=fc,edgecolor=ec,lw=.9))
        if text:ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,weight='bold' if bold else 'normal')
    def arrow(x,y,u,v,color=NAVY,style='-'):
        ax.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=9,lw=1,color=color,linestyle=style,connectionstyle='arc3'))
    ax.text(1,99,'R-ECMoE with executable flexibility',fontsize=15,weight='bold',va='top')
    # Compact iconography and labels are native vector geometry.
    for i,(t,sub) in enumerate([('Market','DA · RT · history'),('Weather','PV · PUE'),('Workload','arrivals · capacity'),('Service','release · deadline')]):
        x=1+25*i;box(x,86,23,8,t+'\n'+sub,fc='#F7F9FA',ec='#BDCDD3',size=8.4)
    box(1,63,67,18,fc='#EDF6F6');ax.text(3,78.4,'Frozen world backbone',weight='bold',fontsize=11)
    ax.text(66,78.4,'59.16 M parameters',ha='right',fontsize=8)
    for i,t in enumerate(['Block 1','Block 2','Block 3','Blocks 4–10']):
        x=[4,17,30,43][i];w=11 if i<3 else 22
        box(x,68,w,6,t,fc='white' if i%2==0 else '#C5E6E5',size=8)
        if i<3:arrow(x+w,71,[17,30,43][i],71)
    ax.text(3,64.9,'Width 256  ·  10 blocks  ·  sparse experts in even blocks',fontsize=7.9)
    for x in [12,37,62]:arrow(x,85.5,min(x,64),81.5)
    box(72,64,26,17,'Live service state\n\nContinuous queue\nCapacity calendar',fc='#FFF6E8',ec=GOLD,size=8.6,bold=False)
    arrow(88,85.5,88,81.5,color=GOLD)
    ax.text(1,59.3,'Cached site, global and expert features',fontsize=8.3)
    arrow(64,63,64,56.4)
    box(1,35,67,21,fc='#F5F2F9',ec=PURPLE)
    ax.text(3,53.2,'R2 market intelligence',fontsize=11,weight='bold')
    ax.text(66,53.2,'248.4 k trained parameters',fontsize=8,ha='right')
    box(4,42.3,30,7,'DA + DART distributions\n7 ordered quantiles',fc='white',ec=PURPLE,size=8.5)
    box(38,42.3,27,7,'Expert-aware adapter\nRegime · tail probability',fc='white',ec=PURPLE,size=8.5)
    ax.text(34.5,37.5,'RT = DA − DART  ·  publication-aware anchoring',ha='center',fontsize=8.4)
    arrow(18,42.2,25,39.2,color=PURPLE);arrow(51,42.2,45,39.2,color=PURPLE)
    arrow(34,34.5,34,30.5);arrow(85,63.5,85,30.5,color=GOLD)
    box(1,10,97,20,fc='#F1F7F7')
    ax.text(3,27.4,'Executable flexibility',fontsize=11,weight='bold')
    ax.text(96,27.4,'Explicit constraints',ha='right',fontsize=8)
    box(4,17.4,41,6.5,'Continuous reservation',fc='white',size=9,bold=True)
    box(53,17.4,41,6.5,'Deadline-priority recourse',fc='white',size=9,bold=True)
    arrow(45.6,20.6,52.6,20.6)
    for i in range(12):
        ax.add_patch(Rectangle((5+i*3.15,12),2.9,3,facecolor=[TEAL,'#CCE7E4',TEAL,'#E3E9ED'][i%4],lw=.4,edgecolor='white'))
    for i,(width,color) in enumerate([(7,RED),(12,GOLD),(19,TEAL)]):
        ax.add_patch(Rectangle((55+12*i,12),min(width,11),3,facecolor=color,lw=0))
    arrow(25,9.5,25,3.8);arrow(75,9.5,75,3.8)
    ax.text(25,1.8,'Physical execution',ha='center',fontsize=10,weight='bold')
    ax.text(75,1.8,'Complete-cycle settlement',ha='center',fontsize=10,weight='bold')
    export(fig,'Figure1_architecture')
    (S/'architecture.json').write_text(json.dumps(dict(frozen_parameters=59156961,R2_trained_with_calibration=248426,width=256,blocks=10,MoE_blocks=[2,4,6,8,10],retained_experts=7,top_k=2,display_folded_blocks=[4,5,6,7,8,9,10],R2_width=128,quantiles=[.05,.1,.25,.5,.75,.9,.95],W0_W1='training phases of one frozen snapshot; not two serial backbones',EF='explicit non-neural reservation and recourse',live_queue_enters_R2=False),indent=2))
def forecast():
    f=pd.read_csv(ROOT/'results/R2_forecast_metrics.csv');names=['Seasonal_quantiles','Ridge_residual_quantiles','RASC','R2_no_world_native','R2_full_native']
    p=f[f.gate.eq('post_DA')&f.target.eq('dart')].set_index('model').loc[names]
    p.to_csv(S/'Figure2_forecast.csv')
    colors=[GREY,GREY,NAVY,PURPLE,TEAL];labels=['Seasonal','Ridge','RASC','R2 no world','R2 full']
    fig,axs=plt.subplots(1,3,figsize=(7.1,2.8),gridspec_kw={'width_ratios':[1.1,1,1]})
    a=axs[0];a.barh(np.arange(5),p.pinball,color=colors,height=.62);a.set(yticks=np.arange(5),yticklabels=labels,xlabel='Pinball loss (US$/MWh)');a.invert_yaxis();a.set_xlim(0,7.6);label(a,'a')
    for i,v in enumerate(p.pinball):a.text(v+.08,i,f'{v:.2f}',va='center',fontsize=7.5)
    a=axs[1]
    for j,(model,color) in enumerate(zip(names,colors)):
        a.plot([50,80,90],[p.loc[model,f'coverage{n}']*100 for n in [50,80,90]],'o-',lw=1.1,ms=3,color=color)
    a.plot([35,96],[35,96],'--',lw=.8,color=GREY);a.set(xlabel='Nominal coverage (%)',ylabel='Observed coverage (%)',xlim=(45,96),ylim=(35,96),xticks=[50,80,90]);label(a,'b')
    a=axs[2];a.bar(np.arange(5),100*p.tail_coverage90,color=colors,width=.65);a.set(xticks=np.arange(5),xticklabels=['S','L','R','R2−','R2'],ylabel='Tail coverage of 90% interval (%)',ylim=(0,100));a.axhline(90,ls='--',color=GREY,lw=.8);label(a,'c')
    for i,v in enumerate(p.tail_coverage90):a.text(i,100*v+2,f'{100*v:.0f}',ha='center',size=7)
    fig.subplots_adjust(wspace=.62,left=.14,right=.99,bottom=.2,top=.93);export(fig,'Figure2_forecast')
def economic():
    f=pd.read_csv(ROOT/'results/policy_comparison_2025.csv');p=f[f.window_h.eq(12)&f.primary_comparison].copy();p.to_csv(S/'Figure4_value_risk.csv',index=False)
    colors={'Published_DA':GREY,'Old_RASC':NAVY,'R2_full_native':TEAL,'R2_no_world_native':PURPLE,'Old_R_action':RED}
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.4));a=axs[0]
    for _,r in p.iterrows():
        marker='D' if r.policy=='robust' else 'o';sz=28+max(r.mtm_daily_loss_cvar95_usd,0)/7
        a.scatter(r.shifted_IT_mwh,r.value_usd/1000,s=sz,marker=marker,color=colors[r.model],edgecolors='white',lw=.7,zorder=4)
        if r.model in ['R2_full_native','Old_R_action','Published_DA']:
            dx,dy=(-10,-18) if r.model=='Published_DA' else ((8,8) if r.policy=='robust' else (0,12))
            a.annotate({'R2_full_native':'R2','Published_DA':'DA','Old_R_action':'Old R'}[r.model]+(' interval' if r.policy=='robust' else ''),(r.shifted_IT_mwh,r.value_usd/1000),xytext=(dx,dy),textcoords='offset points',size=7.5,ha='right' if dx<0 else 'left')
    a.set(xlabel='Shifted IT energy (MWh)',ylabel='Annual incremental value (US$1,000)');a.grid(alpha=.15);label(a,'a')
    mix=pd.read_csv(ROOT/'results/heterogeneous_deadline_R2_results.csv');mix.to_csv(S/'Figure4_mixed.csv',index=False)
    a=axs[1];mp=mix[mix.policy.eq('robust')&mix.model.isin(['Old_RASC','R2_full_native'])]
    for i,m in enumerate(['Old_RASC','R2_full_native']):
        part=mp[mp.model.eq(m)].sort_values('recourse');a.plot([0,1],part.late_IT_mwh,'o-',color=colors[m],lw=2,label='RASC' if m=='Old_RASC' else 'R2')
    a.set(xticks=[0,1],xticklabels=['Fixed','Recourse'],ylabel='Deadline-missed IT energy (MWh)',xlim=(-.25,1.25));a.legend(frameon=False);label(a,'b')
    fig.subplots_adjust(wspace=.43,left=.10,right=.99,bottom=.18,top=.91);export(fig,'Figure4_value_risk')
    fig,axs=plt.subplots(1,3,figsize=(7.1,2.9))
    for model,policy,col,ls in [('Published_DA','point',GREY,'--'),('Old_RASC','robust',NAVY,'-'),('R2_full_native','point',TEAL,'--'),('R2_full_native','robust',TEAL,'-')]:
        q=f[f.model.eq(model)&f.policy.eq(policy)].sort_values('window_h')
        for a,metric,scale in zip(axs,['value_usd','shifted_IT_mwh','late_IT_mwh'],[1000,1,1]):a.plot(q.window_h,q[metric]/scale,'o',ls=ls,color=col,ms=3.5,lw=1.2)
    for i,(a,y) in enumerate(zip(axs,['Annual value (US$1,000)','Shifted IT energy (MWh)','Deadline-missed IT energy (MWh)'])):
        a.set(xlabel='Service window (h)',ylabel=y,xticks=[7,12,24]);a.grid(alpha=.14);label(a,'abc'[i])
    axs[0].plot([],[],color=GREY,ls='--',label='Published DA');axs[0].plot([],[],color=NAVY,label='RASC interval');axs[0].plot([],[],color=TEAL,ls='--',label='R2 point');axs[0].plot([],[],color=TEAL,label='R2 interval')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncols=4,frameon=False,bbox_to_anchor=(.5,1.03),fontsize=8)
    fig.subplots_adjust(wspace=.54,left=.095,right=.99,bottom=.2,top=.83);export(fig,'Figure5_service_windows');f.to_csv(S/'Figure5_service_windows.csv',index=False)
    w=pd.read_csv(ROOT/'results/attribution_waterfall.csv');w=w[w.chain.eq('execution_forecast_risk')]
    if w.empty:return
    w.to_csv(S/'Figure3_attribution.csv',index=False)
    fig,ax=plt.subplots(figsize=(7.1,3.3));levels=w.cumulative_value_usd.to_numpy()/1000;steps=w.increment_usd.to_numpy()/1000
    for i,(level,delta) in enumerate(zip(levels,steps)):
        base=0 if i in [0,len(w)-1] else levels[i-1];height=level if i in [0,len(w)-1] else delta
        ax.bar(i,height,bottom=base,width=.65,color=TEAL if height>=0 else RED,edgecolor='white',lw=.6)
        top=base+height
        text=f'US${height*1000:.2f}' if abs(height)<.01 else ((('' if i in [0,len(w)-1] else '+')+f'{height:.1f}') if height>=0 else f'{height:.1f}')
        ax.text(i,max(base,top)+.9,text,ha='center',size=8)
        if i<len(w)-1:ax.plot([i+.33,i+.67],[level,level],color=GREY,lw=.7)
    ax.set(xticks=np.arange(len(w)),xticklabels=['Old R','Executable\n+ DA','Old RASC\npoint','R2\npoint','Interval\ngate','R2 interval'],ylabel='Annual incremental value (US$1,000)');ax.axhline(0,lw=.7,color=GREY);ax.margins(y=.18);fig.subplots_adjust(left=.10,right=.99,bottom=.23,top=.96);export(fig,'Figure3_attribution')
if __name__=='__main__':
    architecture();forecast()
    if (ROOT/'results/policy_comparison_2025.csv').exists():economic()
