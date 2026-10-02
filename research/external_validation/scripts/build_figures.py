"""Publication figures 2--6 with the user's fixed semantic palette.
Figure 1 is never read, redrawn or exported by this script.
"""
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from paths import *
F=FINAL/'02_figures';S=F/'source_data'
OLD=PAPER/'revision_20260920_r2_executable';ROB=PAPER/'revision_20260920_r2_robustness'
R2='#4F63B8';RASC='#C47A44';DA='#778493';MYOPIC='#A6B2BF';BASELINE='#444A50';RECOURSE='#4359A5';PV='#719A77'
INK='#222222';SECONDARY='#5B616A';GRID='#E5E7EB';UNC='#BCC2CA';WIDTH=175/25.4
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8.7,'xtick.labelsize':8,'ytick.labelsize':8,
 'text.color':INK,'axes.labelcolor':INK,'xtick.color':SECONDARY,'ytick.color':SECONDARY,'axes.edgecolor':UNC,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'lines.linewidth':1.3,'lines.markersize':5.5,
 'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white','figure.facecolor':'white'})
def panel(ax,label):ax.text(-.015,1.065,label,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')
def grid(ax,axis='y'):ax.grid(axis=axis,color=GRID,lw=.6);ax.set_axisbelow(True)
def savefig(fig,name):
    for ext in ['svg','pdf','png']:fig.savefig(F/f'{name}.{ext}',dpi=600,facecolor='white')
    plt.close(fig)
def source(number,frames):
    pd.concat([df.assign(panel=label) for label,df in frames],ignore_index=True,sort=False).to_csv(S/f'fig{number}_source.csv',index=False)

def fig2():
    src=ROB/'figures/source_data'
    trace=pd.read_csv(src/'Figure2_workload_capacity.csv');floor=pd.read_csv(src/'Figure2_capacity_floor.csv');late=pd.read_csv(src/'Figure2_deadline_outcomes.csv')
    source(2,[('a',trace),('b',floor),('c',late)])
    fig=plt.figure(figsize=(WIDTH,4.7));gs=fig.add_gridspec(2,2,height_ratios=[1.08,1],hspace=.7,wspace=.47)
    a=fig.add_subplot(gs[0,:]);x=trace.hours
    released=trace.cumulative_released_and_opening_IT_mwh;served=trace.cumulative_earliest_service_IT_mwh
    a.fill_between(x,served,released,color=R2,alpha=.07,step='post')
    a.step(x,released,where='post',color=R2,label='Released + opening work')
    a.step(x,served,where='post',color=BASELINE,label='Earliest service',ls='--')
    a.plot(x,trace.cumulative_available_capacity_IT_mwh,color=MYOPIC,ls=':',label='Available capacity')
    a.set(xlim=(x.min(),x.max()),ylabel='Cumulative IT energy (MWh)',xlabel='Hours from 22 August 2025 07:30 New York time')
    a.legend(loc='upper left',bbox_to_anchor=(0,1.3),ncols=3,frameon=False,fontsize=7.6);grid(a);panel(a,'a')
    b=fig.add_subplot(gs[1,0]);q=floor.sort_values('capacity_fraction')
    b.plot(q.capacity_fraction*100,q.minimum_feasible_h,'o-',color=BASELINE)
    for _,r in q.iterrows():b.annotate(f'{r.minimum_feasible_h:g} h',(r.capacity_fraction*100,r.minimum_feasible_h),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
    b.set(xlim=(45,105),ylim=(4,18),xticks=[50,75,100],xlabel='Available service capacity (%)',ylabel='Physical service floor (h)');grid(b);panel(b,'b')
    a=fig.add_subplot(gs[1,1])
    for model,policy,col,mark,ls,label in [('Old_RASC','point',RASC,'s','--','RASC point'),('R2_full_native','point',R2,'o','--','R2 point'),('R2_full_native','robust',R2,'D','-','R2 interval')]:
        q=late[late.model.eq(model)&late.policy.eq(policy)].sort_values('window_h');a.plot(q.window_h,q.late_IT_mwh,marker=mark,color=col,ls=ls,label=label)
    a.axvline(6.5,color=UNC,ls=':',lw=.9);a.text(7.2,11.3,'6.5 h floor',color=SECONDARY,fontsize=7.5)
    a.set(xlim=(5,25),ylim=(-.5,13),xticks=[7,12,24],xlabel='Service window (h)',ylabel='Late IT energy (MWh)')
    a.legend(frameon=False,loc='upper right',fontsize=7.5);grid(a);panel(a,'c')
    fig.subplots_adjust(left=.11,right=.98,bottom=.13,top=.88);savefig(fig,'Fig2_service_floor')

def fig3():
    cost=pd.read_csv(PAPER/'revision_20260922_completion/results/economic_value_ladder_12h.csv')
    base=pd.read_csv(OLD/'results/R2_forecast_metrics.csv');base=base[base.gate.eq('post_DA')&base.target.eq('dart')].set_index('model')
    seeds=pd.read_csv(ROB/'results/multiseed_forecast_metrics.csv');seeds=seeds[seeds.gate.eq('post_DA')&seeds.target.eq('dart')]
    inc=pd.read_csv(OLD/'results/paired_policy_differences.csv');inc=inc[inc.role.eq('forecast')&inc['first'].str.endswith('_point')].sort_values('window_h')
    source(3,[('a',cost),('b',pd.concat([base.loc[['RASC']].reset_index(),seeds],ignore_index=True)),('c',inc)])
    fig=plt.figure(figsize=(WIDTH,6.0));gs=fig.add_gridspec(3,3,height_ratios=[1.65,.86,1.05],hspace=.87,wspace=.72)
    a=fig.add_subplot(gs[0,:]);c0=cost.annual_service_adjusted_cost_usd.iloc[0]/1000
    labels=['Earliest service','Myopic + DA','Persistent + DA','RASC point','R2 point','R2 interval'];cols=[BASELINE,MYOPIC,DA,RASC,R2,R2]
    for i,(_,r) in enumerate(cost.iterrows()):
        v=r.annual_service_adjusted_cost_usd/1000
        a.plot([v,c0],[i,i],color=cols[i],alpha=.45,lw=1.3)
        a.plot(v,i,marker='D' if i==5 else ('s' if i==3 else 'o'),color=cols[i],ms=5.5)
        a.annotate(f'{v:.2f}',(v,i),xytext=(-6,0),textcoords='offset points',va='center',ha='right',fontsize=8,color=cols[i])
        a.text(1.014,i,'Baseline' if i==0 else f'{r.service_adjusted_saving_rate_pct:.2f}%',transform=a.get_yaxis_transform(),va='center',fontsize=8,color=cols[i])
    a.axvline(c0,color=UNC,ls='--',lw=.8);a.text(1.014,1.01,'Saving',transform=a.transAxes,fontsize=8)
    a.set(yticks=range(6),yticklabels=labels,xlim=(502,560),ylim=(5.6,-.7),xticks=[510,520,530,540,550,560],xlabel='Service-adjusted annual electricity cost (US$ thousands)');a.tick_params(axis='y',length=0);grid(a,'x');panel(a,'a')
    for j,(key,title,mul,lim,ticks) in enumerate([('pinball','Pinball loss (US$/MWh)',1,(5.35,6.1),[5.5,5.75,6.0]),('coverage90','90% coverage (%)',100,(84,91),[85,87.5,90]),('tail_coverage90','Tail coverage (%)',100,(38,70),[40,50,60,70])]):
        a=fig.add_subplot(gs[1,j]);v=seeds[key].to_numpy()*mul;b=float(base.loc['RASC',key])*mul
        a.plot([b,v.mean()],[0,1],color=UNC,lw=.8);a.plot(b,0,'s',color=RASC)
        a.errorbar(v.mean(),1,xerr=[[v.mean()-v.min()],[v.max()-v.mean()]],fmt='o',color=R2,ecolor=UNC,elinewidth=1,capsize=3)
        a.text(b,.23,f'{b:.3f}' if j==0 else f'{b:.2f}',ha='center',color=RASC,fontsize=7.8)
        a.text(v.mean(),1.23,f'{v.mean():.3f}' if j==0 else f'{v.mean():.2f}',ha='center',color=R2,fontsize=7.8)
        a.set(yticks=[0,1],yticklabels=['RASC','R2'] if j==0 else [],xlim=lim,xticks=ticks,ylim=(-.3,1.65),xlabel=title);grid(a,'x')
        if j==0:panel(a,'b')
    a=fig.add_subplot(gs[2,:])
    for i,(_,r) in enumerate(inc.iterrows()):
        a.errorbar(r.difference_usd/1000,i,xerr=[[(r.difference_usd-r.lower95)/1000],[(r.upper95-r.difference_usd)/1000]],fmt='o',color=R2,mfc='white' if r.window_h==7 else R2,ecolor=UNC,elinewidth=1,capsize=3)
        a.text(1.015,i,f'US${r.difference_usd:,.0f}',transform=a.get_yaxis_transform(),ha='left',va='center',fontsize=7.8,color=R2)
    a.axvline(0,color='#9CA3AF',ls='--',lw=.8);a.set(yticks=range(3),yticklabels=['7 h','12 h','24 h'],ylim=(2.55,-.5),xlim=(-1.5,7.6),xticks=[0,2,4,6],xlabel='R2 point minus RASC point saving (US$ thousands)');grid(a,'x');panel(a,'c')
    fig.subplots_adjust(left=.22,right=.88,bottom=.085,top=.94);savefig(fig,'Fig3_economic_value')

def fig4():
    data=pd.read_csv(OLD/'results/policy_comparison_2025.csv');q=data[data.window_h.eq(12)&data.primary_comparison&~data.model.eq('Old_R_action')].copy()
    p=q[q.model.eq('R2_full_native')&q.policy.eq('point')].iloc[0];r=q[q.model.eq('R2_full_native')&q.policy.eq('robust')].iloc[0]
    ratios=[100*r.value_usd/p.value_usd,100*r.shifted_IT_mwh/p.shifted_IT_mwh,100*(1-r.mtm_daily_loss_cvar95_usd/p.mtm_daily_loss_cvar95_usd)]
    source(4,[('a',q),('b',pd.DataFrame(dict(metric=['Value retained','Energy used','CVaR reduction'],percentage=ratios)))])
    fig,(a,b)=plt.subplots(1,2,figsize=(WIDTH,3.25),gridspec_kw={'width_ratios':[1.45,1]})
    offsets={('R2_full_native','point'):(1620,42.5),('R2_full_native','robust'):(1150,33.2),('Old_RASC','point'):(1090,38.1),('Old_RASC','robust'):(1030,28.8),('Published_DA','point'):(1920,35.1)}
    for _,row in q.iterrows():
        col={'R2_full_native':R2,'Old_RASC':RASC,'Published_DA':DA}[row.model]
        mark='D' if row.policy=='robust' else ('s' if row.model=='Old_RASC' else 'o')
        name=('R2' if row.model=='R2_full_native' else 'RASC' if row.model=='Old_RASC' else 'Published DA')+(' interval' if row.policy=='robust' else ' point' if row.model!='Published_DA' else '')
        a.plot(row.shifted_IT_mwh,row.value_usd/1000,marker=mark,ls='',color=col,ms=5.5)
        a.annotate(name,(row.shifted_IT_mwh,row.value_usd/1000),xytext=offsets[(row.model,row.policy)],fontsize=7.8,color=col,arrowprops=dict(arrowstyle='-',color=col,lw=.8))
    a.set(xlim=(600,2350),ylim=(27,44.5),xlabel='Shifted IT energy (MWh)',ylabel='Annual electricity-cost saving\n(US$ thousands)');grid(a);panel(a,'a')
    for i,v in enumerate(ratios):b.barh(i,v,color=R2,height=.45);b.text(v+2,i,f'{v:.1f}%',va='center',fontsize=8,color=R2)
    b.set(yticks=range(3),yticklabels=['Value\nretained','Energy\nused','CVaR\nreduction'],xlim=(0,103),xticks=[0,50,100],ylim=(2.6,-.6),xlabel='Interval relative to point (%)');grid(b,'x');panel(b,'b')
    fig.subplots_adjust(left=.12,right=.97,bottom=.23,top=.87,wspace=.67);savefig(fig,'Fig4_risk_selection')

def fig5():
    mix=pd.read_csv(OLD/'results/heterogeneous_deadline_R2_results.csv');q=mix[mix.model.eq('R2_full_native')&mix.policy.eq('robust')].sort_values('recourse');fixed,rec=q.iloc[0],q.iloc[1]
    policy=pd.read_csv(OLD/'results/policy_comparison_2025.csv');w=policy[policy.model.eq('R2_full_native')&policy.policy.eq('robust')].sort_values('window_h')
    seeds=pd.read_csv(ROB/'results/multiseed_economic_results.csv');seeds=seeds[seeds.policy.eq('robust')]
    source(5,[('a',pd.DataFrame(dict(window_h=[4,12,24],work_share_pct=[25,50,25]))),('bc',q),('d_primary',w),('d_seeds',seeds)])
    fig,axs=plt.subplots(2,2,figsize=(WIDTH,4.9),gridspec_kw={'wspace':.55,'hspace':.85});a,b,c,dax=axs.ravel()
    for i,(window,share) in enumerate([(4,25),(12,50),(24,25)]):a.barh(i,window,color=DA,height=.45);a.text(window+.7,i,f'{share}%',va='center',fontsize=8)
    a.set(yticks=range(3),yticklabels=['4 h','12 h','24 h'],ylim=(2.6,-.6),xlim=(0,31),xticks=[0,12,24],xlabel='Release-to-deadline window (h)');grid(a,'x');panel(a,'a')
    for i,(v,col) in enumerate([(fixed.late_IT_mwh,RASC),(rec.late_IT_mwh,RECOURSE)]):b.plot(i,v,'o',color=col,ms=5.5);b.text(i,v+.24,f'{v:.3f}' if v else '0',ha='center',color=col,fontsize=8)
    b.plot([0,1],[fixed.late_IT_mwh,rec.late_IT_mwh],color=UNC,lw=1);b.set(xticks=[0,1],xticklabels=['Fixed','Recourse'],xlim=(-.4,1.4),ylim=(-.4,6.4),ylabel='Late IT energy (MWh)');grid(b);panel(b,'b')
    c.plot([fixed.value_usd,rec.value_usd],[0,1],color=UNC,lw=1)
    for i,(row,col) in enumerate([(fixed,RASC),(rec,RECOURSE)]):c.plot(row.value_usd,i,'o',color=col);c.text(row.value_usd,i+.2,f'US${row.value_usd:,.2f}',ha='center',color=col,fontsize=7.8)
    c.set(yticks=[0,1],yticklabels=['Fixed','Recourse'],xlim=(36687,36700),ylim=(-.35,1.55),xticks=[36690,36695,36700],xlabel='Annual saving (US$; magnified scale)');c.ticklabel_format(axis='x',useOffset=False,style='plain');grid(c,'x');panel(c,'c')
    dax.plot(w.window_h,w.value_usd/1000,'D-',color=R2)
    for window,g in seeds.groupby('window_h'):
        centre=w[w.window_h.eq(window)].value_usd.iloc[0]/1000
        dax.errorbar(window,centre,yerr=[[centre-g.value_usd.min()/1000],[g.value_usd.max()/1000-centre]],fmt='none',ecolor=UNC,elinewidth=1,capsize=3)
    dax.set(xticks=[7,12,24],xlim=(5,26),ylim=(20,47),xlabel='Common service window (h)',ylabel='Annual electricity-cost saving\n(US$ thousands)');dax.text(.02,.94,'Zero late work',transform=dax.transAxes,fontsize=7.7);grid(dax);panel(dax,'d')
    fig.subplots_adjust(left=.12,right=.97,bottom=.13,top=.91);savefig(fig,'Fig5_recourse')

def fig6():
    order=[('Eagle',2024),('Eagle',2025),('Frontier',2024),('Frontier',2025)]
    labels=['Eagle\n2024','Eagle\n2025','Frontier\n2024','Frontier\n2025']
    f=pd.read_csv(RESULTS/'trace_service_floor.csv').set_index(['trace','market_year']).loc[order].reset_index()
    s=pd.read_csv(RESULTS/'external_validity_summary.csv').set_index(['trace','market_year']).loc[order].reset_index()
    source(6,[('a',f),('bcd',s)])
    fig,axs=plt.subplots(2,2,figsize=(WIDTH,5.7),gridspec_kw={'wspace':.50,'hspace':.66});a,b,c,dax=axs.ravel();x=np.arange(4)
    marks=['o','o','s','^'];filled=[False,True,False,True]
    for key,col,ls,label in [('physical_floor_h_100pct',BASELINE,'-','100% capacity'),('physical_floor_h_75pct',DA,'--','75% capacity'),('physical_floor_h_50pct',MYOPIC,':','50% capacity')]:
        a.plot(x,f[key],color=col,ls=ls,lw=1.3,label=label)
        for i,v in enumerate(f[key]):a.plot(i,v,marker=marks[i],color=col,mfc=col if filled[i] else 'white',ms=5.5)
    a.set(xticks=x,xticklabels=labels,ylabel='Minimum common service window (h)',ylim=(0,max(f.physical_floor_h_50pct)*1.24))
    a.legend(frameon=False,fontsize=7.8,loc='upper right');grid(a);panel(a,'a')
    for i,r in s.iterrows():
        b.plot([i-.12,i+.12],[r.myopic_da_saving_pct,r.persistent_da_saving_pct],color=UNC,lw=1)
        for dx,v,col in [(-.12,r.myopic_da_saving_pct,MYOPIC),(.12,r.persistent_da_saving_pct,DA)]:b.plot(i+dx,v,marker=marks[i],color=col,mfc=col if filled[i] else 'white',ms=5.5)
    b.set(xticks=x,xticklabels=labels,ylabel='Saving relative to earliest (%)',ylim=(0,max(s.persistent_da_saving_pct.max(),s.myopic_da_saving_pct.max())*1.25))
    b.legend(handles=[Line2D([],[],marker='o',ls='',color=MYOPIC,label='Myopic + DA'),Line2D([],[],marker='o',ls='',color=DA,label='Persistent + DA')],frameon=False,fontsize=7.4,loc='upper left');grid(b);panel(b,'b')
    for i,r in s.iterrows():
        c.plot(r.persistence_increment/1000,i,marker=marks[i],color=DA,mfc=DA if filled[i] else 'white',ms=5.5)
        c.annotate(f'US${r.persistence_increment:,.0f}',(r.persistence_increment/1000,i),xytext=(6,0),textcoords='offset points',va='center',fontsize=7.7)
    c.axvline(0,color=UNC,lw=.8,ls='--');c.set(yticks=x,yticklabels=[s.replace('\n',' ') for s in labels],ylim=(3.5,-.5),xlim=(0,max(s.persistence_increment)/1000*1.47),xlabel='12 h persistence increment\n(US$ thousands)');grid(c,'x');panel(c,'c')
    for i,r in s.iterrows():
        dax.plot([r.fixed_mixed_late,r.recourse_mixed_late],[i,i],color=UNC,lw=1)
        for v,col in [(r.fixed_mixed_late,RASC),(r.recourse_mixed_late,RECOURSE)]:dax.plot(v,i,marker=marks[i],color=col,mfc=col if filled[i] else 'white',ms=5.5)
        if abs(r.fixed_mixed_late)<1e-9:dax.annotate('0 → 0',(0,i),xytext=(7,8),textcoords='offset points',fontsize=7.7)
    dax.set(yticks=x,yticklabels=[s.replace('\n',' ') for s in labels],ylim=(3.5,-.6),xlim=(-.08,max(s.fixed_mixed_late)*1.18),xlabel='Mixed-contract late work (MWh)')
    dax.legend(handles=[Line2D([],[],marker='o',ls='',color=RASC,label='Fixed'),Line2D([],[],marker='o',ls='',color=RECOURSE,label='Recourse')],frameon=False,fontsize=7.4,loc='upper right');grid(dax,'x');panel(dax,'d')
    fig.subplots_adjust(left=.13,right=.98,bottom=.12,top=.93);savefig(fig,'Fig6_external_validation')

if __name__=='__main__':
    fig2();fig3();fig4();fig5();fig6()
    save(F/'figure_manifest.json',[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(F.glob('Fig*.*'))])
    print('Figures 2--6 and source CSVs complete; Figure 1 untouched')
