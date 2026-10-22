"""Redraw every main-text figure from retained source rows and final outcomes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

ROOT=Path(__file__).resolve().parents[2]/'WorldMoE_Revision_20260928'
D=ROOT/'source_data';OUT=ROOT/'figures'
S=json.loads((ROOT/'results_summary.json').read_text());assert S['status']=='complete'
OUT.mkdir(exist_ok=True)
BLUE='#246080';OLD='#7795A5';GOLD='#BB8B3F';ROSE='#AF6F80';GRAY='#ACB3BA';INK='#24333F';GRID='#E4E9ED'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8.5,'axes.labelsize':8,
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.spines.top':False,
    'axes.spines.right':False,'axes.linewidth':.6,'axes.edgecolor':'#8A959E','text.color':INK,
    'axes.labelcolor':INK,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
SEQUENTIAL=LinearSegmentedColormap.from_list('blue_scale',['#F7FAFC','#D7E5EB','#9BBBCB','#477F9B','#153E54'])
STAT=['Seasonal-ME','Ridge-ME','LGB-Full-ME','LGB-Recent-ME','LGB-NoState-ME','GlobalCal-ME']
ORDER=['World-ME','RASC-ME',*STAT,'World-price','Legacy-price','RASC-direct','DA-persistent','DA-myopic','RT-lag','DAC']
SHORT={'World-ME':'WorldMoE','RASC-ME':'RASC-ME','Seasonal-ME':'Seasonal-ME','Ridge-ME':'Ridge-ME',
       'LGB-Full-ME':'LightGBM full','LGB-Recent-ME':'LightGBM recent','LGB-NoState-ME':'LightGBM no state',
       'GlobalCal-ME':'Global calibration','World-price':'World price ranking','Legacy-price':'Price-only head',
       'RASC-direct':'RASC direct','DA-persistent':'DA persistent','DA-myopic':'DA myopic','RT-lag':'RT lag','DAC':'DAC'}
COLORS={m:GRAY for m in ORDER};COLORS.update({'World-ME':OLD,'RASC-ME':GOLD,'Seasonal-ME':'#7D8970','DAC':INK})
FIGURES=[]

def panel(ax,letter,title,axis='y'):
    ax.set_title(f'{letter}  {title}',loc='left',fontweight='bold',pad=9)
    ax.set_axisbelow(True);ax.grid(axis=axis,color=GRID,lw=.6)

def save(fig,name,sources,scope):
    for ext in ['png','svg','pdf']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=360 if ext=='png' else None,bbox_inches='tight',pad_inches=.07)
    plt.close(fig);FIGURES.append(dict(figure=name,sources=sources,scope=scope,formats=['png','svg','pdf']))

def architecture():
    fig=plt.figure(figsize=(7.2,6.0));ax=fig.add_axes([.01,.01,.98,.98]);ax.set(xlim=(0,100),ylim=(0,100));ax.axis('off')
    def box(x,y,w,h,title,body,fill='#F1F5F7',edge=OLD):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.25,rounding_size=1',fc=fill,ec=edge,lw=.8))
        ax.text(x+w/2,y+h-2.6,title,ha='center',va='top',fontsize=8,fontweight='bold')
        ax.text(x+w/2,y+h-7.1,body,ha='center',va='top',fontsize=7.1,linespacing=1.35)
    def arrow(a,b,dash=False,color=INK,rad=0):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,lw=.85,color=color,
                                   linestyle='--' if dash else '-',connectionstyle=f'arc3,rad={rad}'))
    ax.text(1,99,'a  WorldMoE: prediction, state valuation and executable service',va='top',fontweight='bold',fontsize=10)
    box(1,76,28,17,'Causal market information','Issued forecasts and price lags\nTarget-day DA publication gate')
    box(36,76,28,17,'Frozen WorldMoE + R2 features','W0 / W1: 59.16 M parameters\nCached market representation: 128')
    box(71,76,28,17,'Market residual adapter','Trainable 128 → 64 → 128\nAPIC-preserving price shift',fill='#E8F2F7',edge=BLUE)
    arrow((29,84),(36,84));arrow((64,84),(71,84))
    box(1,53,26,16,'Physical observations','PV / PUE priors and seasonality\nOnly completed observations')
    box(1,31,26,16,'Work obligations','Released work and booked capacity\nDeadline-prefix queue state')
    ax.add_patch(FancyBboxPatch((33,30),66,40,boxstyle='round,pad=.3,rounding_size=1.4',fc='#FAFCFD',ec=BLUE,lw=1.1))
    ax.text(66,68,'State-conditioned decision processing',ha='center',va='top',fontsize=8.2,fontweight='bold',color=BLUE)
    box(36,48,28,13,'Market–physics attention','4 heads; 24 hourly tokens',fill='#E8F2F7',edge=BLUE)
    box(70,48,26,13,'Temporal context','2 attention layers; width 64',fill='#E8F2F7',edge=BLUE)
    box(36,33,28,11,'Work-to-time attention','96 admissible quarter-hours',fill='#E8F2F7',edge=BLUE)
    box(70,33,26,11,'State-cost decoder','PV CDF; cross moment; slot value',fill='#E8F2F7',edge=BLUE)
    arrow((27,61),(36,55));arrow((27,39),(36,39));arrow((85,76),(85,72));arrow((85,72),(38,72));arrow((38,72),(38,61))
    arrow((64,54),(70,54));arrow((83,48),(83,46));arrow((83,46),(50,46));arrow((50,46),(50,44));arrow((64,38),(70,38))
    box(1,10,45,16,'Training: conserved differentiable allocation','16 arriving cohorts; box / equality QP\nPrice + PV scoring, marginal cost and realised cost',fill='#F3F6F8')
    box(53,10,46,16,'Execution and independent settlement','Execute current service; enforce capacity and deadlines\nRetain all work, fees and closing liabilities',fill='#F9F2E8',edge=GOLD)
    ax.text(66,31.1,'227,527 trainable parameters including the market adapter',ha='center',va='center',fontsize=6.7,color=BLUE)
    arrow((47,30),(25,26),dash=True,color=BLUE);arrow((84,30),(76,26),color=GOLD)
    ax.text(1,5,'b  Evidence: certified ceiling  •  publication timing  •  electrical delivery  •  event influence',fontsize=8.2,fontweight='bold')
    ax.text(1,0,'2021–2023: parameter fitting     →     2024: checkpoint selection     →     2025: temporal evaluation',fontsize=7.9,va='bottom')
    save(fig,'Fig1_WorldMoE_architecture',['provenance.json'],'Final WorldMoE architecture; dashed arrow denotes differentiable training, not annual replay backpropagation')





def grid_execution():
    g=pd.read_csv(D/'worldmoe_grid_summary.csv');order=['cap75','cap50','ramp25','ramp125','cap75_ramp25'];labels=['75%\ncap','50%\ncap','25%\nramp','12.5%\nramp','Cap +\nramp'];x=np.arange(5)
    base=S['metrics']['saving_usd']['mean'];ref=S['fixed_reference_cost_usd'];cost=S['metrics']['adjusted_cost_usd']['mean']
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.2));fig.subplots_adjust(left=.12,right=.97,bottom=.10,top=.94,wspace=.47,hspace=.65)
    for exp,offset,color,label in [('CdfI3',-.17,GRAY,'Projection only'),('CdfI3G',.17,BLUE,'Deadline-aware')]:
        f=g[g.experiment.eq(exp)].set_index('scenario').loc[order]
        axs[0,0].bar(x+offset,f.saving_usd/1000,.34,color=color,label=label);axs[0,1].bar(x+offset,f.late_mean,.34,color=color,label=label)
        if exp=='CdfI3':
            for j,value in enumerate(f.saving_usd):axs[0,0].text(j+offset,value/1000+1.5,'†',ha='center',va='bottom',fontsize=8)
    ax=axs[0,0];ax.axhline(base/1000,color=GOLD,ls='--',lw=.9,label='Unconstrained');ax.set(xticks=x,xticklabels=labels,ylabel='Annual savings (US$k)',ylim=(0,140));ax.legend(frameon=False,fontsize=6.2,loc='upper right');panel(ax,'a','Electrical constraint cases')
    ax=axs[0,1];ax.set(xticks=x,xticklabels=labels,ylabel='Late IT (MWh)');ax.legend(frameon=False,fontsize=6.4);panel(ax,'b','Deadline completion')
    ax=axs[1,0];a=g[g.experiment.eq('CdfI3G')].set_index('scenario');r=[a.loc['ramp125'],a.loc['ramp25']];ramps=[z.upward_ramp_mw for z in r]+[S['metrics']['max_upward_site_ramp_mw']['max']];values=[z.saving_usd/1000 for z in r]+[base/1000]
    ax.plot(ramps,values,'o-',color=BLUE,ms=4,lw=1.2)
    for i,label in enumerate(['12.5% ramp','25% ramp','Unconstrained']):ax.annotate(label,(ramps[i],values[i]),xytext=(4,7 if i!=2 else -13),textcoords='offset points',fontsize=6.4,ha='left' if i<2 else 'right')
    ax.set(xlabel='Upward site ramp (MW / 15 min)',ylabel='Savings (US$k)',ylim=(0,max(values)*1.15));panel(ax,'c','Measured ramp–value combinations')
    ax=axs[1,1];a=a.loc[order]
    ax.bar(x-.17,(a.policy_cost_usd-cost)/1000,.34,color=BLUE,label='Policy cost change');ax.bar(x+.17,(a.reference_cost_usd-ref)/1000,.34,color=GOLD,label='Reference cost change');ax.axhline(0,color=INK,lw=.6)
    ax.set(xticks=x,xticklabels=labels,ylabel='Cost change (US$k)');ax.legend(frameon=False,fontsize=6.2);panel(ax,'d','Policy and reference cost changes')
    save(fig,'Fig5_electrical_delivery',['worldmoe_grid_summary.csv','worldmoe_grid_raw.csv'],'Innovation 3; causal frozen-request projection with carried service debt; not a retrained constrained policy')

def primary_results():
    f=pd.read_csv(D/'worldmoe_three_seed_results.csv');history=pd.read_csv(D/'worldmoe_training_history.csv')
    daily=pd.read_csv(D/'paired_daily_comparisons.csv',parse_dates=['date']);seeds=pd.read_csv(D/'worldmoe_daily_by_seed.csv').set_index('date')
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.65));fig.subplots_adjust(left=.12,right=.97,bottom=.1,top=.94,wspace=.46,hspace=.65)
    ax=axs[0,0];x=np.arange(3);ax.bar(x,f.saving_usd/1000,color=BLUE,width=.58)
    ax.axhline(S['certified_savings_ceiling_usd']/1000,color=GRAY,ls='--',lw=1,label='Certified savings ceiling')
    ax.axhline(S['target_80_usd']/1000,color=GOLD,ls=':',lw=1,label='80% of ceiling')
    for i,r in enumerate(f.itertuples()):ax.text(i,r.saving_usd/1000-12,f'{r.saving_usd/1000:.2f}\n{100*r.capture_lower_fraction:.2f}%',ha='center',va='top',color='white',fontsize=7)
    ax.set(xticks=x,xticklabels=[str(s)[-2:] for s in f.seed],xlabel='Optimization seed suffix',ylabel='Annual savings (US$k)',ylim=(0,150));ax.legend(frameon=False,fontsize=6.4,loc='upper left');panel(ax,'a','Three-seed annual value (2025)')
    ax=axs[0,1]
    for i,(seed,g) in enumerate(history.groupby('seed',sort=True)):
        z=g.dropna(subset=['economic_saving_usd']);xx=np.r_[0,z.epoch];yy=np.r_[52768.029471363436,z.economic_saving_usd]/1000;color=BLUE
        ax.plot(xx,yy,marker=['o','s','^'][i],ls=['-','--',':'][i],ms=3,lw=1,color=color,label=str(seed));r=f[f.seed.eq(seed)].iloc[0]
        ax.scatter(r.selected_epoch,r.selection_savings_usd/1000,s=70,marker='*',color=color,edgecolor='white',lw=.4,zorder=5)
    ax.set(xlabel='Refinement epoch',ylabel='2024 savings (US$k)',xticks=[0,2,4,6,8,10,12]);ax.legend(frameon=False,fontsize=6.3);panel(ax,'b','Economic checkpoint selection')
    ax=axs[1,0]
    for key,label,color,ls in [('worldmoe_joint','WorldMoE',BLUE,'-'),('rasc_me_original_executor','RASC-ME',GOLD,'--'),('seasonal_me_original_executor','Seasonal-ME',GRAY,':')]:ax.plot(daily.date,daily[key].cumsum()/1000,color=color,ls=ls,lw=1.2,label=label)
    ax.set(xlabel='2025 local calendar',ylabel='Cumulative savings (US$k)',xticks=pd.to_datetime(['2025-01-01','2025-07-01','2025-12-31']),xticklabels=['1 Jan','1 Jul','31 Dec']);ax.legend(frameon=False,fontsize=6.5,loc='upper left');panel(ax,'c','Continuous operating outcomes')
    ax=axs[1,1]
    for i,seed in enumerate(f.seed):
        values=np.sort(seeds[str(seed)].to_numpy());ax.plot(values,np.arange(1,len(values)+1)/len(values),color=BLUE,ls=['-','--',':'][i],lw=1,label=str(seed))
    ax.axvline(0,color=INK,ls=':',lw=.7);ax.set_xscale('symlog',linthresh=50);ax.set_xticks([-1000,-100,0,100,1000],['−1000','−100','0','100','1000'])
    ax.set(xlabel='Daily savings (US$; symmetric-log scale)',ylabel='Empirical cumulative probability',ylim=(0,1.02));panel(ax,'d','Daily service-adjusted savings')
    save(fig,'Fig2_WorldMoE_results',['worldmoe_three_seed_results.csv','worldmoe_training_history.csv','paired_daily_comparisons.csv','worldmoe_daily_by_seed.csv'],'Final model; same initialization, three refinement seeds; external controls compare whole pipelines')

def benchmarks_and_bounds():
    c=pd.read_csv(D/'external_benchmarks_24h.csv');f=pd.read_csv(D/'worldmoe_three_seed_results.csv')
    floor=pd.read_csv(D/'original/Fig2_capacity_floor.csv');bounds=pd.read_csv(D/'extensions/certified_value_bounds.csv').query("year==2025 and trace=='Eagle'")
    ap=pd.read_csv(D/'original/apic_and_forecast_controls.csv').set_index('model');forecast=pd.read_csv(D/'worldmoe_forecast_scores.csv')
    c=c.sort_values('saving',ascending=False);labels=['WorldMoE']+[SHORT[m] for m in c.model];values=np.r_[f.saving_usd.mean(),c.saving]/1000
    fig=plt.figure(figsize=(7.2,6.8));ax=fig.add_axes([.265,.47,.295,.47]);ax.barh(range(len(values)),values,color=[BLUE]+[GOLD if m=='RASC-ME' else GRAY for m in c.model],height=.55)
    ax.errorbar(f.saving_usd.mean()/1000,0,xerr=[[(f.saving_usd.mean()-f.saving_usd.min())/1000],[(f.saving_usd.max()-f.saving_usd.mean())/1000]],fmt='none',ecolor=INK,lw=.8,capsize=2)
    ax.set_yticks(range(len(labels)),labels,fontsize=6.6);ax.set_ylim(len(labels)-.5,-.5);ax.tick_params(axis='y',length=0);ax.set(xlim=(0,122),xlabel='Annual savings (US$k)')
    for i,v in enumerate(values):ax.text(v+2,i,f'{v:.1f}',fontsize=6.2,va='center')
    panel(ax,'a','24 h operating benchmarks',axis='x')
    ax=fig.add_axes([.735,.75,.23,.195]);ax.plot(floor.capacity_fraction*100,floor.minimum_feasible_h,'o-',color=BLUE,mfc='white',ms=3,lw=1)
    ax.set(xlabel='Capacity (%)',ylabel='Service floor (h)',xticks=[50,75,100]);panel(ax,'b','Minimum feasible service')
    ax=fig.add_axes([.735,.415,.23,.18]);ax.plot(bounds.window_h,bounds.maximum_saving_upper_usd/1000,'D--',color=GRAY,ms=4,label='Certified ceiling')
    ax.scatter(24,f.saving_usd.mean()/1000,s=65,marker='*',color=BLUE,label='WorldMoE (24 h)');ax.set(xlabel='Contract window (h)',ylabel='Savings (US$k)',xticks=[12,16,24],ylim=(0,150));ax.legend(frameon=False,fontsize=6,loc='lower right');panel(ax,'c','Certified savings ceiling')
    ax=fig.add_axes([.265,.055,.70,.23]);models=['RASC-ME','Legacy-price',*STAT];fv=[forecast.post_rt_pinball_usd_mwh.mean()]+[ap.loc[m,'pinball'] for m in models];names=['WorldMoE']+[SHORT[m] for m in models]
    ax.scatter(fv,np.arange(len(fv)),s=22,color=[BLUE]+[GOLD if m=='RASC-ME' else GRAY for m in models],zorder=3)
    ax.errorbar(fv[0],0,xerr=[[(fv[0]-forecast.post_rt_pinball_usd_mwh.min())],[(forecast.post_rt_pinball_usd_mwh.max()-fv[0])]],fmt='none',ecolor=INK,lw=.8,capsize=2)
    for i,v in enumerate(fv):ax.annotate(f'{v:.3f}',(v,i),xytext=(5,0),textcoords='offset points',va='center',fontsize=6.7)
    ax.set_yticks(range(len(names)),names,fontsize=6.6);ax.set_ylim(len(names)-.5,-.5);ax.tick_params(axis='y',length=0);ax.set(xlabel='Post-publication RT pinball loss (US$/MWh)',xlim=(min(fv)-.25,max(fv)+.18));panel(ax,'d','Post-publication price forecast quality',axis='x')
    save(fig,'Fig3_benchmarks_and_bounds',['external_benchmarks_24h.csv','worldmoe_forecast_scores.csv','original/apic_and_forecast_controls.csv','original/Fig2_capacity_floor.csv','extensions/certified_value_bounds.csv'],'Innovation 1 and external benchmarks; no old-versus-new WorldMoE mechanism contrast')

def information_value():
    info=pd.read_csv(D/'worldmoe_information_by_seed.csv');daily=pd.read_csv(D/'worldmoe_daily_by_seed.csv',parse_dates=['date']).set_index('date');post=pd.read_csv(D/'worldmoe_post_publication_daily_by_seed.csv',parse_dates=['date']).set_index('date');delta=daily-post
    result=pd.read_csv(D/'paired_financial_intervals.csv').set_index('contrast').loc['Joint WorldMoE early information pathway']
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.6));fig.subplots_adjust(left=.12,right=.97,bottom=.10,top=.94,wspace=.46,hspace=.65)
    ax=axs[0,0];x=np.arange(3);ax.bar(x-.16,info.native_usd/1000,.32,color=BLUE,label='Native access');ax.bar(x+.16,info.post_only_usd/1000,.32,color=GRAY,label='Post-publication only')
    ax.set(xticks=x,xticklabels=[str(s)[-2:] for s in info.seed],xlabel='Optimization seed suffix',ylabel='Annual savings (US$k)',ylim=(0,130));ax.legend(frameon=False,fontsize=6.7,loc='upper left');panel(ax,'a','Information restriction')
    ax=axs[0,1];vals=info.early_path_usd.to_numpy();ax.scatter(vals,np.arange(3),color=BLUE,s=30,label='Individual seeds')
    mean=result.difference_usd;lo=result.lower95_usd;hi=result.upper95_usd;ax.plot([lo,hi],[3.2,3.2],color=GOLD,lw=1.6);ax.scatter(mean,3.2,color=GOLD,s=35,marker='D')
    ax.axvline(0,color=INK,ls=':',lw=.7);ax.set(yticks=[0,1,2,3.2],yticklabels=[str(s)[-2:] for s in info.seed]+['Mean (95%)'],xlabel='Native minus restricted (US$)',ylim=(3.8,-.6));panel(ax,'b','Seed and calendar uncertainty',axis='x')
    ax=axs[1,0];monthly=delta.mean(1).groupby(delta.index.month).sum();ax.bar(monthly.index,monthly,color=[BLUE if x>=0 else ROSE for x in monthly]);ax.axhline(0,color=INK,lw=.6)
    ax.set(xticks=np.arange(1,13),xticklabels=list('JFMAMJJASOND'),xlabel='2025 month',ylabel='Information increment (US$)');panel(ax,'c','Monthly information increments')
    ax=axs[1,1];cumulative=delta.cumsum();ax.plot(cumulative.index,cumulative.mean(1),color=BLUE,lw=1.3);ax.fill_between(cumulative.index,cumulative.min(1),cumulative.max(1),color=BLUE,alpha=.2,label='Three-seed range');ax.axhline(0,color=INK,ls=':',lw=.7)
    ax.set(xticks=pd.to_datetime(['2025-01-01','2025-07-01','2025-12-31']),xticklabels=['1 Jan','1 Jul','31 Dec'],xlabel='2025 local calendar',ylabel='Cumulative increment (US$)');ax.legend(frameon=False,fontsize=6.3);panel(ax,'d','Cumulative information value')
    save(fig,'Fig4_information_timing',['worldmoe_information_by_seed.csv','worldmoe_daily_by_seed.csv','worldmoe_post_publication_daily_by_seed.csv','paired_financial_intervals.csv'],'Innovation 2; identical frozen checkpoints, current model and ledger; early pathway includes associated state-cost availability')

def event_influence():
    daily=pd.read_csv(D/'paired_daily_comparisons.csv').set_index('date');states=pd.read_csv(D/'market_state_definition_inputs.csv');event=json.loads((D/'worldmoe_event_summary.json').read_text())
    key='Joint WorldMoE vs original RASC-ME';ev=event[key];spread=states[states.year.eq(2025)].set_index('date').loc[daily.index,'spread_range'];threshold=ev['threshold'];flag=spread>threshold
    gap=daily.worldmoe_joint-daily.rasc_me_original_executor;early=daily.worldmoe_joint-daily.worldmoe_joint_post
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.65));fig.subplots_adjust(left=.12,right=.97,bottom=.10,top=.94,wspace=.48,hspace=.65)
    ax=axs[0,0];ax.scatter(spread[~flag],gap[~flag],s=10,c=GRAY,alpha=.7,label='Ordinary dates');ax.scatter(spread[flag],gap[flag],s=19,c=BLUE,alpha=.8,label='Extreme dates');ax.axvline(threshold,color=GOLD,ls='--',lw=.8);ax.axhline(0,color=INK,lw=.6)
    ax.set(xlabel='Daily spread range (US$/MWh)',ylabel='WorldMoE − RASC-ME (US$/day)');ax.legend(frameon=False,fontsize=6.3);panel(ax,'a','Price states and daily differences')
    ax=axs[0,1];values=[float(gap[flag].sum()),float(gap[~flag].sum())]
    ax.bar([0,1],np.array(values)/1000,color=[BLUE,GRAY],width=.6)
    for i,value in enumerate(values):ax.text(i,value/1000+1,f'{value/1000:.2f}\n{100*value/gap.sum():.1f}%',ha='center',va='bottom',fontsize=7)
    ax.set(xticks=[0,1],xticklabels=[f'Extreme\n{int(flag.sum())} dates',f'Ordinary\n{int((~flag).sum())} dates'],ylabel='Annual increment (US$k)',ylim=(0,max(values)/1000+7));panel(ax,'b','Extreme and ordinary contributions')
    ax=axs[1,0];top=daily.index==ev['top_date'];week=(daily.index>=ev['top_week_start'])&(daily.index<=ev['top_week_end'])
    values=[float(gap.sum()),float(gap[~top].sum()),float(gap[~week].sum())]
    ax.bar(np.arange(3),np.array(values)/1000,color=BLUE,width=.58)
    for i,value in enumerate(values):ax.text(i,value/1000+1,f'{value/1000:.2f}',ha='center',va='bottom',fontsize=7)
    daylabel=pd.Timestamp(ev['top_date']).strftime('%d %b');weeklabel=pd.Timestamp(ev['top_week_start']).strftime('%d')+'–'+pd.Timestamp(ev['top_week_end']).strftime('%d %b')
    ax.set(xticks=np.arange(3),xticklabels=['Full year','Exclude\n'+daylabel,'Exclude\n'+weeklabel],ylabel='Retained increment (US$k)',ylim=(0,50));panel(ax,'c','Financial exclusions retain positive gains')
    ax=axs[1,1];bs=pd.read_csv(D/'worldmoe_event_block_intervals.csv');bs=bs[bs.contrast.eq(key)]
    for i,r in enumerate(bs.itertuples()):ax.plot([r.lower95_usd/1000,r.upper95_usd/1000],[i,i],color=BLUE,lw=1.3);ax.scatter(r.extreme_usd/1000,i,color=BLUE,s=25)
    ax.axvline(0,color=INK,ls=':',lw=.7);ax.set(yticks=np.arange(len(bs)),yticklabels=[f'{b} days' for b in bs.block_days],xlabel='Extreme-date increment (US$k)',ylim=(-.6,len(bs)-.4));panel(ax,'d','Uncertainty across block lengths',axis='x')
    save(fig,'Fig6_event_influence',['paired_daily_comparisons.csv','market_state_definition_inputs.csv','worldmoe_event_summary.json','worldmoe_event_block_intervals.csv'],'Innovation 5; descriptive influence and robustness only; no blanket extreme-state superiority claim')

if __name__=='__main__':
    for function in [architecture,primary_results,benchmarks_and_bounds,information_value,grid_execution,event_influence]:function()
    (OUT/'figure_manifest.json').write_text(json.dumps(FIGURES,indent=2))
    print('Six main figures redrawn in PNG, SVG and PDF; no old-versus-new mechanism comparison.')



