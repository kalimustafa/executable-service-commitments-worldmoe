"""Compact displays requested by the author; existing results only."""
import json,shutil,subprocess,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from final_figures import ROOT,D,F,W,REF,INK,GRID,COLORS,read,lab,col,panel,save,MANIFEST

def axes(height=4,left=.17,wspace=.72):
    fig,ax=plt.subplots(1,2,figsize=(7.2,height));fig.subplots_adjust(left=left,right=.98,bottom=.17,top=.89,wspace=wspace);return fig,ax

def figure2():
    b=read('baseline_all_ontime_24h.csv');w=read('current/worldmoe_three_seed_results.csv')
    q=pd.DataFrame([dict(model='WorldMoE',policy='final',saving=w.saving_usd.mean(),saving_min=w.saving_usd.min(),saving_max=w.saving_usd.max()),*b.to_dict('records'),dict(model='No-flex',policy='earliest',saving=0,saving_min=0,saving_max=0)]).sort_values('saving',ascending=False).reset_index(drop=True)
    labels=[lab(r.model)+((' / '+r.policy) if r.policy not in ['final','earliest'] else '') for r in q.itertuples()]
    fig,a=plt.subplots(figsize=(7.2,6.25));fig.subplots_adjust(left=.34,right=.98,bottom=.1,top=.93);yy=np.arange(len(q))
    a.barh(yy,q.saving/1000,color=[col(m) for m in q.model],height=.66)
    for i,r in enumerate(q.itertuples()):
        if r.saving_max>r.saving_min:a.hlines(i,r.saving_min/1000,r.saving_max/1000,color=INK,lw=.8)
        a.text(r.saving/1000+1.7,i,f'{r.saving/1000:.1f}',va='center',fontsize=6.3)
    a.scatter([0],[len(q)-1],c=REF,marker='D',s=17,zorder=4)
    b=read('current/extensions/certified_value_bounds.csv');bound=float(b[(b.year==2025)&(b.window_h==24)&(b.trace=='Eagle')].maximum_saving_upper_usd.iloc[0])/1000
    a.axvline(bound,c=INK,ls='--',lw=.8);a.text(bound-1,-1.1,'Certified ceiling',ha='right',fontsize=6.7)
    a.set(yticks=yy,yticklabels=labels,xlim=(-1,143),xlabel='Annual saving against immediate execution (US$k)');a.invert_yaxis();a.grid(axis='x',color=GRID,lw=.5);a.set_axisbelow(True);a.set_title('Operating savings across retained model policies',loc='left',fontweight='bold',pad=16)
    save(fig,'Fig2',['baseline_all_ontime_24h.csv','current/worldmoe_three_seed_results.csv','current/extensions/certified_value_bounds.csv'],'26 historical cells and current WorldMoE; closing work charged; historical pipelines are Tier 2.')

def figure3():
    h=read('current/extfig1_heterogeneous_deadlines.csv');hist=read('mixed_deadline_baselines_25_50_25.csv');g=h.groupby('pathway').mean(numeric_only=True)
    keys=['H0 homogeneous 24 h request fifo','H0 homogeneous 24 h request edf','H1 mixed deadline aware WorldMoE edf'];fig,(a,b)=axes(4.25,left=.18,wspace=.98)
    a.barh([0,1,2],g.loc[keys].late_work_it_mwh,color=[REF,COLORS['RASC-ME'],W],height=.45)
    a.set(yticks=[0,1,2],yticklabels=['24 h request / FIFO','24 h request / EDF','Deadline-aware'],xlabel='Late IT under mixed contract (MWh)',ylim=(2.75,-.75));a.set_xscale('symlog',linthresh=1);panel(a,'a','Service compliance','x')
    vals=np.r_[g.loc[keys[2],'annual_saving_usd'],hist.saving]/1000;b.barh(np.arange(len(vals)),vals,color=[W]+[col(m) for m in hist.model],height=.65)
    b.set(yticks=np.arange(len(vals)),yticklabels=['WorldMoE']+[lab(m) for m in hist.model],xlabel='Annual saving (US$k)');b.invert_yaxis();panel(b,'b','On-time mixed-contract models','x')
    save(fig,'Fig3',['current/extfig1_heterogeneous_deadlines.csv','mixed_deadline_baselines_25_50_25.csv'],'Two panels: mixed-contract compliance and actual on-time financial comparisons.')

def figure4():
    q=read('grid_metrics_all_models.csv').sort_values('top10_net_mwh',ascending=False);raw=read('grid_metrics_all_models_by_seed.csv');fig,ax=plt.subplots(1,2,figsize=(7.2,6.25),sharey=True);fig.subplots_adjust(left=.32,right=.98,bottom=.1,top=.93,wspace=.34)
    yy=np.arange(len(q));names=[lab(r.model)+((' / '+r.policy) if r.policy not in ['final','earliest'] else '') for r in q.itertuples()]
    for a,field,l,title,xlabel in [(ax[0],'top10_net_mwh','a','High-price hours','Net import reduction (MWh)'),(ax[1],'p95_abs_ramp_mw','b','Typical ramps','95th percentile |Δimport| (MW / 15 min)')]:
        a.scatter(q[field],yy,c=[col(m) for m in q.model],s=[28 if m=='WorldMoE' else 17 for m in q.model],edgecolor=INK,lw=.25)
        for i,r in enumerate(q.itertuples()):
            v=raw[(raw.model==r.model)&(raw.policy==r.policy)][field]
            if len(v)>1:a.hlines(i,v.min(),v.max(),color=col(r.model),lw=.85)
        a.axvline(float(q[q.model=='No-flex'][field].iloc[0]),c=INK,ls=':',lw=.7);a.set_xlabel(xlabel);panel(a,l,title,'x')
    ax[0].set(yticks=yy,yticklabels=names);ax[0].invert_yaxis();save(fig,'Fig4',['grid_metrics_all_models.csv','grid_metrics_all_models_by_seed.csv'],'Two panels: high-price displacement and typical ramps. Peak/max-ramp limitations remain in Results and source data.')

def figure5():
    q=read('paired_all_ontime_comparators_24h.csv').sort_values('worldmoe_minus_baseline_usd');fig=plt.figure(figsize=(7.2,7.0));gs=fig.add_gridspec(2,1,height_ratios=[3.2,1.35],left=.30,right=.98,bottom=.09,top=.95,hspace=.37)
    a=fig.add_subplot(gs[0]);yy=np.arange(len(q));a.errorbar(q.worldmoe_minus_baseline_usd/1000,yy,xerr=np.array([q.worldmoe_minus_baseline_usd-q.bootstrap_lower95_usd,q.bootstrap_upper95_usd-q.worldmoe_minus_baseline_usd])/1000,fmt='none',ecolor=REF,lw=.8,capsize=1.5)
    a.scatter(q.worldmoe_minus_baseline_usd/1000,yy,c=[col(m) for m in q.model],s=20,edgecolor=INK,lw=.25);a.set(yticks=yy,yticklabels=[lab(r.model)+' / '+r.policy for r in q.itertuples()],xlabel='Annual WorldMoE advantage (US$k; paired 95% interval)');a.axvline(0,c=INK,ls=':',lw=.7);panel(a,'a','Paired model comparisons','x')
    e=read('cross_market_event_concentration.csv').set_index('portfolio').loc[['NYISO','ISO-NE-SOUTH','ISO-NE-NORTH']];b=fig.add_subplot(gs[1]);xx=np.arange(3)
    for off,f,c,l in [(-.24,'annual_saving_usd',W,'All dates'),(0,'after_largest_day_usd',COLORS['World-ME'],'Largest day removed'),(.24,'after_largest_7day_usd',REF,'Largest seven days removed')]:b.bar(xx+off,e[f]/1000,.23,color=c,label=l)
    b.set(xticks=xx,xticklabels=['NYISO','ISO-NE South','ISO-NE North'],ylabel='Saving vs immediate\nexecution (US$k)',ylim=(0,138));b.legend(frameon=False,loc='upper right',fontsize=5.9);panel(b,'b','Cross-market financial concentration','y')
    save(fig,'Fig5',['paired_all_ontime_comparators_24h.csv','cross_market_event_concentration.csv'],'Two panels: paired model intervals and three-market financial exclusions, with immediate-execution references.')

def figure6():
    h=read('history/fluid_model_window_summary.csv');cur=read('current/fig3_service_windows.csv');jobs=read('historical_frontier.csv');jc=read('current/extfig1_frontier.csv');fig,(a,b)=axes(4.0,left=.1,wspace=.42);fig.subplots_adjust(bottom=.34)
    for m,g in h.groupby('model',sort=False):
        policy='interval' if (g.policy=='interval').any() else g.policy.iloc[0];g=g[g.policy==policy].sort_values('window_h');a.plot(g.window_h,np.where(g.late<=1e-8,g.saving/1000,np.nan),c=col(m),lw=.85)
    q=cur.groupby('window_h').saving_usd.agg(['mean','min','max']);a.errorbar(q.index,q['mean']/1000,yerr=np.array([q['mean']-q['min'],q['max']-q['mean']])/1000,c=W,marker='D',lw=1.7,ms=4,capsize=2)
    a.set(xlabel='Service window (h)',ylabel='Annual saving (US$k)');panel(a,'a','Value across service windows','y')
    old=jobs.groupby('window_h').late.agg(['min','max']);b.fill_between(old.index,old['min'],old['max'],color=REF,alpha=.5,label='Earlier pipeline range');prev=jobs[jobs.model=='World-ME'].sort_values('window_h')
    b.plot(prev.window_h,prev.late,c=COLORS['World-ME'],ls='--',marker='o',ms=3,label='Previous WorldMoE');b.plot(jc.window_h,jc.late_jobs,c=W,marker='D',lw=1.7,ms=4,label='WorldMoE');b.set_yscale('symlog',linthresh=1);b.set(xlabel='Frontier contract (h)',ylabel='Late jobs',xticks=[24,30,36,42,48]);panel(b,'b','Frontier deadline compliance','y');b.legend(frameon=False,fontsize=5.7,loc='upper right')
    handles=[Line2D([0],[0],color=col(m),lw=1,label=lab(m)) for m in h.model.unique()]+[Line2D([0],[0],color=W,lw=1.7,marker='D',label='WorldMoE')];fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.02),ncol=4,frameon=False,fontsize=5.5)
    save(fig,'Fig6',['history/fluid_model_window_summary.csv','current/fig3_service_windows.csv','historical_frontier.csv','current/extfig1_frontier.csv'],'Two panels: service-window economics and Frontier lateness. All historical Frontier methods enter the range.')

def figure7():
    q=read('zero_shot_isone_seed_results.csv');rows=[]
    for market in ['NYISO-Eagle native','ISO-NE-SOUTH','ISO-NE-NORTH']:
        for policy,label,color in [('Frozen WorldMoE native','WorldMoE',W),('Frozen WorldMoE zero-shot','WorldMoE',W),('published_DA_myopic','Published DA',COLORS['DA-myopic']),('lagged_RT_24h_myopic','Lagged RT',COLORS['RT-lag'])]:
            g=q[(q.portfolio==market)&(q.policy==policy)]
            if policy=='Frozen WorldMoE zero-shot':g=g[g.embedding_rule=='neutral_mean']
            if len(g):rows.append(dict(market=market,label=label,color=color,mean=g.saving_pct.mean(),low=g.saving_pct.min(),high=g.saving_pct.max()))
    r=pd.DataFrame(rows);fig,a=plt.subplots(figsize=(7.2,3.65));fig.subplots_adjust(left=.32,right=.97,bottom=.17,top=.9);yy=np.arange(len(r))
    names=[{'NYISO-Eagle native':'NYISO','ISO-NE-SOUTH':'ISO-NE South','ISO-NE-NORTH':'ISO-NE North'}[x.market]+' / '+x.label for x in r.itertuples()]
    a.barh(yy,r['mean'],color=r.color,height=.6)
    for i,x in enumerate(r.itertuples()):a.hlines(i,x.low,x.high,color=INK,lw=.8);a.text(x.mean+.25,i,f'{x.mean:.2f}%',va='center',fontsize=7)
    a.set(yticks=yy,yticklabels=names,xlabel='Saving against matched immediate execution (%)',xlim=(0,22));a.invert_yaxis();a.grid(axis='x',color=GRID,lw=.5);a.set_axisbelow(True);a.set_title('Frozen market transfer and same-market controls',loc='left',fontweight='bold',pad=10)
    save(fig,'Fig7',['zero_shot_isone_seed_results.csv'],'One panel: three market portfolios and measured ISO-NE controls, using matched variable-settlement references.')

def ed1():
    q=read('cross_market_monthly_mtm.csv');fig,a=plt.subplots(figsize=(7.2,3.2));fig.subplots_adjust(left=.1,right=.98,bottom=.16,top=.9)
    for m,c,style,mark,label in [('NYISO',W,'-','D','NYISO'),('ISO-NE-SOUTH',COLORS['RASC-ME'],'--','s','ISO-NE South'),('ISO-NE-NORTH',COLORS['World-ME'],':','^','ISO-NE North')]:
        g=q[q.portfolio==m].groupby('month').saving_usd.mean().sort_index();a.plot(range(12),g/1000,c=c,ls=style,marker=mark,ms=4,label=label)
    a.set(xticks=range(12),xticklabels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],ylabel='Monthly saving (US$k)',xlabel='2025');a.legend(frameon=False,ncol=3);a.grid(axis='y',color=GRID,lw=.5)
    save(fig,'ExtendedDataFig1',['cross_market_monthly_mtm.csv'],'One panel: monthly MTM means across three refinements with condition-matched immediate execution.',True)

def ed2():
    s=read('ED3_source.csv');g=read('G3_SUMMARY.csv');g=g[g.case!='G3-NATIVE'];fig,(a,b)=axes(3.45,left=.1,wspace=.54);q=s[s.panel=='a']
    a.plot(pd.to_numeric(q.x_value),q.y_value,'o-',c=COLORS['RASC-ME'],ms=4);a.set_yscale('symlog',linthresh=1);a.set(xlabel='Uniformly scaled utilization (%)',ylabel='Immediate-execution late IT (MWh)');panel(a,'a','Feasibility under burst scaling','y')
    for share,marker,color,label in [(.1,'o',W,'10% flexible'),(.3,'s',COLORS['RASC-ME'],'30% flexible')]:
        q=g[np.isclose(g.flex_share,share)].sort_values('utilization');b.plot(q.utilization*100,q.saving_per_flex_mwh,c=color,marker=marker,label=label)
        for r in q[q.structurally_feasible.astype(bool)].itertuples():b.errorbar(r.utilization*100,r.saving_per_flex_mwh,yerr=[[r.saving_per_flex_mwh-r.saving_per_flex_mwh_min],[r.saving_per_flex_mwh_max-r.saving_per_flex_mwh]],fmt='none',ecolor=color,capsize=2,lw=.7)
    b.set(xlabel='Total variable-IT utilization (%)',ylabel='Saving / flexible IT (US dollars / MWh)',ylim=(0,35));b.legend(frameon=False,loc='lower left');panel(b,'b','Value with a rigid background','y')
    save(fig,'ExtendedDataFig2',['ED3_source.csv','G3_SUMMARY.csv'],'Two panels: separate burst-scaling and rigid-background experiments. Infeasible 60%/30% has no saving point.',True)

def main():
    MANIFEST.clear();(F/'main').mkdir(parents=True,exist_ok=True)
    builder=ROOT/'figure_source/Fig1_reference_20261010_v3/build_fig1.py'
    subprocess.run([sys.executable,str(builder),'--output-dir',str(F/'main')],check=True)
    for ext in ['png','pdf','svg']:
        shutil.copy2(F/'main'/f'Fig1_mechanism_20261010_v3.{ext}',F/'main'/f'Fig1.{ext}')
        shutil.copy2(F/'main'/f'WorldMoE_architecture_detail_20261010_v3.{ext}',F/'main'/f'WorldMoE_architecture_detail.{ext}')
    MANIFEST.append(dict(figure='Fig1',source_script='figure_source/Fig1_reference_20261010_v3/build_fig1.py',sources=['figure_inputs.json','comparison_models.csv','comparison_diurnal_imports.csv','service_windows.csv'],comparison_scope='Archived full-pipeline comparisons; no architecture-isolated effect',panels='WorldMoE-centred architecture and parallel inputs',palette='Original eight-colour palette',png_dpi=450))
    for fn in [figure2,figure3,figure4,figure5,figure6,figure7,ed1,ed2]:print('Building',fn.__name__,flush=True);fn()
    for r in MANIFEST[1:]:r['source_script']='figure_source/compact_figures.py';r['palette']='Pink and purple';r['panels']=1 if r['figure'] in ['Fig2','Fig7','ExtendedDataFig1'] else 2
    (ROOT/'FIGURE_MANIFEST.json').write_text(json.dumps(MANIFEST,indent=2),encoding='utf-8')
if __name__=='__main__':main()
