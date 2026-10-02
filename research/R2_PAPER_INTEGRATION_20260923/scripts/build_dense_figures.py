"""Measured observations only; editable standalone journal figures."""
from pathlib import Path
import json,argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as path_effects
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
OUT=ROOT/'figures';DATA=OUT/'source_data';DATA.mkdir(exist_ok=True)
CARD=json.loads((ROOT/'configs/visual_palette.json').read_text())
TERMS=json.loads((ROOT/'configs/terminology_contract.json').read_text())['display_labels']
R2=CARD['R2_point'];INTERVAL=CARD['R2_interval'];RASC=CARD['RASC'];DA=CARD['Published_DA'];MYOPIC=CARD['released_work'];BASE=CARD['Earliest_service'];UNC=CARD['uncertainty'];PV='#F6DFD6'
INK='#222222';GRID='#E5E7EB';WIDTH=7.15
plt.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
    'text.color':INK,'axes.labelcolor':INK,'xtick.color':'#5B616A','ytick.color':'#5B616A',
    'axes.edgecolor':UNC,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
    'lines.linewidth':2.0,'lines.markersize':4.2,'svg.fonttype':'none','pdf.fonttype':42})
W=list(range(6,25));SEEDS=[20260920,20260921,20260922]

def records():return pd.DataFrame([json.loads(p.read_text()) for p in (ROOT/'results/paper_runs').glob('*.json')])
def main_rows(all_seeds=True):
    f=records();f=f[(f.year==2025)&(f.trace=='Eagle')&(f.model=='R2')&(f.physical_branch=='particle_trailing4')&(f.ablation=='normal')&f.window_h.isin(W)]
    if not all_seeds:f=f[f.seed==20260920]
    assert not f.duplicated(['seed','window_h','policy']).any()
    expected=114 if all_seeds else 38
    assert len(f)==expected,f'Window experiments incomplete: {len(f)}/{expected}'
    return f
def baselines():
    f=records();f=f[f.name.str.startswith('baseline_')]
    assert len(f)==95,f'Baseline windows incomplete: {len(f)}/95'
    return f
def grid(ax,axis='y'):ax.grid(axis=axis,color=GRID,lw=.6);ax.set_axisbelow(True)
def label(ax,letter):ax.text(-.015,1.065,letter,transform=ax.transAxes,weight='bold',size=11,va='bottom')
def export(fig,name,frames):
    # Thin neutral keylines keep the exact light card colors legible on white.
    def keyline(line):
        if isinstance(line,Line2D) and to_hex(line.get_color()).upper() in CARD['hex']:
            width=max(line.get_linewidth(),1.6);line.set_linewidth(width)
            line.set_path_effects([path_effects.Stroke(linewidth=width+.35,foreground='#625E68',alpha=.28),path_effects.Normal()])
    for ax in fig.axes:
        for line in ax.lines:keyline(line)
        if ax.get_legend():
            for handle in ax.get_legend().legend_handles:keyline(handle)
    for legend in fig.legends:
        for handle in legend.legend_handles:keyline(handle)
    for extension in ['png','svg','pdf']:fig.savefig(OUT/f'{name}.{extension}',dpi=450,facecolor='white',bbox_inches='tight',pad_inches=.045)
    for panel,frame in frames.items():frame.to_csv(DATA/f'{name}_{panel}.csv',index=False)
    plt.close(fig)
def ordered(f,policy):
    q=f[f.policy==policy].sort_values('window_h');assert q.window_h.tolist()==W
    return q

def fig2():
    measured=main_rows(False);base=baselines()
    with np.load(WORK/'external_validation/data/forcing_2025.npz') as z:
        arrivals=z['eagle_arrivals'];k=z['k'];times=pd.DatetimeIndex(pd.to_datetime(z['times'],utc=True))
    queue=np.zeros(4);backlog=[];service=[]
    for a in arrivals:
        ready=queue+a;take=np.minimum(ready,1.);queue=ready-take
        backlog.append(float(.25*ready.dot(k)));service.append(take)
    peak=int(np.argmax(backlog));start=max(0,(peak-96)//4*4);end=min(len(times),start+288)
    full=pd.DataFrame({'timestamp':times,'released_MWh':.25*arrivals.dot(k),
        'served_MWh':.25*np.asarray(service).dot(k),'outstanding_MWh':backlog})
    hour=full.iloc[start:end].set_index('timestamp').resample('1h').agg({'released_MWh':'sum','served_MWh':'sum','outstanding_MWh':'max'}).reset_index()
    hour['hours']=(hour.timestamp-hour.timestamp.iloc[0]).dt.total_seconds()/3600
    floors=pd.read_csv(ROOT/'results/physical_floors_Eagle_2025_registered.csv');assert len(floors)==11
    fig=plt.figure(figsize=(WIDTH,5.55));gs=fig.add_gridspec(3,2,height_ratios=[.9,.58,1.22],hspace=.58,wspace=.4)
    a=fig.add_subplot(gs[0,:]);x=hour.hours
    a.bar(x,hour.released_MWh,width=.82,color=MYOPIC,alpha=.65,label='Released work')
    a.plot(x,hour.served_MWh,color=BASE,label=TERMS['Earliest'])
    a.axhline(k.sum(),color=RASC,ls=':',lw=1.4,label='Hourly service capacity')
    a.set(xlim=(0,x.max()),ylabel='Hourly IT service\n(MWh)',xticks=np.arange(0,73,12));a.tick_params(labelbottom=False)
    a.legend(frameon=False,ncols=3,loc='upper center',bbox_to_anchor=(.5,1.4),fontsize=8);grid(a);label(a,'a')
    a2=fig.add_subplot(gs[1,:],sharex=a)
    a2.fill_between(x,hour.outstanding_MWh,color=R2,alpha=.13);a2.plot(x,hour.outstanding_MWh,color=R2)
    a2.set(ylabel='Outstanding work\n(MWh)',xlabel=f'Hours from {hour.timestamp.iloc[0].strftime("%d %b %Y %H:%M")} UTC');grid(a2)
    b=fig.add_subplot(gs[2,0]);b.plot(floors.capacity_fraction*100,floors.minimum_feasible_h,'o-',color=DA,mfc='white',mew=1.2)
    b.set(xlabel='Available service capacity (%)',ylabel='Minimum service window (h)',xticks=[50,60,70,80,90,100]);grid(b);label(b,'b')
    c=fig.add_subplot(gs[2,1])
    for order,(f,p,col,mark,ls,name,ms) in enumerate([(base[base.model=='RASC'],'point',RASC,'s','--',TERMS['RASC_point'],6.2),(measured,'point',R2,'o','-',TERMS['R2_point'],4.9),(measured,'interval',INTERVAL,'D',':',TERMS['R2_interval'],3.0)]):
        q=ordered(f,p);c.plot(q.window_h,q.late_IT_mwh,marker=mark,color=col,ls=ls,label=name,
            ms=ms,mfc='white' if p=='point' else col,mew=1.0,lw=2.1 if order==0 else 1.3,zorder=3+order)
    floor=float(floors.loc[floors.capacity_fraction==1,'minimum_feasible_h'].iloc[0]);c.axvline(floor,color=UNC,ls=':')
    c.set(xlabel='Service window (h)',ylabel='Late IT energy (MWh)',xticks=[6,9,12,15,18,21,24]);grid(c);label(c,'c')
    c.legend(frameon=False,fontsize=7.4,loc='upper right')
    fig.subplots_adjust(left=.105,right=.99,bottom=.08,top=.88)
    export(fig,'Fig2_service_resource',{'a_hourly':hour,'b_capacity':floors,'c_windows':pd.concat([measured,base[base.model=='RASC']])})

def paired_intervals(primary,base):
    rng=np.random.default_rng(20260920);rows=[]
    for w in W:
        r=primary[(primary.window_h==w)&(primary.policy=='point')].iloc[0]
        b=base[(base.window_h==w)&(base.policy=='point')&(base.model=='RASC')].iloc[0]
        folder=ROOT/'results/economic_trajectories'
        x=pd.read_csv(folder/f'{r["name"]}_daily.csv').set_index('date').mtm_value_usd
        y=pd.read_csv(folder/f'{b["name"]}_daily.csv').set_index('date').mtm_value_usd
        assert x.index.equals(y.index);difference=(x-y).to_numpy();n=len(difference)
        starts=rng.integers(n,size=(2000,int(np.ceil(n/7))))
        index=((starts[...,None]+np.arange(7))%n).reshape(2000,-1)[:,:n]
        sampled=difference[index].sum(1);lo,hi=np.quantile(sampled,[.025,.975])
        rows.append(dict(window_h=w,difference_usd=difference.sum(),lower95=lo,upper95=hi,late_it_mwh=r.late_IT_mwh))
    return pd.DataFrame(rows)

def forecast_weekly():
    out=[];taus=np.array([.05,.1,.25,.5,.75,.9,.95])
    for seed in SEEDS:
        f=pd.read_parquet(ROOT/'results/predictions_2025'/f'R2_seed{seed}_normal_native.parquet');f=f[f.gate=='post_DA'].copy()
        if seed==20260920:truth=f[['timestamp_utc','zone','date','spread']].copy()
        q=f[[f'dart_q{t:.2f}' for t in taus]].to_numpy();err=f.spread.to_numpy()[:,None]-q
        f['loss']=np.maximum(taus*err,(taus-1)*err).mean(1);f['week']=(pd.to_datetime(f.date)-pd.Timestamp('2025-01-01')).dt.days//7
        out.extend(dict(model='R2',seed=seed,week=int(week),pinball=float(g.loss.mean()),rows=len(g)) for week,g in f.groupby('week'))
    path=WORK/'新论文_市场可交付灵活性_20260917/revision_20260918/forecast_2025/bridge_forecast_only_hourly.parquet'
    f=pd.read_parquet(path);f=f[f.model=='RASC'].copy()
    f=f.drop(columns=['spread','date'],errors='ignore').merge(truth,on=['timestamp_utc','zone'],how='inner',validate='one_to_one')
    assert len(f)==35040
    q=f[[f'q{t:.2f}' for t in taus]].to_numpy()
    err=f.spread.to_numpy()[:,None]-q;f['loss']=np.maximum(taus*err,(taus-1)*err).mean(1)
    f['week']=(pd.to_datetime(f.date)-pd.Timestamp('2025-01-01')).dt.days//7
    out.extend(dict(model='RASC',seed=0,week=int(week),pinball=float(g.loss.mean()),rows=len(g)) for week,g in f.groupby('week'))
    return pd.DataFrame(out)

def fig3():
    seeds=main_rows(False);primary=seeds; base=baselines()
    weekly=forecast_weekly();paired=paired_intervals(primary,base)
    fig=plt.figure(figsize=(WIDTH,7.15));outer=fig.add_gridspec(3,1,height_ratios=[1.5,1.35,1],hspace=.48)
    ag=outer[0].subgridspec(2,1,height_ratios=[3,1],hspace=.08);bg=outer[1].subgridspec(2,1,height_ratios=[2.8,1],hspace=.08)
    a=fig.add_subplot(ag[0]);ad=fig.add_subplot(ag[1],sharex=a)
    b=fig.add_subplot(bg[0]);bd=fig.add_subplot(bg[1],sharex=b);c=fig.add_subplot(outer[2])
    for f,policy,col,ls,mark,name in [(primary,'point',R2,'-','o',TERMS['R2_point']),(primary,'interval',INTERVAL,'--','D',TERMS['R2_interval']),(base[base.model=='RASC'],'point',RASC,'--','^',TERMS['RASC_point']),(base[base.model=='Published_DA'],'point',DA,'-.','s',TERMS['Published_DA_persistent'])]:
        q=ordered(f,policy);a.plot(q.window_h,q.net_operating_cost_usd/1000,marker=mark,color=col,ls=ls,label=name,
            mfc='white' if name in [TERMS['RASC_point'],TERMS['Published_DA_persistent']] else col,mew=1.1)
    a.axhline(primary.reference_cost_usd.iloc[0]/1000,color=BASE,ls=':',label='Earliest service')
    a.set(ylabel='Service-adjusted cost\n(US$ thousands)',xticks=[6,9,12,15,18,21,24]);a.tick_params(labelbottom=False);grid(a);label(a,'a')
    a.legend(frameon=False,ncols=3,fontsize=7.7,loc='lower center',bbox_to_anchor=(.5,1.08))
    costgap=ordered(base[base.model=='Published_DA'],'point').net_operating_cost_usd.to_numpy()-ordered(base[base.model=='RASC'],'point').net_operating_cost_usd.to_numpy()
    ad.axhline(0,color=BASE,lw=.7);ad.plot(W,costgap,color=DA,marker='o',ms=2.8,lw=1.2)
    ad.set(xlabel='Service window (h)',ylabel='DA − RASC\ncost (US$)');ad.tick_params(axis='y',labelsize=7);grid(ad)
    q=weekly[weekly.model=='R2'].groupby('week').pinball.agg(['mean','min','max']);b.fill_between(q.index+1,q['min'],q['max'],color=R2,alpha=.13)
    b.plot(q.index+1,q['mean'],color=R2,marker='o',ms=2.5,label=TERMS['R2_mean'])
    old=weekly[weekly.model=='RASC'];b.plot(old.week+1,old.pinball,color=RASC,marker='s',mfc='white',ms=3.3,ls='--',lw=1.25,label='RASC')
    b.set(ylabel='Spread pinball loss\n(US$/MWh)',xlim=(1,53));b.tick_params(labelbottom=False);b.legend(frameon=False,ncols=2,fontsize=8);grid(b);label(b,'b')
    old_loss=old.set_index('week').pinball.reindex(q.index);gap=q['mean']-old_loss
    bd.fill_between(q.index+1,q['min']-old_loss,q['max']-old_loss,color=INTERVAL,alpha=.13)
    bd.axhline(0,color=BASE,lw=.7);bd.plot(q.index+1,gap,color=INTERVAL,marker='o',ms=2.2,lw=1.1)
    bd.set(xlabel='Week of 2025',ylabel='R2 − RASC\nloss',xticks=[1,13,26,39,53]);bd.tick_params(axis='y',labelsize=7);grid(bd)
    c.fill_between(paired.window_h,paired.lower95/1000,paired.upper95/1000,color=UNC,alpha=.38)
    c.plot(paired.window_h,paired.difference_usd/1000,color=R2)
    for _,r in paired.iterrows():c.plot(r.window_h,r.difference_usd/1000,'o',color=R2,mfc='white' if r.late_it_mwh>1e-8 else R2)
    c.axhline(0,color=BASE,ls=':',lw=1);c.set(xlabel='Service window (h)',ylabel='R2 − RASC saving\n(US$ thousands)',xticks=[6,9,12,15,18,21,24]);grid(c);label(c,'c')
    fig.subplots_adjust(left=.14,right=.985,bottom=.055,top=.93)
    export(fig,'Fig3_cost_and_information',{'a_costs':pd.concat([seeds,base]),'a_cost_gap':pd.DataFrame(dict(window_h=W,published_DA_minus_RASC_cost_usd=costgap)),
        'b_weekly':weekly,'b_loss_gap':pd.DataFrame(dict(week=q.index+1,R2_minus_RASC_pinball=gap.to_numpy())),'c_paired':paired})

def fig4():
    seeds=main_rows();primary=seeds[seeds.seed==20260920];base=baselines();ratios=[]
    for seed,g in seeds.groupby('seed'):
        for w,g2 in g.groupby('window_h'):
            p=g2[g2.policy=='point'].iloc[0];r=g2[g2.policy=='interval'].iloc[0]
            ratios.append(dict(seed=seed,window_h=w,value_retained=100*r.value_usd/p.value_usd,energy_used=100*r.shifted_IT_mwh/p.shifted_IT_mwh,
                cvar_reduction=100*(1-r.mtm_daily_loss_cvar95_usd/p.mtm_daily_loss_cvar95_usd)))
    ratios=pd.DataFrame(ratios)
    fig,(a,b)=plt.subplots(1,2,figsize=(WIDTH,3.55),gridspec_kw={'width_ratios':[1.12,1],'wspace':.4})
    for f,model,col in [(primary,'R2',R2),(base[base.model=='RASC'],'RASC',RASC)]:
        for policy,mark,ls in [('point','o','-'),('interval','D','--')]:
            linecol=INTERVAL if model=='R2' and policy=='interval' else col
            q=ordered(f,policy);a.plot(q.shifted_IT_mwh,q.value_usd/1000,color=linecol,marker=mark,ls=ls,label=TERMS[model+'_'+policy],
                mfc='white' if model=='RASC' else linecol,mew=1.1)
            failed=q[q.late_IT_mwh>1e-8]
            a.scatter(failed.shifted_IT_mwh,failed.value_usd/1000,marker='x',s=32,color=INK,linewidths=.75,zorder=8)
            for w in [12,24]:
                row=q[q.window_h==w].iloc[0];a.annotate(f'{w} h',(row.shifted_IT_mwh,row.value_usd/1000),xytext=(4,4 if policy=='point' else -9),textcoords='offset points',fontsize=7,color=linecol)
    a.set(xlabel='Shifted IT energy (MWh)',ylabel='Annual cost saving (US$ thousands)');grid(a);label(a,'a');a.legend(frameon=False,fontsize=7.3,loc='upper left')
    for field,col,ls,name,mark in [('value_retained',R2,'-','Cost saving retained','o'),('energy_used',DA,'--','Shifted IT energy used','s'),('cvar_reduction',RASC,':','CVaR95 reduction','D')]:
        q=ratios.groupby('window_h')[field].agg(['mean','min','max']);b.fill_between(q.index,q['min'],q['max'],color=col,alpha=.1)
        b.plot(q.index,q['mean'],color=col,ls=ls,marker=mark,ms=3.6,label=name,mfc='white' if field=='energy_used' else col)
    b.axhline(0,color=BASE,lw=.6);b.set(xlabel='Service window (h)',ylabel='Interval relative to point (%)',xticks=[6,9,12,15,18,21,24]);grid(b);label(b,'b');b.legend(frameon=False,fontsize=7.3,loc='lower left',bbox_to_anchor=(0,.085))
    fig.subplots_adjust(left=.095,right=.99,bottom=.17,top=.91)
    export(fig,'Fig4_value_and_shifted_energy',{'a_frontier':pd.concat([seeds,base[base.model=='RASC']]),'b_ratios':ratios})

def fig5():
    allruns=records();mixed=allruns[allruns.name.str.startswith('Eagle_2025_R2_seed20260920_mix')].copy()
    assert len(mixed)==32,f'Mixed experiments incomplete: {len(mixed)}/32'
    mixed['urgent_pct']=mixed.shares.apply(lambda v:100*v[0]);mixed['long_pct']=mixed.shares.apply(lambda v:100*v[2])
    sweep=mixed[mixed.long_pct==25].copy();primary=main_rows();ratios=[]
    fig,axs=plt.subplots(2,2,figsize=(WIDTH,5.65),gridspec_kw={'hspace':.65,'wspace':.42});a,b,c,d=axs.ravel()
    spec=a.get_subplotspec();a.remove()
    inner=spec.subgridspec(2,1,height_ratios=[3,1],hspace=.12)
    a=fig.add_subplot(inner[0]);detail=fig.add_subplot(inner[1],sharex=a)
    for policy,mark,ls in [('point','o','--'),('interval','D','-')]:
        for recourse,col in [(False,RASC),(True,R2)]:
            q=sweep[(sweep.policy==policy)&(sweep.recourse==recourse)].sort_values('urgent_pct')
            linecol=INTERVAL if recourse and policy=='interval' else col
            a.plot(q.urgent_pct,q.late_IT_mwh,marker=mark,ls=ls,color=linecol,label=TERMS[('recourse_' if recourse else 'fixed_')+policy],
                mfc='white' if policy=='point' else linecol,ms=5.3 if policy=='point' else 3.7,mew=1.1)
        fixed=sweep[(sweep.policy==policy)&(sweep.recourse==False)].set_index('urgent_pct')
        repaired=sweep[(sweep.policy==policy)&(sweep.recourse==True)].set_index('urgent_pct')
        change=(repaired.value_usd-fixed.value_usd).sort_index()
        b.plot(change.index,change,marker=mark,ls=ls,color=R2 if policy=='point' else INTERVAL,label=TERMS['R2_'+policy],mfc='white' if policy=='point' else INTERVAL)
        ratios.extend(dict(policy=policy,urgent_pct=x,value_change_usd=v) for x,v in change.items())
    a.set(ylabel='Late IT energy (MWh)');a.tick_params(labelbottom=False);grid(a);label(a,'a')
    handles,labels=a.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,1.0),ncols=4,frameon=False,fontsize=7.2)
    for policy,col,mark,size in [('point',R2,'o',4.3),('interval',INTERVAL,'D',2.7)]:
        q=sweep[(sweep.policy==policy)&(sweep.recourse==True)].sort_values('urgent_pct')
        detail.plot(q.urgent_pct,q.late_IT_mwh,color=col,marker=mark,ms=size,mfc='white' if policy=='point' else col,lw=1.1)
    detail.set(xticks=[10,20,30,40,50,60],yticks=[0,.05],ylim=(-.006,.061),xlabel='Work assigned a 4 h deadline (%)',ylabel='Recourse\n(MWh)')
    detail.tick_params(labelsize=6.8,length=2);detail.yaxis.label.set_size(7);grid(detail)
    b.axhline(0,color=BASE,lw=.7);b.set(xlabel='Work assigned a 4 h deadline (%)',ylabel='Recourse saving change (US$)');b.legend(frameon=False,fontsize=7.7);grid(b);label(b,'b')
    with np.load(WORK/'external_validation/data/forcing_2025.npz') as z:
        k=z['k'];start=pd.Timestamp(z['times'][0],tz='UTC');n=len(z['times'])
    curves=[]
    for recourse,col in [(False,RASC),(True,INTERVAL)]:
        row=mixed[(mixed.policy=='interval')&(mixed.urgent_pct==25)&(mixed.long_pct==25)&(mixed.recourse==recourse)].iloc[0]
        events=pd.read_parquet(ROOT/'results/economic_trajectories'/f'{row["name"]}_assignments.parquet')
        late=events[events.execution>events.deadline]
        energy=.25*late.work_units.to_numpy()*k[late.zone.to_numpy(dtype=int)]
        series=np.bincount(late.deadline.to_numpy(dtype=int),weights=energy,minlength=n)
        time=pd.date_range(start,periods=len(series),freq='15min')
        hour=pd.Series(series,index=time).resample('1h').sum().cumsum()
        c.plot(hour.index,hour,color=col,label='Recourse' if recourse else 'Fixed')
        curves.append(pd.DataFrame(dict(timestamp=hour.index,cumulative_late_MWh=hour.to_numpy(),recourse=recourse)))
    from zoneinfo import ZoneInfo
    ticks=[pd.Timestamp(d,tz='America/New_York') for d in ['2025-01-01','2025-04-01','2025-07-01','2025-10-01','2025-12-31 23:00']]
    c.set_xticks(ticks);c.set_xlim(ticks[0],ticks[-1]);c.xaxis.set_major_formatter(mdates.DateFormatter('%b',tz=ZoneInfo('America/New_York')))
    c.set(ylabel='Cumulative late IT energy (MWh)',xlabel='2025 calendar');c.legend(frameon=False,fontsize=7.7);grid(c);label(c,'c')
    q=primary[primary.policy=='interval'].groupby('window_h').value_usd.agg(['mean','min','max'])
    d.fill_between(q.index,q['min']/1000,q['max']/1000,color=INTERVAL,alpha=.13)
    d.plot(q.index,q['mean']/1000,'D-',color=INTERVAL)
    d.set(xlabel='Common service window (h)',ylabel='Annual cost saving (US$ thousands)',xticks=[6,9,12,15,18,21,24]);grid(d);label(d,'d')
    fig.subplots_adjust(left=.11,right=.985,bottom=.09,top=.90)
    export(fig,'Fig5_service_recourse',{'a_mixes':mixed,'b_value_changes':pd.DataFrame(ratios),'c_hourly':pd.concat(curves),'d_windows':primary})

def fig6():
    pairs=[('Eagle',2024),('Eagle',2025),('Frontier',2024),('Frontier',2025)]
    allruns=records();runs=allruns[allruns.name.str.startswith('external_')&allruns.proxy.isin(['registered','requested_nodes'])]
    curves=runs[runs.policy.isin(['myopic_da','persistent_da'])&runs.window_h.isin(W)]
    assert len(curves)==152,f'External window matrix incomplete: {len(curves)}/152'
    mixed=runs[runs.name.str.contains('_mix')].copy();assert len(mixed)==64
    mixed['urgent_pct']=mixed.shares.apply(lambda v:100*v[0]);mixed['long_pct']=mixed.shares.apply(lambda v:100*v[2])
    fig,axs=plt.subplots(2,2,figsize=(WIDTH,5.5),gridspec_kw={'hspace':.65,'wspace':.43});a,b,c,d=axs.ravel();floors=[];increments=[]
    fig.subplots_adjust(left=.115,right=.94,bottom=.1,top=.88)
    palette={('Eagle',2024):R2,('Eagle',2025):INTERVAL,('Frontier',2024):RASC,('Frontier',2025):DA}
    handles=[]
    for trace,year in pairs:
        col=palette[(trace,year)];mark='o' if trace=='Eagle' else 's';ls='--' if year==2024 else '-';fill='white' if year==2024 else col
        proxy='registered' if trace=='Eagle' else 'requested_nodes';style=dict(color=col,marker=mark,ls=ls,mfc=fill,ms=3.8)
        floor=pd.read_csv(ROOT/'results'/f'physical_floors_{trace}_{year}_{proxy}.csv');assert len(floor)==11;floors.append(floor)
        a.plot(floor.capacity_fraction*100,floor.minimum_feasible_h,**style)
        sub=curves[(curves.trace==trace)&(curves.year==year)]
        persistent=ordered(sub,'persistent_da');myopic=ordered(sub,'myopic_da')
        b.plot(persistent.window_h,persistent.saving_pct,**style)
        diff=persistent.value_usd.to_numpy()-myopic.value_usd.to_numpy();c.plot(W,diff/1000,**style)
        increments.append(pd.DataFrame(dict(trace=trace,year=year,window_h=W,persistence_increment_usd=diff)))
        handles.append(Line2D([],[],label=f'{trace} ({year})',**style))
    a.set(xlabel='Available service capacity (%)',ylabel='Minimum service window (h)',xticks=[50,60,70,80,90,100])
    b.set(xlabel='Service window (h)',ylabel='Published DA (persistent)\ncost saving (%)',xticks=[6,9,12,15,18,21,24])
    c.axhline(0,color=UNC,lw=.8);c.set(xlabel='Service window (h)',ylabel='Persistence increment\n(US$ thousands)',xticks=[6,9,12,15,18,21,24])
    for ax,letter in zip([a,b,c],'abc'):grid(ax);label(ax,letter)
    from matplotlib.colors import LinearSegmentedColormap
    sub=d.get_subplotspec();d.remove();dg=sub.subgridspec(2,1,hspace=.46)
    urgency=[10,20,25,30,40,50,60];axes_d=[];mats=[]
    for recourse in [False,True]:
        matrix=[]
        for trace,year in pairs:
            sub=mixed[(mixed.trace==trace)&(mixed.year==year)&(mixed.long_pct==25)&(mixed.recourse==recourse)].set_index('urgent_pct')
            matrix.append(sub.late_IT_mwh.reindex(urgency).to_numpy())
        mats.append(np.array(matrix));assert np.isfinite(mats[-1]).all()
    cmap=LinearSegmentedColormap.from_list('user_card_late_work',CARD['heatmap'])
    vmax=max(float(m.max()) for m in mats)
    for index,(matrix,title) in enumerate(zip(mats,['Fixed reservations','Priority recourse'])):
        ax=fig.add_subplot(dg[index]);axes_d.append(ax)
        im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=vmax,cmap=cmap)
        ax.set(yticks=range(4),yticklabels=[f'{trace} ({year})' for trace,year in pairs],xticks=range(7),xticklabels=urgency)
        ax.tick_params(length=0,labelsize=6.8);ax.set_title(title,fontsize=8,pad=3,loc='left')
        if index==0:
            ax.tick_params(labelbottom=False)
            ax.text(-.015,1.27,'d',transform=ax.transAxes,weight='bold',size=11,va='bottom')
        else:ax.set_xlabel('4 h work share (%)',fontsize=8,labelpad=3)
        for i in range(4):
            for j in range(7):
                v=matrix[i,j]
                val='0' if v<1e-8 else f'{v:.0e}'.replace('e-0','e−') if v<.005 else f'{v:.3f}'.lstrip('0') if v<.1 else f'{v:.2f}' if v<10 else f'{v:.1f}'
                ax.text(j,i,val,ha='center',va='center',fontsize=6.7,color=INK)
        for spine in ax.spines.values():spine.set_visible(False)
    cb=fig.colorbar(im,ax=axes_d,fraction=.035,pad=.025);cb.set_label('Late IT energy (MWh)',fontsize=7.5);cb.ax.tick_params(labelsize=7,length=2)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,1.01),frameon=False,ncols=4,fontsize=7.6)
    export(fig,'Fig6_external_mechanisms',{'a_capacity':pd.concat(floors),'b_windows':curves,'c_increments':pd.concat(increments),'d_mixes':mixed})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--figures',nargs='+',type=int,default=[2,3,4,5,6]);args=p.parse_args()
    for number in args.figures:globals()[f'fig{number}']()
