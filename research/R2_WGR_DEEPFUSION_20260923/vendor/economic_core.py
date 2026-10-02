"""Unmodified formal accounting functions; no legacy data-loading main."""
import numpy as np
import pandas as pd
from src.runtime import ROOT
N=0
DT=.25
FEE=1.
MIGRATION_COST=5.
DATA=ROOT/'results/economic_trajectories'
DATA.mkdir(exist_ok=True)
DAILY=[]
ROWS=[]
TRAJ={}
def risk(v):
    daily=np.bincount(day_index,weights=v,minlength=len(days))
    loss=np.sort(-daily)[::-1];mass=.05*len(daily);whole=int(mass)
    cvar=(loss[:whole].sum()+(mass-whole)*loss[whole])/mass
    cumulative=np.r_[0.,np.cumsum(v)]
    return dict(value_usd=float(v.sum()),loss_days=int((daily<-1e-8).sum()),
        daily_loss_cvar95_usd=float(cvar),max_drawdown_usd=float((np.maximum.accumulate(cumulative)-cumulative).max())),daily

def cash(book):
    load=pue*(idle+k*book[:N]);used=np.minimum(pv,load);grid=load-used
    return DT*(rt*grid+(pv-used)).sum(1),grid

def record(name,route,book,reference,meta,moved=None,late=0.,fee_override=None,grid_override=None):
    costs,grid=cash(book);refcost,refgrid=cash(reference)
    if grid_override is not None:
        grid,refgrid=grid_override
        used=pue*(idle+k*book[:N])-grid
        refused=pue*(idle+k*reference[:N])-refgrid
        costs=DT*(rt*grid+pv-used).sum(1)
        refcost=DT*(rt*refgrid+pv-refused).sum(1)
    fee=FEE*DT*np.abs(grid-refgrid).sum(1) if fee_override is None else fee_override
    mig=np.zeros(N) if moved is None else moved
    value=refcost-costs-fee-MIGRATION_COST*mig
    # Unfinished year-end work is retained and valued using pre-2024 medians.
    liability=DT*np.sum((book[N:].sum(0)-reference[N:].sum(0))*k*np.max(c['pue_max'])*train_rt)
    adjusted=value.copy();adjusted[-1]-=liability
    met,daily=risk(adjusted)
    cumulative_work_difference=DT*np.cumsum((reference[:N]-book[:N])*k,axis=0)
    if route=='spatial':
        service_debt=cumulative_work_difference.sum(1)*np.mean(train_rt)*np.max(c['pue_max'])
    else:service_debt=(cumulative_work_difference*train_rt*np.max(c['pue_max'])).sum(1)
    mtm=value-np.diff(np.r_[0.,service_debt])
    mtm_met,mtm_daily=risk(mtm)
    row=dict(name=name,route=route,**meta,**met,cash_value_usd=float(value.sum()),terminal_liability_difference_usd=float(liability),
        final_queue_IT_mwh=float(DT*np.sum(book[N:]*k)),reference_final_queue_IT_mwh=float(DT*np.sum(reference[N:]*k)),
        IT_energy_mwh=float(DT*np.sum(book[:N]*k)),shifted_IT_mwh=float(DT*np.maximum(reference[:N]-book[:N],0).dot(k).sum()),
        migrated_IT_mwh=float(mig.sum()),late_IT_mwh=float(late),fee_usd=float(fee.sum()),
        migration_cost_usd=float(MIGRATION_COST*mig.sum()),grid_import_mwh=float(DT*grid.sum()),
        max_compute=float(book[:N].max()),max_grid_mw=float(grid.max()),electrical_shortage_mwh=0.)
    row.update({f'mtm_{key}':v for key,v in mtm_met.items()})
    assert book.min()>-1e-9 and book.max()<=1+1e-8
    assert np.all(grid<=np.array(c['grid_import_limit'])+1e-8)
    ROWS.append(row)
    DAILY.extend(dict(name=name,date=str(day),value_usd=float(v),mtm_value_usd=float(m)) for day,v,m in zip(days,daily,mtm_daily))
    TRAJ[name]=dict(compute=book[:N],grid=grid,daily=daily,value=adjusted,mtm=mtm,service_debt=service_debt)
    np.savez_compressed(DATA/(name+'_trajectory.npz'),**TRAJ[name])
    return row
