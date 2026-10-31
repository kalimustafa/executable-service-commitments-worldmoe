"""Causal fluid execution extensions; no job runtime or discrete-job analysis."""
from collections import defaultdict
import numpy as np
from src.context import c
from vendor.forecast_provider import TZ

def project_grid(requests,arr,grid_fraction=1.,ramp_fraction=None,deadline_guard=False,window=24):
    """Project a frozen causal request stream, carrying every deferred unit forward.

    The request at t is issued by the frozen controller. The projection only
    reads current PV/PUE and arrived work, and serves cohorts in FIFO order.
    It does not use future realized forcing to promise a feasible schedule.
    """
    n=c.N;book=np.zeros_like(requests);released=np.zeros(4);requested=np.zeros(4);served=np.zeros(4)
    static=np.asarray(c.c['grid_import_limit'])*grid_fraction
    ramp=np.asarray(c.c['grid_import_limit'])*ramp_fraction if ramp_fraction is not None else None
    previous=np.maximum(c.pue[0]*c.idle-c.pv[0],0.)
    guard_grid=static if ramp is None else np.minimum(static,ramp)
    guaranteed_capacity=np.clip((guard_grid/max(c.c['pue_max'])-c.idle)/c.k,0,1)
    cumulative_arrivals=np.cumsum(arr,axis=0)
    max_excess=0.;max_ramp_excess=0.;binding=0
    for t in range(len(book)):
        if t<n:released+=arr[t];requested+=requests[t];pue=c.pue[t];pv=c.pv[t]
        else:
            requested=released.copy();pue=np.full(4,max(c.c['pue_max']));pv=np.zeros(4)
        limit=static.copy()
        if ramp is not None:limit=np.minimum(limit,previous+ramp)
        floor=np.maximum(pue*c.idle-pv,0.)
        max_excess=max(max_excess,float(np.maximum(floor-limit,0).max()))
        capacity=np.clip(((limit+pv)/pue-c.idle)/c.k,0,1)
        demand=np.maximum(np.minimum(requested,released)-served,0.)
        if deadline_guard and t<n:
            releases=np.arange(max(0,t-int(window*4)+1),t+1)
            remaining_steps=np.maximum(releases+int(window*4)-1-t,0)
            required=np.maximum(cumulative_arrivals[releases]-served-remaining_steps[:,None]*guaranteed_capacity,0).max(axis=0)
            demand=np.minimum(released-served,np.maximum(demand,required))
        take=np.minimum(demand,capacity);book[t]=take;served+=take
        actual=np.maximum(pue*(c.idle+c.k*take)-pv,0.)
        if t<n:
            binding+=int(np.sum(demand>capacity+1e-9))
            max_excess=max(max_excess,float(np.maximum(actual-static,0).max()))
            if ramp is not None:max_ramp_excess=max(max_ramp_excess,float(np.maximum(actual-previous-ramp,0).max()))
        previous=actual
        if t>=n and np.max(released-served)<1e-9:break
    if np.max(released-served)>1e-7:raise RuntimeError('Projection calendar exhausted')
    return book,dict(binding_site_intervals=binding,max_import_cap_excess_mw=max_excess,max_upward_ramp_excess_mw=max_ramp_excess,deadline_guard=deadline_guard)

def current_rank(model,t,z,amount,book):
    """Read current price/physical information without updating an observer book."""
    model.current_decision_index=t;ids=t+np.arange(model.h);now=model.base.times.asi8[t]
    post=model.post_issue[ids]<=now;pre=model.pre_issue[ids]<=now
    available=post|pre
    quant=np.where(post[:,None],model.post[ids,z],np.where(pre[:,None],model.pre[ids,z],model.base.historical[z]))
    physical=np.where(post,model.physical_post[ids],np.where(pre,model.physical_pre[ids],False))
    booked=book[ids,z];free=np.maximum(1-booked,0.)
    delta=np.where(free>1e-12,np.minimum(amount,free),1e-6)
    cost=model.cost_quantiles(quant,ids,z,booked,delta,physical).astype(np.float32)
    score=cost[:,4].copy();permitted=(cost[0,2]-score>2.)&available
    score[1:]=np.where(permitted[1:],score[1:],np.inf);score[0]=cost[0,3]
    return np.argsort(score,kind='stable').astype(np.int32)

def adaptive(arr,model,lock_h,pressure_threshold,window,event_myopic=False):
    n=c.N;h=int(window*4);lock=int(lock_h*4)
    book=np.zeros((n+8192,4));slots=[defaultdict(list) for _ in range(4)];last=np.zeros(4,dtype=int)
    local=c.times.tz_convert(TZ)
    issue_event=((local.hour==5)|(local.hour==11))&(local.minute==0)
    replan_calls=0;protected_calls=0;moved_work=0.;late=0.;max_error=0.
    def allocate(release,amount,z,now,earliest):
        if amount<=1e-12:return
        due=release+h-1
        if now<n:
            ranks=current_rank(model,now,z,amount,book)
            ids=now+ranks;ids=ids[(ids>=earliest)&(ids<=due)]
        else:ids=np.arange(max(now,earliest),due+1)
        if len(ids):
            free=np.maximum(1-book[ids,z],0.)
            take=np.minimum(free,np.maximum(amount-np.r_[0.,np.cumsum(free[:-1])],0.))
        else:take=np.empty(0)
        rem=amount-take.sum()
        if rem>1e-9:
            extra=np.arange(max(earliest,now,due+1),len(book));free=np.maximum(1-book[extra,z],0.)
            additional=np.minimum(free,np.maximum(rem-np.r_[0.,np.cumsum(free[:-1])],0.))
            if abs(additional.sum()-rem)>1e-7:raise RuntimeError('Adaptive calendar exhausted')
            ids=np.r_[ids,extra];take=np.r_[take,additional]
        use=take>1e-12;ids=ids[use];take=take[use];book[ids,z]+=take
        for target,units in zip(ids,take):slots[z][int(target)].append((release,float(units)))
        if len(ids):last[z]=max(last[z],int(ids.max()))
    for t in range(n):
        for z in range(4):
            amount=float(arr[t,z])
            if amount>1e-12:allocate(t,amount,z,t,t)
            if issue_event[t] and slots[z]:
                outstanding=defaultdict(float)
                for at,pieces in slots[z].items():
                    if at>=t:
                        for release,units in pieces:outstanding[release]+=units
                cumulative=0.;pressure=0.
                for release in sorted(outstanding):
                    cumulative+=outstanding[release]
                    pressure=max(pressure,cumulative/max(1,release+h-t))
                if event_myopic or pressure<pressure_threshold:
                    start=t+(0 if event_myopic else lock)
                    pending=defaultdict(float);end=int(last[z])+1
                    before=book[start:end,z].copy()
                    for at in range(start,end):
                        for release,units in slots[z].pop(at,[]):pending[release]+=units
                    if pending:
                        book[start:end,z]=0.;last[z]=max(t,start-1)
                        for release in sorted(pending):allocate(release,pending[release],z,t,start)
                        new_end=max(end,int(last[z])+1)
                        after=book[start:new_end,z]
                        prior=np.pad(before,(0,new_end-end))
                        moved_work+=.5*float(np.abs(after-prior).sum())*c.DT*c.k[z]
                        max_error=max(max_error,abs(float(sum(pending.values()))-float(book[start:max(end,last[z]+1),z].sum())))
                        replan_calls+=1
                else:protected_calls+=1
            for release,units in slots[z].pop(t,[]):
                if t>release+h-1:late+=units*c.DT*c.k[z]
    for z in range(4):
        for at,pieces in slots[z].items():
            for release,units in pieces:
                if at>release+h-1:late+=units*c.DT*c.k[z]
    if abs(book.sum()-arr.sum())>1e-5 or book.max()>1+1e-8:raise RuntimeError('Adaptive conservation/capacity')
    if max_error>1e-7:raise RuntimeError('Adaptive replan work mismatch')
    return book,late,dict(replan_calls=replan_calls,pressure_protected_events=protected_calls,
        reassigned_it_mwh=moved_work,lock_h=lock_h,pressure_threshold=pressure_threshold,max_replan_conservation_error=max_error,
        replan_calendar='05:00 and 11:00 local market-information events',event_myopic=event_myopic)
