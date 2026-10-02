import numpy as np
import pandas as pd
from collections import defaultdict
from types import SimpleNamespace
from vendor import economic_core as c
from vendor import reservation as d
H=SimpleNamespace(c=c)
CLASSES=[(.25,4),(.5,12),(.25,24)]

def myopic_execute(arr, window, rank_matrix, capacity=1.0, hedge=True):
    """Rolling-myopic executor E0 (no persistent reservation claims).

    Each interval, released-but-unexecuted work (merged by deadline, divisible)
    is re-planned with the information available at decision time t (the same
    provider rank rows the persistent kernel uses, evaluated at t). Only the
    current interval of each plan is executed. With hedge=True, deferral is
    limited by the fluid EDF feasibility condition so deadlines remain feasible;
    with hedge=False the executor follows the price argmin and any overflow at
    the deadline is retained post-deadline and counted late (kernel convention).
    """
    c = H.c
    N, DT, k = c.N, c.DT, c.k
    h = int(window * 4)
    INF = 1 << 30
    rank = np.asarray(rank_matrix, dtype=np.int32)
    assert rank.shape == (N, h, 4)
    pos0 = np.argmax(rank == 0, axis=1)
    m = np.full((N, 4), INF, dtype=np.int32)
    for i in range(h):
        upd = i < pos0
        if upd.any():
            m[upd] = np.minimum(m[upd], rank[:, i, :][upd])
    book = np.zeros((N + 8192, 4)); late = 0.0
    queues = [{} for _ in range(4)]
    for t in range(len(book)):
        arrivals = arr[t] if t < N else None
        for z in range(4):
            if arrivals is not None and arrivals[z] > 1e-12:
                dq = queues[z]; dl = t + h - 1
                dq[dl] = dq.get(dl, 0.) + arrivals[z]
        if t >= N and not any(queues):
            break
        for z in range(4):
            dq = queues[z]
            if not dq:
                continue
            free = capacity - book[t, z]
            mz = m[t, z] if t < N else INF
            dls = sorted(dq)
            if hedge and t < N:
                # required-now = max prefix surplus against capacity available
                # AFTER the current slot (latest-departure feasibility hedge)
                cum = 0.0; required = 0.0
                for dl in dls:
                    cum += dq[dl]
                    cap_left = capacity * (dl - t)
                    if cum - cap_left > required:
                        required = cum - cap_left
                if required > 1e-12 and free > 1e-12:
                    need = min(required, free)
                    for dl in dls:
                        if need <= 1e-12 or free <= 1e-12:
                            break
                        amt = dq[dl]
                        take = min(amt, need, free)
                        book[t, z] += take; free -= take; amt -= take
                        dq[dl] = amt; need -= take
            if free > 1e-12:
                for dl in dls:
                    if dl - t >= mz:
                        continue
                    amt = dq[dl]
                    if amt <= 1e-12:
                        continue
                    take = amt if amt < free else free
                    book[t, z] += take; free -= take; amt -= take
                    dq[dl] = amt
                    if free <= 1e-12:
                        break
            dq = {dl: a for dl, a in dq.items() if a > 1e-12}
            queues[z] = dq
            if t in dq:
                rem = dq.pop(t)
                if rem > 1e-9:
                    if t < N:
                        late += rem * DT * k[z]
                    extra = np.arange(t + 1, len(book))
                    freec = np.maximum(capacity - book[extra, z], 0.)
                    take = np.minimum(freec, np.maximum(rem - np.r_[0., np.cumsum(freec[:-1])], 0.))
                    book[extra, z] += take
                    if abs(take.sum() - rem) > 1e-7:
                        raise ValueError('calendar exhausted')
    assert abs(book.sum() - arr.sum()) < 1e-5
    return book, dict(late_IT_mwh=late)

def run(arr,policy):
    rank={w:d.rank(w,policy) for _,w in CLASSES}
    book=np.zeros((c.N+8192,4));slots=[defaultdict(list) for _ in range(4)]
    last=np.zeros(4,dtype=int);executed=[];repairs=0;max_repair_error=0.
    def allocate(job,amount,z,now,allow_overdue):
        release,due,w=job
        ids=release+rank[w][release,:,z].astype(np.int32);ids=ids[ids>=now]
        if allow_overdue:ids=np.r_[ids,np.arange(max(now,due+1),len(book))]
        free=np.maximum(1-book[ids,z],0)
        take=np.minimum(free,np.maximum(amount-np.r_[0.,np.cumsum(free[:-1])],0))
        use=take>1e-12;ids=ids[use];take=take[use]
        book[ids,z]+=take
        for target,units in zip(ids,take):slots[z][int(target)].append((job,float(units)))
        if len(ids):last[z]=max(last[z],int(ids.max()))
        return amount-take.sum()
    for t in range(c.N):
        for z in range(4):
            for fraction,w in CLASSES:
                amount=arr[t,z]*fraction
                if amount<=1e-12:continue
                job=(t,t+w*4-1,w);rem=allocate(job,amount,z,t,False)
                if rem>1e-9:
                    repairs+=1;pending=defaultdict(float)
                    before=float(book[t:last[z]+1,z].sum()+rem)
                    for at in range(t,last[z]+1):
                        for old,units in slots[z].pop(at,[]):pending[old]+=units
                    pending[job]+=rem;book[t:last[z]+1,z]=0.;last[z]=t
                    for old in sorted(pending,key=lambda q:(q[1],q[0])):
                        rest=allocate(old,pending[old],z,t,True)
                        if rest>1e-7:raise ValueError('Outstanding work lost')
                    max_repair_error=max(max_repair_error,abs(before-book[t:last[z]+1,z].sum()))
            for job,units in slots[z].pop(t,[]):
                executed.append((job[0],job[1],t,z,units,job[2]))
    future=[]
    for z in range(4):
        for at,pieces in slots[z].items():
            for job,units in pieces:future.append((job[0],job[1],at,z,units,job[2]))
    events=pd.DataFrame(executed+future,columns=['release_step','deadline_step','execution_step','site','work_units','window_h'])
    weight=events.work_units.to_numpy()*c.DT*c.k[events.site.to_numpy()]
    late=(events.execution_step>events.deadline_step)&(events.deadline_step<c.N)
    delivered=events.execution_step<c.N;delay=(events.execution_step-events.release_step)*c.DT
    idx=np.argsort(delay[delivered].to_numpy());dur=delay[delivered].to_numpy()[idx];energy=weight[delivered][idx];cw=np.cumsum(energy)/energy.sum()
    stats=dict(late_IT_mwh=float(weight[late].sum()),mean_delay_hours=float(np.sum(energy*dur)/energy.sum()),
        p50_delay_hours=float(dur[np.searchsorted(cw,.5)]),p95_delay_hours=float(dur[np.searchsorted(cw,.95)]),
        p99_delay_hours=float(dur[np.searchsorted(cw,.99)]),repair_calls=repairs,max_repair_conservation_error=max_repair_error)
    assert abs(book.sum()-arr.sum())<1e-6 and max_repair_error<1e-7
    return book,stats,events

def edf_reference(arr):
    # Independent work-conserving EDF executor, with real cohort deadlines.
    queues=[[] for _ in range(4)];records=[];book=np.zeros((c.N+8192,4))
    for t in range(len(book)):
        for z in range(4):
            if t<c.N:
                for fraction,w in CLASSES:
                    if arr[t,z]*fraction>1e-12:queues[z].append([t,t+4*w-1,arr[t,z]*fraction,w])
            queues[z].sort(key=lambda q:(q[1],q[0]));remaining=1.
            for job in queues[z]:
                take=min(job[2],remaining)
                if take>1e-12:records.append((job[0],job[1],t,z,take,job[3]))
                job[2]-=take;remaining-=take;book[t,z]+=take
                if remaining<1e-12:break
            queues[z]=[job for job in queues[z] if job[2]>1e-12]
        if t>=c.N and all(not q for q in queues):break
    f=pd.DataFrame(records,columns=['release_step','deadline_step','execution_step','site','work_units','window_h'])
    missed=f[(f.execution_step>f.deadline_step)&(f.deadline_step<c.N)]
    return book,float(np.sum(missed.work_units*c.DT*c.k[missed.site]))
