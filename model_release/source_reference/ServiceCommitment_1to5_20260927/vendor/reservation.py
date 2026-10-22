"""Exact formal reservation, capacity calendar and late-work handling."""
import numpy as np
from vendor import economic_core as c
def reference(arr,capacity):
    b=np.zeros((c.N+8192,4));queue=np.zeros(4)
    for t in range(c.N):
        queue+=arr[t];b[t]=np.minimum(queue,capacity);queue-=b[t]
    t=c.N
    while queue.max()>1e-10:b[t]=np.minimum(queue,capacity);queue-=b[t];t+=1
    return b

def rank(window,policy):
    if policy in ['point','robust']:return c.ranks(window,policy)
    D=int(window*4);ids=np.arange(c.N)[:,None]+np.arange(D)[None,:]
    if policy=='DA_only':
        prices=np.vstack([c.da,np.tile(c.train_rt,(D,1))])
        known=np.r_[c.known,np.full(D,np.iinfo(np.int64).max,dtype=np.int64)]
        scores=prices[ids].copy();avail=known[ids]<=c.times.asi8[:,None]
        scores[~avail]=np.broadcast_to(c.train_rt,scores.shape)[~avail]
    elif policy=='persistence':
        # Twenty-four ACTUAL hours before the target, only if that completed
        # interval precedes the current decision; no future persistence input.
        source=ids-96
        valid=(source>=0)&(source+4<=np.arange(c.N)[:,None])
        scores=np.broadcast_to(c.train_rt,(c.N,D,4)).copy()
        scores[valid]=c.rt[np.clip(source,0,c.N-1)][valid]
    else:raise ValueError(policy)
    scores[:,1:]+=2*c.FEE
    return np.argsort(scores,axis=1,kind='stable').astype(np.int16)

def schedule(arr,classes,policy,capacity=1.,keep_events=False):
    book=np.zeros((c.N+8192,4));hist=np.zeros(len(book));late=0.;delay_censored=0.;events=[]
    ranks={w:rank(w,policy) for _,w in classes}
    for t in range(c.N):
        for z in range(4):
            for fraction,w in sorted(classes,key=lambda x:x[1]):
                amount=arr[t,z]*fraction
                if amount<1e-12:continue
                deadline=t+int(w*4)-1;ids=t+ranks[w][t,:,z].astype(np.int32)
                free=np.maximum(capacity-book[ids,z],0)
                take=np.minimum(free,np.maximum(amount-np.r_[0.,np.cumsum(free[:-1])],0))
                book[ids,z]+=take
                remainder=amount-take.sum()
                if remainder>1e-9:
                    extra=np.arange(deadline+1,len(book));free=np.maximum(capacity-book[extra,z],0)
                    extra_take=np.minimum(free,np.maximum(remainder-np.r_[0.,np.cumsum(free[:-1])],0))
                    book[extra,z]+=extra_take
                    if abs(extra_take.sum()-remainder)>1e-7:raise ValueError('Insufficient calendar')
                    if deadline<c.N:late+=remainder*c.DT*c.k[z]
                    ids=np.r_[ids,extra];take=np.r_[take,extra_take]
                nz=take>1e-12;ids=ids[nz];take=take[nz]
                delivered=ids<c.N
                np.add.at(hist,ids[delivered]-t,take[delivered]*c.DT*c.k[z])
                delay_censored+=float(np.sum(take[~delivered]*c.DT*c.k[z]*(c.N-t)*c.DT))
                if keep_events:
                    events.extend((t,deadline,int(slot),z,float(work),w) for slot,work in zip(ids,take))
    assert abs(book.sum()-arr.sum())<1e-5
    assert book.max()<=capacity+1e-9
    weights=hist/hist.sum();cum=weights.cumsum()
    stats=dict(late_IT_mwh=late,mean_delay_hours=float(np.dot(weights,np.arange(len(hist))*.25)),
        p50_delay_hours=float(np.searchsorted(cum,.5)*.25),p95_delay_hours=float(np.searchsorted(cum,.95)*.25),
        p99_delay_hours=float(np.searchsorted(cum,.99)*.25),max_delivered_delay_hours=float(np.flatnonzero(hist>1e-10)[-1]*.25),
        delivered_delay_IT_mwh_hours=float(np.dot(hist,np.arange(len(hist))*.25)),
        closing_censored_delay_IT_mwh_hours=delay_censored,delivered_IT_mwh=float(hist.sum()))
    return book,stats,events
