"""World-conditioned incremental marginal-cost allocation with persistent claims.

This changes the cost-to-allocation interface, not physical capacity, the
service contract, price publication chronology, or financial accounting.
No future realized information is used. Allocations are never reopened.
"""
import numpy as np
from src.context import c

def allocate(arr,model,window,quantum=.1,risk=.5,margin=2.,close_year=False):
    n=c.N;h=int(window*4);book=np.zeros((n+8192,4));late=0.;pieces=0
    tau=np.array([.05,.1,.25,.5,.75,.9,.95])
    for t in range(n):
        model.current_decision_index=t;ids=t+np.arange(h);now=model.base.times.asi8[t]
        post=model.post_issue[ids]<=now;pre=model.pre_issue[ids]<=now;available=post|pre
        physical=np.where(post,model.physical_post[ids],np.where(pre,model.physical_pre[ids],False))
        for z in range(4):
            remaining=float(arr[t,z])
            if remaining<1e-12:continue
            q=np.where(post[:,None],model.post[ids,z],np.where(pre[:,None],model.pre[ids,z],model.base.historical[z]))
            if getattr(model,'oracle_price_diagnostic',False):
                if int(c.times[0].year)!=2024:raise RuntimeError('Oracle diagnostic restricted to 2024 selection analysis')
                inside=ids<n;q[inside]=c.rt[ids[inside],z,None]
            if hasattr(model,'price_update'):
                update=model.price_update[t//4,ids//4-t//4,z]
                valid_update=post&np.isfinite(update)
                shift=np.where(valid_update,model.price_update_weight*(update-q[:,3]),0.)
                q=q+shift[:,None]
            while remaining>1e-9:
                free=np.maximum(1-book[ids,z],0.)
                if close_year:free[ids>=n]=0.
                if free.sum()<1e-9:break
                delta=np.maximum(np.minimum(np.minimum(free,quantum),remaining),1e-8)
                cost=model.cost_quantiles(q,ids,z,book[ids,z],delta,physical).astype(np.float32)
                if risk==0:
                    score=cost[:,3].copy();score[1:]+=margin
                else:
                    lower=(1-risk)*cost[0,3]+risk*cost[0,2]
                    upper=(1-risk)*cost[:,3]+risk*cost[:,4]
                    score=upper.copy();permitted=(lower-upper>margin)&available
                    score[1:]=np.where(permitted[1:],score[1:],np.inf);score[0]=cost[0,3]
                valid=free>1e-9;score[~valid]=np.inf
                if np.isfinite(score).any():j=int(np.argmin(score))
                else:j=int(np.flatnonzero(valid)[0])
                take=min(remaining,float(free[j]),quantum);book[ids[j],z]+=take;remaining-=take;pieces+=1
            if remaining>1e-9:
                extra=np.arange(t+h,len(book));free=np.maximum(1-book[extra,z],0.)
                take=np.minimum(free,np.maximum(remaining-np.r_[0.,np.cumsum(free[:-1])],0.))
                if abs(take.sum()-remaining)>1e-7:raise RuntimeError('Marginal allocator exhausted calendar')
                book[extra,z]+=take;late+=remaining*c.DT*c.k[z]
    if abs(book.sum()-arr.sum())>1e-5 or book.max()>1+1e-8:raise RuntimeError('Marginal allocation conservation')
    return book,late,dict(marginal_quantum=quantum,risk_fraction=risk,transaction_margin=margin,allocation_pieces=pieces,persistent_claims=True,year_end_completion_requested=close_year)
