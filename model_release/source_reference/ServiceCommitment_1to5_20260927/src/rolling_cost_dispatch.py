"""Queue-aware WorldMoE cost planning, with strictly lagged physical feedback.

Only the current planned service is executed. There is no adaptive lock rule.
The future plan uses issued forecasts, never realized future trajectories.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def execute(arr,timestamps,pre,post,pre_issue,post_issue,prior_pv,prior_pue,physical_pre,physical_post,pv_prediction,pv_weight,observed_pv,observed_pue,idle,k,pue_ref,historical,safety):
    n=len(arr);book=np.zeros((n+8192,4));remaining=np.zeros((n,4));oldest=np.zeros(4,np.int64)
    for t in range(n):
        for z in range(4):
            remaining[t,z]=arr[t,z]
            while oldest[z]<=t and remaining[oldest[z],z]<1e-10:oldest[z]+=1
            if oldest[z]>t:continue
            h=min(96,n-t);cost=np.empty(2*h);capacity=np.empty(2*h)
            for j in range(h):
                target=t+j;published=post_issue[target]<=timestamps[t];issued=pre_issue[target]<=timestamps[t]
                price=post[target,z] if published else pre[target,z] if issued else historical[z]
                valid=physical_post[target] if published else physical_pre[target] if issued else False
                ppue=prior_pue[target,z] if valid else pue_ref[z]
                ppv=prior_pv[target,z] if valid else 0.
                if valid and np.isfinite(pv_prediction[t,j,z]):ppv=(1-pv_weight)*ppv+pv_weight*pv_prediction[t,j,z]
                if j==0 and t>0:
                    ppv=observed_pv[t-1,z];ppue=observed_pue[t-1,z]
                solar=min(1.,max(0.,(ppv/ppue-idle[z])/k[z]))
                ratio=ppue/pue_ref[z];extra=0. if j==0 else 2.
                csolar=-ratio+extra;cgrid=price*ratio+extra
                # Optimistic linear envelope only for a predicted negative-price
                # nonconvex segment. Realized accounting remains unchanged.
                if cgrid<csolar:
                    average=solar*csolar+(1-solar)*cgrid;csolar=average;cgrid=average
                cost[2*j]=csolar;cost[2*j+1]=cgrid
                capacity[2*j]=solar;capacity[2*j+1]=1-solar
            order=np.argsort(cost);planned_now=0.;prefix=0.;required=0.
            for release in range(oldest[z],t+1):
                demand=remaining[release,z]
                if demand<1e-10:continue
                due=min(release+95,n-1);prefix+=demand
                required=max(required,prefix-safety*max(0,due-t))
                for ii in range(len(order)):
                    index=order[ii];slot=index//2
                    if t+slot>due or capacity[index]<1e-10:continue
                    take=min(demand,capacity[index]);capacity[index]-=take;demand-=take
                    if slot==0:planned_now+=take
                    if demand<1e-10:break
            service=min(1.,max(planned_now,required));done=0.
            for release in range(oldest[z],t+1):
                take=min(remaining[release,z],service-done)
                if take>0:remaining[release,z]-=take;done+=take
                if done>=service-1e-10:break
            book[t,z]=done
    for z in range(4):
        total=remaining[:,z].sum();t=n
        while total>1e-9:
            take=min(1.,total);book[t,z]=take;total-=take;t+=1
    return book
