"""Causal R2 cost decoder conditioned on already-known reservations.

The real reservation executor is unchanged. An indexable forecast adapter
supplies its existing rank API and maintains an exact observer of previously
executed deterministic reservation updates. Only the current released job is
read when its prediction is requested; no future realized arrivals are used.
"""
import numpy as np

class ReservationStateRanks:
    def __init__(self,base_provider,arrivals,window_h,policy,*,pv,pue,physical_pre,physical_post,
                 idle,k,capacity,pue_reference,identity=False):
        self.base=base_provider;self.arrivals=arrivals;self.n=len(arrivals);self.h=int(window_h*4)
        self.policy=policy;self.idle=np.asarray(idle);self.k=np.asarray(k);self.capacity=float(capacity)
        self.pue_reference=np.asarray(pue_reference);self.identity=identity
        self.book=np.zeros((self.n+8192,4));self.last_request=(-1,-1);self.requests=0;self.maximum_arrival_index_read=-1
        historical=np.broadcast_to(base_provider.historical,(self.h,4))
        tail=np.repeat(historical[...,None],7,axis=-1)
        self.pre=np.concatenate([base_provider.pre,tail]);self.post=np.concatenate([base_provider.post,tail])
        self.pre_issue=np.r_[base_provider.pre_issue,np.full(self.h,np.iinfo(np.int64).max,dtype=np.int64)]
        self.post_issue=np.r_[base_provider.post_issue,np.full(self.h,np.iinfo(np.int64).max,dtype=np.int64)]
        self.pv=np.vstack([pv,np.zeros((self.h,4))])
        self.pue=np.vstack([pue,np.tile(self.pue_reference,(self.h,1))])
        self.physical_pre=np.r_[physical_pre,np.zeros(self.h,bool)]
        self.physical_post=np.r_[physical_post,np.zeros(self.h,bool)]
        self.identity_ranks=base_provider.ranked_slots(window_h,policy) if identity else None

    def cost_quantiles(self,q,ids,z,booked,delta,physical):
        q=self.point_input(q,ids,z)
        denominator=self.k[z]*self.pue_reference[z]*delta
        pue=self.pue[ids,z];pv=self.pv[ids,z]
        load0=pue*(self.idle[z]+self.k[z]*booked)
        increment=pue*self.k[z]*delta
        additional_pv=np.minimum(np.maximum(pv-load0,0.),increment)
        covered=np.clip(additional_pv/increment,0.,1.)
        ratio=pue/self.pue_reference[z]
        a=np.where(physical,ratio*(1-covered),1.)
        b=np.where(physical,-ratio*covered,0.)
        return q*a[:,None]+b[:,None]

    def point_input(self,q,ids,z):
        if self.policy!='point' or not hasattr(self,'point_pre_mean'):return q
        now=self.base.times.asi8[self.current_decision_index]
        mean=np.where(self.post_issue[ids]<=now,self.point_post_mean[ids,z],
            np.where(self.pre_issue[ids]<=now,self.point_pre_mean[ids,z],self.base.historical[z]))
        return np.repeat(mean[:,None],7,axis=1)

    def set_conditional_mean_point_head(self,pre_mean,post_mean):
        tail=np.broadcast_to(self.base.historical,(self.h,4))
        self.point_pre_mean=np.vstack([pre_mean,tail]);self.point_post_mean=np.vstack([post_mean,tail])

    def __getitem__(self,index):
        t,sl,z=index
        if not isinstance(sl,slice) or sl!=slice(None):raise IndexError('Expected [decision, :, zone]')
        if (t,z)<=self.last_request:raise RuntimeError('Forecast observer requires chronological one-pass reservation queries')
        self.last_request=(int(t),int(z));self.requests+=1
        self.current_decision_index=int(t)
        self.maximum_arrival_index_read=max(self.maximum_arrival_index_read,int(t))
        amount=float(self.arrivals[t,z])  # Only the job released NOW is observed.
        ids=t+np.arange(self.h)
        if self.identity:
            ranks=self.identity_ranks[t,:,z].copy()
        else:
            now=self.base.times.asi8[t]
            post_available=self.post_issue[ids]<=now;pre_available=self.pre_issue[ids]<=now
            available=post_available|pre_available
            q=np.where(post_available[:,None],self.post[ids,z],
                np.where(pre_available[:,None],self.pre[ids,z],self.base.historical[z]))
            physical=np.where(post_available,self.physical_post[ids],np.where(pre_available,self.physical_pre[ids],False))
            booked=self.book[ids,z]
            free=np.maximum(self.capacity-booked,0.)
            # A full slot has no allocatable quantity. Use its marginal limit
            # for prediction; the unchanged executor still assigns zero there.
            delta=np.where(free>1e-12,np.minimum(amount,free),1e-6)
            cost=self.cost_quantiles(q,ids,z,booked,delta,physical).astype(np.float32)
            if not np.isfinite(cost).all() or (np.diff(cost,axis=1)<0).any():raise RuntimeError('Cost output violates finite ordered FP32 contract')
            median=cost[:,3]
            if self.policy=='point':
                score=median.copy();score[1:]+=2.
            elif self.policy=='robust':
                lower=cost[:,2];score=cost[:,4].copy()
                permitted=(lower[0]-score>2.)&available
                score[1:]=np.where(permitted[1:],score[1:],np.inf);score[0]=median[0]
            else:raise ValueError(self.policy)
            ranks=np.argsort(score,kind='stable').astype(np.int16)
        # Read-only state observer: mirrors the exact, unmodified allocation
        # that the real executor will immediately perform with this rank.
        ordered=t+ranks.astype(np.int32)
        free=np.maximum(self.capacity-self.book[ordered,z],0.)
        take=np.minimum(free,np.maximum(amount-np.r_[0.,np.cumsum(free[:-1])],0.))
        self.book[ordered,z]+=take;remainder=amount-take.sum()
        if remainder>1e-9:
            deadline=t+self.h-1;extra=np.arange(deadline+1,len(self.book))
            free=np.maximum(self.capacity-self.book[extra,z],0.)
            additional=np.minimum(free,np.maximum(remainder-np.r_[0.,np.cumsum(free[:-1])],0.))
            if abs(additional.sum()-remainder)>1e-7:raise RuntimeError('Observer calendar is insufficient')
            self.book[extra,z]+=additional
        return ranks

class DistributionalReservationRanks(ReservationStateRanks):
    def __init__(self,*args,pv_quantiles,pv_levels,**kwargs):
        super().__init__(*args,**kwargs)
        from scipy.stats import qmc
        draws=qmc.Sobol(2,scramble=True,seed=20260920).random_base2(5)
        self.price_u=draws[:,0];self.pv_u=draws[:,1];self.pv_levels=pv_levels
        self.pv_quantiles=np.concatenate([pv_quantiles,np.zeros((self.h,4,len(pv_levels)))])
        self.tau=np.asarray([.05,.10,.25,.50,.75,.90,.95])
        self.price_ix=np.searchsorted(self.tau,self.price_u,side='right').clip(1,6)-1
        self.pv_ix=np.searchsorted(pv_levels,self.pv_u,side='right').clip(1,len(pv_levels)-1)-1

    def cost_quantiles(self,q,ids,z,booked,delta,physical):
        ix=self.price_ix
        price=q[:,ix]+(q[:,ix+1]-q[:,ix])*(self.price_u-self.tau[ix])/(self.tau[ix+1]-self.tau[ix])
        quant=self.pv_quantiles[ids,z];ix=self.pv_ix;levels=self.pv_levels
        pv=quant[:,ix]+(quant[:,ix+1]-quant[:,ix])*(self.pv_u-levels[ix])/(levels[ix+1]-levels[ix])
        pue=self.pue[ids,z];increment=pue*self.k[z]*delta
        load0=pue*(self.idle[z]+self.k[z]*booked)
        used=np.minimum(np.maximum(pv-load0[:,None],0.),increment[:,None])
        denominator=self.k[z]*self.pue_reference[z]*delta
        covered=np.clip(used/increment[:,None],0.,1.);ratio=pue[:,None]/self.pue_reference[z]
        a=ratio*(1-covered);b=-ratio*covered
        samples=a*price+b
        result=np.quantile(samples,self.tau,axis=1).T
        # Preserve exact affine/identity cases instead of introducing Sobol
        # quantile approximation error where physics creates no uncertainty.
        constant=np.ptp(used,axis=1)<1e-10
        result[constant]=q[constant]*a[constant,:1]+b[constant,:1]
        result[~physical]=q[~physical]
        if not np.isfinite(result).all() or (np.diff(result,axis=1)<-1e-8).any():raise RuntimeError('Invalid cost quantiles')
        return result

class NowcastReservationRanks(ReservationStateRanks):
    def __init__(self,*args,previous_PV_observation,coefficients,pv_capacity,**kwargs):
        super().__init__(*args,**kwargs)
        self.previous_PV_observation=previous_PV_observation
        self.coefficients=np.asarray(coefficients);self.pv_capacity=np.asarray(pv_capacity)
        self.fast=np.exp(-np.arange(self.h)*.25)
        self.slow=np.exp(-np.arange(self.h)*.25/6)

    def cost_quantiles(self,q,ids,z,booked,delta,physical):
        q=self.point_input(q,ids,z)
        t=self.current_decision_index
        prior=self.pv[ids,z]
        if t>0:
            last_actual=self.previous_PV_observation(t,z)
            innovation=last_actual-self.pv[t-1,z]
        else:innovation=0.
        gain=self.coefficients[z,0]*self.fast+self.coefficients[z,1]*self.slow
        pv=np.clip(prior+gain*innovation,0,self.pv_capacity[z])
        pv=np.where(prior>1e-3,pv,prior)
        denominator=self.k[z]*self.pue_reference[z]*delta
        pue=self.pue[ids,z];load0=pue*(self.idle[z]+self.k[z]*booked)
        increment=pue*self.k[z]*delta
        used=np.minimum(np.maximum(pv-load0,0.),increment)
        covered=np.clip(used/increment,0.,1.);ratio=pue/self.pue_reference[z]
        a=np.where(physical,ratio*(1-covered),1.)
        b=np.where(physical,-ratio*covered,0.)
        return q*a[:,None]+b[:,None]

class ParticleReservationRanks(ReservationStateRanks):
    def __init__(self,*args,source_days,quarter_slots,prior_paths,observed_prefix,sigma,context='completed_day_prefix',**kwargs):
        super().__init__(*args,**kwargs)
        self.source_days=np.r_[source_days,np.full(self.h,'',dtype=source_days.dtype)]
        self.quarter_slots=np.r_[quarter_slots,np.zeros(self.h,dtype=int)]
        self.prior_paths=prior_paths;self.observed_prefix=observed_prefix;self.sigma=np.asarray(sigma);self.context=context
        self.weight_cache={};self.posterior_reads=0

    def posterior(self,day,z):
        key=(day,z,self.current_decision_index)
        if key in self.weight_cache:return self.weight_cache[key]
        pool=self.prior_paths(day)[:,z,:]  # 96 x scenarios, issued before use.
        slots,values=self.observed_prefix(day,self.current_decision_index,z,self.context)
        if len(slots):
            error=pool[slots]-values[:,None]
            logweight=-.5*np.sum(error**2,axis=0)/(self.sigma[z]**2)
            logweight-=logweight.max();weights=np.exp(logweight);weights/=weights.sum()
        else:weights=np.full(pool.shape[1],1/pool.shape[1])
        self.weight_cache={key:weights};self.posterior_reads+=1
        return weights

    def cost_quantiles(self,q,ids,z,booked,delta,physical):
        q=self.point_input(q,ids,z)
        denominator=self.k[z]*self.pue_reference[z]*delta
        pue=self.pue[ids,z];load0=pue*(self.idle[z]+self.k[z]*booked)
        increment=pue*self.k[z]*delta
        expected_coverage=np.zeros(len(ids))
        days=self.source_days[ids]
        for day in np.unique(days[physical]):
            if not day:continue
            positions=np.flatnonzero((days==day)&physical)
            pool=self.prior_paths(str(day))[self.quarter_slots[ids[positions]],z,:]
            weights=self.posterior(str(day),z)
            covered=np.clip((pool-load0[positions,None])/increment[positions,None],0.,1.)
            expectation=np.clip(covered@weights,0.,1.)
            constant=np.ptp(covered,axis=1)==0
            expectation[constant]=covered[constant,0]
            expected_coverage[positions]=expectation
        ratio=pue/self.pue_reference[z]
        a=np.where(physical,ratio*(1-expected_coverage),1.)
        b=np.where(physical,-ratio*expected_coverage,0.)
        return q*a[:,None]+b[:,None]
