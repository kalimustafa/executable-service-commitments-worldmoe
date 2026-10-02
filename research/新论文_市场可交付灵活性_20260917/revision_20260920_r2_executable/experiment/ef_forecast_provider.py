"""Causal forecast-provider boundary for the unchanged reservation algorithm."""
from pathlib import Path
import numpy as np,pandas as pd
ZONES=['WEST','CAPITL','N.Y.C.','LONGIL'];TZ='America/New_York'
TAU=(.05,.10,.25,.5,.75,.9,.95)

class ForecastProvider:
    def __init__(self,times,da,historical_rt,*,kind,frame=None):
        self.times=pd.DatetimeIndex(times);self.n=len(times);self.da=np.asarray(da);self.historical=np.asarray(historical_rt)
        self.kind=kind;self.rank_cache={};self.max_identity_error=0.
        dates=self.times.tz_convert(TZ).tz_localize(None).normalize()
        self.post_issue=(dates-pd.Timedelta(days=1)+pd.Timedelta(hours=11)).tz_localize(TZ).tz_convert('UTC').asi8
        self.pre_issue=(dates-pd.Timedelta(days=1)+pd.Timedelta(hours=4,minutes=55)).tz_localize(TZ).tz_convert('UTC').asi8
        index=pd.MultiIndex.from_product([self.times.floor('h'),ZONES],names=['timestamp_utc','zone'])
        if kind=='r2':
            tensors={}
            for gate in ['pre_DA','post_DA']:
                f=frame[frame.gate.eq(gate)].set_index(['timestamp_utc','zone']).reindex(index)
                q=f[[f'rt_q{x:.2f}' for x in TAU]].to_numpy().reshape(self.n,4,7)
                assert np.isfinite(q).all() and np.min(np.diff(q,axis=-1))>=-1e-5
                tensors[gate]=q
                if gate=='post_DA':
                    dart=f[[f'dart_q{x:.2f}' for x in TAU]].to_numpy().reshape(self.n,4,7)
                    self.max_identity_error=float(np.max(np.abs(q-(self.da[...,None]-dart[...,::-1]))))
            self.pre=tensors['pre_DA'];self.post=tensors['post_DA']
        elif kind=='old_rasc':
            f=frame[frame.model.eq('RASC')].set_index(['timestamp_utc','zone']).reindex(index)
            q=f[[f'q{x:.2f}' for x in TAU]].to_numpy().reshape(self.n,4,7)
            assert np.isfinite(q).all();self.post=self.da[...,None]-q[...,::-1];self.pre=None
        elif kind=='published_da':
            self.post=np.repeat(self.da[...,None],7,axis=-1);self.pre=None
        else:raise ValueError(kind)

    def ranked_slots(self,window,policy):
        key=(window,policy)
        if key in self.rank_cache:return self.rank_cache[key]
        horizon=int(window*4);ids=np.arange(self.n)[:,None]+np.arange(horizon)[None,:]
        extended=np.broadcast_to(self.historical,(horizon,4))
        post=np.concatenate((self.post,np.repeat(extended[...,None],7,axis=-1)),axis=0)
        post_time=np.r_[self.post_issue,np.full(horizon,np.iinfo(np.int64).max,dtype=np.int64)]
        post_available=post_time[ids]<=self.times.asi8[:,None]
        if self.kind=='r2':
            pre=np.concatenate((self.pre,np.repeat(extended[...,None],7,axis=-1)),axis=0)
            pre_time=np.r_[self.pre_issue,np.full(horizon,np.iinfo(np.int64).max,dtype=np.int64)]
            pre_available=pre_time[ids]<=self.times.asi8[:,None]
            available=post_available|pre_available
            def channel(j):
                return np.where(post_available[...,None],post[ids,:,j],np.where(pre_available[...,None],pre[ids,:,j],self.historical))
        else:
            available=post_available
            def channel(j):return np.where(post_available[...,None],post[ids,:,j],self.historical)
        median=channel(3)
        if policy=='point':
            score=median;score[:,1:]+=2.
        elif policy=='robust':
            lower=channel(2);score=channel(4)
            permitted=(lower[:,:1]-score>2.)&available[...,None]
            score[:,1:]=np.where(permitted[:,1:],score[:,1:],np.inf);score[:,0]=median[:,0]
        else:raise ValueError(policy)
        rank=np.argsort(score,axis=1,kind='stable').astype(np.int16);self.rank_cache[key]=rank
        return rank
