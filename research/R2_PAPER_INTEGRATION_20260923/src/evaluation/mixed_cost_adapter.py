"""Apply the frozen cost function to the real mixed-contract calendar.

Each released cohort's rank is computed once with its actual admitted amount
and currently known reservations. Priority recourse reuses that original rank,
as in the unchanged paper executor. Only the forecast-input binding differs
from the homogeneous observer interface; no allocation rule is changed.
"""
import ast,inspect
import numpy as np
from src.candidate_runtime import build_candidate
from src.evaluation.paper_economics import physical_kwargs,forecast_file
from vendor import reservation as engine
from vendor import paper_controls as control
from vendor import economic_core as core
from vendor.forecast_provider import ForecastProvider
import pandas as pd

class MixedCostProvider:
    def __init__(self,arr,year,seed,policy,windows=(4,12,24)):
        source=forecast_file(year,seed);frame=pd.read_parquet(source)
        base=ForecastProvider(core.times,core.da,core.train_rt,kind='r2',frame=frame)
        kwargs=physical_kwargs(year,base);self.models={};self.cache={}
        for window in windows:
            row=dict(name='mixed_particle_trailing4',model='R2_StateCost',market_branch='C1_ProfileResidual',seed=seed,year=year,
                physical_branch='particle_trailing4',window_h=window,policy=policy,price_forecast_source=str(source))
            self.models[window]=build_candidate(row,arr,kwargs)[0]

    def request(self,t,z,window,amount,book,policy):
        key=(int(t),int(z),int(window))
        if key in self.cache:return self.cache[key]
        model=self.models[window];model.current_decision_index=int(t)
        ids=t+np.arange(model.h);now=model.base.times.asi8[t]
        post=model.post_issue[ids]<=now;pre=model.pre_issue[ids]<=now
        q=np.where(post[:,None],model.post[ids,z],np.where(pre[:,None],model.pre[ids,z],model.base.historical[z]))
        physical=np.where(post,model.physical_post[ids],np.where(pre,model.physical_pre[ids],False))
        booked=book[ids,z];free=np.maximum(model.capacity-booked,0.)
        delta=np.where(free>1e-12,np.minimum(amount,free),1e-6)
        cost=model.cost_quantiles(q,ids,z,booked,delta,physical).astype(np.float32)
        assert np.isfinite(cost).all() and (np.diff(cost,axis=1)>=0).all()
        if policy=='point':score=cost[:,3].copy();score[1:]+=2.
        else:
            score=cost[:,4].copy();permitted=(cost[0,2]-score>2.)&(post|pre)
            score[1:]=np.where(permitted[1:],score[1:],np.inf);score[0]=cost[0,3]
        rank=np.argsort(score,kind='stable').astype(np.int16);self.cache[key]=rank
        return rank

def bound_executors(provider,classes):
    """Retain source allocation bodies, replacing only the rank lookup."""
    namespace=dict(np=np,c=core,rank=lambda w,p:None,CLASSES=classes,
        pd=pd,defaultdict=control.defaultdict,d=type('RankInterface',(),{'rank':staticmethod(lambda w,p:None)}),cost_provider=provider)
    fixed=inspect.getsource(engine.schedule)
    old='ranks[w][t,:,z]';new='cost_provider.request(t,z,w,amount,book,policy)'
    assert fixed.count(old)==1;fixed=fixed.replace(old,new).replace('def schedule(','def schedule_with_cost(',1)
    repaired=inspect.getsource(control.run)
    old='rank[w][release,:,z]';new='cost_provider.request(release,z,w,amount,book,policy)'
    assert repaired.count(old)==1;repaired=repaired.replace(old,new).replace('def run(','def recourse_with_cost(',1)
    exec(compile(ast.parse(fixed+'\n'+repaired),'<frozen_executor_cost_input_binding>','exec'),namespace)
    return namespace['schedule_with_cost'],namespace['recourse_with_cost']
