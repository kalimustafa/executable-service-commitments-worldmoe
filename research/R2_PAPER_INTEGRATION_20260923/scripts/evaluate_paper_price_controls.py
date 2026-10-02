"""Retain the original paper's causal price-lag and matched base controls."""
from src.evaluation.paper_economics import *
arr,reference,shared=configure(2025);canonical_rank=engine.rank;rows=[]
for window in [7,12,24]:
    name=f'control_RT_lag_2025_{window}h_point';saved=ROOT/'results/paper_runs'/f'{name}.json'
    if saved.exists():rows.append(json.loads(saved.read_text()));continue
    engine.rank=lambda w,p:canonical_rank(w,'persistence')
    began=time.perf_counter();book,stats,events=engine.schedule(arr,[(1.,window)],'point',keep_events=True)
    rows.append(measure(name,book,reference,arr,shared,dict(model='RT_lag',year=2025,trace='Eagle',proxy='registered',seed=0,window_h=window,policy='point'),stats,events,began,saved))
pd.DataFrame(rows).to_csv(ROOT/'results/causal_price_lag_control.csv',index=False)
source=ROOT/'results/predictions_2025/A0_Base_seed20260920_native.parquet'
frame=pd.read_parquet(source);base=ForecastProvider(core.times,core.da,core.train_rt,kind='r2',frame=frame);rows=[]
for window in [12,24]:
    for policy in ['point','interval']:
        name=f'control_Base_price_cost_2025_{window}h_{policy}';saved=ROOT/'results/paper_runs'/f'{name}.json'
        if saved.exists():rows.append(json.loads(saved.read_text()));continue
        meta=dict(model='Base_price_cost',year=2025,trace='Eagle',proxy='registered',seed=20260920,window_h=window,policy=policy)
        spec=dict(name=name,**meta,market_branch='A0_Base',physical_branch='particle_trailing4',price_forecast_source=str(source))
        observer,_,_=build_candidate(spec,arr,physical_kwargs(2025,base));engine.rank=lambda w,p:observer
        began=time.perf_counter();book,stats,events=engine.schedule(arr,[(1.,window)],'robust' if policy=='interval' else policy,keep_events=True)
        assert np.max(np.abs(book-observer.book))<1e-10
        rows.append(measure(name,book,reference,arr,shared,meta,stats,events,began,saved))
pd.DataFrame(rows).to_csv(ROOT/'results/base_price_cost_control.csv',index=False)
print('ORIGINAL PRICE CONTROLS COMPLETE',flush=True)
