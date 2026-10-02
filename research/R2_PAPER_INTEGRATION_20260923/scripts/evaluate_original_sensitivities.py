"""Paper's original workload/capacity/overhead scenarios, now with frozen R2."""
from src.evaluation.paper_economics import *
import inspect

original_record=core.record
arr0,_,shared=configure(2025)
frame=pd.read_parquet(forecast_file(2025));base=ForecastProvider(core.times,core.da,core.train_rt,kind='r2',frame=frame)
da_provider=ForecastProvider(core.times,core.da,core.train_rt,kind='published_da')
cases=[('workload_075',.75,1.),('workload_125',1.25,1.),('capacity_075',1.,.75),('capacity_125',1.,1.25),('overhead_002',1.,.98),('overhead_005',1.,.95)]
rows=[]
for case,work_scale,capacity in cases:
    arr=arr0*work_scale;ref=engine.reference(arr,capacity)
    # Identical accounting; the declared capacity scenario gets its own bound.
    # Interconnection excess remains an explicit sensitivity outcome.
    code=inspect.getsource(original_record).replace('book.max()<=1+1e-8','book.max()<=capacity+1e-8')
    code=code.replace("assert np.all(grid<=np.array(c['grid_import_limit'])+1e-8)","row['grid_limit_exceeded']=bool(np.any(grid>np.array(c['grid_import_limit'])+1e-8))")
    for window in [12,24]:
        for model,policy in [('Published_DA','point'),('R2','point'),('R2','interval')]:
            name=f'sensitivity_{case}_{model}_{window}h_{policy}';saved=ROOT/'results/paper_runs'/f'{name}.json'
            if saved.exists():rows.append(json.loads(saved.read_text()));continue
            began=time.perf_counter();observer=None
            if model=='Published_DA':engine.rank=lambda w,p:da_provider.ranked_slots(w,p)
            else:
                spec=dict(name=name,model='R2_StateCost',market_branch='C1_ProfileResidual',seed=20260920,year=2025,
                    physical_branch='particle_trailing4',window_h=window,policy=policy,price_forecast_source=str(forecast_file(2025)))
                observer,_,_=build_candidate(spec,arr,physical_kwargs(2025,base,capacity))
                engine.rank=lambda w,p:observer
            book,stats,events=engine.schedule(arr,[(1.,window)],'robust' if policy=='interval' else 'point',capacity=capacity,keep_events=True)
            if observer is not None:assert np.max(np.abs(book-observer.book))<1e-10
            namespace=dict(core.__dict__,capacity=capacity);exec(compile(code,'<declared_capacity_sensitivity_account>','exec'),namespace)
            core.record=namespace['record']
            meta=dict(year=2025,trace='Eagle',proxy='sensitivity',seed=20260920,window_h=window,policy=policy,model=model,
                scenario=case,workload_scale=work_scale,available_capacity=capacity,primary_boundary_changed=False,
                scenario_reference='same scaled demand and available capacity')
            rows.append(measure(name,book,ref,arr,shared,meta,stats,events,began,saved))
            core.record=original_record
            pd.DataFrame(rows).to_csv(ROOT/'results/original_parameter_sensitivities.csv',index=False)
print('ORIGINAL PARAMETER SENSITIVITIES COMPLETE',flush=True)
