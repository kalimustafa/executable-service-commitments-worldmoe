from src.evaluation.paper_economics import *
from src.runtime import PAPER
arr,reference,shared=configure(2025)
protocol=json.loads((ROOT/'configs/paper_support_protocol.json').read_text())
rows=[]
for model in ['Published_DA','RASC','Legacy_R2']:
    if model=='Published_DA':provider=ForecastProvider(core.times,core.da,core.train_rt,kind='published_da')
    elif model=='RASC':
        f=pd.read_parquet(PAPER/'revision_20260918/forecast_2025/bridge_forecast_only_hourly.parquet')
        provider=ForecastProvider(core.times,core.da,core.train_rt,kind='old_rasc',frame=f)
    else:
        f=pd.read_parquet(LEGACY/'forecast_2025/R2_full_native.parquet')
        provider=ForecastProvider(core.times,core.da,core.train_rt,kind='r2',frame=f)
    engine.rank=lambda w,p:provider.ranked_slots(w,p)
    for window in protocol['window_sweep_h']:
        for policy in (['point'] if model=='Published_DA' else ['point','interval']):
            name=f'baseline_{model}_2025_{window}h_{policy}';saved=ROOT/'results/paper_runs'/f'{name}.json'
            if saved.exists():rows.append(json.loads(saved.read_text()));continue
            began=time.perf_counter()
            book,stats,events=engine.schedule(arr,[(1.,window)],'robust' if policy=='interval' else policy,keep_events=True)
            meta=dict(year=2025,trace='Eagle',proxy='registered',seed=20260920 if model=='Legacy_R2' else 0,window_h=window,policy=policy,model=model)
            rows.append(measure(name,book,reference,arr,shared,meta,stats,events,began,saved))
        provider.rank_cache.clear()
    pd.DataFrame(rows).to_csv(ROOT/'results/final_baseline_window_sweep.csv',index=False)
print('FINAL BASELINES COMPLETE',flush=True)
