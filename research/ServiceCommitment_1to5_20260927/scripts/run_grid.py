import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.context import *
from src.controllers import project_grid

protocol=json.loads((ROOT/'configs/protocol.json').read_text());rows=[]
cases=[(2025,'Eagle',s,w) for s in protocol['seeds'] for w in [12,16,24]]+[(2024,'Eagle',20260920,24),(2024,'Frontier',20260920,24),(2025,'Frontier',20260920,24)]
for year,trace,seed,window in cases:
    arr,reference,shared=configure(year,trace)
    book=cached_book(year,trace,seed,window)
    if book is None:book=persistent(arr,year,seed,window)
    baseline=measure(f'I3_{trace}_{year}_s{seed}_w{window}_unconstrained',book,reference,arr,shared,dict(experiment='I3',year=year,trace=trace,seed=seed,window_h=window,scenario='unconstrained'))
    rows.append(baseline)
    for scenario in protocol['grid_constraints']:
        name=f'I3_{trace}_{year}_s{seed}_w{window}_{scenario["name"]}';saved=RESULTS/f'{name}.json'
        if saved.exists():rows.append(json.loads(saved.read_text()));continue
        start=time.perf_counter();kwargs={k:scenario[k] for k in ['grid_fraction','ramp_fraction']}
        ref,refextra=project_grid(reference,arr,**kwargs);policy,extra=project_grid(book,arr,**kwargs)
        extra.update(reference_late_it_mwh=fifo_lateness(ref,arr,window),delta_cost_vs_unconstrained_usd=0.,seconds=time.perf_counter()-start)
        row=measure(name,policy,ref,arr,shared,dict(experiment='I3',year=year,trace=trace,seed=seed,window_h=window,scenario=scenario['name'],**kwargs),extra=extra)
        row['delta_cost_vs_unconstrained_usd']=row['adjusted_cost_usd']-baseline['adjusted_cost_usd']
        saved.write_text(json.dumps(row,indent=2));rows.append(row)
pd.DataFrame(rows).to_csv(RESULTS/'grid_constraints.csv',index=False)
