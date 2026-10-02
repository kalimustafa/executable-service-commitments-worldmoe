import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.context import *
from src.controllers import project_grid
protocol=json.loads((ROOT/'configs/protocol.json').read_text());rows=[]
cases=[(2025,'Eagle',s,24) for s in protocol['seeds']]+[(2025,'Eagle',20260920,12),(2024,'Eagle',20260920,24),(2025,'Frontier',20260920,24)]
for year,trace,seed,window in cases:
    arr,reference,shared=configure(year,trace);book=cached_book(year,trace,seed,window)
    if book is None:book=persistent(arr,year,seed,window)
    for scenario in protocol['grid_constraints']:
        name=f'I3G_{trace}_{year}_s{seed}_w{window}_{scenario["name"]}';saved=RESULTS/f'{name}.json'
        if saved.exists():rows.append(json.loads(saved.read_text()));continue
        kwargs={k:scenario[k] for k in ['grid_fraction','ramp_fraction']}
        ref,_=project_grid(reference,arr,**kwargs)
        protected,extra=project_grid(book,arr,deadline_guard=True,window=window,**kwargs)
        extra['reference_late_it_mwh']=fifo_lateness(ref,arr,window)
        rows.append(measure(name,protected,ref,arr,shared,dict(experiment='I3G',year=year,trace=trace,seed=seed,window_h=window,scenario=scenario['name'],**kwargs),extra=extra))
pd.DataFrame(rows).to_csv(RESULTS/'grid_deadline_guard.csv',index=False)
