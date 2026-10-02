import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.context import *
from src.optimized_allocator import allocate
import argparse
p=argparse.ArgumentParser();p.add_argument('--year',type=int,default=2024);p.add_argument('--seed',type=int,default=20260920);p.add_argument('--quantum',type=float,nargs='+',default=[.25,.1,.05]);p.add_argument('--risk',type=float,nargs='+',default=[1.]);a=p.parse_args()
O=ROOT/'optimization80';O.mkdir(exist_ok=True)
for risk in a.risk:
 for quantum in a.quantum:
    tag=f'O80_Eagle_{a.year}_s{a.seed}_q{quantum:g}_r{risk:g}'
    saved=RESULTS/f'{tag}.json'
    if saved.exists():print('CACHED',tag,flush=True);continue
    arr,reference,shared=configure(a.year);model=observer(a.year,a.seed,arr,24);start=time.perf_counter()
    book,late,extra=allocate(arr,model,24,quantum,risk)
    extra['seconds']=time.perf_counter()-start
    row=measure(tag,book,reference,arr,shared,dict(experiment='O80',year=a.year,trace='Eagle',seed=a.seed,window_h=24,role='development' if a.year==2024 else 'exploratory_frozen_evaluation'),late=late,extra=extra)
    bound=json.loads((RESULTS/f'I1_Eagle_{a.year}_w24.json').read_text())
    row['capture_lower_fraction']=row['saving_usd']/bound['maximum_saving_upper_usd'];row['target_80_usd']=.8*bound['maximum_saving_upper_usd']
    saved.write_text(json.dumps(row,indent=2));print('CAPTURE',row['capture_lower_fraction'],'CLOSING',row['closing_it_mwh'],flush=True)
