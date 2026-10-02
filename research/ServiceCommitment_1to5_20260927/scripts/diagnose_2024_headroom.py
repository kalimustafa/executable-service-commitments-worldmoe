import sys,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.context import *
from src.optimized_allocator import allocate
for label in ['price','physical','both']:
    name='ORACLE_DIAGNOSTIC_2024_'+label;saved=RESULTS/f'{name}.json'
    if saved.exists():continue
    arr,ref,shared=configure(2024);model=observer(2024,20260920,arr,24)
    model.oracle_price_diagnostic=label in ['price','both']
    if label in ['physical','both']:
        def perfect_physical(self,q,ids,z,booked,delta,physical):
            inside=ids<c.N;pue=np.full(len(ids),max(c.c['pue_max']));pv=np.zeros(len(ids));pue[inside]=c.pue[ids[inside],z];pv[inside]=c.pv[ids[inside],z]
            increment=pue*c.k[z]*delta;load0=pue*(c.idle[z]+c.k[z]*booked)
            coverage=np.clip((pv-load0)/increment,0,1);ratio=pue/self.pue_reference[z]
            return q*(ratio*(1-coverage))[:,None]-(ratio*coverage)[:,None]
        model.cost_quantiles=types.MethodType(perfect_physical,model)
    book,late,extra=allocate(arr,model,24,.05,0.,close_year=True)
    row=measure(name,book,ref,arr,shared,dict(experiment='NONDEPLOYABLE_DIAGNOSTIC',year=2024,trace='Eagle',window_h=24,label=label,not_a_candidate=True),late=late,extra=extra)
    print('DIAGNOSTIC ONLY',label,row['saving_usd'],flush=True)
