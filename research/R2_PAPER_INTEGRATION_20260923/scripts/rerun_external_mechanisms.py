"""Repeat paper external controls; enrich capacity and contract sampling."""
from src.evaluation.paper_economics import *
from vendor import paper_controls as control
import argparse

def floor(arr,capacity):
    queue=np.zeros(4);steps=0;peak=0;step=0
    for t,a in enumerate(arr):
        ready=queue+a
        if ready.max()>peak:peak=float(ready.max());step=t
        steps=max(steps,int(np.ceil(ready.max()/capacity-1e-10)))
        queue=np.maximum(ready-capacity,0.)
    return dict(minimum_feasible_h=steps*core.DT,peak_prefix=peak,peak_step=step)

def run_case(year,trace,proxy):
    arr,reference,shared=configure(year,trace,proxy)
    base=ForecastProvider(core.times,core.da,core.train_rt,kind='published_da')
    engine.rank=lambda w,p:base.ranked_slots(w,p)
    protocol=json.loads((ROOT/'configs/paper_support_protocol.json').read_text())
    floors=[dict(year=year,trace=trace,proxy=proxy,capacity_fraction=f,**floor(arr,f)) for f in protocol['capacity_floor_fractions']]
    pd.DataFrame(floors).to_csv(ROOT/'results'/f'physical_floors_{trace}_{year}_{proxy}.csv',index=False)
    rows=[];windows=[12] if proxy=='field7_sensitivity' else protocol['window_sweep_h']
    for window in windows:
        ranks=base.ranked_slots(window,'point')
        for policy in ['myopic_da','persistent_da']:
            name=f'external_{trace}_{year}_{proxy}_{window}h_{policy}'
            saved=ROOT/'results/paper_runs'/f'{name}.json'
            if saved.exists():rows.append(json.loads(saved.read_text()));continue
            began=time.perf_counter()
            if policy=='persistent_da':book,stats,events=engine.schedule(arr,[(1.,window)],'point',keep_events=True)
            else:
                book,stats=control.myopic_execute(arr,window,ranks)
                # The myopic control retains no per-cohort events. Reconstruct
                # its homogeneous FIFO service matching for all-cohort deadlines.
                events=[]
                for z in range(4):
                    cumulative=np.cumsum(book[:,z]);released=0.
                    for t,a in enumerate(arr[:,z]):
                        if a<=1e-12:continue
                        begin=np.searchsorted(cumulative,released+1e-10)
                        end=np.searchsorted(cumulative,released+a-1e-10)
                        for slot in range(begin,end+1):
                            left=max(released,cumulative[slot-1] if slot else 0.)
                            units=max(0.,min(released+a,cumulative[slot])-left)
                            if units>1e-12:events.append((t,t+window*4-1,slot,z,units,window))
                        released+=a
            meta=dict(year=year,trace=trace,proxy=proxy,window_h=window,policy=policy,seed=0,model='Published_DA')
            rows.append(measure(name,book,reference,arr,shared,meta,stats,events,began,saved))
        base.rank_cache.clear()
    # Original paper includes the unhedged diagnostic and resource-field test.
    name=f'external_{trace}_{year}_{proxy}_12h_unhedged_da';saved=ROOT/'results/paper_runs'/f'{name}.json'
    if not saved.exists():
        book,stats=control.myopic_execute(arr,12,base.ranked_slots(12,'point'),hedge=False)
        costs,_=core.cash(book);refcost,_=core.cash(reference)
        row=core.record(name,'external',book,reference,dict(year=year,trace=trace,proxy=proxy,window_h=12,policy='unhedged_da'),late=stats['late_IT_mwh'])
        row.update(stats);saved.write_text(json.dumps(row,indent=2));core.TRAJ.pop(name,None);core.DAILY=[]
    mixed=[]
    for shares in ([ [.25,.5,.25] ] if proxy=='field7_sensitivity' else protocol['mixed_shares']):
        control.CLASSES=[(fraction,window) for fraction,window in zip(shares,[4,12,24])]
        ref,ref_late=control.edf_reference(arr)
        for recourse in [False,True]:
            code='_'.join(str(round(x*100)) for x in shares)
            name=f'external_{trace}_{year}_{proxy}_mix{code}_recourse{int(recourse)}'
            saved=ROOT/'results/paper_runs'/f'{name}.json'
            if saved.exists():mixed.append(json.loads(saved.read_text()));continue
            began=time.perf_counter()
            if recourse:
                book,stats,ev=control.run(arr,'point');events=list(ev.itertuples(index=False,name=None))
            else:book,stats,events=engine.schedule(arr,control.CLASSES,'point',keep_events=True)
            meta=dict(year=year,trace=trace,proxy=proxy,window_h='mixed',policy='Published_DA',shares=shares,recourse=recourse,reference_late=ref_late)
            mixed.append(measure(name,book,ref,arr,shared,meta,stats,events,began,saved))
    pd.DataFrame(rows).to_csv(ROOT/'results'/f'external_mechanisms_{trace}_{year}_{proxy}.csv',index=False)
    pd.DataFrame(mixed).to_csv(ROOT/'results'/f'external_recourse_{trace}_{year}_{proxy}.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--year',type=int,required=True);args=p.parse_args()
    for trace,proxy in [('Eagle','registered'),('Frontier','requested_nodes'),('Frontier','field7_sensitivity')]:run_case(args.year,trace,proxy)
    print('EXTERNAL MECHANISM REPLICATION COMPLETE',args.year,flush=True)
