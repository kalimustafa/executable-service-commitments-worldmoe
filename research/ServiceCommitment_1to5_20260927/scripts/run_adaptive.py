import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.context import *
from src.controllers import adaptive

protocol=json.loads((ROOT/'configs/protocol.json').read_text());rows=[]
def run(year,trace,seed,window,lock,pressure,role,event=False):
    label='event_replan' if event else f'adaptive_lock{lock}'
    name=f'I4_{trace}_{year}_s{seed}_w{window}_{label}';saved=RESULTS/f'{name}.json'
    if saved.exists():return json.loads(saved.read_text())
    arr,ref,shared=configure(year,trace);start=time.perf_counter();model=observer(year,seed,arr,window)
    book,late,extra=adaptive(arr,model,lock,pressure,window,event_myopic=event)
    extra.update(seconds=time.perf_counter()-start,role=role)
    return measure(name,book,ref,arr,shared,dict(experiment='I4',year=year,trace=trace,seed=seed,window_h=window,policy=label),late=late,extra=extra)

def main():
    development=[]
    for candidate in protocol['adaptive_candidates']:
        row=run(2024,'Eagle',20260920,24,candidate['lock_h'],candidate['pressure'],'development')
        row['selection_score']=row['saving_usd']-.1*366*row['cvar95_usd'];development.append(row);rows.append(row)
    winner=sorted(development,key=lambda q:(q['late_IT_mwh']>1e-8,-q['selection_score'],q['lock_h']))[0]
    selection=dict(selected_lock_h=winner['lock_h'],pressure=winner['pressure_threshold'],selection_data='2024 Eagle full calendar primary seed',criterion='zero lateness first, then savings minus 0.1 times days times CVaR95',candidates=[dict(lock_h=r['lock_h'],saving=r['saving_usd'],late=r['late_IT_mwh'],score=r['selection_score']) for r in development])
    (RESULTS/'adaptive_selection.json').write_text(json.dumps(selection,indent=2));print('SELECTED',selection,flush=True)
    cases=[(2025,'Eagle',s,w) for s in protocol['seeds'] for w in [12,16,24]]+[(2024,'Eagle',20260920,24),(2024,'Frontier',20260920,24),(2025,'Frontier',20260920,12),(2025,'Frontier',20260920,24)]
    for year,trace,seed,window in cases:
        rows.append(run(year,trace,seed,window,winner['lock_h'],winner['pressure_threshold'],'evaluation' if year==2025 else 'development_transfer'))
        rows.append(run(year,trace,seed,window,0,1.,'matched_event_replan',event=True))
    pd.DataFrame(rows).drop_duplicates('name').to_csv(RESULTS/'adaptive_results.csv',index=False)

if __name__=='__main__':main()
