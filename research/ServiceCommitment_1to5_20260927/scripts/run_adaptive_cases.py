from run_adaptive import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--year',type=int,default=2025);p.add_argument('--trace',default='Eagle');p.add_argument('--seed',type=int,default=20260922);p.add_argument('--windows',type=int,nargs='+',default=[12,16,24]);a=p.parse_args()
s=json.loads((RESULTS/'adaptive_selection.json').read_text())
for w in a.windows:
    run(a.year,a.trace,a.seed,w,s['selected_lock_h'],s['pressure'],'evaluation' if a.year==2025 else 'development_transfer')
    run(a.year,a.trace,a.seed,w,0,1.,'matched_event_replan',event=True)
