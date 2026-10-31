"""Advance the three 48 h runs concurrently; the shared runner lock avoids duplicates."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os,subprocess,sys,json,time
ROOT=Path(__file__).resolve().parents[1]
def run(pair):
    seed,ep=pair;env=os.environ.copy();env.update(PYTHONUTF8='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    args=[sys.executable,str(ROOT/'code/run_window_extension.py'),'--year','2025','--window','48','--seed',str(seed),'--cdf-epoch',str(ep),'--eta','0','--safety','0.9','--current-coupling','--continuous-cost']
    with (ROOT/'logs'/f's{seed}_w48_external.log').open('w',encoding='utf-8') as f:
        p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
        (ROOT/'logs'/f's{seed}_w48_external_pid.json').write_text(json.dumps({'pid':p.pid,'started_epoch':time.time(),'command':args},indent=2),encoding='utf-8')
        code=p.wait()
    return {'seed':seed,'window_h':48,'exit_code':code,'log':f'logs/s{seed}_w48_external.log'}
with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run,[(20260920,6),(20260921,2),(20260922,0)]))
(ROOT/'extra48_status.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
if any(r['exit_code'] for r in rows):raise SystemExit(1)
