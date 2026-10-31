"""Run three single-threaded seed chains in parallel; retain all process logs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/'suite_status.json'
SEEDS=[(20260920,6),(20260921,2),(20260922,0)]
WINDOWS=[24,8,20,36,48]
state={'run_date':'2026-10-31','protocol':'fixed 96-slot neural forecast with explicitly declared deterministic continuation beyond slot 95','started_epoch':time.time(),'python_executable':sys.executable,'runs':{}}
import threading
lock=threading.Lock()
def update(key,record):
    with lock:
        state['runs'][key]=record;state['updated_epoch']=time.time()
        temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2),encoding='utf-8');os.replace(temp,STATUS)
def chain(pair):
    seed,ep=pair
    for window in WINDOWS:
        key=f's{seed}_w{window}'
        command=[sys.executable,str(ROOT/'code/run_window_extension.py'),'--year','2025','--window',str(window),'--seed',str(seed),'--cdf-epoch',str(ep),'--eta','0','--safety','0.9','--current-coupling','--continuous-cost']
        env=os.environ.copy();env.update(PYTHONUTF8='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        started=time.time();log=ROOT/'logs'/f'{key}.log'
        with log.open('w',encoding='utf-8') as f:
            proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,env=env)
            update(key,{'status':'running','pid':proc.pid,'command':command,'log':str(log.relative_to(ROOT)),'started_epoch':started})
            code=proc.wait()
        update(key,{'status':'complete' if code==0 else 'failed','exit_code':code,'seconds':time.time()-started,'log':str(log.relative_to(ROOT)),'started_epoch':started,'completed_epoch':time.time()})
        if code:raise RuntimeError(f'{key} failed, inspect {log}')
    return seed
with ThreadPoolExecutor(max_workers=3) as pool:
    futures=[pool.submit(chain,s) for s in SEEDS]
    errors=[]
    for f in futures:
        try:print('SEED CHAIN COMPLETE',f.result(),flush=True)
        except Exception as e:errors.append(str(e));traceback.print_exc()
with lock:
    state['status']='failed' if errors else 'complete';state['errors']=errors;state['completed_epoch']=time.time();STATUS.write_text(json.dumps(state,indent=2),encoding='utf-8')
if errors:raise SystemExit(1)
