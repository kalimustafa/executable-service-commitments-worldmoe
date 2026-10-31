"""Derive a separate frozen-checkpoint window runner without changing the archive."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parents[1]
WORKSPACE=HERE.parent
source=WORKSPACE/'ExecutableServiceCommitments_Revision_20260928/engine/scripts/run_service_window.py'
s=source.read_text(encoding='utf-8')
replacements=[
("import sys,argparse,json,time,hashlib\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parents[1]))\nfrom src.runtime import ROOT,WORKSPACE,PREVIOUS,EXTENSION,ARCHIVE",
 "import sys,argparse,json,time,hashlib,os,platform\nfrom pathlib import Path\nRUN_ROOT=Path(__file__).resolve().parents[1]\nENGINE_ROOT=Path(os.environ.get('WORLDMOE_ENGINE_ROOT',str(RUN_ROOT.parent/'ExecutableServiceCommitments_Revision_20260928/engine')))\nsys.dont_write_bytecode=True\nos.environ['NUMBA_CACHE_DIR']=str(RUN_ROOT/'runtime_cache/numba')\nos.environ['OMP_NUM_THREADS']='1'\nos.environ['MKL_NUM_THREADS']='1'\nsys.path.insert(0,str(ENGINE_ROOT))\nfrom src.runtime import ROOT,WORKSPACE,PREVIOUS,EXTENSION,ARCHIVE"),
("choices=[12,16,24],required=True","choices=[8,12,16,20,24,36,48],required=True"),
("name+=f'_w{args.window}'\nsaved=ROOT/'results'/f'{name}.json'","name=f'WindowExtension20261031_{name}_w{args.window}'\nsaved=RUN_ROOT/'results'/f'{name}.json'"),
("solar=prepare(times,c.pv);fit_observer()","assert (ROOT/'cache/causal_observer.npz').exists(), 'Frozen observer missing; no fitting permitted'\nsolar=prepare(times,c.pv)"),
("probabilities=np.array([.075,.1,.2,.25,.2,.1,.075]);queue=np.zeros((96,4));ref_queue=np.zeros(4);book=np.zeros((n+8192,4))",
 "plan_horizon=max(96,args.window*4)\nprobabilities=np.array([.075,.1,.2,.25,.2,.1,.075]);queue=np.zeros((plan_horizon,4));ref_queue=np.zeros(4);book=np.zeros((n+8192,4))\n# All obligations remain in this queue. Neural forecasts stay at 96 slots.\n# Tail priors depend only on frozen pre-2024 quantities, never future realisations.\ntail_rt=np.array(c.train_rt,dtype=float)\ntail_pue=np.array(c.c['pue_max'],dtype=float)\nrun_manifest=dict(run_date='2026-10-31',window_h=args.window,seed=seed,forecast_horizon_slots=96,planning_horizon_slots=plan_horizon,\n    tail_rule='after slot 95: frozen pre-2024 zonal median RT, zero PV, declared maximum PUE, zero learned opportunity, unchanged service capacity',\n    queue_rule='all obligations retained; year-end early-completion target only; original exclusive deadline retained for independent audit; complete queue enters prefix guard and continuation',\n    neural_state_rule='original 96x22 coordinates and 4/8/12/24 h deadline bins; total backlog includes distant obligations',\n    terminal_rule='same original year-end deadline clipping and final-hour earliest-service guard',\n    checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),\n    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),\n    forcing_sha256=hashlib.sha256((WORKSPACE/'external_validation/data/forcing_2025.npz').read_bytes()).hexdigest(),\n    numpy_version=np.__version__,torch_version=torch.__version__,python_version=platform.python_version(),\n    new_training_performed=False,new_fitting_performed=False,original_archive_mutated=False,\n    tail_rt_usd_mwh=tail_rt.tolist(),tail_pue=tail_pue.tolist(),status='running')\n(RUN_ROOT/'results'/f'{name}_run_manifest.json').write_text(json.dumps(run_manifest,indent=2),encoding='utf-8')\nprint('WINDOW EXTENSION START',json.dumps(run_manifest),flush=True)"),
("            h=min(96,n-t)","            h=min(plan_horizon,n-t)"),
("                values=(queue[:h,z],pv_scenarios[a,:,:h].copy(),price_scenarios[a,:,:h].copy(),probabilities,puehat[:h,z].copy(),reference_known,float(c.idle[z]),float(c.k[z]),opportunity[a,:h].copy(),float(pue_ref[z]),float(capacity[z]))",
 "                if h<=96:\n                    plan_pv=pv_scenarios[a,:,:h].copy();plan_prices=price_scenarios[a,:,:h].copy();plan_pue=puehat[:h,z].copy();plan_opportunity=opportunity[a,:h].copy()\n                else:\n                    extra=h-96\n                    plan_pv=np.concatenate([pv_scenarios[a,:,:96],np.zeros((7,extra))],axis=1)\n                    plan_prices=np.concatenate([price_scenarios[a,:,:96],np.full((7,extra),tail_rt[z])],axis=1)\n                    plan_pue=np.r_[puehat[:96,z],np.full(extra,tail_pue[z])]\n                    plan_opportunity=np.r_[opportunity[a,:96],np.zeros(extra)]\n                assert queue[h:,z].sum()==0., 'Future obligations omitted from continuation'\n                values=(queue[:h,z],plan_pv,plan_prices,probabilities,plan_pue,reference_known,float(c.idle[z]),float(c.k[z]),plan_opportunity,float(pue_ref[z]),float(capacity[z]))"),
("ctx.RESULTS=ROOT/'results';ctx.TRAJECTORIES=ROOT/'results/trajectories';ctx.TRAJECTORIES.mkdir(exist_ok=True);c.DATA=ctx.TRAJECTORIES",
 "ctx.RESULTS=RUN_ROOT/'results';ctx.TRAJECTORIES=RUN_ROOT/'results/trajectories';ctx.TRAJECTORIES.mkdir(exist_ok=True);c.DATA=ctx.TRAJECTORIES"),
]
for old,new in replacements:
    assert s.count(old)==1,(old[:90],s.count(old))
    s=s.replace(old,new)
old="bound=json.loads((EXTENSION/'results'/f'I1_Eagle_{args.year}_w{args.window}.json').read_text());row['capture_lower_fraction']=row['saving_usd']/bound['maximum_saving_upper_usd'];row['target_80_usd']=.8*bound['maximum_saving_upper_usd'];row['certified_savings_ceiling_usd']=bound['maximum_saving_upper_usd'];saved.write_text(json.dumps(row,indent=2));print('SCENARIO COMPLETE',json.dumps({k:row[k] for k in ['saving_usd','capture_lower_fraction','late_IT_mwh','closing_it_mwh']}),flush=True)"
assert old in s
s=s.replace(old,"print('SCENARIO COMPLETE',json.dumps({k:row[k] for k in ['saving_usd','late_IT_mwh','closing_it_mwh']}),flush=True)")
s+='''\nrow.update(run_date='2026-10-31',evidence_type='new frozen-checkpoint evaluation',forecast_horizon_slots=96,planning_horizon_slots=plan_horizon,tail_rule=run_manifest['tail_rule'],neural_state_rule=run_manifest['neural_state_rule'],new_training_performed=False,new_fitting_performed=False,original_source_checkpoint_hash=run_manifest['checkpoint_sha256'])
saved.write_text(json.dumps(row,indent=2),encoding='utf-8')
run_manifest.update(status='complete',seconds=row['seconds'],saving_usd=row['saving_usd'],late_IT_mwh=row['late_IT_mwh'],closing_it_mwh=row['closing_it_mwh'])
(RUN_ROOT/'results'/f'{name}_run_manifest.json').write_text(json.dumps(run_manifest,indent=2),encoding='utf-8')
'''
s=s.replace("RUN_ROOT=Path(__file__).resolve().parents[1]", "RUN_ROOT=Path(__file__).resolve().parents[1]\nRUNNER_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()",1)
s=s.replace("runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()", "runner_sha256=RUNNER_SOURCE_SHA256",1)
old_guard="if saved.exists():print(saved.read_text());raise SystemExit"
new_guard="""lock_file=RUN_ROOT/'results'/f'{name}.lock'
while True:
    if saved.exists():
        completed=json.loads(saved.read_text(encoding='utf-8'))
        if completed.get('evidence_type')=='new frozen-checkpoint evaluation':
            print('REUSING COMPLETE NEW RUN',saved,flush=True);print(saved.read_text());raise SystemExit
    try:
        lock_handle=os.open(str(lock_file),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        os.write(lock_handle,str(os.getpid()).encode('ascii'));os.close(lock_handle)
        break
    except FileExistsError:
        print('WAITING FOR EXISTING RUN',name,flush=True)
        time.sleep(15)
"""
assert s.count(old_guard)==1;s=s.replace(old_guard,new_guard)
s+="\nlock_file.unlink(missing_ok=True)\n"
out=HERE/'code/run_window_extension.py';out.write_text(s,encoding='utf-8')
(HERE/'code/runner_derivation.json').write_text(json.dumps({'original_source':str(source),'original_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'extension_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'source_preserved':True,'target_windows_h':[8,12,16,20,24,36,48]},indent=2),encoding='utf-8')
print(out)
