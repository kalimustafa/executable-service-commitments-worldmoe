"""Prepare or execute the approved 27-stage study without test-set selection."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import yaml
from multiseed_common import (REPO,ROOT,SEEDS,VARIANTS,W0_BUDGETS,config_path,jobs,now,
    publish,read_json,r_root,r_run,world_phase,world_run,write_json)

def prepare():
    existing=read_json(ROOT/'study_status.json')
    can_amend=not existing or (not existing.get('implementation_verified') and
                              all(j['status']=='pending' for j in existing['jobs']))
    data_root=ROOT/'world_moe/data'
    data_root.mkdir(parents=True,exist_ok=True)
    data_lineage={}
    for name in ('train_only_typed_normalization_v1.json','data_lineage_v1.json'):
        source=REPO/'artifacts/vpp_world_moe_r5_v2_1_2025final/data'/name
        target=data_root/name
        if not target.exists():shutil.copyfile(source,target)
        with source.open('rb') as f:source_hash=hashlib.file_digest(f,'sha256').hexdigest()
        with target.open('rb') as f:target_hash=hashlib.file_digest(f,'sha256').hexdigest()
        if source_hash!=target_hash:raise RuntimeError('Frozen training data metadata changed')
        data_lineage[str(target)]={'source':str(source),'sha256':source_hash,'fit_years':[2022,2023]}
    write_json(ROOT/'training_data_lineage.json',data_lineage)
    for seed in SEEDS:
        w0_epochs=W0_BUDGETS[seed]
        for phase in ('w0','w1'):
            is_w0=phase=='w0'
            schedule={'enabled':True,'first_formal_epoch':32 if is_w0 else 48,
                      'recovery_interval_seconds':1800,'progress_interval_seconds':60,
                      'fixed_budget':is_w0,'early_stop_start_epoch':w0_epochs if is_w0 else 0,
                      'early_stop_evaluations':4 if is_w0 else 2,'relative_improvement':0.001 if is_w0 else 0.0,
                      'defer_w1_reference':not is_w0,'selection_policy':'validation_score_only',
                      'checkpoint_policy':'best_only'}
            phase_config={'max_epochs':w0_epochs if is_w0 else 80,
                          'formal_validation_interval_short_epochs':16 if is_w0 else 4,
                          'efficient_schedule':schedule}
            if is_w0:
                phase_config.update({'early_formal_interval_short_epochs':16,
                    'early_formal_until_short_epoch':0,
                    'target_update_budgets':{'minimum':122,'maximum':(w0_epochs//8)*61,'patience_equivalent':122}})
            config={'extends':f'vpp_world_moe_R5_v2_{1 if is_w0 else 2}.yaml',
                    'project':{'output_root':str(ROOT/'world_moe'),'seed':seed},
                    'training':{'seed':seed,'run_id':world_run(seed),'phases':{phase:phase_config}}}
            if is_w0:
                config['data']={'normalization':{'artifact_path':str(data_root/'train_only_typed_normalization_v1.json')},
                                'lineage_path':str(data_root/'data_lineage_v1.json')}
            path=config_path(seed,phase)
            text=yaml.safe_dump(config,allow_unicode=True,sort_keys=False)
            same_config=path.exists() and yaml.safe_load(path.read_text(encoding='utf-8'))==config
            if path.exists() and not same_config and not can_amend:
                raise RuntimeError(f'Existing registered configuration differs: {path}')
            if not same_config:path.write_text(text,encoding='utf-8')
        for variant in VARIANTS:
            config={'extends':'r_ecmoe_balanced_continue_earlystop.yaml',
                    'source':{'world_config':str(config_path(seed,'w1')),
                              'w1_checkpoint':str(world_phase(seed,'w1')/'checkpoints/formal_primary_best.pt'),
                              'w1_checkpoint_role':'independently_trained_formal_primary_best'},
                    'model':{'mechanism_variant':variant,'active_top_k':2},
                    'training':{'seed':seed,'epochs':208,'first_full_validation_epoch':78,
                      'selection_policy':'validation_score_only',
                      'enable_sentinel_validation':False,'checkpoint_interval_updates':0,
                      'save_latest_every_epoch':False,'resume_checkpoint_epochs':[],
                      'early_stop_start_epoch':0,'full_validation_interval_epochs':13,
                      'early_stop_patience':2,'early_stop_min_relative_improvement':0.0,
                      'coalesce_checkpoint_roles':True,'save_only_on_improvement':True,'save_on_exception':True},
                    'outputs':{'root':str(ROOT/'r'),'save_best_speed_feasible':False}}
            path=config_path(seed,'r',variant); text=yaml.safe_dump(config,allow_unicode=True,sort_keys=False)
            same_config=path.exists() and yaml.safe_load(path.read_text(encoding='utf-8'))==config
            if path.exists() and not same_config and not can_amend:
                raise RuntimeError(f'Existing registered configuration differs: {path}')
            if not same_config:path.write_text(text,encoding='utf-8')
    if not (ROOT/'study_status.json').exists():
        state={'schema':'dcvpp.multiseed-study.v1','status':'implementation_in_progress',
               'created_at':now(),'jobs':jobs(),'implementation_verified':False,
               'models_frozen':False,'evaluation_completed':False,'report_completed':False,
               'protocol':{'seeds':SEEDS,'variants':list(VARIANTS),'train_years':[2022,2023],
                           'validation_year':2024,'test_year':2025,'expected_test_days':362,
                           'expected_test_replay_rows':45612,'prior_2025_already_evaluated':True}}
        publish(state)
    elif can_amend:
        existing['jobs']=jobs();publish(existing)
    return read_json(ROOT/'study_status.json')

def command(job):
    root=Path(job['run_root']); latest=root/'checkpoints/latest.pt'
    if job['phase'] in ('w0','w1'):
        cmd=[sys.executable,'-u',str(REPO/'scripts/run_vpp_world_moe.py'),
             '--config',job['config'],'--phase',job['phase'],'--device','cuda',
             '--seed',str(job['seed']),'--run-id',world_run(job['seed']),'--enforce-phase-gate']
        if latest.exists():cmd+=['--resume']
        elif job['phase']=='w1':cmd+=['--parent-checkpoint',str(world_phase(job['seed'],'w0')/'checkpoints/formal_primary_best.pt')]
    else:
        cmd=[sys.executable,'-u',str(REPO/'scripts/run_r_ecmoe.py'),'--config',job['config'],
             '--mode','resume' if latest.exists() else 'train','--device','cuda',
             '--run-id',r_run(job['seed'],job['variant'])]
    return cmd

def progress(job):
    root=Path(job['run_root'])
    for name in ('progress.json','run_progress.json','state.json','latest_metrics.json'):
        value=read_json(root/name)
        if value:
            if name=='progress.json':return value
            if isinstance(value.get('resume_state'),dict):value={**value,**value['resume_state']}
            return {k:value[k] for k in ('epoch','short_epoch','global_update','updates','update','status',
                    'best_checkpoint','formal_primary_score','decision_value') if k in value}
    return {}

def process_alive(pid):
    if not pid:return False
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.windll.kernel32
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:return False
        try:
            code=wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
        finally:kernel.CloseHandle(handle)
    try:os.kill(int(pid),0);return True
    except ProcessLookupError:return False

def qualified(job):
    root=Path(job['run_root'])
    if job['phase'] in ('w0','w1'):
        state=read_json(root/'state.json',{})
        gate=read_json(root/'gate.json',{})
        if state.get('selection_policy')=='validation_score_only':
            return (state.get('status')=='VALIDATION_SCORE_SELECTED'
                    and state.get('stage_transition_allowed') is True
                    and (root/'checkpoints/formal_primary_best.pt').exists())
        released=state.get('next_phase_released') is True or gate.get('passed') is True
        return released and (root/'checkpoints/formal_primary_best.pt').exists()
    summary=read_json(root/'summary.json',{})
    return (root/'checkpoints/best_decision_value.pt').exists() and bool(summary)

def retain_best_only(job):
    """Remove obsolete training snapshots only after durable stage completion."""
    if job['status']!='completed':return
    folder=(Path(job['run_root'])/'checkpoints').resolve()
    if not folder.is_relative_to(ROOT.resolve()):
        raise RuntimeError('Checkpoint retention target is outside this study')
    keep={'best.pt','formal_primary_best.pt'} if job['phase'] in ('w0','w1') else {'best_decision_value.pt'}
    primary='formal_primary_best.pt' if job['phase'] in ('w0','w1') else 'best_decision_value.pt'
    if not (folder/primary).is_file():raise RuntimeError('Cannot prune without a selected best checkpoint')
    for path in folder.glob('*.pt'):
        if path.name not in keep:
            if not path.resolve().is_relative_to(ROOT.resolve()):raise RuntimeError('Checkpoint escapes study directory')
            path.unlink()

def run_child(state, cmd, log_root, callback=None):
    log_root.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); env['PYTHONPATH']=str(REPO/'src')+os.pathsep+env.get('PYTHONPATH','')
    env['PYTHONUNBUFFERED']='1'
    with (log_root/'stdout.log').open('a',encoding='utf-8') as out, (log_root/'stderr.log').open('a',encoding='utf-8') as err:
        proc=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=out,stderr=err)
        state['active_pid']=proc.pid; state['active_command']=cmd
        last_publish=0.0
        try:
            while proc.poll() is None:
                if time.monotonic()-last_publish >= 60:
                    if callback:callback()
                    publish(state)
                    last_publish=time.monotonic()
                time.sleep(10)
        except BaseException:
            state['status']='controller_interrupted_child_may_be_running';publish(state)
            raise
    state.pop('active_pid',None)
    if callback:callback()
    return proc.returncode

def freeze(state):
    if len(state['jobs'])!=27 or any(j['status']!='completed' for j in state['jobs']):
        raise RuntimeError('All 27 training stages must complete before freezing or opening 2025')
    records=[]
    for job in state['jobs']:
        if job['phase']!='r':continue
        config=yaml.safe_load(Path(job['config']).read_text(encoding='utf-8'))
        checkpoint=Path(job['run_root'])/'checkpoints/best_decision_value.pt'
        paths=[checkpoint,Path(config['source']['w1_checkpoint']),Path(job['config']),Path(config['source']['world_config'])]
        hashes={}
        for path in paths:
            with path.open('rb') as f:hashes[str(path)]=hashlib.file_digest(f,'sha256').hexdigest()
        records.append({'seed':job['seed'],'variant':job['variant'],'checkpoint':str(checkpoint),
                        'config':job['config'],'w1_checkpoint':config['source']['w1_checkpoint'],
                        'world_config':config['source']['world_config'],'sha256':hashes})
    write_json(ROOT/'freeze_manifest.json',{'created_at':now(),'models':records,
                'all_27_training_stages_completed':True,'test_selection_feedback_allowed':False})
    state['models_frozen']=True;publish(state)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    if (ROOT/'inference_protocol.json').exists():
        from run_inference_study import main as inference_main
        return inference_main(['--execute'] if args.execute else ['--prepare'])
    state=prepare()
    if not args.execute:
        print(str(ROOT/'PLAN.md'));return 0
    if not state.get('implementation_verified'):
        raise SystemExit('Focused implementation checks must be recorded before training starts')
    lock=ROOT/'controller.lock'
    if lock.exists():
        try:old_pid=int(lock.read_text(encoding='utf-8').strip())
        except ValueError:old_pid=None
        if process_alive(old_pid) or process_alive(state.get('active_pid')):
            raise SystemExit('A study controller or child is still active; do not launch a duplicate')
        lock.unlink()
    try:
        with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    except FileExistsError:
        raise SystemExit('Study controller lock exists; inspect the recorded process before resuming')
    state['controller_pid']=os.getpid();state['status']='training';state.pop('error',None);publish(state)
    try:
        for job in state['jobs']:
            if job['status']=='completed':retain_best_only(job);continue
            if qualified(job):job['status']='completed';publish(state);retain_best_only(job);continue
            job['status']='running';job.setdefault('started_at',now())
            started=time.monotonic();prior=job.get('elapsed_seconds',0)
            def update():
                job['elapsed_seconds']=prior+time.monotonic()-started;job['progress']=progress(job)
            cmd=command(job);job['command']=cmd
            code=run_child(state,cmd,ROOT/'logs'/str(job['seed'])/(job.get('variant') or job['phase']),update)
            job['exit_code']=code
            if code or not qualified(job):
                job['status']='failed' if code else 'completion_evidence_missing'
                state['status']='needs_attention';state['error']=f'{job["id"]}: exit={code}; inspect stage logs and completion evidence'
                publish(state);return 1
            job['status']='completed';job['completed_at']=now();publish(state);retain_best_only(job)
        if not state['models_frozen']:freeze(state)
        state['status']='evaluating_2025';publish(state)
        if not state['evaluation_completed']:
            code=run_child(state,[sys.executable,'-u',str(REPO/'scripts/evaluate_multiseed_ablation_2025.py')],ROOT/'logs/evaluation_2025')
            if code:raise RuntimeError('2025 evaluation failed; retained per-model results are resumable')
            state['evaluation_completed']=True;publish(state)
        state['status']='building_report';publish(state)
        code=run_child(state,[sys.executable,'-u',str(REPO/'scripts/report_multiseed_ablation.py')],ROOT/'logs/report')
        if code:raise RuntimeError('Report generation or result validation failed')
        state['report_completed']=True;state['status']='completed';state['completed_at']=now();publish(state)
        return 0
    except Exception as exc:
        state['status']='needs_attention';state['error']=f'{type(exc).__name__}: {exc}';publish(state);raise
    finally:
        if not process_alive(state.get('active_pid')):
            lock.unlink(missing_ok=True)

if __name__=='__main__':raise SystemExit(main())
