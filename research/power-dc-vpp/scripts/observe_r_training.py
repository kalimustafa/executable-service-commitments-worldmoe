"""Low-frequency host timing and progress without additional neural forwards."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import time

def attach_progress(runner):
    root=Path(runner.run_root)
    timing={'train_step_seconds':0.0,'train_step_count':0,'validation_seconds':0.0,
            'validation_count':0,'checkpoint_seconds':0.0,'checkpoint_write_count':0,
            'checkpoint_alias_seconds':0.0,'checkpoint_alias_count':0}
    last=0.0
    status='initializing'
    last_validation={}
    def write(path,payload):
        tmp=path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')
        tmp.replace(path)
    def publish(force=False):
        nonlocal last
        if not force and time.monotonic()-last<60:return
        last=time.monotonic()
        write(root/'progress.json',{'run_id':runner.run_id,'status':status,'epoch':runner.epoch,
              'global_update':runner.global_update,'mechanism_variant':runner.mechanism_variant,
              'last_validation':last_validation,'timing':dict(timing),
              'timing_basis':'host wall-clock within existing operations; no additional CUDA synchronization',
              'updated_at_utc':datetime.now(timezone.utc).isoformat()})
    def wrap(method_name,prefix):
        original=getattr(runner,method_name)
        def observed(*args,**kwargs):
            nonlocal status,last_validation
            status={'train_step':'training','validation':'validating',
                    'checkpoint':'saving','checkpoint_alias':'saving'}[prefix]
            start=time.perf_counter()
            try:result=original(*args,**kwargs)
            finally:
                timing[prefix+'_seconds']+=time.perf_counter()-start
                count_key={'checkpoint':'checkpoint_write_count','checkpoint_alias':'checkpoint_alias_count'}.get(prefix,prefix+'_count')
                timing[count_key]+=1
            if prefix=='validation':
                last_validation={'epoch':runner.epoch,'decision_value':result.decision_value,
                                 'selection_eligible':result.selection_eligible}
            if prefix in ('checkpoint','checkpoint_alias'):
                write(root/'timing.json',{'run_id':runner.run_id,'epoch':runner.epoch,
                                        'global_update':runner.global_update,'timing':dict(timing)})
            status='training';publish(force=prefix!='train_step')
            return result
        setattr(runner,method_name,observed)
    for method,prefix in [('train_update','train_step'),('validate','validation'),
                          ('_save_checkpoint','checkpoint'),('_reuse_checkpoint','checkpoint_alias')]:
        wrap(method,prefix)
    original_resume=runner.resume
    def resume(*args,**kwargs):
        result=original_resume(*args,**kwargs)
        try:
            saved=json.loads((root/'timing.json').read_text(encoding='utf-8'))
            if saved.get('global_update')==runner.global_update and saved.get('epoch')==runner.epoch:
                timing.update({k:v for k,v in saved['timing'].items() if k in timing})
        except (FileNotFoundError,json.JSONDecodeError):pass
        publish(True);return result
    runner.resume=resume
    def finish(success):
        nonlocal status
        status='completed' if success else 'failed';publish(True)
    return finish
