"""Train each seed's full model, run inference ablations, then advance seeds."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
from multiseed_common import (REPO,ROOT,SEEDS,VARIANTS,W0_BUDGETS,now,read_json,write_json,
                              config_path,r_root,world_phase)
import run_multiseed_ablation as training

STATUS=ROOT/'inference_study_status.json'
PROTOCOL=ROOT/'inference_protocol.json'

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def configuration_hashes():
    hashes={}
    def visit(path):
        path=Path(path).resolve()
        if str(path) in hashes:return
        hashes[str(path)]=digest(path)
        cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
        if cfg.get('extends'):
            parent=Path(cfg['extends'])
            visit(parent if parent.is_absolute() else path.parent/parent)
    for seed in SEEDS:
        for phase,variant in [('w0',None),('w1',None),('r','full')]:visit(config_path(seed,phase,variant))
    return hashes

def render(state):
    done=sum(j['status']=='completed' for j in state['jobs'])
    evaluated=sum(e['status']=='completed' for e in state['evaluations'])
    lines=['# 三种子完整训练与推理端消融计划','',
           f'状态：**{state["status"]}**；完整模型训练 **{done}/9**；推理消融测试 **{evaluated}/3个种子**。',
           f'更新时间：{state["updated_at"]}。','',
           '每个种子按 W0 → W1 → 完整R → 2025推理消融 → 单种子报告 执行，再进入下一个种子。',
           '推理条件为完整模型加6项干预；同一种子共享同一完整R权重，不训练消融模型，不含Top-1/Top-2对照。','',
           '| 种子 | 阶段 | 状态 | 轮数 | 更新数 |','|---|---|---|---:|---:|']
    for j in state['jobs']:
        p=j.get('progress',{}) or {}
        lines.append(f'| {j["seed"]} | {j["phase"].upper()} | {j["status"]} | {p.get("epoch","—")} | {p.get("global_update",p.get("update","—"))} |')
    lines+=['','| 种子 | 2025推理测试 | 报告 |','|---|---|---|']
    for e,r in zip(state['evaluations'],state['seed_reports']):
        lines.append(f'| {e["seed"]} | {e["status"]} | {r["status"]} |')
    protocol=read_json(PROTOCOL,{})
    budgets=protocol.get('w0_epoch_budgets',W0_BUDGETS)
    budget_text='；'.join(f'{seed}为{budgets.get(str(seed),budgets.get(seed))}短轮' for seed in SEEDS)
    chronology=('- 2025此前已见，本轮为补充评估。初始配置在首次测试前冻结；第三种子W0按用户后续指令改为96短轮，修订在该种子训练前登记，历史协议保留。'
                if protocol.get('configuration_amendments') else
                '- 2025此前已见；本轮逐种子测试后才训练下一种子，后续配置已预先冻结，不允许测试结果反馈调参。')
    lines+=['',f'- W0预算：{budget_text}。首次验证32，此后每16；96短轮为12遍数据/732更新。',
            '- W1首次验证48、每4轮；R首次验证78、每13轮。首次最优后两次未刷新即停，无额外起算轮数或最小改善幅度。',
            '- 正常训练仅最优刷新时保存；暂停/异常可保存恢复点。各阶段选模以正式验证评分为准，资格阈值仅诊断。',
            chronology,
            '- 共3个训练种子、每种子7个推理条件，合计45,612条2025决策回放。已取消重训练任务仅保留历史，不计入结果。',
            f'- [{"x" if state.get("final_report_completed") else " "}] 三种子成本、CVaR、SLA等综合报告及CSV/JSON原始结果。']
    if state.get('error'):lines+=['','## 当前问题','',state['error']]
    tmp=ROOT/f'PLAN.md.{os.getpid()}.tmp';tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8');tmp.replace(ROOT/'PLAN.md')

def publish(state):
    state['updated_at']=now();write_json(STATUS,state);render(state)
    old=read_json(ROOT/'study_status.json',{})
    old.update(status=state['status'],updated_at=state['updated_at'],active_pid=state.get('active_pid'),
               controller_pid=state.get('controller_pid'),error=state.get('error'),
               active_plan='inference_study_status.json',report_completed=state.get('final_report_completed',False))
    old.setdefault('protocol',{}).update(design='same_checkpoint_inference_intervention',training_stages=9)
    lookup={j['id']:j for j in state['jobs']}
    old['jobs']=[lookup.get(j['id'],j) for j in old.get('jobs',[])]
    write_json(ROOT/'study_status.json',old)

def prepare():
    if not PROTOCOL.exists():
        records=[{'seed':seed,'variant':'full','config':str(config_path(seed,'r','full')),
                  'checkpoint':str(r_root(seed,'full')/'checkpoints/best_decision_value.pt'),
                  'w1_checkpoint':str(world_phase(seed,'w1')/'checkpoints/formal_primary_best.pt'),
                  'world_config':str(config_path(seed,'w1'))} for seed in SEEDS]
        write_json(PROTOCOL,{'schema':'dcvpp.inference-study.protocol.v1','created_at':now(),
             'seeds':list(SEEDS),'variants':list(VARIANTS),'full_model_records':records,
             'comparison_design':'same_checkpoint_inference_intervention','separately_retrained':False,
             'final_results_may_modify_selection':False,'no_test_to_training_feedback':True,
             'future_seed_configurations_frozen_before_first_test':True,
             'later_seed_training_after_prior_seed_test':True,'prior_2025_already_evaluated':True,
             'configuration_sha256':configuration_hashes(),'w0_epoch_budgets':W0_BUDGETS,
             'expected_days':362,'decisions_per_day':6,'expected_replay_rows_per_seed':15204,
             'expected_total_replay_rows':45612,'training_stages':9})
    if not STATUS.exists():
        old=read_json(ROOT/'study_status.json')
        base=[dict(j) for j in old['jobs'] if j['phase'] in ('w0','w1') or j.get('variant')=='full']
        state={'schema':'dcvpp.inference-study.status.v1','status':'implementation_in_progress',
               'created_at':now(),'jobs':base,'implementation_verified':False,
               'evaluations':[{'seed':seed,'status':'pending','output_root':str(ROOT/'inference_evaluation_2025'/f'seed_{seed}')} for seed in SEEDS],
               'seed_reports':[{'seed':seed,'status':'pending','output_root':str(ROOT/'inference_report'/f'seed_{seed}')} for seed in SEEDS],
               'final_report_completed':False}
        publish(state)
    return read_json(STATUS)

def check_configuration():
    p=read_json(PROTOCOL)
    for path,value in p['configuration_sha256'].items():
        if digest(path)!=value:raise RuntimeError(f'Frozen training configuration changed: {path}')

def child(state,cmd,logdir,on_progress=None):
    logdir.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['PYTHONPATH']=str(REPO/'src')+os.pathsep+env.get('PYTHONPATH','')
    env['PYTHONIOENCODING']='utf-8';env['PYTHONUNBUFFERED']='1'
    with (logdir/'stdout.log').open('a',encoding='utf-8') as out,(logdir/'stderr.log').open('a',encoding='utf-8') as err:
        proc=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=out,stderr=err)
        state['active_pid']=proc.pid;state['active_command']=cmd;last=0
        while proc.poll() is None:
            if time.monotonic()-last>=60:
                if on_progress:on_progress()
                publish(state);last=time.monotonic()
            time.sleep(5)
    state.pop('active_pid',None)
    if on_progress:on_progress()
    return proc.returncode

def refresh_plan():
    """Refresh amended text while an older controller continues its live child."""
    previous=None
    while True:
        stamp=STATUS.stat().st_mtime_ns
        if stamp!=previous:
            time.sleep(1)  # Let the controller finish its status/plan publish pair.
            state=read_json(STATUS)
            if state:render(state)
            previous=stamp
        state=read_json(STATUS,{})
        if not training.process_alive(state.get('controller_pid')):return 0
        time.sleep(5)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--refresh-plan',action='store_true',help='Refresh plan text only; never starts training')
    args=parser.parse_args(argv)
    if args.refresh_plan:return refresh_plan()
    state=prepare()
    if not args.execute:print(str(ROOT/'PLAN.md'));return 0
    if not state.get('implementation_verified'):raise SystemExit('Inference workflow checks are not yet complete')
    if training.process_alive(state.get('controller_pid')) or training.process_alive(state.get('active_pid')):
        raise SystemExit('Inference controller or child is already active')
    lock=ROOT/'inference_controller.lock'
    if lock.exists():
        try:pid=int(lock.read_text())
        except ValueError:raise RuntimeError('Invalid controller lock')
        if training.process_alive(pid):raise SystemExit('Inference controller already active')
        lock.unlink()
    with lock.open('x') as f:f.write(str(os.getpid()))
    state['controller_pid']=os.getpid();state.pop('error',None)
    try:
        for seed in SEEDS:
            check_configuration()
            for job in [j for j in state['jobs'] if j['seed']==seed]:
                if job['status']=='completed':continue
                if training.qualified(job):job['status']='completed';publish(state);continue
                state['status']='training';job['status']='running';job.setdefault('started_at',now())
                job.pop('exit_code',None);started=time.monotonic();prior=job.get('elapsed_seconds',0) or 0
                def update():
                    job['elapsed_seconds']=prior+time.monotonic()-started;job['progress']=training.progress(job)
                cmd=training.command(job);job['command']=cmd
                code=child(state,cmd,ROOT/'inference_logs'/str(seed)/job['phase'],update)
                job['exit_code']=code
                if code or not training.qualified(job):
                    job['status']='failed';raise RuntimeError(f'{job["id"]}: exit {code}; inspect actual stage evidence')
                job['status']='completed';job['completed_at']=now();publish(state);training.retain_best_only(job)
            evaluation=next(e for e in state['evaluations'] if e['seed']==seed)
            if evaluation['status']!='completed':
                check_configuration();state['status']='evaluating_inference_ablations';evaluation['status']='running'
                code=child(state,[sys.executable,'-u',str(REPO/'scripts/evaluate_inference_ablation_2025.py'),'--seed',str(seed)],ROOT/'inference_logs'/str(seed)/'evaluation')
                if code:evaluation['status']='failed';raise RuntimeError(f'Seed {seed} inference evaluation failed')
                meta=read_json(Path(evaluation['output_root'])/'metadata.json',{})
                if str(meta.get('status','')).lower()!='completed':raise RuntimeError('Evaluation did not produce completion metadata')
                evaluation['status']='completed';evaluation['completed_at']=now();publish(state)
            report=next(r for r in state['seed_reports'] if r['seed']==seed)
            if report['status']!='completed':
                state['status']='building_seed_report';report['status']='running';publish(state)
                code=child(state,[sys.executable,'-u',str(REPO/'scripts/report_inference_ablation.py'),'--seed',str(seed)],ROOT/'inference_logs'/str(seed)/'report')
                if code:report['status']='failed';raise RuntimeError(f'Seed {seed} report failed')
                if not (Path(report['output_root'])/'REPORT.md').is_file():raise RuntimeError('Seed report is missing')
                report['status']='completed';report['completed_at']=now();publish(state)
        state['status']='building_final_report';publish(state)
        code=child(state,[sys.executable,'-u',str(REPO/'scripts/report_inference_ablation.py')],ROOT/'inference_logs/final_report')
        if code:raise RuntimeError('Final inference report failed')
        if not (ROOT/'inference_report/REPORT.md').is_file():raise RuntimeError('Final report is missing')
        state['final_report_completed']=True;state['status']='completed';state['completed_at']=now();publish(state)
        return 0
    except BaseException as exc:
        state['status']='needs_attention';state['error']=f'{type(exc).__name__}: {exc}';publish(state);raise
    finally:
        if not training.process_alive(state.get('active_pid')):lock.unlink(missing_ok=True)

if __name__=='__main__':raise SystemExit(main())
