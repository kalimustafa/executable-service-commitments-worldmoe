"""Matched W1-MPC and R-ECMoE measurements on the registered suffix replay.

The deterministic MPC planning problem is solved afresh at each registered
origin. Whole suffix plans are replayed, exactly as for the learned controller;
this is not a newly claimed continuous closed-loop dispatch experiment.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
import pyarrow.dataset  # noqa: E402,F401 -- verified Windows DLL import order
import numpy as np
import torch
import evaluate_inference_ablation_2025 as core
from dcvpp.vpp_world_moe.data import WorldMoECorpus
from dcvpp.vpp_world_moe.r_ecmoe_replay import RealizedHistoricalReplay
from dcvpp.vpp_world_moe.r_ecmoe_runtime import RECMoEInferenceRuntime
from dcvpp.vpp_world_moe.mpc_baseline import W1MPCController
from multiseed_common import read_json, write_json, now

ROOT = REPO/'artifacts/r_ecmoe_mpc_benchmark_20260912'
STUDY = REPO/'artifacts/r_ecmoe_multiseed_20260910'
LEGACY = REPO/'artifacts/r_ecmoe_legacy_aligned_20260912'
RUN_IDS = ('legacy_seed20240813','current_seed20240813','current_seed20240829','current_seed20240917')
POLICIES = ('full','w1_mpc')
NAMES = {'full':'R-ECMoE','w1_mpc':'W1-MPC'}
OPTIONS = {'time_limit_seconds':2.0,'mip_gap':1e-4,'threads':1}
DESIGN = 'matched_forecaster_independent_suffix_optimization_comparison'
MPC_DESCRIPTOR = SimpleNamespace(name='w1_mpc', description='Deterministic W1-MPC horizon optimization with matched causal inputs and action assembly')


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def save_csv(path, rows):
    rows=list(rows)
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    normalized=[{k:row.get(k) for k in fields} for row in rows]
    core.evaluation._atomic_csv(Path(path),normalized)


def input_record(run_id):
    seed=int(run_id[-8:])
    folder=LEGACY if run_id.startswith('legacy_') else STUDY/'inference_evaluation_2025'/f'seed_{seed}'
    meta=read_json(folder/'metadata.json')
    if meta.get('status')!='completed':raise RuntimeError('Existing model evaluation identity is incomplete')
    record=dict(meta['full_model_record'])
    if run_id.startswith('legacy_'):
        snapshot=read_json(REPO/'artifacts/r_ecmoe_mechanisms_2025_20260910/epoch130-seed20240813-20260910T204056/launch_inputs.json')
        world_config=snapshot['world_config_resolved']; model_config=snapshot['main_config_resolved']['model']
        config_source=REPO/'artifacts/r_ecmoe_mechanisms_2025_20260910/epoch130-seed20240813-20260910T204056/launch_inputs.json'
    else:
        world_config=core.base._load_extended_yaml(Path(record['world_config']))
        model_config=core.base._load_extended_yaml(Path(record['config']))['model']
        config_source=Path(record['config'])
    w1_state=read_json(Path(record['w1_checkpoint']).parent.parent/'state.json')
    w1_best=w1_state['training_control']['formal_primary_record']['epoch']
    record.update(run_id=run_id,run_label=f'Run {RUN_IDS.index(run_id)+1}',seed=seed,
        w1_selected_epoch=w1_best,source_metadata=str(folder/'metadata.json'),config_source=str(config_source))
    return record,world_config,model_config


def runtime(record, world_config, model_config, device):
    result=RECMoEInferenceRuntime.from_w1_checkpoint(world_config,model_config,record['w1_checkpoint'],device,
        r_checkpoint_path=record['checkpoint'])
    model=result.model
    model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):raise RuntimeError('Weights are not frozen')
    return model


def make_protocol(root):
    path=root/'protocol.json'
    if path.exists():
        protocol=read_json(path)
        if protocol['mpc_options']!=OPTIONS:raise RuntimeError('Frozen MPC options differ')
        for name,digest in protocol['input_sha256'].items():
            if core.base._sha256(Path(name))!=digest:raise RuntimeError(f'Frozen benchmark input changed: {name}')
        return protocol
    if not read_json(root/'pilot_2024.json',{}).get('completed'):
        raise RuntimeError('Complete the development-only pilot before freezing the final benchmark')
    inputs={str(Path(__file__)):core.base._sha256(Path(__file__)),
        str(REPO/'src/dcvpp/vpp_world_moe/mpc_baseline.py'):core.base._sha256(REPO/'src/dcvpp/vpp_world_moe/mpc_baseline.py')}
    records=[]
    for run_id in RUN_IDS:
        record,wc,mc=input_record(run_id)
        records.append({'record':record,'world_config':wc,'model_config':mc})
        for key in ['checkpoint','w1_checkpoint','config_source','source_metadata']:
            p=Path(record[key]);inputs[str(p)]=core.base._sha256(p)
        normalizer=core.base._path(wc['data']['normalization']['artifact_path'])
        inputs[str(normalizer)]=core.base._sha256(normalizer)
    import scipy
    protocol={'schema':'dcvpp.matched-mpc.protocol.v1','created_at':now(),'records':records,
        'mpc_options':OPTIONS,'input_sha256':inputs,'policies':NAMES,'comparison_design':DESIGN,
        'forecast':'same frozen W1 point-observation packet within each run',
        'controller_input_boundary':'WorldInputPacket-derived reference, physical constraints and published action history only; no batch targets',
        'optimization_objective':'same forecast raw composite cost with original degradation, PV-curtailment, unserved-power and expired-work penalties; deterministic point forecast',
        'no_added_terminal_soc_requirement':True,'matched_assembler_storage_energy_budgets':True,
        'shared_constraint_assembler_and_realized_replay':True,'continuous_closed_loop_claim':False,
        'expected_days':362,'decisions':list(map(list,core.DECISIONS)),'expected_rows_per_run':4344,
        'expected_total_rows':17376,'expected_formal_mpc_solves':8688,
        'training_runs':4,'unique_seed_count':3,'training_performed':False,'checkpoint_selection_performed':False,
        'prior_2025_already_evaluated':True,'test_used_to_tune_mpc':False,
        'solver':'SciPy HiGHS MILP','gurobi_size_limit_observed':True,
        'versions':{'torch':torch.__version__,'scipy':scipy.__version__,'numpy':np.__version__}}
    write_json(path,protocol)
    return protocol


def make_corpus(record,wc,root,protocol):
    marker={'schema':core.base.FROZEN_MARKER_SCHEMA,'status':'FROZEN_FOR_ONE_SHOT_FINAL_EVALUATION',
        'final_year':2025,'best_checkpoint_frozen':True,'model_selection_frozen':True,'thresholds_frozen':True,
        'data_statistics_frozen':True,'development_evaluation_complete':True,'training_closed':True,
        'locked_final_opened':False,'one_shot_runs_allowed':1,'final_results_may_modify_selection':False,
        'expected_evaluable_days':362,'best_checkpoint':record['checkpoint'],
        'run_id':record.get('checkpoint_run_id',record['run_id']),
        'one_shot_id':f'matched-mpc-20260912-{record["run_id"]}',
        'evaluation_role':'supplemental_matched_mpc_benchmark','prior_2025_already_evaluated':True,
        'comparison_design':DESIGN,'separately_retrained':False,'protocol_sha256':core.base._sha256(root/'protocol.json')}
    write_json(root/record['run_id']/'frozen_evaluation_marker.json',marker)
    corpus=core.base.build_sealed_final_corpus(config=wc,repository_root=REPO,frozen_evaluation_marker=marker)
    core.base._validate_sealed_corpus(corpus,marker)
    return corpus


def encode(model,batch):
    return model.encode_expert_context(batch['world_input'],da_anchor=batch['forecast_anchors']['da'],
        da_residual_cap=batch['forecast_residual_caps']['da'])


def outcome_rows(decision,realized,policy,record,batch,stage,origin,horizon,equivalent_ms):
    descriptor=core.registered_interventions()['full'] if policy=='full' else MPC_DESCRIPTOR
    result=core.evaluation._execution_rows(decision,realized,variant=descriptor,
        dates=batch['operating_dates'],stage_id=stage,origin=origin,horizon=horizon,
        cost_scale=float(batch['train_frozen_baseline_daily_cost'].detach().cpu()),
        equivalent_latency_ms_per_sample=equivalent_ms)
    sla=core.base._sla_deadline_severity(realized)[1].detach().cpu().tolist()
    constraint=core.base._constraint_severity(realized)[1].detach().cpu().tolist()
    for i,row in enumerate(result):
        row.update(variant=policy,policy_name=NAMES[policy],seed=record['seed'],run_id=record['run_id'],
            comparison_design=DESIGN,separately_retrained=False,
            sla_deadline_severity=float(sla[i]),constraint_severity=float(constraint[i]),
            hard_violation=float(row['hard_residual_max'])>core.base.HARD_TOLERANCE)
    return result


def pilot(root,device):
    record,wc,mc=input_record(RUN_IDS[0]);model=runtime(record,wc,mc,device)
    corpus=WorldMoECorpus(wc,REPO)
    controller=W1MPCController(model,**OPTIONS)
    replay=RealizedHistoricalReplay.from_model(model)
    measurements=[]
    with torch.inference_mode():
        for stage,origin,horizon in [(0,0,96),(1,0,96),(1,95,1)]:
            batch=next(iter(corpus.batches('validation',1,device=device,shuffle_seed=None,max_days=1,
                origin=origin,require_h96=False,include_next_day=False)))
            if any(not str(day).startswith('2024-') for day in batch['operating_dates']):raise RuntimeError('Pilot must use 2024')
            risk=core.base.RiskConditionPacket.defaults(1,device=device,horizon=horizon,stage=stage)
            with core.base._precision(device):
                packet=encode(model,batch)
                decision,elapsed=core.base._timed(device,lambda:controller.decide(packet,risk))
            realized=replay(decision,batch=batch,risk=risk,source_split='validation')
            row=outcome_rows(decision,realized,'w1_mpc',record,batch,stage,origin,horizon,elapsed)[0]
            if not math.isfinite(row['normalized_operating_cost']):raise RuntimeError('Pilot produced nonfinite outcomes')
            measurements.append({'stage_id':stage,'origin':origin,'horizon':horizon,
                'controller_wall_ms':elapsed,'outcome':row,'solver_records':controller.last_solver_records})
            write_json(root/'pilot_progress.json',{'completed_windows':len(measurements),'total_windows':3,'updated_at':now()})
    write_json(root/'pilot_2024.json',{'completed':True,'split':'validation','year':2024,'mpc_options':OPTIONS,
        'parameter_search':False,'training_performed':False,'measurements':measurements,'completed_at':now()})
    print(json.dumps({'pilot_completed':True,'windows':3,'controller_ms':[r['controller_wall_ms'] for r in measurements]}))


def benchmark_latency(model,controller,corpus,device,folder):
    stored=read_json(folder/'latency.json')
    if stored:return stored
    batch=next(iter(corpus.batches(core.base.FINAL_SPLIT,1,device=device,shuffle_seed=None,max_days=None,
        origin=0,require_h96=False,include_next_day=False)))
    risk=core.base.RiskConditionPacket.defaults(1,device=device,horizon=96,stage=1)
    variant=core.registered_interventions()['full']
    samples={name:[] for name in POLICIES};solver_samples=[]
    with torch.inference_mode(),core.base._precision(device):
        for index in range(35):
            for policy in (POLICIES if index%2==0 else tuple(reversed(POLICIES))):
                def call():
                    packet=encode(model,batch)
                    if policy=='full':return core.evaluation._execute_variant(model,packet,risk,variant)
                    return controller.decide(packet,risk)
                _,elapsed=core.base._timed(device,call)
                if index>=5:
                    samples[policy].append(elapsed)
                    if policy=='w1_mpc':solver_samples.extend(controller.last_solver_records)
            write_json(folder/'progress.json',{'status':'benchmarking_latency','completed_repeats':index+1,
                'total_repeats_including_warmup':35,'updated_at':now()})
    result={name:{'p95_latency_ms_batch1':core.evaluation._percentile(sample,.95),
        'p50_latency_ms_batch1':core.evaluation._percentile(sample,.5),'mean_latency_ms_batch1':sum(sample)/len(sample),
        'latency_warmup':5,'latency_repeats':30,'latency_measurement':'batch1_h96_end_to_end_causal_forward_without_realized_replay',
        'raw_samples_ms':sample} for name,sample in samples.items()}
    result['w1_mpc']['solver_raw_records']=solver_samples
    write_json(folder/'latency.json',result)
    return result


def execute_run(root,entry,protocol,device,batch_size):
    record=entry['record'];run_id=record['run_id'];folder=root/run_id;folder.mkdir(parents=True,exist_ok=True)
    if read_json(folder/'metadata.json',{}).get('status')=='completed':return
    started=time.perf_counter();wc=entry['world_config'];mc=entry['model_config']
    model=runtime(record,wc,mc,device);controller=W1MPCController(model,**OPTIONS)
    corpus=make_corpus(record,wc,root,protocol)
    replay=core.base.SealedRealizedHistoricalReplay.from_model(model)
    latency=benchmark_latency(model,controller,corpus,device,folder)
    partial={name:core.base.read_partial(folder/'partial'/f'{name}.jsonl') for name in POLICIES}
    keys={name:{core.base._row_key(row) for row in data} for name,data in partial.items()}
    solver_records=read_json(folder/'solver_records.json',[])
    solver_keys={tuple(row[key] for key in ['operating_date','stage_id','decision_origin','horizon']) for row in solver_records}
    full_variant=core.registered_interventions()['full']
    with torch.inference_mode():
        for stage,origin,horizon in core.DECISIONS:
            for batch in corpus.batches(core.base.FINAL_SPLIT,batch_size,device=device,shuffle_seed=None,max_days=None,
                    origin=origin,require_h96=False,include_next_day=False):
                dates=tuple(map(str,batch['operating_dates']))
                pending=[p for p in POLICIES if any((day,stage,origin,horizon) not in keys[p] for day in dates)]
                if not pending:continue
                if any(not day.startswith('2025-') for day in dates):raise RuntimeError('Wrong formal split')
                risk=core.base.RiskConditionPacket.defaults(len(dates),device=device,horizon=horizon,stage=stage)
                with core.base._precision(device):
                    packet,encode_ms=core.base._timed(device,lambda:encode(model,batch))
                    full,full_ms=core.base._timed(device,lambda:core.evaluation._execute_variant(model,packet,risk,full_variant))
                    mpc,mpc_ms=core.base._timed(device,lambda:controller.decide(packet,risk))
                # Neither controller sees the realized targets until both decisions are fixed.
                for policy,decision,elapsed in [('full',full,full_ms),('w1_mpc',mpc,mpc_ms)]:
                    if policy not in pending:continue
                    realized=replay(decision,batch=batch,risk=risk)
                    emitted=outcome_rows(decision,realized,policy,record,batch,stage,origin,horizon,(encode_ms+elapsed)/len(dates))
                    selected=[row for row in emitted if core.base._row_key(row) not in keys[policy]]
                    core.base._append_batch(folder/'partial'/f'{policy}.jsonl',selected)
                    partial[policy].extend(selected);keys[policy].update(core.base._row_key(row) for row in selected)
                if len(controller.last_solver_records)!=len(dates):raise RuntimeError('Solver record count mismatch')
                for day,solver in zip(dates,controller.last_solver_records):
                    key=(day,stage,origin,horizon)
                    if key not in solver_keys:
                        solver_records.append({**solver,'run_id':run_id,'seed':record['seed'],'operating_date':day,
                            'stage_id':stage,'decision_origin':origin,'horizon':horizon});solver_keys.add(key)
                write_json(folder/'solver_records.json',solver_records)
                write_json(folder/'progress.json',{'status':'running','run_id':run_id,
                    'observed_replay_rows':sum(map(len,partial.values())),'expected_replay_rows':4344,
                    'formal_solver_calls_recorded':len(solver_records),'last_date':dates[-1],
                    'stage_id':stage,'decision_origin':origin,'elapsed_seconds':time.perf_counter()-started,'updated_at':now()})
    all_rows=sorted([row for data in partial.values() for row in data],key=lambda row:(row['variant'],*core.base._row_key(row)))
    if len(all_rows)!=4344 or len(solver_records)!=2172:raise RuntimeError('Incomplete matched benchmark coverage')
    model_records=[{**record,'variant':'full'}, {**record,'variant':'w1_mpc','checkpoint':record['w1_checkpoint'],
        'checkpoint_epoch':record['w1_selected_epoch'],'selection_policy':'fixed_MPC_no_checkpoint_selection'}]
    scalar_latency={name:{k:v for k,v in latency[name].items() if not isinstance(v,(list,dict))} for name in POLICIES}
    daily,summaries,dates=core.base.aggregate_models(all_rows,model_records,scalar_latency,expected_days=362)
    for row in daily:
        row.update(run_id=run_id,policy_name=NAMES[row['variant']],separately_retrained=False,comparison_design=DESIGN)
    for row in summaries:
        row.update(run_id=run_id,policy_name=NAMES[row['variant']],separately_retrained=False,comparison_design=DESIGN)
        raw=np.asarray([r['realized_operating_cost'] for r in daily if r['variant']==row['variant']])
        row['mean_realized_operating_cost']=float(raw.mean());row['cvar95_realized_operating_cost']=float(np.sort(raw)[-19:].mean())
    for filename,data in [('per_decision_models.csv',all_rows),('per_day_models.csv',daily),('model_summary.csv',summaries),('solver_records.csv',solver_records)]:
        save_csv(folder/filename,data)
    write_json(folder/'metadata.json',{'status':'completed','run_id':run_id,'seed':record['seed'],
        'comparison_design':DESIGN,'policies':NAMES,'dataset_dates':dates,'observed_replay_rows':len(all_rows),
        'formal_mpc_solves':len(solver_records),'full_model_record':record,'mpc_options':OPTIONS,
        'same_frozen_forecaster_within_run':True,'all_actions_fixed_before_realized_target_read':True,
        'training_performed':False,'checkpoint_selection_performed':False,'test_used_to_tune_mpc':False,
        'prior_2025_already_evaluated':True,'continuous_closed_loop_claim':False,
        'protocol_sha256':core.base._sha256(root/'protocol.json'),'data_lineage':corpus.sealed_final_lineage,
        'elapsed_seconds':time.perf_counter()-started,'completed_at':now()})
    write_json(folder/'progress.json',{'status':'completed','run_id':run_id,'observed_replay_rows':4344,'updated_at':now()})
    del model,controller,corpus;gc.collect();torch.cuda.empty_cache()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--run-id',choices=RUN_IDS)
    parser.add_argument('--device',default='cuda');parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args(argv);root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    if args.pilot:pilot(root,device);return 0
    if not args.execute:parser.error('Select --pilot or --execute')
    import run_multiseed_ablation as process_tools
    lock=root/'controller.lock'
    if lock.exists():
        owner=int(lock.read_text())
        if process_tools.process_alive(owner):raise RuntimeError('Matched MPC controller already active')
        lock.unlink()
    with lock.open('x') as f:f.write(str(os.getpid()))
    protocol=make_protocol(root)
    state={'status':'running','controller_pid':os.getpid(),'updated_at':now(),'jobs':[]}
    try:
        for entry in protocol['records']:
            run_id=entry['record']['run_id']
            if args.run_id and args.run_id!=run_id:continue
            state['active_run']=run_id;state['updated_at']=now();write_json(root/'status.json',state)
            execute_run(root,entry,protocol,device,args.batch_size)
            state['jobs'].append({'run_id':run_id,'status':'completed'})
        state['status']='completed' if not args.run_id else 'selected_run_completed'
        state.pop('active_run',None);state['updated_at']=now();write_json(root/'status.json',state)
    except BaseException as exc:
        state.update(status='needs_attention',error=f'{type(exc).__name__}: {exc}',updated_at=now())
        write_json(root/'status.json',state);raise
    finally:
        if lock.exists() and int(lock.read_text())==os.getpid():lock.unlink()
    return 0


if __name__=='__main__':raise SystemExit(main())
