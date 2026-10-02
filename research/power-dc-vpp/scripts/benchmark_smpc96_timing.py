"""Six fixed-input H96 full-chain timing probes; no cost/SLA outcome comparison."""
from __future__ import annotations
import argparse
from dataclasses import fields,is_dataclass,replace
import json
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import pyarrow.dataset  # noqa: F401
import numpy as np
import torch
import benchmark_smpc20_timing as previous
import run_matched_mpc_benchmark as common
from train_grid_sensitivity import sha
from multiseed_common import read_json,now

ROOT=REPO/'artifacts/r_ecmoe_smpc96_timing_20260913'
DATES=('2025-01-01','2025-04-01','2025-05-03','2025-06-24','2025-07-01','2025-10-01')


def save(path,value):previous.save(path,value)


def on_device(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if is_dataclass(value):return replace(value,**{f.name:on_device(getattr(value,f.name),device) for f in fields(value)})
    if isinstance(value,dict):return {k:on_device(v,device) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(on_device(v,device) for v in value)
    if isinstance(value,list):return [on_device(v,device) for v in value]
    return value


def freeze(root,record,wc,mc):
    root.mkdir(exist_ok=True,parents=True)
    files=[Path(__file__),REPO/'scripts/benchmark_smpc20_timing.py',
        REPO/'src/dcvpp/vpp_world_moe/scenario_mpc.py',REPO/'src/dcvpp/vpp_world_moe/scenario_mpc_inputs.py',
        Path(record['checkpoint']),Path(record['w1_checkpoint']),Path(record['config_source'])]
    protocol=dict(schema='dcvpp.smpc96.timing.v1',frozen_at=now(),model_record=record,world_config=wc,r_model_config=mc,
        cases=[dict(case_id=f'{day}_RT_o0_H96',date=day,origin=0,horizon=96,stage=1) for day in DATES],
        case_rule='first calendar day of each quarter plus the already fixed routine and rare dates; no timing-outcome selection',
        batch_size=1,scenario_count=20,scenario_seed=20240813,scenario_weighting='uniform native frozen-W1 Monte Carlo',
        objective='expected forecast base composite cost + 0.25 CVaR95; no added ramp or curvature term',
        target_relative_gap=0.,absolute_gap_tolerance=1e-10,time_limit_seconds_per_call=600.,accepted_timeout_gap=.05,threads=1,
        baseline='W1-SMPC-20',predictive_horizon_identical_for_both_models=96,active_intervals_identical=True,
        r_warmups_per_case=5,r_measured_repeats_per_case=30,mpc_measured_calls_per_case=1,scenario_kernel_warmups_per_case=2,
        persistent_solver_across_dates=True,solver_reset_between_dates=False,
        native_model_build_included_in_first_call=True,native_model_build_reported_separately=True,
        no_repeated_solve_of_same_unchanged_problem=True,
        all_cpu_input_packets_prepared_before_timing=True,
        timing_includes=['fresh frozen W1 encoding','learned R and auxiliary heads OR 20-path scenario generation and persistent CVaR MILP','causal assembly and first-action extraction'],
        timing_excludes=['checkpoint loading','CPU input construction','disk I/O','realized outcome scoring'],
        W0_is_inherited_by_W1_not_a_second_forward=True,
        realized_cost_or_SLA_evaluated=False,closed_loop_execution=False,first_control_only_extracted=True,
        no_annual_outcome_or_latency_percentile_inference=True,
        short_H15_to4_results_allowed_in_primary_comparison=False,
        prior_2025_already_evaluated=True,parameters_or_checkpoints_selected_from_test=False,
        input_sha256={str(f):sha(f) for f in files})
    old=read_json(root/'protocol.json')
    if old:
        a=dict(old);b=dict(protocol);a.pop('frozen_at');b.pop('frozen_at')
        if a!=b:raise RuntimeError('Frozen H96 timing inputs changed')
        return old
    save(root/'protocol.json',protocol);return protocol


def prepare_inputs(root,record,wc):
    marker=dict(schema=common.core.base.FROZEN_MARKER_SCHEMA,status='FROZEN_FOR_ONE_SHOT_FINAL_EVALUATION',
        final_year=2025,best_checkpoint_frozen=True,model_selection_frozen=True,thresholds_frozen=True,
        data_statistics_frozen=True,development_evaluation_complete=True,training_closed=True,
        locked_final_opened=False,one_shot_runs_allowed=1,final_results_may_modify_selection=False,
        expected_evaluable_days=362,best_checkpoint=record['checkpoint'],run_id=record.get('checkpoint_run_id',record['run_id']),
        one_shot_id='six-fixed-H96-timing-20260913',evaluation_role='supplemental_full_H96_timing_only',
        prior_2025_already_evaluated=True,comparison_design='same_inputs_H96_twenty_joint_scenarios_timing_only',
        protocol_sha256=sha(root/'protocol.json'))
    save(root/'frozen_evaluation_marker.json',marker)
    corpus=common.core.base.build_sealed_final_corpus(config=wc,repository_root=REPO,frozen_evaluation_marker=marker)
    common.core.base._validate_sealed_corpus(corpus,marker)
    selected={}
    for batch in corpus.batches(common.core.base.FINAL_SPLIT,1,device='cpu',shuffle_seed=None,origin=0,
            require_h96=False,include_next_day=False):
        date=str(batch['operating_dates'][0])
        if date in DATES:selected[date]=batch
    if set(selected)!=set(DATES):raise RuntimeError('Incomplete fixed six-date inventory')
    return selected


def run(root,device):
    record,wc,mc=common.input_record('legacy_seed20240813')
    protocol=freeze(root,record,wc,mc)
    if read_json(root/'completed.json',{}).get('status')=='completed':return
    save(root/'progress.json',dict(status='preparing_all_inputs_before_timing',completed_cases=0,total_cases=6,updated_at=now()))
    inputs_by_date=prepare_inputs(root,record,wc)
    model=common.runtime(record,wc,mc,device)
    controller=previous.solver(model)
    variant=common.core.registered_interventions()['full']
    risk=common.core.base.RiskConditionPacket.defaults(1,device=device,horizon=96,stage=1,action_range=1.)
    cases=[]
    with torch.inference_mode(),common.core.base._precision(device):
        for case in protocol['cases']:
            date=case['date'];batch=on_device(inputs_by_date[date],device);samples=[]
            save(root/'progress.json',dict(status='measuring_R_H96',date=date,completed_cases=len(cases),total_cases=6,updated_at=now()))
            for repeat in range(35):
                def call():
                    packet=common.encode(model,batch)
                    output=common.core.evaluation._execute_variant(model,packet,risk,variant)
                    return previous.first_from_actions(output.actions,output.day_ahead_commitment,output.target_soc,0)
                _,ms=common.core.base._timed(device,call)
                if repeat>=5:samples.append(ms)
            for _ in range(2):previous.scenario_call(model,batch,risk,device)
            save(root/'progress.json',dict(status='measuring_SMPC20_H96',date=date,completed_cases=len(cases),total_cases=6,
                per_call_solver_upper_limit_seconds=600.,updated_at=now()))
            common.core.evaluation._synchronize(device);started=time.perf_counter()
            packet,inputs,encode_ms,generation_ms=previous.scenario_call(model,batch,risk,device)
            if inputs['horizon']!=96 or inputs['origin']!=0:raise RuntimeError('Short horizon entered primary timing')
            begun=time.perf_counter();result=previous.optimize(controller,inputs);optimize_ms=(time.perf_counter()-begun)*1000
            begun=time.perf_counter();assembled=previous.assemble_current(model,packet,risk,result,inputs)
            common.core.evaluation._synchronize(device)
            assembly_ms=(time.perf_counter()-begun)*1000;total_ms=(time.perf_counter()-started)*1000
            if assembled is None:raise RuntimeError('No executable first action, including fallback')
            row=dict(**case,model_run_id=record['run_id'],r_warmups=5,r_repeats=30,mpc_measured_calls=1,
                r_raw_samples_ms=samples,r_mean_ms=float(np.mean(samples)),r_median_ms=float(np.median(samples)),r_p95_ms=float(np.quantile(samples,.95)),
                mpc_full_chain_ms=total_ms,mpc_encode_ms=encode_ms,mpc_scenario_generation_ms=generation_ms,
                mpc_optimize_wrapper_ms=optimize_ms,mpc_assembly_ms=assembly_ms,mpc_solverstats=result['solverstats'],
                scenario_diagnostics=inputs['diagnostics'],realized_outcome_scoring_performed=False,completed_at=now())
            cases.append(row)
            save(root/'timing_cases.json',dict(status='running',cases=cases,completed_cases=len(cases),total_cases=6))
            print(json.dumps(dict(date=date,H=96,R_mean_ms=row['r_mean_ms'],SMPC_full_ms=total_ms,
                status=result['solverstats']['status'],actual_gap=result['solverstats']['actual_gap'])),flush=True)
    save(root/'timing_cases.json',dict(status='completed',cases=cases,completed_cases=6,total_cases=6,completed_at=now()))
    save(root/'completed.json',dict(status='completed',cases=6,mpc_calls=6,R_measured_calls=180,H=96,
        protocol_sha256=sha(root/'protocol.json'),completed_at=now()))
    save(root/'progress.json',dict(status='completed',completed_cases=6,total_cases=6,updated_at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--device',default='cuda');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    try:run(a.root,torch.device(a.device))
    except Exception as e:
        save(a.root/'error.json',dict(error=repr(e),updated_at=now()));raise
