"""Two fixed price-selected, genuinely state-propagated 12-step RT episodes.

No 2025 value is supplied to the controller before its corresponding action.
The initial DA schedule is shared and fixed; this is not DA market clearing.
"""
from __future__ import annotations
import argparse
from dataclasses import replace, fields, is_dataclass
import gc
import json
import sys
import time
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import pyarrow.dataset  # noqa: F401
import numpy as np
import torch
import run_matched_mpc_benchmark as common
import benchmark_smpc20_timing as timing
from train_grid_sensitivity import sha, ROOT, WEIGHTS
from multiseed_common import read_json,write_json,now
from dcvpp.vpp_world_moe.r_ecmoe import RiskConditionPacket
from dcvpp.vpp_world_moe.scenario_mpc_sensitivity import ScenarioMPCSensitivity
from dcvpp.vpp_world_moe.scenario_mpc_inputs import generate_w1_joint_scenarios
from dcvpp.vpp_world_moe.closed_loop_replay import ClosedLoopEpisode


def save(path,payload):
    timing.save(path,payload)


def make_final_corpus(root,record,wc):
    marker=dict(schema=common.core.base.FROZEN_MARKER_SCHEMA,status='FROZEN_FOR_ONE_SHOT_FINAL_EVALUATION',
        final_year=2025,best_checkpoint_frozen=True,model_selection_frozen=True,thresholds_frozen=True,
        data_statistics_frozen=True,development_evaluation_complete=True,training_closed=True,
        locked_final_opened=False,one_shot_runs_allowed=1,final_results_may_modify_selection=False,
        expected_evaluable_days=362,best_checkpoint=record['checkpoint'],
        run_id=record.get('checkpoint_run_id',record['run_id']),one_shot_id='closedloop-sensitivity-20260913',
        evaluation_role='supplemental_two_case_closedloop_and_fixed_weight_sensitivity',
        prior_2025_already_evaluated=True,comparison_design='two_price_selected_12_step_RT_episodes',
        separately_retrained=False,all_short_adaptation_checkpoints_frozen=True,
        protocol_sha256=sha(root/'execution_protocol.json'))
    save(root/'frozen_evaluation_marker.json',marker)
    corpus=common.core.base.build_sealed_final_corpus(config=wc,repository_root=REPO,frozen_evaluation_marker=marker)
    common.core.base._validate_sealed_corpus(corpus,marker)
    corpus._closedloop_case_dates={case['date'] for case in read_json(root/'execution_protocol.json')['cases']}
    return corpus


def policy_list(root):
    return [dict(policy_id='R_original',family='R-ECMoE',lambda_grid=None,checkpoint=None)]+[
        dict(policy_id=f'R_gamma_{lam:g}',family='R-ECMoE',lambda_grid=lam,
             checkpoint=read_json(root/'adaptations'/f'lambda_{lam:g}'/'completed.json')['checkpoint']) for lam in WEIGHTS]+[
        dict(policy_id=f'SMPC_gamma_{lam:g}',family='W1-SMPC-20',lambda_grid=lam,checkpoint=None) for lam in WEIGHTS]


def freeze(root,record):
    if read_json(root/'training_completed.json',{}).get('status')!='completed':raise RuntimeError('All four train-only arms must finish first')
    if read_json(root/'pilot_closedloop_2024.json',{}).get('status')!='passed':raise RuntimeError('Technical closed-loop pilot must pass first')
    sources=[Path(__file__),REPO/'src/dcvpp/vpp_world_moe/closed_loop_replay.py',
        REPO/'src/dcvpp/vpp_world_moe/scenario_mpc_sensitivity.py',REPO/'src/dcvpp/vpp_world_moe/scenario_mpc.py',
        REPO/'src/dcvpp/vpp_world_moe/scenario_mpc_inputs.py',root/'case_protocol.json',root/'analysis_schema_v1_1.json',root/'training_protocol.json',
        Path(record['checkpoint']),Path(record['w1_checkpoint'])]
    policies=policy_list(root)
    sources += [Path(p['checkpoint']) for p in policies if p['checkpoint']]
    protocol=dict(schema='dcvpp.closedloop-sensitivity.execution.v1',frozen_at=now(),
        cases=read_json(root/'case_protocol.json')['cases'],policies=policies,source_record=record,
        scenario_count=20,scenario_seed=20240813,uniform_native_W1_scenarios=True,
        risk_objective='expected raw composite cost + 0.25 CVaR95(raw composite) + gamma * train_daily_D * H/96 * expected grid penalty',
        scenario_recourse='current battery/compute/PV fraction shared; future scenario recourse',
        planning_horizon='min(16,96-origin)',executed_steps_per_case=12,
        threads=1,target_relative_gap=0.,absolute_gap_tolerance=1e-10,time_limit_seconds_per_solve=600.,
        accepted_timeout_gap=.05,prior_2025_already_evaluated=True,cases_are_posthoc_stress_examples=True,
        test_used_for_fitting_or_checkpoint_or_weight_selection=False,
        full_day_or_annual_closedloop=False,RT_only_fixed_shared_DA_schedule=True,
        matched_initial_states=True,policy_specific_executed_state_feedback=True,
        timing_includes=['fresh frozen W1','R policy and auxiliary heads or native 20-path scenarios and persistent CVaR MILP','causal assembly and first-action extraction'],
        timing_excludes=['checkpoint loading','disk I/O','prepared-input construction and feedback rewrite','realized environment execution and cohort accounting'],
        per_episode_first_native_build_included=True,model_warmups_on_2024=5,scenario_warmups_on_2024=2,
        R_adaptation_scope='one existing R130 lineage; four fixed train-only epochs per gamma with matched gamma0 continuation',
        sample_size='18 policy-case trajectories, 216 executed decisions, 96 SMPC calls; two nonrandomly selected days',
        input_sha256={str(p):sha(p) for p in sources})
    prior=read_json(root/'execution_protocol.json')
    if prior:
        expected=dict(prior);expected.pop('frozen_at');candidate=dict(protocol);candidate.pop('frozen_at')
        if expected!=candidate:raise RuntimeError('Frozen execution protocol changed')
        return prior
    save(root/'execution_protocol.json',protocol);return protocol


def make_solver(model,scale,lam):
    c=model.world_model.config
    return ScenarioMPCSensitivity(train_cost_scale=scale,grid_weight=float(lam),planning_horizon=16,
        time_limit_seconds=600.,mip_gap=0.,threads=1,accepted_timeout_gap=.05,
        battery_degradation_weight=float(c.battery_degradation_weight),sla_weight=float(c.unserved_power_weight),
        curtailment_weight=1.,deadline_intervals=model.config.deadline_intervals,
        action_residual_fraction=model.config.action_residual_fraction)


def to_device(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if is_dataclass(value):return replace(value,**{f.name:to_device(getattr(value,f.name),device) for f in fields(value)})
    if isinstance(value,dict):return {k:to_device(v,device) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(to_device(v,device) for v in value)
    if isinstance(value,list):return [to_device(v,device) for v in value]
    return value


def exact_batch(corpus,split,date,origin,device):
    records=corpus.records(split,require_h96=False)
    indices=[i for i,r in enumerate(records) if str(r['operating_date'])==date]
    if len(indices)!=1:raise RuntimeError(f'Expected unique date {date}')
    if split==common.core.base.FINAL_SPLIT:
        # Keep the sealed loader's complete-ordered-split contract intact. Its
        # formal target metadata is preserved; only the two preregistered case
        # packets are retained, and no other date receives model inference.
        allowed=getattr(corpus,'_closedloop_case_dates',set())
        if date not in allowed:raise RuntimeError('Date is not in the frozen two-case protocol')
        cache=getattr(corpus,'_closedloop_cpu_packet_cache',{})
        if origin not in cache:
            selected={}
            for prepared in corpus.batches(split,1,device='cpu',shuffle_seed=None,origin=origin,
                    require_h96=False,include_next_day=False):
                key=str(prepared['operating_dates'][0])
                if key in allowed:selected[key]=prepared
            if set(selected)!=allowed:raise RuntimeError('Incomplete fixed-case packet cache')
            cache[origin]=selected;corpus._closedloop_cpu_packet_cache=cache
        batch=to_device(cache[origin][date],device)
    else:
        batch=next(iter(corpus.batches(split,1,device=device,shuffle_seed=None,origin=origin,
            batch_ordinals=indices,require_h96=False,include_next_day=False)))
    assert str(batch['operating_dates'][0])==date
    return batch,records[indices[0]]['base_record']['state_sequence']


def risk_for(origin,device):return RiskConditionPacket.defaults(1,device=device,horizon=min(16,96-origin),stage=1,action_range=1.)


def neural(model,batch,risk,device):
    def call():
        packet=common.encode(model,batch)
        decision=common.core.evaluation._execute_variant(model,packet,risk,common.core.registered_interventions()['full'])
        origin=int(batch['world_input'].origin_index[0])
        first=timing.first_from_actions(decision.actions,decision.day_ahead_commitment,decision.target_soc,origin)
        return decision,first
    return common.core.base._timed(device,call)


def smpc(model,batch,risk,controller,episode,device):
    common.core.evaluation._synchronize(device);started=time.perf_counter()
    packet,inputs,encode_ms,generation_ms=timing.scenario_call(model,batch,risk,device)
    history=episode.mpc_history()
    result=controller.solve(inputs['scenarios'],inputs['weights'],inputs['constraints'],inputs['previous_da'],inputs['origin'],stage=1,
        horizon=inputs['horizon'],**history)
    assembled=timing.assemble_current(model,packet,risk,result,inputs)
    if assembled is None:raise RuntimeError('MPC omitted even its registered fallback action')
    common.core.evaluation._synchronize(device);elapsed=(time.perf_counter()-started)*1000
    stats=dict(result['solverstats'],encode_ms=encode_ms,scenario_generation_ms=generation_ms)
    return packet,result,elapsed,stats


def episode_run(root,model,corpus,split,case,policy,device,pilot=False):
    date=case['date'];start=case['start_origin'];stop=case['stop_origin_exclusive']
    batch,history=exact_batch(corpus,split,date,start,device)
    scale=float(batch['train_frozen_baseline_daily_cost'].detach().cpu())
    episode=ClosedLoopEpisode(model,corpus.statistics,policy['policy_id'],date,batch['world_input'],historical_state_sequence=history)
    controller=make_solver(model,scale,policy['lambda_grid']) if policy['family']=='W1-SMPC-20' else None
    rows=[];started=time.perf_counter()
    with torch.inference_mode(),common.core.base._precision(device):
        for origin in range(start,stop):
            batch,_=exact_batch(corpus,split,date,origin,device)
            batch=dict(batch,world_input=episode.prepare_packet(batch['world_input']))
            risk=risk_for(origin,device)
            save(root/('pilot_progress.json' if pilot else 'execution_progress.json'),dict(status='deciding',policy_id=policy['policy_id'],date=date,origin=origin,
                completed_steps=len(rows),episode_steps=stop-start,updated_at=now()))
            if controller is None:
                (decision,_),elapsed=neural(model,batch,risk,device)
                # Only now admit the current realized 21-channel vector.
                actual=batch['stochastic_target'][0,origin].detach().clone()
                row=episode.execute_neural(decision,risk,actual);stats=None
            else:
                packet,result,elapsed,stats=smpc(model,batch,risk,controller,episode,device)
                actual=batch['stochastic_target'][0,origin].detach().clone()
                row=episode.execute_mpc(packet,risk,result['first_action'],actual)
            row.update(policy_id=policy['policy_id'],family=policy['family'],lambda_grid=policy['lambda_grid'],
                case_type=case.get('case_type','technical_pilot'),case_id=case.get('case_id',date),date=date,
                horizon=min(16,96-origin),full_chain_ms=elapsed,solverstats=stats,train_daily_cost_scale=scale)
            rows.append(row)
            folder=root/('pilot_steps' if pilot else 'partial');folder.mkdir(exist_ok=True)
            with (folder/f'{date}_{policy["policy_id"]}.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(timing.jsonable(row),allow_nan=False)+'\n')
            print(json.dumps(dict(date=date,policy=policy['policy_id'],origin=origin,full_chain_ms=elapsed)),flush=True)
    summary=episode.summary()
    payload=dict(schema='dcvpp.closedloop-sensitivity.episode.v1',status='completed',completed=True,
        case=case,**policy,rows=rows,summary=summary,final_state=episode.state.to_dict(),
        train_daily_cost_scale=scale,elapsed_seconds=time.perf_counter()-started,completed_at=now())
    if not pilot:
        folder=root/'episodes';folder.mkdir(exist_ok=True);save(folder/f'{date}_{policy["policy_id"]}.json',payload)
    return payload


def run(root,device,pilot=False):
    record,wc,mc=common.input_record('legacy_seed20240813');model=common.runtime(record,wc,mc,device)
    original={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('world_model.')}
    development=common.WorldMoECorpus(wc,REPO)
    warm_batch,_=exact_batch(development,'validation','2024-01-01',81,device)
    warm_risk=risk_for(81,device)
    with torch.inference_mode(),common.core.base._precision(device):
        for _ in range(5):neural(model,warm_batch,warm_risk,device)
        for _ in range(2):timing.scenario_call(model,warm_batch,warm_risk,device)
    if pilot:
        case=dict(date='2024-01-01',start_origin=81,stop_origin_exclusive=83,case_type='technical_pilot')
        results=[]
        for policy in [dict(policy_id='R_original',family='R-ECMoE',lambda_grid=None),dict(policy_id='SMPC_gamma_0',family='W1-SMPC-20',lambda_grid=0.)]:
            results.append(episode_run(root,model,development,'validation',case,policy,device,pilot=True))
        save(root/'pilot_closedloop_2024.json',dict(status='passed',split='validation',year=2024,results=results,completed_at=now()))
        return
    protocol=freeze(root,record)
    del development,warm_batch;gc.collect()
    corpus=make_final_corpus(root,record,wc)
    for policy in protocol['policies']:
        state=original
        if policy.get('checkpoint'):state=torch.load(policy['checkpoint'],map_location='cpu',weights_only=False)['r_state_dict']
        missing,unexpected=model.load_state_dict(state,strict=False)
        if unexpected or any(not key.startswith('world_model.') for key in missing):raise RuntimeError('Unexpected adaptation state mismatch')
        model.eval()
        for case in protocol['cases']:
            path=root/'episodes'/f'{case["date"]}_{policy["policy_id"]}.json'
            if read_json(path,{}).get('completed'):continue
            episode_run(root,model,corpus,common.core.base.FINAL_SPLIT,case,policy,device)
    files=list((root/'episodes').glob('*.json'))
    counts=[len(read_json(p)['rows']) for p in files]
    if len(files)!=18 or sum(counts)!=216 or set(counts)!={12}:raise RuntimeError('Incomplete experiment inventory')
    save(root/'execution_completed.json',dict(status='completed',episodes=18,executed_decisions=216,SMPC_solves=96,
        completed_at=now(),protocol_sha256=sha(root/'execution_protocol.json')))
    save(root/'execution_progress.json',dict(status='completed',episodes=18,executed_decisions=216,updated_at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--device',default='cuda')
    a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    try:run(a.root,torch.device(a.device),a.pilot)
    except Exception as error:
        save(a.root/('pilot_error.json' if a.pilot else 'execution_error.json'),dict(status='error',error=repr(error),updated_at=now()))
        raise
