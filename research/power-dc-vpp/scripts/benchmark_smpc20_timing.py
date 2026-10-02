"""Measure risk-aware 20-scenario MPC without substituting point-forecast MPC.

This records initial-model and updated-origin calls. It returns only the current
control, and does not claim that a complete closed-loop day was evaluated.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import pyarrow.dataset  # noqa: E402,F401
import numpy as np
import torch
import run_matched_mpc_benchmark as common
from dcvpp.vpp_world_moe.scenario_mpc import ScenarioMPC20
from dcvpp.vpp_world_moe.scenario_mpc_inputs import generate_w1_joint_scenarios
from dcvpp.vpp_world_moe.r_ecmoe_evaluation import _safe_raw
from multiseed_common import read_json,write_json,now

ROOT=REPO/'artifacts/r_ecmoe_smpc20_timing_20260912'
CASES=((0,96),(1,95))
SCENARIO_SEED=20240813


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    return value


def save(path,value):
    value=jsonable(value)
    json.dumps(value,allow_nan=False)
    write_json(path,value)


def solver(model,time_limit=600.0):
    config=model.world_model.config
    return ScenarioMPC20(time_limit_seconds=time_limit,mip_gap=0.0,threads=1,accepted_timeout_gap=.05,
        battery_degradation_weight=float(config.battery_degradation_weight),
        sla_weight=float(config.unserved_power_weight),curtailment_weight=1.0,
        deadline_intervals=model.config.deadline_intervals,
        action_residual_fraction=model.config.action_residual_fraction)


def first_from_actions(actions,day_ahead,target_soc,origin):
    return {name:jsonable(getattr(actions,name)[0,origin]) for name in
            ['battery','compute','grid','pv_used','pv_curtailed','unserved_power']} | {
            'day_ahead':jsonable(day_ahead[0,origin]),'target_soc':jsonable(target_soc[0,origin])}


def assemble_current(model,packet,risk,result,inputs):
    """Pack only the shared first command through the existing physical map.

    The scenario mean is a causal forecast, not a realized trajectory. Future
    filler commands are masked out and are never passed off as an MPC plan.
    """
    first=result.get('first_action')
    if first is None:return None
    device=packet.state_latent.device;origin=inputs['origin']
    forecasts={name:torch.as_tensor(np.einsum('s,stn->tn',inputs['weights'],array),
        dtype=torch.float32,device=device)[None] for name,array in inputs['scenarios'].items()}
    dart=forecasts['da']-forecasts['rt']
    public=torch.cat((forecasts['da'],forecasts['rt'],dart,forecasts['arrivals'].sum(-1,keepdim=True),
        forecasts['pv'],forecasts['pue']),dim=-1)
    reference=replace(packet.reference,da=forecasts['da'],rt=forecasts['rt'],dart=dart,
        arrivals=forecasts['arrivals'],pv=forecasts['pv'],pue=forecasts['pue'],public_21=public)
    local_packet=replace(packet,reference=reference)
    raw=_safe_raw(local_packet)
    def tensor(value):return torch.as_tensor(value,device=device,dtype=torch.float32)
    eps=1e-6
    def logit(value):
        value=value.clamp(eps,1-eps)
        return torch.log(value)-torch.log1p(-value)
    c=packet.constraints
    battery=tensor(first['battery'])/c.battery_power_limit[0].clamp_min(eps)
    compute=tensor(first['compute'])/c.compute_capacity[0].clamp_min(eps)
    raw['battery'][0,origin]=torch.atanh(battery.clamp(-1+eps,1-eps))
    raw['compute'][0,origin]=logit(compute)
    raw['pv'][0,origin]=logit(tensor(first['pv_fraction']))-2.0
    raw['target_soc'][0,origin]=logit(tensor(first['target_soc'])/c.battery_energy_capacity[0].clamp_min(eps))
    residual=(tensor(first['day_ahead'])-packet.previous_actions[0,origin,:,0]) / (
        model.config.action_residual_fraction*c.grid_import_limit[0].clamp_min(eps))
    raw['day_ahead'][0,origin]=torch.atanh(residual.clamp(-1+eps,1-eps))
    first_risk=replace(risk,horizon=torch.ones_like(risk.horizon))
    assembled=model.assembler(raw,local_packet,first_risk)
    if int(assembled.decision_active_mask.sum())!=1:raise RuntimeError('Only the current first action may execute')
    return first_from_actions(assembled.actions,assembled.day_ahead_commitment,assembled.target_soc,origin)


def prepare_model(device):
    record,wc,mc=common.input_record(common.RUN_IDS[0])
    model=common.runtime(record,wc,mc,device)
    return record,wc,mc,model


def make_protocol(root,record,wc,mc):
    path=root/'protocol.json'
    sources=[Path(__file__),REPO/'src/dcvpp/vpp_world_moe/scenario_mpc.py',
        REPO/'src/dcvpp/vpp_world_moe/scenario_mpc_inputs.py',
        Path(record['checkpoint']),Path(record['w1_checkpoint']),Path(record['config_source'])]
    hashes={str(p):common.core.base._sha256(p) for p in sources}
    payload={'schema':'dcvpp.smpc20.timing.protocol.v1','primary_baseline':'W1-SMPC-20',
        'model_record':record,'world_config':wc,'r_model_config':mc,'input_sha256':hashes,
        'scenario_count':20,'scenario_seed':SCENARIO_SEED,'scenario_weights':'native frozen-W1 Monte Carlo weights',
        'scenario_source':'native W1 latent joint rollout, decoded with causal issue-time APIC anchors; not original R3 scenario head',
        'cvar_alpha':.95,'expected_cost_weight':1.0,'cvar_weight':.25,'target_relative_gap':0.0,
        'absolute_gap_tolerance':1e-10,'time_limit_seconds_per_call':600.0,'threads':1,
        'accepted_timeout_relative_gap':.05,'backend':'native persistent HiGHS',
        'cases':[{'date':'2025-01-01','stage':1,'origin':o,'horizon':h} for o,h in CASES],
        'r_warmups_per_case':5,'r_measured_repeats_per_case':30,'mpc_measured_calls_per_case':1,
        'mpc_second_case_role':'same persistent model updated to next origin on historical inputs',
        'first_action_only':True,'complete_closed_loop_day_executed':False,
        'empirical_cost_performance_comparison_completed':False,
        'same_current_R_physical_and_cost_contract':True,
        'legacy_implementation_changes':['current-period truth is not revealed before decisions',
            'same current R deadline and cost coefficients','same current R battery budgets and no added terminal SOC reserve',
            'grid is derived under the current R action contract rather than an additional nonanticipative requested control',
            'native W1 joint scenarios replace the old R3 scenario-model interface','native HiGHS replaces unavailable full-size Gurobi license'],
        'timing_includes':['fresh W1 encoding','20-path scenario generation','initial model loading or parameter update',
            'risk-aware MILP solve','current first-control extraction and shared assembly'],
        'timing_excludes':['checkpoint loading','disk I/O','input batch construction','realized outcome scoring'],
        'initial_model_construction_reported_separately':True,'prior_2025_already_evaluated':True,
        'training_performed':False,'checkpoint_selection_performed':False,'test_used_for_parameter_tuning':False}
    prior=read_json(path)
    if prior is not None and prior!=payload:raise RuntimeError('Frozen risk-MPC benchmark inputs changed')
    if prior is None:save(path,payload)
    return payload


def scenario_call(model,batch,risk,device):
    packet,encode_ms=common.core.base._timed(device,lambda:common.encode(model,batch))
    inputs,generation_ms=common.core.base._timed(device,lambda:generate_w1_joint_scenarios(model,packet,risk,
        da_anchor=batch['forecast_anchors']['da'],da_residual_cap=batch['forecast_residual_caps']['da'],
        scenario_seed=SCENARIO_SEED))
    return packet,inputs,encode_ms,generation_ms


def final_corpus(root,record,wc):
    marker={'schema':common.core.base.FROZEN_MARKER_SCHEMA,'status':'FROZEN_FOR_ONE_SHOT_FINAL_EVALUATION',
        'final_year':2025,'best_checkpoint_frozen':True,'model_selection_frozen':True,'thresholds_frozen':True,
        'data_statistics_frozen':True,'development_evaluation_complete':True,'training_closed':True,
        'locked_final_opened':False,'one_shot_runs_allowed':1,'final_results_may_modify_selection':False,
        'expected_evaluable_days':362,'best_checkpoint':record['checkpoint'],
        'run_id':record.get('checkpoint_run_id',record['run_id']),
        'one_shot_id':f'smpc20-timing-20260912-{record["run_id"]}',
        'evaluation_role':'supplemental_risk_scenario_mpc_timing','prior_2025_already_evaluated':True,
        'comparison_design':'single_call_risk_MPC_timing_not_full_day_performance',
        'separately_retrained':False,'protocol_sha256':common.core.base._sha256(root/'protocol.json')}
    save(root/'frozen_evaluation_marker.json',marker)
    corpus=common.core.base.build_sealed_final_corpus(config=wc,repository_root=REPO,frozen_evaluation_marker=marker)
    common.core.base._validate_sealed_corpus(corpus,marker)
    return corpus


def optimize(controller,inputs):
    return controller.solve(inputs['scenarios'],inputs['weights'],inputs['constraints'],
        inputs['previous_da'],inputs['origin'],stage=inputs['stage'])


def pilot(root,device,pilot_seconds):
    record,wc,mc,model=prepare_model(device)
    corpus=common.WorldMoECorpus(wc,REPO)
    batch=next(iter(corpus.batches('validation',1,device=device,shuffle_seed=None,max_days=1,
        origin=0,require_h96=False,include_next_day=False)))
    if not str(batch['operating_dates'][0]).startswith('2024-'):raise RuntimeError('Pilot requires 2024')
    risk=common.core.base.RiskConditionPacket.defaults(1,device=device,horizon=96,stage=1)
    controller=solver(model,pilot_seconds)
    with torch.inference_mode(),common.core.base._precision(device):
        packet,inputs,encode_ms,generation_ms=scenario_call(model,batch,risk,device)
        save(root/'pilot_progress.json',{'status':'optimizing_2024','scenario_count':20,'horizon':96,
            'pilot_solver_budget_seconds':pilot_seconds,'formal_budget_unchanged_seconds':600,'updated_at':now()})
        start=time.perf_counter();result=optimize(controller,inputs);decision_ms=(time.perf_counter()-start)*1000
        assembled=assemble_current(model,packet,risk,result,inputs)
    save(root/'pilot_2024.json',{'pipeline_executed':True,'scope':'development-only technical pilot; not final latency measurement',
        'date':str(batch['operating_dates'][0]),'scenario_count':20,'horizon':96,
        'pilot_solver_budget_seconds':pilot_seconds,'formal_solver_budget_seconds':600,'encode_ms':encode_ms,
        'scenario_generation_ms':generation_ms,'decision_ms':decision_ms,'scenario_diagnostics':inputs['diagnostics'],
        'solverstats':result['solverstats'],'assembled_first_action':assembled,'completed_at':now()})
    save(root/'pilot_progress.json',{'status':'completed','formal_latency_measurement':False,'updated_at':now()})
    print(json.dumps(jsonable({'pilot_pipeline_executed':True,'encode_ms':encode_ms,'generation_ms':generation_ms,
        'solverstats':result['solverstats']}),ensure_ascii=True))


def run(root,device):
    if not read_json(root/'pilot_2024.json',{}).get('pipeline_executed'):
        raise RuntimeError('Run the development-only risk-MPC pipeline pilot first')
    record,wc,mc,model=prepare_model(device)
    protocol=make_protocol(root,record,wc,mc)
    corpus=final_corpus(root,record,wc)
    controller=solver(model)
    variant=common.core.registered_interventions()['full']
    results=[]
    with torch.inference_mode(),common.core.base._precision(device):
        for origin,horizon in CASES:
            batch=next(iter(corpus.batches(common.core.base.FINAL_SPLIT,1,device=device,shuffle_seed=None,max_days=None,
                origin=origin,require_h96=False,include_next_day=False)))
            if str(batch['operating_dates'][0])!='2025-01-01':raise RuntimeError('Frozen timing date changed')
            risk=common.core.base.RiskConditionPacket.defaults(1,device=device,horizon=horizon,stage=1)
            samples=[]
            save(root/'progress.json',{'status':'measuring_R','origin':origin,'horizon':horizon,'updated_at':now()})
            for repeat in range(35):
                def neural_call():
                    packet=common.encode(model,batch)
                    output=common.core.evaluation._execute_variant(model,packet,risk,variant)
                    return first_from_actions(output.actions,output.day_ahead_commitment,output.target_soc,origin)
                first,elapsed=common.core.base._timed(device,neural_call)
                if repeat>=5:samples.append(elapsed)
            # Warm the neural scenario kernels without solving or altering the
            # optimization problem. No MPC incumbent is supplied by R.
            for _ in range(2):scenario_call(model,batch,risk,device)
            save(root/'progress.json',{'status':'measuring_full_risk_MPC_pipeline','origin':origin,'horizon':horizon,
                'scenario_count':20,'target_relative_gap':0.0,'solver_budget_seconds':600,
                'pipeline_started_at':now(),'updated_at':now()})
            common.core.evaluation._synchronize(device);total_start=time.perf_counter()
            packet,inputs,encode_ms,generation_ms=scenario_call(model,batch,risk,device)
            optimize_start=time.perf_counter();result=optimize(controller,inputs)
            optimize_ms=(time.perf_counter()-optimize_start)*1000
            assembly_start=time.perf_counter();assembled=assemble_current(model,packet,risk,result,inputs)
            common.core.evaluation._synchronize(device)
            assembly_ms=(time.perf_counter()-assembly_start)*1000
            total_ms=(time.perf_counter()-total_start)*1000
            row={'case_id':f'2025-01-01_RT_o{origin}_h{horizon}','origin':origin,'horizon':horizon,
                'model_run_id':record['run_id'],'r_repeats':30,'r_warmups':5,'r_raw_samples_ms':samples,
                'r_mean_ms':float(np.mean(samples)),'r_median_ms':float(np.median(samples)),
                'r_p95_ms':float(np.percentile(samples,95)),'r_first_action':first,
                'mpc_measured_calls':1,'mpc_full_chain_ms':total_ms,'mpc_encode_ms':encode_ms,
                'mpc_scenario_generation_ms':generation_ms,'mpc_optimization_pipeline_ms':optimize_ms,
                'mpc_shared_assembly_ms':assembly_ms,'mpc_solverstats':result['solverstats'],
                'mpc_requested_first_action':result.get('first_action'),'mpc_assembled_first_action':assembled,
                'scenario_diagnostics':inputs['diagnostics'],'full_day_rolling_completed':False,
                'measured_input_case_not_latency_population':True,'recorded_at':now()}
            results.append(row);save(root/'timing_cases.json',results)
            save(root/'progress.json',{'status':'case_completed','cases_completed':len(results),
                'cases_planned':2,'origin':origin,'updated_at':now()})
    save(root/'completed.json',{'timing_calls_completed':True,'cases_completed':2,'primary_baseline':'W1-SMPC-20',
        'full_day_rolling_completed':False,'annual_performance_comparison_completed':False,
        'controller_outcome_equivalence_established':False,'completed_at':now()})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--device',default='cuda')
    parser.add_argument('--pilot',action='store_true');parser.add_argument('--pilot-seconds',type=float,default=10.0)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv);root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    try:
        if args.pilot:pilot(root,device,args.pilot_seconds)
        elif args.execute:run(root,device)
        else:parser.error('Use --pilot or --execute')
    except BaseException as exc:
        save(root/'failure.json',{'error':f'{type(exc).__name__}: {exc}','updated_at':now(),
            'partial_evidence_preserved':True});raise
    return 0


if __name__=='__main__':raise SystemExit(main())
