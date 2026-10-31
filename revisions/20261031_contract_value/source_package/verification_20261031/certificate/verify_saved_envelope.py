"""Solver-free replay of saved annual original-cost duals and robust schedules."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from envelope_certificate import (q,rational_text,reference_envelopes,original_points,
    exact_dual,up_units,down_units)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    args=p.parse_args();old=json.loads((args.results/'envelope_results.json').read_text())
    digest=hashlib.sha256(args.inputs.read_bytes()).hexdigest()
    if digest!=old['input_sha256']:raise ValueError('Annual input hash mismatch')
    with np.load(args.inputs,allow_pickle=False) as f:z={k:f[k] for k in f.files}
    config=args.repo/'research/R2_PAPER_INTEGRATION_20260923/configs/economic_constraints.json'
    if hashlib.sha256(config.read_bytes()).hexdigest()!=old['constraints_sha256']:raise ValueError('Physical constraints hash mismatch')
    constraints=json.loads(config.read_text())['constraints'];n=len(z['reference_compute']);scale=10**12
    report={'scope':'Solver-free exact replay of saved NEW annual envelope certificates and exact robust schedules; unknown-admission repair remains a uniform constructive proof',
            'solver_called':False,'dual_records':[],'robust_schedule_records':[],
            'input_sha256':digest,'source_admissions_recovered':False,'original_29oct_certificates_reproduced':False}
    for site in range(4):
        b,R,clear,E,Amin,Amax,runs=reference_envelopes(z['reference_compute'][:,site])
        points=[];caps=[];funcs=[];baseline_lo=0;baseline_hi=0
        for t in range(n):
            pts,base,cap,cost=original_points(z['rt_lmp_usd_mwh'][t,site],z['pue'][t,site],z['pv_mw'][t,site],
                z['idle_mw'][site],z['work_power_mw'][site],z['reference_compute'][t,site],constraints['grid_import_limit'][site])
            points.append(pts);caps.append(cap);funcs.append(cost)
            baseline_lo+=down_units(base,scale);baseline_hi+=up_units(base,scale)
        for record in (r for r in old['site_results'] if r['site']==site):
            hours=record['window_hours'];D=4*hours
            file=args.results/f'outer_w{hours}_zone{site}.npz'
            if hashlib.sha256(file.read_bytes()).hexdigest()!=record['saved_multiplier_sha256']:raise ValueError('Multiplier artifact hash mismatch')
            with np.load(file,allow_pickle=False) as f:
                if str(f['input_sha256'])!=digest:raise ValueError('Multiplier provenance hash mismatch')
                lam=f['multipliers']
                if not np.array_equal(f['reference'],z['reference_compute'][:,site]):raise ValueError('Retained reference differs')
            L=[F(0)]+[Amin[max(0,t-D+1)] for t in range(1,n+1)]
            lower,loss=exact_dual(points,L,Amax,lam)
            if lower!=q(record['slot_cost_lower_exact']):raise AssertionError('Exact slot-cost lower bound differs')
            if 21*E!=q(record['terminal_offset_allowance_exact']):raise AssertionError('Terminal offset allowance differs')
            lower-=21*E
            if lower!=q(record['bound_cost_exact']):raise AssertionError('Exact original-cost lower bound differs')
            if F(baseline_lo,scale)!=q(record['reference_cost_down_exact']) or F(baseline_hi,scale)!=q(record['reference_cost_up_exact']):raise AssertionError('Exact baseline enclosure differs')
            report['dual_records'].append({'site':site,'window_hours':hours,'exact_reproduction':True,'cost_lower_exact':rational_text(lower)})
        w=next(r for r in old['witness_results'] if r['site']==site)
        raw=json.loads((args.results/f'robust_w24_zone{site}.json').read_text())
        x=[q(v) for v in raw['execution_exact']]
        if len(x)!=n:raise ValueError('Robust schedule length')
        S=F(0);L=[F(0)]+[R[clear[max(0,t-96+1)]] for t in range(1,n+1)];L[-1]=R[-1]
        for t,(v,c) in enumerate(zip(x,caps),1):
            if not 0<=v<=c:raise AssertionError('Exact physical service box violation')
            S+=v
            if not L[t]<=S<=R[t]:raise AssertionError('Exact robust prefix violation')
        if S!=R[-1]:raise AssertionError('Exact surrogate endpoint differs')
        cost=F(sum(up_units(fn(v),scale) for fn,v in zip(funcs,x)),scale)
        if cost!=q(w['robust_execution_cost_upper_exact']):raise AssertionError('Original nonlinear robust-schedule cost differs')
        M=max(q(z['work_power_mw'][site])*q(z['pue'][t,site])*(abs(q(z['rt_lmp_usd_mwh'][t,site]))+2)/4 for t in range(n))
        if M*2*n*E!=q(w['uniform_actual_admission_repair_money_upper_exact']):raise AssertionError('Uniform monetary allowance differs')
        if 21*E!=q(w['fixed_reference_terminal_offset_allowance_exact']):raise AssertionError('Witness terminal offset allowance differs')
        if cost+M*2*n*E+21*E!=q(w['candidate_plus_uniform_repair_cost_upper_exact']):raise AssertionError('Full original-account cost upper differs')
        if any(E>=1-b[r['clearing_slot_zero_based']] for r in runs):raise AssertionError('Exact reference clearing proof guard fails')
        if max(r['saturated_slots'] for r in runs)+1>48:raise AssertionError('Reference feasibility for all displayed rights is not proved')
        report['robust_schedule_records'].append({'site':site,'all_35040_service_boxes_and_prefixes_exact':True,
            'surrogate_final_conservation_exact':True,'original_nonlinear_cost_reproduced':True,
            'source_compatible_actual_admission_feasibility':'Uniform exact-repair proof; original admissions not materialized'})
        print('solver-free original-cost and robust-schedule replay passed site',site,flush=True)
    report['all_passed']=True
    (args.results/'solver_free_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'all_passed':True,'annual_dual_records':len(report['dual_records']),
                      'annual_exact_robust_schedules':len(report['robust_schedule_records'])}),flush=True)


if __name__=='__main__':main()
