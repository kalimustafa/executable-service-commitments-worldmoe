"""Small boundary tests against unmodified public source functions.

The functions are extracted with Python AST to avoid data-loading side effects.
This is structural verification, not a reproduction of annual model results.
"""
from __future__ import annotations
import ast
import argparse
import heapq
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = None
OUT = Path(__file__).resolve().parent


def load_functions(rel, names, namespace=None):
    path=ROOT/rel
    tree=ast.parse(path.read_text())
    nodes=[]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list=[]
            nodes.append(node)
    if {n.name for n in nodes} != set(names):
        raise RuntimeError('Requested public-source function not found')
    ns={'np':np,'heapq':heapq}
    if namespace:ns.update(namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),ns)
    return ns


def main():
    global ROOT, OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True,help='Path to the inspected public source checkout')
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    ROOT=args.repo.resolve()
    OUT=args.output_dir.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    tests=[]
    def record(name, condition, detail):
        if not condition:raise AssertionError(name)
        tests.append({'test':name,'passed':True,'detail':detail})
    context='research/ServiceCommitment_1to5_20260927/src/context.py'
    c=SimpleNamespace(N=1,DT=.25,k=np.ones(4))
    ns=load_functions(context,['fifo_lateness'],{'c':c})
    arr=np.zeros((1,4));arr[0,0]=1
    for hours in (8,12,16,24,48):
        slots=4*hours
        book=np.zeros((slots+2,4));book[slots-1,0]=1
        timely=ns['fifo_lateness'](book,arr,hours)
        book[slots-1,0]=0;book[slots,0]=1
        late=ns['fifo_lateness'](book,arr,hours)
        record(f'homogeneous_{hours}h_last_slot_included',timely==0,
               {'last_eligible_slot_zero_based':slots-1})
        record(f'homogeneous_{hours}h_deadline_boundary_excluded',late==.25,
               {'first_ineligible_slot_zero_based':slots,'late_it_mwh_fixture':late})
    hetero='research/ExecutableServiceCommitments_Revision_20260928/experiments/E03_heterogeneous_deadlines/audit_hetero_book.py'
    hs=load_functions(hetero,['audit','weighted_quantile'],{'WINDOWS':(4,12,24),'SHARES':(.25,.5,.25),'DT':.25})
    book=np.zeros((100,4));book[15,0]=.25;book[47,0]=.5;book[95,0]=.25
    mixed=hs['audit'](book,arr,np.ones(4),'edf')
    record('mixed_class_deadline_endpoints',mixed['late_work_it_mwh']==0,mixed)
    book[:]=0;book[95,0]=1
    mixed=hs['audit'](book,arr,np.ones(4),'edf')
    record('homogeneous_24h_book_fails_mixed_contract',mixed['late_work_it_mwh']==.1875,
           {'late_it_mwh_fixture':mixed['late_work_it_mwh']})
    reservation='research/ServiceCommitment_1to5_20260927/vendor/reservation.py'
    def ranking(window,policy):
        return np.broadcast_to(np.arange(int(window*4))[None,:,None],(c.N,int(window*4),4)).copy()
    rs=load_functions(reservation,['schedule'],{'c':c,'rank':ranking})
    large=np.zeros((1,4));large[0,0]=5
    book,stats,_=rs['schedule'](large,[(1.,1.)],'fixture')
    independent=ns['fifo_lateness'](book,large,1.)
    record('legacy_tail_lateness_censor_confirmed',stats['late_IT_mwh']==0 and independent==.25,
           {'legacy_late_it_mwh':stats['late_IT_mwh'],'full_deadline_audit_it_mwh':independent})
    frontier='research/ExecutableServiceCommitments_Revision_20260928/experiments/E04_frontier_nonpreemptive/run_frontier_jobs.py'
    fs=load_functions(frontier,['first_fit','choose_start'])
    book=np.zeros((10,4));book[0,0]=1
    selected=fs['choose_start'](book,0,0,2,1.,0,np.zeros(4))
    record('frontier_infeasible_earliest_fallback_confirmed',selected==1,
           {'latest_start':0,'returned_start':int(selected),'late_requires_outcome_gate':True})
    controls='research/ServiceCommitment_1to5_20260927/src/controllers.py'
    physics=SimpleNamespace(N=1,c={'grid_import_limit':[1.]*4,'pue_max':[2.]*4},
                            pue=np.full((1,4),2.),pv=np.zeros((1,4)),idle=np.full(4,.1),k=np.ones(4))
    cs=load_functions(controls,['project_grid'],{'c':physics})
    requests=np.zeros((3,4));requests[0]=1
    admissions=np.ones((1,4))
    without,_=cs['project_grid'](requests,admissions)
    physics.pv[:]=1.
    with_pv,_=cs['project_grid'](requests,admissions)
    record('legacy_grid_projection_reads_current_physics',not np.array_equal(without[0],with_pv[0]),
           {'current_service_no_pv':without[0].tolist(),'current_service_with_pv':with_pv[0].tolist()})
    # Exact source expression, evaluated on a last-slot example.
    horizon,slot,hours=100,99,24
    offset=min(hours*4-1,horizon-1-slot)
    record('legacy_year_end_deadline_clamp_confirmed',offset==0 and hours*4-1==95,
           {'clamped_due_offset':offset,'true_contract_due_offset':95})
    summary={'status':'passed','test_count':len(tests),'tests':tests,
             'annual_model_replay_performed':False,
             'original_source_modified':False,
             'meaning':'Expected source behavior verified on explicit fixtures; identified limitations are not repaired annual results.'}
    (OUT/'source_boundary_test_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'status':summary['status'],'test_count':len(tests)},indent=2))


if __name__=='__main__':main()
