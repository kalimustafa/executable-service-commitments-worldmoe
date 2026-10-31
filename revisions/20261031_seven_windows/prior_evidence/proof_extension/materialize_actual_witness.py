"""Materialise the proved S27 admission-dependent witness on recovered source arrivals.

No optimiser is called. Every input float is interpreted as its exact binary
rational; the retained robust schedules are repaired with the proved corridor
map and all release, deadline, capacity, terminal and physical constraints are
checked exactly. Paths are configurable for execution from the evidence bundle.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse, ast, gzip, hashlib, json, types
import numpy as np


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(path,obj):
    with gzip.open(path,'wt',encoding='utf-8') as f: json.dump(obj,f,separators=(',',':'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-package',type=Path,required=True)
    p.add_argument('--forcing',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();s=args.source_package;o=args.output_dir;o.mkdir(parents=True,exist_ok=True)
    code=s/'verification_20261031/certificate/envelope_certificate.py'
    tree=ast.parse(code.read_text(encoding='utf-8')); names={'q','rational_text','down_units','up_units','ceil_cents','floor_cents','reference_envelopes','original_points','exact_clamp'}; ns={'F':F,'np':np}; exec(compile(ast.Module(body=[v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name in names],type_ignores=[]),str(code),'exec'),ns); e=types.SimpleNamespace(**ns)
    q=e.q;txt=e.rational_text
    pub=s/'source_snapshot/source_data/grid_metrics_inputs_2025.npz'
    with np.load(args.forcing,allow_pickle=False) as f: f={k:f[k].copy() for k in f.files}
    with np.load(pub,allow_pickle=False) as g: g={k:g[k].copy() for k in g.files}
    a=f['eagle_arrivals'];n,zs=a.shape;assert (n,zs)==(35040,4)
    matches={}
    for fk,gk in [('rt','rt_lmp_usd_mwh'),('pv','pv_mw'),('pue','pue'),('k','work_power_mw'),('idle','idle_mw')]:
        matches[fk]=bool(np.array_equal(f[fk],g[gk]));assert matches[fk]
    queue=np.zeros(zs);ref=np.zeros_like(a)
    for t in range(n):queue+=a[t];ref[t]=np.minimum(queue,1.);queue-=ref[t]
    assert np.array_equal(ref,g['reference_compute']) and not queue.any()
    constraints=json.loads((s/'source_snapshot/research/R2_PAPER_INTEGRATION_20260923/configs/economic_constraints.json').read_text())['constraints']
    saved=json.loads((s/'verification_20261031/certificate/results/envelope_results.json').read_text())
    res={'scope':'Exact recovered-arrival materialisation of the S27 uniform witness, nominal homogeneous 24 h rights, static capacity/import limits and fixed source settlement ledger',
      'forcing_sha256':sha(args.forcing),'published_arrays_sha256':sha(pub),'source_code_sha256':sha(code),
      'forcing_fields_identical_to_published':matches,'reference_recurrence_bitwise_identical':True,
      'intervals':n,'sites':zs,'origin_queue_exact':0,'horizon':'All recovered work completes inside observed year; no tail service required',
      'causal_or_nonanticipative_policy_claimed':False,'site_results':[]}
    scale=10**12;cost_lo=F(0);cost_hi=F(0);base_lo=F(0);base_hi=F(0)
    for z in range(zs):
        aa=[q(v) for v in a[:,z]];A=[F(0)]
        for v in aa:assert v>=0;A.append(A[-1]+v)
        b,R,nxt,E,Amin,Amax,runs=e.reference_envelopes(ref[:,z])
        assert all(l<=v<=u for l,v,u in zip(Amin,A,Amax))
        ss=s/f'verification_20261031/certificate/results/robust_w24_zone{z}.json'
        raw=[F(v) for v in json.loads(ss.read_text())['execution_exact']]
        L=[F(0)]+[A[max(0,t-95)] for t in range(1,n+1)];U=list(A);L[-1]=A[-1]
        xx,diag=e.exact_clamp(raw,L,U,[F(1)]*n)
        X=[F(0)]
        for v in xx:X.append(X[-1]+v)
        assert all(F(0)<=v<=1 for v in xx)
        assert all(l<=v<=u for l,v,u in zip(L,X,U)) and X[-1]==A[-1]
        l1=sum((abs(v-w) for v,w in zip(xx,raw)),F(0));assert l1<=2*n*E
        slo=shi=blo=bhi=0;min_grid_slack=None
        for t in range(n):
            pts,bc,cap,fn=e.original_points(f['rt'][t,z],f['pue'][t,z],f['pv'][t,z],f['idle'][z],f['k'][z],ref[t,z],constraints['grid_import_limit'][z])
            assert xx[t]<=cap
            v=fn(xx[t]);slo+=e.down_units(v,scale);shi+=e.up_units(v,scale)
            blo+=e.down_units(bc,scale);bhi+=e.up_units(bc,scale)
            load=q(f['pue'][t,z])*(q(f['idle'][z])+q(f['k'][z])*xx[t])
            grid=max(load-q(f['pv'][t,z]),F(0));slack=q(constraints['grid_import_limit'][z])-grid
            assert slack>=0
            min_grid_slack=slack if min_grid_slack is None else min(min_grid_slack,slack)
        tau=q(.25)*q(f['k'][z])*q(max(constraints['pue_max']))*q(f['historical_rt'][z]);assert 0<tau<21
        signed=tau*(R[-1]-A[-1]);assert abs(signed)<=21*E
        site_lo=F(slo,scale)+signed;site_hi=F(shi,scale)+signed
        assert site_hi<=F(saved['witness_results'][z]['candidate_plus_uniform_repair_cost_upper_exact'])
        cost_lo+=site_lo;cost_hi+=site_hi;base_lo+=F(blo,scale);base_hi+=F(bhi,scale)
        file=o/f'actual_admission_witness_zone{z}.json.gz'
        dump(file,{'site':z,'slot_index':'zero based, eligible r through r+95 inclusive','arrival_source':'forcing_2025.npz:eagle_arrivals',
          'arrivals_exact':[txt(v) for v in aa],'execution_exact':[txt(v) for v in xx],
          'cumulative_arrivals_exact':[txt(v) for v in A],'cumulative_execution_exact':[txt(v) for v in X],
          'surrogate_source_sha256':sha(ss),'surrogate_repair_l1_exact':txt(l1),'source_signed_terminal_offset_exact':txt(signed)})
        row={'site':z,'all_release_prefixes_exact':True,'all_exclusive_deadline_prefixes_exact':True,'all_capacity_boxes_exact':True,'all_static_import_limits_exact':True,
          'terminal_true_backlog_exact':'0/1','arrival_total_exact':txt(A[-1]),'reference_total_minus_arrival_total_exact':txt(R[-1]-A[-1]),
          'repair_l1_exact':txt(l1),'repair_l1_float':float(l1),'uniform_repair_l1_bound_exact':txt(2*n*E),
          'minimum_import_slack_mw':float(min_grid_slack),'terminal_rate_exact':txt(tau),'source_signed_terminal_offset_exact':txt(signed),
          'cost_lower_exact':txt(site_lo),'cost_upper_exact':txt(site_hi),'witness_file':file.name,'witness_sha256':sha(file),
          'within_previously_proved_uniform_cost_bound':True}
        res['site_results'].append(row);print('site',z,'exact witness saved; repair l1',float(l1),'cost upper',float(site_hi),flush=True)
    common=F('2409.7842143056');C0lo=base_lo+common;C0hi=base_hi+common;Jlo=cost_lo+common;Jhi=cost_hi+common
    Hlo=C0lo-Jhi;Hhi=C0hi-Jlo
    res.update(baseline_cost_lower_exact=txt(C0lo),baseline_cost_upper_exact=txt(C0hi),witness_cost_lower_exact=txt(Jlo),witness_cost_upper_exact=txt(Jhi),
      materialised_saving_lower_exact=txt(Hlo),materialised_saving_upper_exact=txt(Hhi),materialised_saving_lower_usd=float(Hlo),materialised_saving_upper_usd=float(Hhi),
      materialised_saving_down_cents=e.floor_cents(Hlo),retained_uniform_floor_down_cents=saved['uniform_witness_value']['savings_floor_down_cents'],
      retained_uniform_floor_is_valid_for_recovered_inputs=bool(Hlo>=F(saved['uniform_witness_value']['savings_floor_exact'])))
    assert res['retained_uniform_floor_is_valid_for_recovered_inputs']
    (o/'actual_witness_results.json').write_text(json.dumps(res,indent=2)+'\n',encoding='utf-8')
    print('materialised saving enclosure',float(Hlo),float(Hhi),'retained floor',res['retained_uniform_floor_down_cents'],flush=True)

if __name__=='__main__':main()

