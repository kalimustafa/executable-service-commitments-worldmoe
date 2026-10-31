"""Independent audit of recovered historical WorldMoE books, without model reruns.

Requires numpy. Run from any directory; reads only the sibling inputs folder.
Original archives are preserved, never repaired or overwritten by this script.
"""
from pathlib import Path
import csv, hashlib, json
from fractions import Fraction
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'inputs'
OUT = ROOT / 'results'
TOL = 1e-8
DT = .25
SEEDS = [(20260920, 6), (20260921, 2), (20260922, 0)]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def exact_prefix_check(a, x, width):
    """All stored binary64 values treated as exact dyadic rationals."""
    ar = [[float(v).as_integer_ratio() for v in a[:, z]] for z in range(4)]
    xr = [[float(v).as_integer_ratio() for v in x[:, z]] for z in range(4)]
    denom = max(d for col in ar + xr for _, d in col)
    release_max = due_max = mass_max = 0
    total_admission = []
    for z in range(4):
        A=[];X=[];s=0
        for v,d in ar[z]: s += v*(denom//d); A.append(s)
        total_admission.append(str(Fraction(s,denom)))
        s=0
        for v,d in xr[z]: s += v*(denom//d); X.append(s)
        release_max=max(release_max,max(X[t]-A[t] for t in range(len(a))))
        due_max=max(due_max,max(A[t]-X[t+width-1] for t in range(len(a))))
        mass_max=max(mass_max,abs(X[-1]-A[-1]))
    return dict(exact_arithmetic='integer accumulation with one common power-of-two denominator',
                max_release_excess_work_exact=str(Fraction(release_max,denom)),
                max_deadline_deficit_work_exact=str(Fraction(due_max,denom)),
                max_total_work_error_exact=str(Fraction(mass_max,denom)),
                max_release_excess_work=float(Fraction(release_max,denom)),
                max_deadline_deficit_work=float(Fraction(due_max,denom)),
                max_total_work_error=float(Fraction(mass_max,denom)),
                exactly_feasible_as_stored=(release_max==due_max==mass_max==0),
                admission_work_exact_by_site=total_admission)

def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(exist_ok=True)
    f=np.load(INPUT/'forcing_2025.npz',allow_pickle=False)
    g=np.load(INPUT/'published_grid_metrics_inputs_2025.npz',allow_pickle=False)
    config=json.loads((INPUT/'economic_constraints.json').read_text(encoding='utf-8'))['constraints']
    a=f['eagle_arrivals'];n=len(a);N=n+8192
    ref=np.zeros((N,4));q=np.zeros(4)
    for t in range(n):
        q+=a[t];ref[t]=np.minimum(q,1.);q-=ref[t]
    tail=n
    while q.max()>1e-10:
        ref[tail]=np.minimum(q,1.);q-=ref[tail];tail+=1
    fields={'rt':'rt_lmp_usd_mwh','pv':'pv_mw','pue':'pue','k':'work_power_mw','idle':'idle_mw'}
    correspondence={k:bool(np.array_equal(f[k],g[v])) for k,v in fields.items()}
    correspondence['regenerated_reference_compute']=bool(np.array_equal(ref[:n],g['reference_compute']))
    cap=np.minimum(1.,np.maximum((f['grid_limit']+f['pv'])/f['pue']-f['idle'],0)/f['k'])
    L0=f['pue']*(f['idle']+f['k']*ref[:n]);G0=np.maximum(L0-f['pv'],0);V0=np.maximum(f['pv']-L0,0)
    refcost=DT*np.sum(f['rt']*G0+V0)+f['common_da_component'].sum()
    month=f['times'].astype('datetime64[M]').astype(str)
    # Use archived market-local daily labels to assign local months, including DST.
    labels=[];financial=[];monthly=[];detailed=[]
    for seed,ep in SEEDS:
        for w in [12,16,24]:
            name=f'CdfR2_Eagle_2025_s{seed}_ep{ep:02d}_coupled_cdf_w{w}'
            j=json.loads((INPUT/f'{name}.json').read_text(encoding='utf-8'))
            z=np.load(INPUT/f'{name}_complete_book.npz',allow_pickle=False)
            tr=np.load(INPUT/f'{name}_trajectory.npz',allow_pickle=False)
            x=z['compute'];X=np.cumsum(x,axis=0);A=np.cumsum(a,axis=0)
            eq=exact_prefix_check(a,x,4*w)
            load=f['pue']*(f['idle']+f['k']*x[:n]);grid=np.maximum(load-f['pv'],0);spill=np.maximum(f['pv']-load,0)
            avoided=DT*(f['rt']*np.maximum(G0-grid,0)).sum(1)
            recovery=DT*(f['rt']*np.maximum(grid-G0,0)).sum(1)
            fee=DT*np.abs(grid-G0).sum(1);spillbenefit=DT*(V0-spill).sum(1)
            value=avoided-recovery-fee+spillbenefit
            liability=DT*np.sum((x[n:].sum(0)-ref[n:].sum(0))*f['k']*max(config['pue_max'])*f['historical_rt'])
            value[-1]-=liability;saving=float(value.sum())
            idx=np.arange(n)+4*w-1
            cohort_late=np.maximum(A-np.maximum(A-a,X[idx]),0)
            audit=dict(seed=seed,window_h=w,source_name=name,
                historical_checkpoint_sha256=j['forecast_checkpoint_sha256'],
                archived_reference_equal=bool(np.array_equal(z['reference'],ref)),
                archived_inyear_compute_equal=bool(np.array_equal(x[:n],tr['compute'])),
                published_24h_compute_equal=(bool(np.array_equal(x[:n],g[f'compute_seed_{seed}'])) if w==24 else None),
                minimum_compute=float(x.min()),max_compute_excess=float(max(0.,x.max()-1.)),
                max_electrical_import_excess_mw=float(max(0.,np.max(grid-f['grid_limit']))),
                maximum_grid_reconstruction_error_mw=float(np.max(np.abs(grid-tr['grid']))),
                maximum_recorded_value_difference_usd=float(np.max(np.abs(value-tr['value']))),
                reference_cost_usd=float(refcost),recomputed_saving_usd=saving,
                recorded_saving_usd=j['saving_usd'],saving_difference_usd=saving-j['saving_usd'],
                terminal_policy_IT_mwh=float(DT*np.sum(x[n:]*f['k'])),
                terminal_reference_IT_mwh=float(DT*np.sum(ref[n:]*f['k'])),
                terminal_cost_difference_usd=float(liability),
                original_fifo_late_IT_mwh=j['late_IT_mwh'],
                recomputed_fifo_late_IT_mwh=float(DT*np.sum(cohort_late*f['k'])),
                **eq)
            audit['passes_declared_numeric_tolerance']=bool(
                max(eq['max_release_excess_work'],eq['max_deadline_deficit_work'],eq['max_total_work_error'])<=TOL
                and audit['max_compute_excess']<=TOL and audit['minimum_compute']>=-TOL
                and audit['max_electrical_import_excess_mw']<=TOL and abs(audit['saving_difference_usd'])<.000001
                and audit['archived_reference_equal'] and audit['archived_inyear_compute_equal'])
            detailed.append(audit)
            financial.append(dict(seed=seed,window_h=w,reference_cost_usd=float(refcost),
                saving_usd=saving,gross_avoided_energy_usd=float(avoided.sum()),recovery_energy_usd=float(recovery.sum()),
                execution_change_charge_usd=float(fee.sum()),curtailment_benefit_usd=float(spillbenefit.sum()),
                terminal_cost_difference_usd=float(liability),recorded_saving_usd=j['saving_usd']))
            labels.append(dict(seed=seed,window_h=w,evidence_label='historical paired policy results with recovered trajectory audit',
                passes_numeric_tolerance=audit['passes_declared_numeric_tolerance'],
                max_release_excess_work=eq['max_release_excess_work'],max_deadline_deficit_work=eq['max_deadline_deficit_work'],
                max_total_work_error=eq['max_total_work_error'],exactly_feasible_as_stored=eq['exactly_feasible_as_stored'],
                tail_IT_mwh=audit['terminal_policy_IT_mwh'],saving_difference_usd=audit['saving_difference_usd']))
            # Quarterly arrays permit direct period-level replay of both service and money.
            np.savez_compressed(OUT/f'audit_s{seed}_w{w}.npz',timestamp_utc=f['times'],
                cumulative_admission=A,cumulative_execution=X[:n],
                deadline_deficit_work=np.maximum(A-X[idx],0),release_excess_work=np.maximum(X[:n]-A,0),
                compute_capacity_excess=np.maximum(x[:n]-cap,0),grid_import_mw=grid,
                gross_avoided_energy_usd=avoided,recovery_energy_usd=recovery,
                execution_change_charge_usd=fee,curtailment_benefit_usd=spillbenefit,saving_usd=value)
            for m in np.unique(month):
                sel=month==m
                monthly.append(dict(seed=seed,window_h=w,utc_month=m,intervals=int(sel.sum()),
                    admissions_work=float(a[sel].sum()),execution_work=float(x[:n][sel].sum()),
                    saving_usd=float(value[sel].sum()),gross_avoided_energy_usd=float(avoided[sel].sum()),
                    recovery_energy_usd=float(recovery[sel].sum()),execution_change_charge_usd=float(fee[sel].sum()),
                    curtailment_benefit_usd=float(spillbenefit[sel].sum())))
    paired=[]
    for seed,_ in SEEDS:
        rows={r['window_h']:r for r in detailed if r['seed']==seed}
        for lo,hi in [(12,16),(16,24),(12,24)]:
            paired.append(dict(seed=seed,from_h=lo,to_h=hi,
                same_checkpoint=rows[lo]['historical_checkpoint_sha256']==rows[hi]['historical_checkpoint_sha256'],
                independently_resettled_increment_usd=rows[hi]['recomputed_saving_usd']-rows[lo]['recomputed_saving_usd']))
    summary=dict(scope='Recovered original historical trajectory audit, not a new policy execution or training run.',
        tolerances=dict(work=TOL,grid_mw=TOL,financial_usd=1e-6),
        intervals=n,sites=4,stored_tail_intervals=8192,observed_timestamps_utc=[str(f['times'][0]),str(f['times'][-1])],
        forcing_sha256=sha(INPUT/'forcing_2025.npz'),published_input_correspondence=correspondence,
        all_nine_pass_numeric_tolerance=all(r['passes_declared_numeric_tolerance'] for r in detailed),
        exact_feasibility_note='The stored binary64 schedules have tiny nonzero rational residuals; they are not claimed exact symbolic witnesses. Exact-rational prefix residuals are reported, not rounded to zero.',
        mean_savings_by_window={str(w):float(np.mean([r['recomputed_saving_usd'] for r in detailed if r['window_h']==w])) for w in [12,16,24]},
        mean_12_to_24_increment_usd=float(np.mean([r['independently_resettled_increment_usd'] for r in paired if r['from_h']==12 and r['to_h']==24])),
        rows=detailed,paired=paired)
    (OUT/'historical_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    write_csv(OUT/'historical_service_audit.csv',labels);write_csv(OUT/'historical_settlement.csv',financial)
    write_csv(OUT/'historical_monthly_ledger.csv',monthly);write_csv(OUT/'historical_paired_increments.csv',paired)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','paired']},indent=2))
    if not summary['all_nine_pass_numeric_tolerance'] or not all(correspondence.values()):
        raise SystemExit('Historical audit failed; inspect output, do not label verified.')

if __name__=='__main__':
    main()
