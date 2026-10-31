"""Join exact policy accounts and certified opportunities without filling missing runs."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal, localcontext
import csv,json,hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
OLD=ROOT/'prior_evidence/historical_extension/results/historical_exact_repair.json'
if not OLD.exists():OLD=ROOT.parent/'EnergyEconomics_Revision_20261031/historical_extension/results/historical_exact_repair.json'
WINDOWS=[8,12,16,20,24,36,48];SEEDS=[20260920,20260921,20260922]
def text(x):return f'{x.numerator}/{x.denominator}'
def dec(x):
    with localcontext() as c:c.prec=60;return format(Decimal(x.numerator)/Decimal(x.denominator),'f')
def down(x,scale=100):return (x.numerator*scale//x.denominator)/scale
def up(x,scale=100):return -((-x.numerator*scale)//x.denominator)/scale
def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as g:
        wr=csv.DictWriter(g,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def main():
    historical=json.loads(OLD.read_text(encoding='utf-8'))
    policy={w:{} for w in WINDOWS};rows=[];anchors=[];feasible={w:True for w in WINDOWS};lateness={w:[] for w in WINDOWS}
    for row in historical['rows']:
        w=row['window_h'];seed=row['seed'];value=F(row['exact_repaired_saving_usd'])
        policy[w][seed]=value
        rows.append(dict(window_h=w,seed=seed,annual_saving_usd=dec(value),evidence_origin='recovered_historical',
            run_id=f'historical_s{seed}_w{w}',source_path='../prior_evidence/historical_extension/results/historical_exact_repair.json',service_feasible='true',late_it_mwh=0.0))
    for seed in SEEDS:
        for w in [8,20,24,36,48]:
            p=ROOT/f'audit/policy_s{seed}_w{w}.json'
            if not p.exists():raise RuntimeError(f'Waiting for audited result: {p.name}')
            j=json.loads(p.read_text());valid=j['all_exact_checks']
            value=F(j['exact_repaired_saving']) if valid else F(str(j['independently_resettled_cash_saving_usd']))
            feasible[w]=feasible[w] and valid;lateness[w].append(j.get('late_IT_mwh',0.))
            if w==24:
                anchors.append(dict(window_h=w,seed=seed,annual_saving_usd=dec(value),source_path=f'../audit/{p.name}'))
                assert value==policy[24][seed], '24h exact repaired anchor differs'
            else:
                policy[w][seed]=value
                rows.append(dict(window_h=w,seed=seed,annual_saving_usd=dec(value),evidence_origin='new_20261031',
                    run_id=j['run_name'],source_path=f'../audit/{p.name}',service_feasible=str(valid).lower(),late_it_mwh=j.get('late_IT_mwh',0.)))
    certfile=ROOT/'certificates/results/seven_window_summary.json'
    cert=json.loads(certfile.read_text());C={r['window_hours']:r for r in cert['windows']}
    certrows=[dict(window_h=w,pi_lower_usd=C[w]['combined_floor_down_cents'],pi_upper_usd=C[w]['combined_ceiling_up_cents'],
        source_path='../certificates/results/seven_window_summary.json') for w in WINDOWS]
    means={w:sum(policy[w].values(),F(0))/3 for w in WINDOWS};summary=[]
    for w in WINDOWS:
        H=F(C[w]['combined_floor_exact']);U=F(C[w]['combined_ceiling_exact']);S=means[w]
        if feasible[w]:assert max(policy[w].values())<=U
        summary.append(dict(window_h=w,policy_mean_usd=float(S),policy_mean_exact=text(S) if feasible[w] else None,
            monetary_scope='exact feasible numerical account' if feasible[w] else 'raw floating cash account with substantive service failure',
            policy_range_usd=[float(min(policy[w].values())),float(max(policy[w].values()))],
            policy_evidence_origin='recovered_historical' if w in [12,16,24] else 'new_20261031',
            pi_lower_usd=C[w]['combined_floor_down_cents'],pi_upper_usd=C[w]['combined_ceiling_up_cents'],
            service_feasible=feasible[w],late_it_mwh_mean=sum(lateness[w])/len(lateness[w]) if lateness[w] else 0.,
            remaining_opportunity_usd=[down(H-S),up(U-S)] if feasible[w] else None,
            capture_percent_approx=[float(100*S/U),float(100*S/H)] if feasible[w] else None,
            capture_percent_outward=[down(100*S/U),up(100*S/H)] if feasible[w] else None,
            certificate_width_up_cents=C[w]['interval_width_up_cents']))
    adjacent=[]
    for a,b in zip(WINDOWS,WINDOWS[1:]):
        inc=[policy[b][seed]-policy[a][seed] for seed in SEEDS];H0,U0=F(C[a]['combined_floor_exact']),F(C[a]['combined_ceiling_exact']);H1,U1=F(C[b]['combined_floor_exact']),F(C[b]['combined_ceiling_exact'])
        pl=max(F(0),H1-U0);pu=U1-H0;delta=b-a
        valid=feasible[a] and feasible[b]
        adjacent.append(dict(from_h=a,to_h=b,added_deadline_hours=delta,policy_mean_increment_usd=float(sum(inc)/3) if valid else None,
            paired_policy_increment_range_usd=[float(min(inc)),float(max(inc))] if valid else None,
            mean_policy_increment_per_deadline_hour=float(sum(inc)/(3*delta)) if valid else None,
            policy_increment_eligible=valid,
            pi_increment_interval_usd=[down(pl),up(pu)],pi_increment_per_deadline_hour_usd=[down(pl/delta),up(pu/delta)],
            exact_policy_mean_increment=text(sum(inc)/3) if valid else None,pi_lower_exact=text(pl),pi_upper_exact=text(pu)))
    H24,U24=F(C[24]['combined_floor_exact']),F(C[24]['combined_ceiling_exact']);H12,U12=F(C[12]['combined_floor_exact']),F(C[12]['combined_ceiling_exact'])
    result={'scope':'Seven matched contract windows on one recovered 2025 path. Historical 12/16/24 policy records and new 2026-10-31 evaluations are explicitly distinguished.',
        'windows':summary,'adjacent':adjacent,'policy_by_seed_exact':{str(w):({str(s):text(v) for s,v in d.items()} if feasible[w] else None) for w,d in policy.items()},
        'pi_12_to_24_interval_usd':[down(H24-U12),up(U24-H12)],
        'new_policy_run_count':15,'historical_policy_rows_retained':9,
        'anchor_complete_book_bitwise_equal':json.loads((ROOT/'results/anchor_comparison.json').read_text()),
        'max_new_policy_repair_cost_bound_usd':max(json.loads(p.read_text())['repair_cost_bound_usd'] or 0. for p in (ROOT/'audit').glob('policy_s*_w*.json')),
        'no_new_training':True,'new_policy_runs_passing_exact_service':12,'new_policy_service_failures':3,
        'source_hashes':{'old_exact_results':hashlib.sha256(OLD.read_bytes()).hexdigest(),'seven_window_certificate':hashlib.sha256(certfile.read_bytes()).hexdigest()}}
    write_csv(OUT/'policy.csv',sorted(rows,key=lambda x:(x['window_h'],x['seed'])))
    write_csv(OUT/'certificates.csv',certrows);write_csv(OUT/'anchor.csv',anchors)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    # Main-table displayed precision is separate from exact source enclosures.
    md=['**Table 5. Policy accounts, service compliance and certified opportunity.** Annual four-site US dollars; three-checkpoint policy means. The 8 h cash account fails service requirements and is excluded from capture calculations. Other policy endpoints use exact feasible numerical counterparts. Historical 12/16/24 h rows and new evaluations are distinguished in Figure 6.','',
        '| Deadline | Policy cash saving | Service | Certified PI opportunity | Capture |','|---|---:|:---|---:|---:|']
    for r in summary:
        capture='—';service='Feasible'
        if r['service_feasible']:
            lo,hi=r['capture_percent_outward'];capture=f'{lo:.2f}%–{hi:.2f}%'
        else:service=f"Fails ({r['late_it_mwh_mean']:.2f} IT MWh late)"
        md.append(f"| {r['window_h']} h | {r['policy_mean_usd']:,.2f} | {service} | {float(r['pi_lower_usd']):,.2f}–{float(r['pi_upper_usd']):,.2f} | {capture} |")
    (OUT/'main_table5.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps({'windows':summary,'pi_12_to_24':result['pi_12_to_24_interval_usd'],'max_repair':result['max_new_policy_repair_cost_bound_usd']},indent=2))
if __name__=='__main__':main()
