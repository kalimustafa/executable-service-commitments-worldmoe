"""Pair the prespecified untouched R with every SMPC setting on declared axes."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'artifacts/r_ecmoe_closedloop_sensitivity_20260913'


def main():
    source=ROOT/'summary.json';s=json.loads(source.read_text(encoding='utf-8'));rows=[]
    for date in s['days']:
        r=next(x for x in s['episodes'] if x['date']==date and x['policy_id']=='R_original')
        for m in [x for x in s['episodes'] if x['date']==date and x['family']=='W1-SMPC-20']:
            d=[r['normalized_composite_cost']-m['normalized_composite_cost'],
               m['on_time_workload_rate']-r['on_time_workload_rate'],
               r['full_chain_mean_ms']-m['full_chain_mean_ms']]
            rows.append(dict(date=date,R=r['policy_id'],comparator=m['policy_id'],
                R_dominates_on_registered_three_axes=all(x<=0 for x in d) and any(x<0 for x in d),
                cost_R_minus_SMPC=d[0],raw_composite_cost_R_minus_SMPC=r['raw_composite_cost']-m['raw_composite_cost'],
                latency_reduction_percent=100*(1-r['full_chain_mean_ms']/m['full_chain_mean_ms']),
                on_time_difference_pp=-100*d[1],
                terminal_soc_R_minus_SMPC=r['terminal_soc_total']-m['terminal_soc_total'],
                terminal_queue_R_minus_SMPC=r['terminal_queue_total']-m['terminal_queue_total'],
                local_ramp_R_minus_SMPC=r['local_ramp_mean']-m['local_ramp_mean'],
                peak_V_rebound_R_minus_SMPC=r['true_v_rebound_max']-m['true_v_rebound_max']))
    assert len(rows)==8
    out=dict(scope='descriptive paired comparisons of untouched R against all four prespecified SMPC settings on each of two dates; not independent trials',
        axes=['normalized base composite cost lower','mature cohort on-time completion higher','mean full-chain latency lower'],
        excluded_from_this_dominance_claim=['local ramp','recovery curvature','terminal SOC','terminal backlog valuation','annual outcomes'],
        new_weighting_or_selection=False,dominance_count=sum(r['R_dominates_on_registered_three_axes'] for r in rows),comparisons=rows,
        summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),analysis_code=str(Path(__file__).resolve()))
    (ROOT/'original_R_paired_dominance.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    with (ROOT/'original_R_paired_dominance.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'paired_comparisons':len(rows),'R_dominance_count':out['dominance_count']}))


if __name__=='__main__':main()
