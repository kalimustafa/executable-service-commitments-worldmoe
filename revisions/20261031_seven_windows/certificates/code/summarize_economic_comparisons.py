from pathlib import Path
from fractions import Fraction as F
import hashlib,json,sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'code'));import exact_certificate_core as e
r=json.loads((root/'results/seven_window_summary.json').read_text());rows={x['window_hours']:x for x in r['windows']}
p=root/'inputs/historical_exact_repair.json'
h=json.loads(p.read_text());S=F(h['mean_repaired_savings_usd_exact']['24']);H=F(rows[24]['combined_floor_exact']);U=F(rows[24]['combined_ceiling_exact'])
H12=F(rows[12]['combined_floor_exact']);U12=F(rows[12]['combined_ceiling_exact'])
out={'scope':'New exact actual-admission PI bounds compared with the previously recovered and exactly repaired three-seed 24 h WorldMoE executions; new policy anchors are checked separately',
 'historical_policy_record_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'mean_worldmoe_saving_exact':e.rational_text(S),
 '24h_remaining_opportunity_lower_exact':e.rational_text(H-S),'24h_remaining_opportunity_upper_exact':e.rational_text(U-S),
 '24h_remaining_opportunity_lower_down_cents':e.floor_cents(H-S),'24h_remaining_opportunity_upper_up_cents':e.ceil_cents(U-S),
 '24h_capture_percent_lower_exact':e.rational_text(100*S/U),'24h_capture_percent_upper_exact':e.rational_text(100*S/H),
 '24h_capture_percent_lower_down_2dp':e.floor_cents(100*S/U),'24h_capture_percent_upper_up_2dp':e.ceil_cents(100*S/H),
 '12_to_24_PI_premium_lower_exact':e.rational_text(max(F(0),H-U12)),'12_to_24_PI_premium_upper_exact':e.rational_text(U-H12),
 '12_to_24_PI_premium_lower_down_cents':e.floor_cents(H-U12),'12_to_24_PI_premium_upper_up_cents':e.ceil_cents(U-H12),
 '24h_witness_ceiling_gap_up_cents':e.ceil_cents(U-H)}
(root/'results/key_economic_comparisons.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if not k.endswith('exact')},indent=2))
text=(root/'README.md').read_text(encoding='utf-8-sig')
text=text.replace('# Actual admission certificates for seven deadline durations','# S31 Newly computed opportunity bounds for seven actual admission contracts')
reps={
 'A_t is the exact cumulative sum of the source arrivals and X_t the cumulative execution. With D=4W eligible slots':'$A_t$ is the exact cumulative sum of the source arrivals and $X_t$ the cumulative execution. With $D=4W$ eligible slots',
 'does not impose X_N=A_N.':'does not impose $X_N=A_N$.',
 'exactly tau_z(A_N-X_N), with tau_z=0.25 k_z PUE_max historical_RT_median_z.':'exactly $\\tau_z(A_N-X_N)$, with $\\tau_z=0.25k_z\\mathrm{PUE}_{\\max}\\operatorname{median}(p_z^{\\rm historical})$.',
 'by N and has exact total R_N=A_N.':'by $N$ and has exact total $R_N=A_N$.',
 'beyond N retain':'beyond $N$ retain',
 'completes by N.':'completes by $N$.',
 'Write f_t(x) for':'Write $f_t(x)$ for',
 'multipliers lambda_t':'multipliers $\\lambda_t$',
 'Set lambda_(N+1)=0, L_t=A_(max(0,t-D+1)) and U_t=A_t.':'Set $\\lambda_{N+1}=0$, $L_t=A_{\\max(0,t-D+1)}$ and $U_t=A_t$.',
 'as tau_z A_N minus tau_z times the sum of executed work, subtract lambda_t(x_t+X_(t-1)-X_t)':'as $\\tau_z A_N$ minus $\\tau_z$ times the sum of executed work, subtract $\\lambda_t(x_t+X_{t-1}-X_t)$',
 'down to 10^-12 dollars':'down to $10^{-12}$ dollars',
 'with X_N=A_N.':'with $X_N=A_N$.',
 're-settled with f_t,':'re-settled with $f_t$,',
 'floor H_W and ceiling U_W.':'floor $H_W$ and ceiling $U_W$.',
 'as W increases.':'as $W$ increases.',
 'durations W_a<W_b,':'durations $W_a<W_b$,',
 'by W_b-W_a gives':'by $W_b-W_a$ gives'
}
for a,b in reps.items():
 assert a in text,a
 text=text.replace(a,b)
section='## Numerical results\n\nAll values are annual four-site US dollars on the recovered actual admission stream. Floors are rounded downward and ceilings upward. No neighbouring window supplied a tighter endpoint than the raw endpoint in these runs; the reported raw and nesting-combined intervals therefore coincide.\n\n'
section+='| Deadline h | Feasible saving floor | Global saving ceiling | Interval width upper |\n|:---|---:|---:|---:|\n'
for x in r['windows']:section+=f"| {x['window_hours']} | {float(x['combined_floor_down_cents']):,.2f} | {float(x['combined_ceiling_up_cents']):,.2f} | {float(x['interval_width_up_cents']):,.2f} |\n"
section+='\nThe per-hour column below is annual value per additional deadline hour, rather than an hourly electricity revenue rate.\n\n| Extension h | Rights premium interval US$ | Annual US$ per extra deadline hour |\n|:---|---:|---:|\n'
for x in r['adjacent_rights_premiums']:
 section+=f"| {x['from_hours']} to {x['to_hours']} | {float(x['premium_lower_down_cents']):,.2f}–{float(x['premium_upper_up_cents']):,.2f} | {float(x['premium_per_hour_lower_down_cents']):,.2f}–{float(x['premium_per_hour_upper_up_cents']):,.2f} |\n"
section+='\nThe marginal annual value per extra deadline hour decreases across the tested intervals through 24–36 h, but rises from US$783.16–823.30 for 24–36 h to US$878.48–922.81 for 36–48 h. Because the later lower endpoint exceeds the earlier upper endpoint, that local rise follows even after allowing for the certificate intervals. The results do not support a global diminishing-returns claim.\n\n'
section+=f"The 24 h actual-stream opportunity is US${rows[24]['combined_floor_down_cents']}–{rows[24]['combined_ceiling_up_cents']}, with a witness-to-ceiling gap below US${out['24h_witness_ceiling_gap_up_cents']}. Relative to the previously recovered, exactly repaired three-seed WorldMoE mean, remaining same-rights opportunity is US${out['24h_remaining_opportunity_lower_down_cents']}–{out['24h_remaining_opportunity_upper_up_cents']} and the mean capture ratio lies in {out['24h_capture_percent_lower_down_2dp']}%–{out['24h_capture_percent_upper_up_2dp']}%. The actual-stream 12–24 h optimal rights premium is US${out['12_to_24_PI_premium_lower_down_cents']}–{out['12_to_24_PI_premium_upper_up_cents']}. These intervals describe newly computed perfect-information values for the historical calendar. They are distinct from historical paired operating-policy increments and from the earlier admission-uniform robustness guarantee.\n\n"
text=text.replace('## Files and reproduction',section+'## Files and reproduction')
text=text.replace(r'\rm historical',r'\mathrm{historical}').replace(r'\rm site',r'\mathrm{site}').replace(r'\rm raw',r'\mathrm{raw}')
text=text.replace('## Files and reproduction','All 28 annual site-duration witnesses passed exact release, exclusive-deadline, capacity, import and terminal checks when constructed. One saved 24 h site-3 multiplier file and full exact witness were additionally reloaded and re-evaluated without an LP solver; the saved-path report binds their hashes and inputs. This replay concerns one selected saved certificate and is not presented as a separate 28-case second-method verification.\n\n## Files and reproduction')
(root/'S31_actual_admission_seven_window_certificates.md').write_text(text,encoding='utf-8')
