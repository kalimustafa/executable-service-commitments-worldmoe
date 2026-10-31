from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'audit').glob('policy_s*_w*.json'))]
assert len(rows)==15
maxbound=max(r.get('repair_cost_bound_usd') or 0 for r in rows)
md=['## S32.8 Independent service and monetary records','',
'All fifteen completed new evaluations were independently re-settled from their saved complete books and the common forcing. The three substantive 8 h deadline failures are retained without repair. The remaining twelve new executions have only roundoff-scale residuals and receive separate exact dyadic service witnesses; their raw outputs are unchanged. Release, exclusive deadline, service capacity, static import and closing-work conditions are checked on each repaired witness. The maximum new-run monetary repair allowance is US$'+f'{maxbound:.14g}'+'. No dollar of materially late 8 h work is hidden inside a numerical allowance.','',
'The 8 h records supply cash-account evidence only, with no feasible-value or capture endpoint. Their source books, original deadline indices, cohort late-work arrays and independent settlement are retained in the service-failure NPZ files. For the twelve service-valid runs, the JSON records give exact rational savings, directed monetary enclosures, repair norms and file hashes; compressed witnesses supply all 35,040 service decisions at each of the four sites. All source and policy tails are zero in these runs.','',
'| Seed | Deadline h | Raw cash saving US$ | Service result | Repair allowance US$ |',
'|---|---:|---:|:---|---:|']
for r in sorted(rows,key=lambda x:(x['window_h'],x['seed'])):
    valid=r['all_exact_checks'];status='Exact feasible counterpart' if valid else f"Fails; {r['late_IT_mwh']:.6f} IT MWh late"
    allowance=f"{r['repair_cost_bound_usd']:.3g}" if valid else 'Not applied'
    md.append(f"| {r['seed']} | {r['window_h']} | {r['recorded_raw_saving_usd']:,.2f} | {status} | {allowance} |")
md += ['', 'The seven-window comparison retains the original exact 12/16/24 h policy counterparts from S30. The three new 24 h anchors equal the original full paths and exact repaired monetary results; they do not add independent replications. Across the 24→36 and 36→48 h intervals, policy gains average US$748.34 and US$737.51 per added deadline hour, respectively, while the certified opportunity increments move in the opposite direction. Both claims use the same annual calendar and account.','',
'Reproduction uses `code/audit_new_policy.py` for the independent new-run ledger checks and exact witnesses, followed by `code/assemble_results.py` for the matched seed comparison. The audit resumes by source-book hash and never substitutes a repaired witness for a materially failing operating trajectory. The 8 h cash-only rows remain separate throughout the tables, plots and interpretation.','']
(ROOT/'manuscript/S32_results_appendix.md').write_text('\n'.join(md),encoding='utf-8')
print('Wrote audit appendix for',len(rows),'runs')
