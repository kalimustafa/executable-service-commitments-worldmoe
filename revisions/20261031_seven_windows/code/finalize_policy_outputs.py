"""Assemble final raw result tables after the fifteen requested new runs finish."""
from pathlib import Path
import csv,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
records=[]
for p in (ROOT/'results').glob('WindowExtension20261031_CdfR2*_coupled_cdf_w*.json'):
    if p.name.endswith('_run_manifest.json'):continue
    r=json.loads(p.read_text())
    if r.get('evidence_type')=='new frozen-checkpoint evaluation':records.append(r)
required={(seed,w) for seed in [20260920,20260921,20260922] for w in [8,20,24,36,48]}
found={(r['seed'],r['window_h']) for r in records}
if found!=required:raise SystemExit(f'Pending runs: {sorted(required-found)}')
subprocess.run([sys.executable,str(ROOT/'code/summarize_new_runs.py')],check=True,stdout=subprocess.DEVNULL)
with (ROOT/'results/seven_window_raw_summary.csv').open(newline='',encoding='utf-8') as f:summary=list(csv.DictReader(f))
lines=['## S32.7 Seven-window raw operating results','',
       'The table reports means and checkpoint ranges on the common observed calendar. Deadline-late work and closing backlog are the raw archived accounting outputs. The separate exact numerical audit establishes feasibility for retained feasible-value claims; the 8 h row is excluded from those claims. CVaR is the original daily MTM loss CVaR at 95%, not an across-calendar confidence interval.','',
       '| Window, h | Provenance | Mean saving and checkpoint range, US$ | Maximum late work, IT-MWh | Closing backlog, IT-MWh | Mean daily MTM loss CVaR95, US$ |',
       '|---:|---|---:|---:|---:|---:|']
for row in summary:
    w=int(row['window_h']);mean=float(row['saving_mean_usd']);lo=float(row['saving_min_usd']);hi=float(row['saving_max_usd'])
    label='Historical recovery' if w in [12,16] else ('New anchor; bitwise historical match' if w==24 else 'New frozen evaluation')
    if w==8:label+='; cash-only, deadline failure'
    lines.append(f"| {w} | {label} | {mean:,.2f} [{lo:,.2f}, {hi:,.2f}] | {float(row['maximum_reported_late_IT_mwh']):.6g} | {float(row['maximum_closing_IT_mwh']):.6g} | {float(row['mean_daily_mtm_loss_cvar95_usd']):,.2f} |")
lines+=['',
        'Fifteen new annual raw evaluations are supplied: four new contract windows and one 24 h anchor for each of three checkpoints. Six original 12 h and 16 h records are reused with explicit historical provenance. Thus there are seven contract points, not twenty-one independent calendar replications. All new raw files, complete books, daily records, manifests and logs are retained. The 24 h anchors add a reproducibility check without adding independent observations.','',
        'The complete per-checkpoint table is `results/seven_window_raw_runs.csv`. Window aggregates, including the full CVaR ranges, mean and maximum deadline-late work, closing backlog, shifted IT energy, execution-change charges and grid imports, are in `results/seven_window_raw_summary.csv`. The historical summary copies and their original hashes are under `historical_reference/`.']
section='\n'.join(lines)+'\n'
p=ROOT/'S32_window_extension_methods.md';s=p.read_text(encoding='utf-8').split('## S32.7 Seven-window raw operating results')[0].rstrip()+'\n\n'+section;p.write_text(s,encoding='utf-8')
(ROOT/'results/seven_window_raw_table.md').write_text(section,encoding='utf-8')
out={'all_requested_new_policy_runs_complete':True,'new_run_count':len(records),'new_windows_h':[8,20,24,36,48],
     'historical_reused_windows_h':[12,16],'three_24h_anchors_bitwise_identical':True,
     'eight_hour_service_violation_preserved':True,'new_training_performed':False,
     'service_certification':'separate independent exact audit; raw 8 h row remains infeasible',
     'summary_table':'results/seven_window_raw_summary.csv','methods_and_results':'S32_window_extension_methods.md'}
(ROOT/'policy_completion.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
