"""Conditional replication, followed by a single frozen 2025 evaluation per seed."""
import sys,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
gate=root/'results/primary_80_gate.json'
if not gate.exists() or not json.loads(gate.read_text()).get('passed',False):
    raise SystemExit('User requires primary-seed 80% capture before any additional seed training. Mere positive improvement is insufficient.')
if json.loads(gate.read_text()).get('family') in ['DeepR2','QPR2','DistR2']:
    raise SystemExit('Replication must use the accepted new architecture; the older StructuredR2-only runner is disabled.')
for seed in [20260921,20260922]:
    completed=root/'results'/f'structured_complete_s{seed}.json'
    if not completed.exists():subprocess.run([sys.executable,'-B','-u',str(root/'scripts/train_structured.py'),'--seed',str(seed)],check=True,cwd=root)
    result=json.loads(completed.read_text());print('REPLICATION COMPLETE',seed,'BEST',result['best_epoch'],'GAIN',result['improvement_usd'],flush=True)
subprocess.run([sys.executable,'-B','-u',str(root/'scripts/evaluate_structured_final.py'),'--seeds','20260920','20260921','20260922'],check=True,cwd=root)
rows=[]
for seed in [20260920,20260921,20260922]:
    result=json.loads((root/'results'/f'structured_complete_s{seed}.json').read_text())
    row=json.loads((root/'results'/f'StructuredR2_Eagle_2025_s{seed}_ep{result["best_epoch"]:02d}.json').read_text());rows.append(row)
average=sum(r['saving_usd'] for r in rows)/3;ceiling=132605.4208413913
summary=dict(status='complete',seeds=[r['seed'] for r in rows],annual_savings_usd=[r['saving_usd'] for r in rows],mean_savings_usd=average,certified_savings_ceiling_usd=ceiling,capture_lower_fraction=average/ceiling,target_savings_usd=.8*ceiling,target_met=average>=.8*ceiling and all(r['late_IT_mwh']<1e-8 and r['closing_it_mwh']<1e-8 for r in rows),all_zero_late=all(r['late_IT_mwh']<1e-8 for r in rows),all_zero_closing=all(r['closing_it_mwh']<1e-8 for r in rows),training_years=[2021,2022,2023],selection_year=2024,evaluation_year=2025,evaluation_role='exploratory; previously examined calendar; checkpoints frozen before these runs',original_frozen_three_seed_savings_usd=71968.45705157731)
(root/'results/structured_final_summary.json').write_text(json.dumps(summary,indent=2));print('FINAL SUMMARY',json.dumps(summary),flush=True)
