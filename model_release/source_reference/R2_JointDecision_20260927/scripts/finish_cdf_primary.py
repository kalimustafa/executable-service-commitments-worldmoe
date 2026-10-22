"""Freeze a completed 2024 checkpoint choice, then run the 2025 temporal test."""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT

p = argparse.ArgumentParser()
p.add_argument('--cash', action='store_true')
p.add_argument('--seed', type=int, default=20260920)
args = p.parse_args()
seed = args.seed
label = 'cash' if args.cash else 'cdf'
prefix = 'CashR2' if args.cash else 'CdfR2'
authorization_path = ROOT / 'results/replication_authorization.json'
authorization = json.loads(authorization_path.read_text()) if authorization_path.exists() else {}
if seed != 20260920:
    assert authorization.get('authorized') and seed in authorization['seeds']
    assert authorization['selected_family'] == prefix
summary_file = f'{label}_primary_complete.json' if seed == 20260920 else f'{label}_complete_s{seed}.json'
summary = json.loads((ROOT / 'results' / summary_file).read_text())
assert summary['status'] == 'complete' and summary['best_restored']
assert summary['seed'] == seed
epoch = summary['best_epoch']
checkpoint = ROOT / 'checkpoints' / f'{prefix}_seed{seed}_epoch{epoch:02d}.pt'
selection_row = json.loads((ROOT / 'results' / f'{prefix}_Eagle_2024_s{seed}_ep{epoch:02d}_coupled_cdf.json').read_text())
checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
selection = dict(
    frozen_at_utc=datetime.now(timezone.utc).isoformat(), selection_year=2024,
    target_year=2025, seed=seed, family=prefix, refinement_epoch=epoch,
    forecast_epoch=epoch, eta=0., safety=.9, opportunity_weight=1.,
    cost_bias_weight=1., exact_action=False, current_coupling=True,
    continuous_cost=True, forecast_checkpoint_sha256=checkpoint_hash,
    observer_sha256=hashlib.sha256((ROOT / 'cache/causal_observer.npz').read_bytes()).hexdigest(),
    selected_2024_saving_usd=selection_row['saving_usd'],
    selection_source=selection_row['name'] + '.json',
    selection_rule='maximum eligible 2024 annual savings after completed training; 50 USD min-delta controls stopping only',
    training_years=[2021, 2022, 2023], physical_training_years=[2022, 2023],
    parent_checkpoint='CdfR2_seed20260920_best.pt' if args.cash else 'QPR2_seed20260920_epoch10.pt',
    replication_scope='optimization seeds from shared pretrained initialization and frozen WorldMoE/R2 feature cache',
    evaluation_role='2025 temporal holdout; excluded from parameter fitting and this seed checkpoint selection',
    development_disclosure='2025 replay outcomes were inspected during earlier method development; this is not a never-inspected blind evaluation',
)
selection_file = ROOT / 'results' / (f'{label}_selection.json' if seed == 20260920 else f'{label}_selection_s{seed}.json')
if selection_file.exists():
    existing = json.loads(selection_file.read_text())
    assert existing['forecast_checkpoint_sha256'] == checkpoint_hash
    selection['frozen_at_utc'] = existing['frozen_at_utc']
selection_file.write_text(json.dumps(selection, indent=2))
print(prefix, 'FROZEN', json.dumps(selection), flush=True)

output = ROOT / 'results' / f'{prefix}_Eagle_2025_s{seed}_ep{epoch:02d}_coupled_cdf.json'
if output.exists():
    row = json.loads(output.read_text())
    assert row['forecast_checkpoint_sha256'] == checkpoint_hash
elif args.cash and epoch == 0:
    import torch
    parent_selection = json.loads((ROOT / 'results/cdf_selection.json').read_text())
    parent_epoch = parent_selection['refinement_epoch']
    parent_path = ROOT / 'checkpoints' / f'CdfR2_seed20260920_epoch{parent_epoch:02d}.pt'
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)['state_dict']
    child = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
    assert all(torch.equal(child[k], v) for k, v in parent.items())
    assert child['price_scale.weight'].count_nonzero() == 0 and child['price_scale.bias'].count_nonzero() == 0
    parent_result = ROOT / 'results' / f'CdfR2_Eagle_2025_s20260920_ep{parent_epoch:02d}_coupled_cdf.json'
    row = json.loads(parent_result.read_text())
    row.update(name=output.stem, seed=seed, forecast_epoch=0, cash_training_epoch=0,
               forecast_checkpoint_sha256=checkpoint_hash, baseline_reused_from=parent_result.name,
               baseline_identity='identical parent tensors and execution; added market scale identically one')
else:
    subprocess.run([sys.executable, '-B', '-u', str(ROOT / 'scripts/evaluate_scenario_recourse.py'),
                    '--year', '2025', '--eta', '0', '--safety', '0.9',
                    f'--{label}-epoch', str(epoch), '--current-coupling', '--continuous-cost',
                    '--seed', str(seed)], check=True, cwd=ROOT)
    row = json.loads(output.read_text())
assert abs(row['reference_cost_usd'] - 555168.2050556969) < 1e-6
assert abs(row['target_80_usd'] - 106084.33667311305) < 1e-6
row.update(evaluation_role=selection['evaluation_role'], development_disclosure=selection['development_disclosure'],
           selection_year=2024, parameter_fitting_years=[2021, 2022, 2023],
           replication_scope=selection['replication_scope'], parameters_updated_during_evaluation=False,
           final_stage_refinement_epoch=epoch)
output.write_text(json.dumps(row, indent=2))
passed = row['saving_usd'] >= row['target_80_usd'] and row['late_IT_mwh'] < 1e-8 and row['closing_it_mwh'] < 1e-8
gate = dict(passed=passed, seed=seed, family=prefix, refinement_epoch=epoch,
            target_year=2025, target_savings_usd=row['target_80_usd'], measured_savings_usd=row['saving_usd'],
            capture_fraction=row['capture_lower_fraction'], late_it_mwh=row['late_IT_mwh'],
            closing_it_mwh=row['closing_it_mwh'], remaining_usd=max(0., row['target_80_usd'] - row['saving_usd']),
            selection_year=2024, threshold_80_waived_by_user=authorization.get('threshold_80_waived', False),
            other_seeds_authorized=passed or authorization.get('authorized', False),
            evaluation_role=selection['evaluation_role'], source_result=row['name'] + '.json',
            forecast_checkpoint_sha256=checkpoint_hash)
gate_path = ROOT / 'results' / ('primary_80_gate.json' if seed == 20260920 else f'{label}_gate_s{seed}.json')
gate_path.write_text(json.dumps(gate, indent=2))
print(prefix, 'TEST COMPLETE', json.dumps(gate), flush=True)
