"""Reproduce historical paired upgrade values and explicit fee sensitivities.

This script does not estimate future-year expectations, tenant reservation
prices, actual contract fees, or a repaired perfect-information annual witness.
CSV arithmetic treats published decimal strings exactly. The separate annual
array re-settlement uses extended precision and is a numerical ledger check.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, localcontext
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path

import numpy as np


SEEDS = (20260920, 20260921, 20260922)
RIGHTS = (12, 16, 24)
UPGRADES = ((12, 16), (16, 24), (12, 24))
ANALYTICAL_TOTAL_COSTS = (0, 5000, 10000, 15000, 20000, 30000, 35000)
ANALYTICAL_OPERATOR_COSTS = (0, 5000, 10000)


def dec(value, places=18):
    value = Q(value)
    with localcontext() as ctx:
        ctx.prec = 70
        text = format(Decimal(value.numerator) / Decimal(value.denominator), f'.{places}f')
    return text.rstrip('0').rstrip('.') if '.' in text else text


def mean(values):
    return sum(values, Q(0)) / len(values)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resettle_arrays(path):
    z = np.load(path, allow_pickle=False)
    ld = np.longdouble
    dt = ld('0.25')
    rt, pue, pv = (z[k].astype(ld) for k in ('rt_lmp_usd_mwh', 'pue', 'pv_mw'))
    idle, power = (z[k].astype(ld) for k in ('idle_mw', 'work_power_mw'))
    x0 = z['reference_compute'].astype(ld)
    load0 = pue * (idle + power*x0)
    g0, v0 = np.maximum(load0-pv, 0), np.maximum(pv-load0, 0)
    rows = []
    for seed in SEEDS:
        x = z[f'compute_seed_{seed}'].astype(ld)
        load = pue*(idle+power*x)
        g, v = np.maximum(load-pv, 0), np.maximum(pv-load, 0)
        energy = dt*np.sum(rt*(g0-g))
        deviation = dt*np.sum(np.abs(g-g0))
        curtailment = dt*np.sum(v0-v)
        saving = energy-deviation+curtailment
        rows.append({
            'seed': seed,
            'observed_year_saving_before_terminal_difference_usd': str(saving),
            'energy_cost_reduction_usd': str(energy),
            'import_deviation_charge_usd': str(deviation),
            'curtailment_value_usd': str(curtailment),
            'maximum_reconstructed_grid_difference_mw': str(np.max(np.abs(g-z[f'grid_seed_{seed}']))),
        })
    return {
        'method': 'Independent longdouble settlement of supplied binary64 execution and physical arrays',
        'scope': 'Observed-year ledger only. Reported zero terminal difference is not independently proved from admissions/recovery records.',
        'intervals': int(rt.shape[0]), 'sites': int(rt.shape[1]),
        'reference_variable_IT_energy_mwh': str(dt*np.sum(power*x0)),
        'reference_cost_usd': str(dt*np.sum(rt*g0+v0)+ld('2409.7842143056')),
        'rows': rows,
        'mean_saving_before_terminal_difference_usd': str(np.mean([ld(r['observed_year_saving_before_terminal_difference_usd']) for r in rows])),
    }


def information_gap_examples():
    """Exact two-state counterexamples; illustrative payoffs, not market data."""
    examples = [
        ('PI_increment_exceeds_adapted_increment', [(0, 0)], [(0, 0), (1, -1), (-1, 1)]),
        ('adapted_increment_exceeds_PI_increment', [(2, 0), (0, 2)], [(2, 0), (0, 2), (2, 2)]),
    ]
    out = []
    for label, actions_a, actions_b in examples:
        values = []
        for actions in (actions_a, actions_b):
            pi = mean([Q(max(a[s] for a in actions)) for s in (0, 1)])
            adapted = max(mean(list(map(Q, action))) for action in actions)
            values.append((pi, adapted, pi-adapted))
        a, b = values
        assert b[1]-a[1] == b[0]-a[0]-(b[2]-a[2])
        out.append({'label': label, 'probabilities': ['1/2', '1/2'],
                    'A_action_payoffs': actions_a, 'B_action_payoffs': actions_b,
                    'expected_PI_increment': dec(b[0]-a[0]),
                    'adapted_expected_increment': dec(b[1]-a[1]),
                    'information_gap_A': dec(a[2]), 'information_gap_B': dec(b[2])})
    return out


def analyze(repo, output, scalar_file):
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        'paired_contract_summary': repo/'source_data/current/fig3_service_windows.csv',
        'published_mean_contrasts': repo/'source_data/current/0927_adapted_window_contrasts.csv',
        'checkpoint_provenance': repo/'source_data/current/current_provenance.json',
        'annual_24h_arrays': repo/'source_data/grid_metrics_inputs_2025.npz',
        'public_Lagrangian_summaries': repo/'source_data/current/extensions/certified_value_bounds.csv',
        'manuscript_reported_certificates': scalar_file,
    }
    source_rows = read_csv(paths['paired_contract_summary'])
    by = {(int(r['seed']), int(r['window_h'])): r for r in source_rows}
    assert set(by) == {(s, w) for s in SEEDS for w in RIGHTS}
    assert len(source_rows) == 9
    ref = {Q(r['reference_cost_usd']) for r in source_rows}
    assert len(ref) == 1
    provenance = json.loads(paths['checkpoint_provenance'].read_text())
    expected_hashes = {int(r['seed']): r['sha256'] for r in provenance['checkpoints']}
    for seed in SEEDS:
        assert {by[seed, w]['forecast_checkpoint_sha256'] for w in RIGHTS} == {expected_hashes[seed]}
    settlement_identity_error = max(abs(Q(r['saving_usd'])-(Q(r['reference_cost_usd'])-Q(r['adjusted_cost_usd']))) for r in source_rows)
    assert settlement_identity_error < Q('0.0000001')
    published_means = {int(r['window_h']): Q(r['worldmoe_saving_usd']) for r in read_csv(paths['published_mean_contrasts'])}
    means = {w: mean([Q(by[s,w]['saving_usd']) for s in SEEDS]) for w in RIGHTS}
    assert max(abs(means[w]-published_means[w]) for w in RIGHTS) < Q('0.0000001')
    paired = []
    summary = []
    for a, b in UPGRADES:
        differences = []
        for seed in SEEDS:
            av, bv = (Q(by[seed,w]['saving_usd']) for w in (a,b))
            difference = bv-av
            differences.append(difference)
            paired.append({'from_h': a, 'to_h': b, 'seed': seed,
                           'same_checkpoint_sha256': expected_hashes[seed],
                           'saving_A_usd': dec(av), 'saving_B_usd': dec(bv),
                           'paired_increment_usd': dec(difference)})
        summary.append({'from_h': a, 'to_h': b,
                        'mean_increment_usd': dec(mean(differences)),
                        'minimum_seed_increment_usd': dec(min(differences)),
                        'maximum_seed_increment_usd': dec(max(differences)),
                        'mean_saving_A_usd': dec(means[a]), 'mean_saving_B_usd': dec(means[b]),
                        'scope': 'Historical same-checkpoint paired annual summary; seed range is not a future-year confidence interval'})
    scenarios, transfer_frontiers = [], []
    for row in summary:
        m, lo, hi = map(Q, (row['mean_increment_usd'], row['minimum_seed_increment_usd'], row['maximum_seed_increment_usd']))
        for total in ANALYTICAL_TOTAL_COSTS:
            scenarios.append({'from_h':row['from_h'], 'to_h':row['to_h'],
                              'assumed_incremental_F_plus_K_usd': total,
                              'mean_historical_net_usd': dec(m-total),
                              'minimum_seed_historical_net_usd': dec(lo-total),
                              'maximum_seed_historical_net_usd': dec(hi-total),
                              'scenario_status':'analytical cost sensitivity; no observed fees or ex-ante probability claim'})
        for operator_cost in ANALYTICAL_OPERATOR_COSTS:
            transfer_frontiers.append({'from_h':row['from_h'], 'to_h':row['to_h'],
                                       'assumed_operator_implementation_cost_usd':operator_cost,
                                       'mean_historical_operator_transfer_ceiling_usd':dec(m-operator_cost),
                                       'all_three_seeds_transfer_ceiling_usd':dec(lo-operator_cost),
                                       'tenant_transfer_floor':'D_tenant + K_tenant (unobserved)',
                                       'scope':'Retrospective conditional IR interval; not a market quote, IC proof, or ex-ante procurement price'})
    settled = resettle_arrays(paths['annual_24h_arrays'])
    array_discrepancies = {str(r['seed']): abs(Q(r['observed_year_saving_before_terminal_difference_usd'])-Q(by[r['seed'],24]['saving_usd'])) for r in settled['rows']}
    assert max(array_discrepancies.values()) < Q('0.000001')
    scalars = json.loads(scalar_file.read_text())
    primary = SEEDS[0]
    intervals = []
    for a,b in UPGRADES:
        hb = Q(scalars['raw_24h_hindsight_saving_usd']) if b == 24 else Q(by[primary,b]['saving_usd'])
        ha = Q(by[primary,a]['saving_usd'])
        ua = min(Q(scalars[k][str(a)]) for k in ('lag_savings_ceiling_usd','cut_savings_ceiling_usd'))
        ub = min(Q(scalars[k][str(b)]) for k in ('lag_savings_ceiling_usd','cut_savings_ceiling_usd'))
        strong_ua = max(Q(scalars[k][str(a)]) for k in ('lag_savings_ceiling_usd','cut_savings_ceiling_usd'))
        intervals.append({'from_h':a,'to_h':b,
                          'nominal_combined_lower_usd':dec(max(Q(0),hb-ua)),
                          'nominal_both_methods_lower_usd':dec(max(Q(0),hb-strong_ua)),
                          'nominal_upper_usd':dec(ub-ha),
                          'A_witness_source':f'primary fixed seed {primary} historical summary, not yet exact repaired',
                          'B_witness_source':'raw perfect-information annual scalar' if b==24 else f'primary fixed seed {primary} historical summary',
                          'correction_rule':'lower=max(0,H_B_raw-delta_B-U_A); upper=U_B-H_A_raw+delta_A, plus any additional ceiling allowance',
                          'status':'conditional mathematical interval; unavailable original witnesses/certificates prevent annual certification here'})
    examples = information_gap_examples()
    energy = Q(settled['reference_variable_IT_energy_mwh'])
    normalizations = [{'from_h':r['from_h'],'to_h':r['to_h'],
                       'mean_historical_budget_usd_per_reference_IT_MWh':dec(Q(r['mean_increment_usd'])/energy),
                       'denominator_mwh':dec(energy),
                       'scope':'Optional normalization by common annual variable-IT execution; not shifted work, avoided MWh, or an observed tariff'} for r in summary]
    results = {
        'status':'Historical numerical evidence and explicitly analytical sensitivities; ex-ante value and actual contract net value not identified',
        'source_commit':'323c2084e7ee668ba512fb1e33dd56b8394b32d4',
        'sources':{k:{'path':str(p),'sha256':sha256(p)} for k,p in paths.items()},
        'checks':{
            'complete_3_by_3_paired_grid':True,
            'same_checkpoint_within_seed_across_rights':True,
            'same_reference_cost_across_all_rows':True,
            'maximum_recorded_saving_identity_error_usd':dec(settlement_identity_error),
            '24h_array_vs_summary_maximum_difference_usd':dec(max(array_discrepancies.values())),
            '12h_16h_full_trajectories_available':False,
            'complete_admission_deadline_replay_done':False,
            'complete_first_release_replay_done':False,
            'observed_fee_or_tenant_cost_data_available':False,
            'ex_ante_scenario_law_supplied':False,
            'original_annual_PI_witness_or_dual_reverified':False,
        },
        'contract_means_usd':{str(w):dec(v) for w,v in means.items()},
        'paired_upgrade_summary':summary,
        'conditional_PI_increment_intervals':intervals,
        'historical_fee_scenarios':scenarios,
        'conditional_transfer_frontiers':transfer_frontiers,
        'optional_energy_normalization':normalizations,
        'independent_24h_settlement':settled,
        'analytical_information_gap_counterexamples':examples,
        'arithmetic_note':'Published CSV strings are treated as exact decimals for descriptive recomputation, not as certified exact physical observations. Annual array settlement is longdouble numerical arithmetic.',
    }
    (output/'economic_value_results.json').write_text(json.dumps(results,indent=2)+'\n')
    write_csv(output/'paired_upgrades.csv',paired)
    write_csv(output/'upgrade_summary.csv',summary)
    write_csv(output/'historical_fee_scenarios.csv',scenarios)
    write_csv(output/'conditional_transfer_frontiers.csv',transfer_frontiers)
    write_csv(output/'conditional_PI_increment_intervals.csv',intervals)
    write_csv(output/'energy_normalization.csv',normalizations)
    print(json.dumps({'paired_upgrade_summary':summary,
                      'conditional_PI_increment_intervals':intervals,
                      'checks':results['checks']},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--reported-scalars',type=Path,default=Path(__file__).with_name('reported_certificate_scalars.json'))
    args = parser.parse_args()
    analyze(args.repo,args.output_dir,args.reported_scalars)
