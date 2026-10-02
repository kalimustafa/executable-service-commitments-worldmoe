from src.runtime import ROOT, REPO, PAPER, LEGACY
from src.data.access_guard import install_guard, register_historical, assert_years, read_development_parquet
import json
import hashlib
import csv
from pathlib import Path
import sys
import os
install_guard()
import torch
torch.set_num_threads(4)
torch.zeros(100000).clone()
import numpy as np
from dcvpp.vpp_world_moe.config import _read_config_with_base, WorldMoEConfig
from dcvpp.vpp_world_moe.model import VPPWorldMoE, load_plannerless_released_w1_state

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def main():
    print('Comparing recorded 2024 W1 evidence', flush=True)
    rows = []
    states = {}
    folders = [('initial_released_w1', 20240813, REPO / 'artifacts/vpp_world_moe_r5_v2_2_2025final/run=r5v22-w1-continuous-action-formal-20260824-seed20240813/phase=w1', REPO / 'configs/vpp_world_moe_R5_v2_2.yaml')]
    folders += [(f'multiseed_{seed}', seed, REPO / f'artifacts/r_ecmoe_multiseed_20260910/world_moe/run=multiseed-20260910-seed{seed}/phase=w1', REPO / f'configs/multiseed_20260910_{seed}_w1.yaml') for seed in (20240813, 20240829, 20240917)]
    base_contract = None
    for name, seed, folder, cfg_path in folders:
        for file in ('state.json', 'gate.json'):
            register_historical(folder / file)
        state = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
        gate = json.loads((folder / 'gate.json').read_text(encoding='utf-8'))
        record = state['training_control']['formal_primary_record']
        cfg = _read_config_with_base(cfg_path)
        model_cfg = WorldMoEConfig.from_mapping(cfg['model'])
        contract = {'data': {k: cfg['data'][k] for k in ('selection_bundle_root', 'official_settlement_root', 'frozen_physical_twin', 'frozen_physical_surrogate')},
                    'splits': {k: cfg['protocol']['splits'][k] for k in ('train', 'validation')},
                    'model': dict(model_cfg.__dict__)}
        if base_contract is None:
            base_contract = contract
        assert contract == base_contract, f'W1 data/model contract mismatch: {name}'
        metrics = record['selection_metrics']
        checkpoint = Path(state['best_checkpoint'])
        assert checkpoint.exists(), f'Missing core W1 checkpoint: {checkpoint}'
        row = dict(model=name, seed=seed, selected_epoch=record['epoch'], primary_score=record['primary_score'],
                   **metrics['horizon_scores'], native_eligible=record['eligible'], native_gate=gate['native_evidence']['status'],
                   selection_role=state['best_checkpoint_selection_role'], checkpoint=str(checkpoint),
                   checkpoint_bytes=checkpoint.stat().st_size, recorded_sha256=gate['checkpoint_id'],
                   validation_days=contract['splits']['validation']['days'], validation_year=2024,
                   blocks=model_cfg.num_blocks, world_width=model_cfg.d_model,
                   h96_latent_floor_fraction=metrics['latent_free_bit_floor_fraction']['h96'],
                   h96_latent_raw_kl=metrics['latent_raw_kl']['h96'])
        rows.append(row)
        states[name] = (state, cfg, cfg_path)
    winner = min(rows, key=lambda row: (row['primary_score'], row['model']))
    assert winner['native_eligible'] and winner['native_gate'] == 'PASS'
    old = rows[0]
    for row in rows[1:]:
        assert states[row['model']][0]['training_control']['formal_primary_record']['selection_metrics']['queue_rollout_valid_counts'] == states[old['model']][0]['training_control']['formal_primary_record']['selection_metrics']['queue_rollout_valid_counts']
    with (ROOT / 'results/w1_foundation_comparison_2024.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    state, cfg, cfg_path = states[winner['model']]
    register_historical(winner['checkpoint'])
    cfg['model']['gradient_checkpointing'] = False
    print('Loading selected frozen W1', flush=True)
    model = VPPWorldMoE(WorldMoEConfig.from_mapping(cfg['model']), include_legacy_planner=False)
    payload = torch.load(winner['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    load_plannerless_released_w1_state(model, payload['model_state'])
    model.eval().requires_grad_(False)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    actual_hash = sha(winner['checkpoint'])
    assert actual_hash == winner['recorded_sha256']
    selected = dict(winner, config=str(cfg_path), parent=state['training_control']['parent_checkpoint'],
                    criterion='minimum common full-2024 worst-horizon proper score, eligible native W1; no downstream outcome tuning',
                    frozen_parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=0,
                    extraction_blocks_zero_based=[2, 5, 9], extraction_blocks_one_based=[3, 6, 10],
                    common_contract=base_contract, locked_targets_accessed=False,
                    checkpoint_payload_keys=list(payload), source_states=[str(f[2] / 'state.json') for f in folders])
    (ROOT / 'configs/selected_w1.json').write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding='utf-8')
    notes = ['# Frozen W1 foundation selection (2024 only)', '',
             'Selected: multiseed 20240829, epoch 80. Preserve its complete W1 weights and W0 parent lineage.', '',
             '| W1 | Epoch | h1 | h4 | h16 | h96 / primary | Native gate |',
             '|---|---:|---:|---:|---:|---:|---|']
    for row in rows:
        notes.append(f"| {row['model']} | {row['selected_epoch']} | {row['h1']:.9f} | {row['h4']:.9f} | {row['h16']:.9f} | {row['h96']:.9f} | {row['native_gate']} |")
    notes += ['', f"The chosen primary score is {(1-winner['primary_score']/old['primary_score'])*100:.3f}% below the initially released W1, and it is lower at all four horizons.",
              'All four use identical configured train/validation windows, the same physical/source contracts, model configuration, and recorded rollout valid counts. The W1 selection metric is the existing proper-score worst-horizon criterion. This comparison does not establish which frozen representation will necessarily give the best downstream R2.',
              'The initial released W1 was explicitly fixed at epoch 56 by the prior user selection; it is not silently replaced by another historical epoch. The three later seeds use validation-score-only selection.',
              'All selected W1 native gates pass. The later W0 parent native gate is FAIL under the historical validation-score-only protocol; this is retained in the selection manifest, not relabeled PASS. No unrelated W0 weights are spliced into W1.',
              'All four have h96 latent free-bit floor fraction 1.0. This remains a limitation of the inherited world model, not evidence that the new gate will help.',
              f"Selected W1 strict-compatible load succeeded; {selected['frozen_parameters']:,} frozen parameters, zero trainable parameters. Legal interaction blocks: 3, 6 and 10 (indices 2, 5 and 9), width 256.",
              'The older R5-V2.1 precursor was not treated as the released continuous-action W1: its state was TRAINED_AWAITING_GATE and it predates the R5-V2.2 dynamics repair.',
              'No 2025 targets, forecasts, economics or world cache were opened. Existing recorded 2024 validation is development evidence, not independent holdout evidence.']
    (ROOT / 'reports/W1_FOUNDATION_SELECTION.md').write_text('\n'.join(notes)+'\n', encoding='utf-8')
    panel_path = PAPER / 'experiments/gate_hourly_panel.parquet'
    frame = read_development_parquet(panel_path, columns=['timestamp_utc','date','zone','split'])
    split = {}
    for name, f in frame.groupby('split'):
        keys = f[['timestamp_utc','zone']].astype(str).to_csv(index=False).encode()
        split[name] = dict(first_date=str(f.date.min()), last_date=str(f.date.max()), rows=len(f),
                           dates=sorted(f.date.unique().tolist()), row_keys_sha256=hashlib.sha256(keys).hexdigest())
    physical = np.load(LEGACY / 'cache/development_physical_reference.npz', allow_pickle=False)
    import pandas as pd
    dates = assert_years(physical['dates'], 'development physical reference')
    eligible = np.flatnonzero(dates >= pd.Timestamp('2024-07-01', tz='UTC'))
    groups = np.split(eligible, np.flatnonzero(np.diff(dates[eligible].asi8) != 86400 * 10**9)+1)
    group = max(groups, key=lambda g: (len(g), -g[0]))
    split['economic_selection'] = dict(start_utc=dates[group[0]].isoformat(), end_exclusive_utc=(dates[group[-1]]+pd.Timedelta(days=1)).isoformat(), days=len(group),
                                        criterion='longest contiguous available physical block after calibration; source select_2024_economics.configure_validation')
    split['provenance'] = dict(price_panel=str(panel_path), panel_sha256=sha(panel_path),
                              split_source=str(PAPER / 'experiments/prepare_gate_data.py'),
                              physical_source=str(LEGACY / 'cache/development_physical_reference.npz'),
                              all_candidates_share_same_rows=True, locked_targets_accessed=False)
    (ROOT / 'configs/splits_2024_frozen.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    print(json.dumps(dict(winner={k:winner[k] for k in ('model','selected_epoch','primary_score')}, frozen_parameters=selected['frozen_parameters'],
                          splits={k:{x:y for x,y in v.items() if x != 'dates'} for k,v in split.items() if k != 'provenance'},
                          runtime=dict(python=sys.version,torch=torch.__version__,cuda=torch.cuda.is_available())), indent=2), flush=True)

if __name__ == '__main__':
    main()
