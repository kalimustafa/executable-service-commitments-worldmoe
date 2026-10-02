"""Supplemental 2025 test evaluation of the frozen epoch-130 mechanisms.

Uses the existing sealed 2025 corpus and replay, with the same intervention
registry and daily statistics as the 2024 study. The prior one-shot model-suite
results remain historical evidence; this run is explicitly supplemental.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
PREPARED = REPO.parent / 'NatureEnergy_R-ECMoE_投稿修订_20260829' / 'round6_cpu_only_phase_20260831' / 'scripts'
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(PREPARED))

# Preserve the established Windows DLL import order.
import pyarrow.dataset  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import r_ecmoe_frozen_backend as backend  # noqa: E402
from frozen_eval_common import DEFAULT_CHECKPOINT, DEFAULT_NORMALIZER, FIXED_RARE_THRESHOLD_USD_PER_MWH  # noqa: E402
from dcvpp.vpp_world_moe.r_ecmoe import RiskConditionPacket  # noqa: E402
from dcvpp.vpp_world_moe import r_ecmoe_evaluation as evaluation  # noqa: E402
from dcvpp.vpp_world_moe.r_ecmoe_sealed_2025 import (  # noqa: E402
    FINAL_SPLIT, RECMoESealed2025Evaluator, SealedRealizedHistoricalReplay,
    _validate_sealed_corpus,
)
from dcvpp.vpp_world_moe.r_ecmoe_sealed_corpus import build_sealed_final_corpus, _locked_sibling  # noqa: E402

MARKER = REPO / 'artifacts/paper_finalization/r_ecmoe_2025_final_suite_marker.json'
PRIOR = REPO / 'artifacts/r_ecmoe_final_2025_suite/one_shot_20260827/suite_meta.json'


class TestMechanismEvaluator(RECMoESealed2025Evaluator):
    """Evaluate every registered intervention on true 2025 batches."""

    __test__ = False
    _aggregate_days = evaluation.RECMoE2024Evaluator._aggregate_days
    _summaries = evaluation.RECMoE2024Evaluator._summaries
    _paired_comparisons = evaluation.RECMoE2024Evaluator._paired_comparisons

    def run_mechanisms(self, record):
        evaluation._validate_variant_registry()
        self.expected_validation_days = self.expected_days  # Shared statistics API.
        self.bootstrap_seed, self.bootstrap_resamples, self.block_days = 20240813, 2000, 7
        corpus = self.corpus_factory(dict(self.marker))
        _validate_sealed_corpus(corpus, self.marker)
        replay = SealedRealizedHistoricalReplay.from_model(self.model)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        decision_rows, reference_dates = [], None
        batch_count = 0
        expected_rows = self.expected_days * len(evaluation.EVALUATION_DECISIONS) * len(evaluation.REGISTERED_VARIANTS)
        record(status='running', expected_days=self.expected_days, expected_replay_rows=expected_rows)
        with torch.inference_mode():
            for stage_id, origin, horizon in evaluation.EVALUATION_DECISIONS:
                decision_dates = []
                for batch in corpus.batches(
                    FINAL_SPLIT, self.batch_size, device=self.runtime.device,
                    shuffle_seed=None, max_days=None, origin=origin,
                    require_h96=False, include_next_day=False,
                ):
                    dates = tuple(str(value) for value in batch.get('operating_dates', ()))
                    packet = batch.get('world_input')
                    if not dates or packet is None or len(dates) != int(packet.fast_values.shape[0]):
                        raise RuntimeError('2025 batch omitted aligned dates/world input')
                    if any(date.fromisoformat(value).year != 2025 for value in dates):
                        raise RuntimeError('2025 mechanism evaluation received another year')
                    if not bool((packet.origin_index == origin).all()):
                        raise RuntimeError('2025 packet origin mismatch')
                    decision_dates.extend(dates)
                    anchor = batch.get('forecast_anchors', {}).get('da')
                    cap = batch.get('forecast_residual_caps', {}).get('da')
                    if not isinstance(anchor, torch.Tensor) or not isinstance(cap, torch.Tensor):
                        raise RuntimeError('2025 batch omitted frozen DA anchors')
                    risk = RiskConditionPacket.defaults(len(dates), device=self.runtime.device, horizon=horizon, stage=stage_id)
                    expert_packet, encode_ms = self._timed_encode(packet, anchor, cap)
                    # Every action is fixed before admitting realized targets to replay.
                    decisions = [(variant, *self._timed_top2(expert_packet, risk, variant))
                                 for variant in evaluation.REGISTERED_VARIANTS]
                    scale = self._cost_scale(batch)
                    for variant, decision, decision_ms in decisions:
                        realized = replay(decision, batch=batch, risk=risk)
                        decision_rows.extend(evaluation._execution_rows(
                            decision, realized, variant=variant, dates=dates,
                            stage_id=stage_id, origin=origin, horizon=horizon, cost_scale=scale,
                            equivalent_latency_ms_per_sample=(encode_ms + decision_ms) / len(dates),
                            queue_event_target=batch.get('queue_risk_event_target'),
                            queue_event_mask=batch.get('queue_risk_event_mask'),
                        ))
                    batch_count += 1
                    record(status='running', completed_replay_rows=len(decision_rows),
                           completed_batches=batch_count, stage_id=stage_id, origin=origin,
                           horizon=horizon, last_date=dates[-1])
                ordered = tuple(decision_dates)
                if len(ordered) != self.expected_days or len(set(ordered)) != self.expected_days:
                    raise RuntimeError('2025 intervention did not cover the frozen unique day set')
                if reference_dates is not None and ordered != reference_dates:
                    raise RuntimeError('2025 decision windows have different dates')
                reference_dates = ordered
        if len(decision_rows) != expected_rows:
            raise RuntimeError('Incomplete 2025 mechanism replay')
        record(status='aggregating', completed_replay_rows=len(decision_rows))
        # Save origin rows before aggregation, making numerical output recoverable.
        evaluation._atomic_csv(self.output_root / 'per_decision_variant.csv', decision_rows)
        daily = self._aggregate_days(decision_rows)
        summaries = self._summaries(daily)
        comparisons = self._paired_comparisons(daily)
        evaluation._atomic_csv(self.output_root / 'per_day_variant.csv', daily)
        evaluation._atomic_csv(self.output_root / 'variant_summary.csv', summaries)
        evaluation._atomic_csv(self.output_root / 'paired_comparisons.csv', comparisons)
        common = {
            'year': 2025, 'split': FINAL_SPLIT, 'evaluation_role': 'supplemental_mechanism_test',
            'checkpoint_epoch': 130, 'seed': 20240813,
            'expected_evaluable_days': self.expected_days, 'dataset_dates': list(reference_dates),
            'decisions_per_day': len(evaluation.EVALUATION_DECISIONS),
            'decision_variant_rows_before_daily_aggregation': len(decision_rows),
            'paired_daily_rows': len(daily), 'batch_size': self.batch_size,
            'comparison_design': 'same_checkpoint_intervention_only',
            'shared_checkpoint_for_all_variants': True, 'separately_retrained_variant_claims': False,
            'variant_definitions': [v.evidence_definition() for v in evaluation.REGISTERED_VARIANTS],
            'outcome_basis': 'realized_historical_replay',
            'cost_normalization': 'train_frozen_baseline_daily_cost * active_horizon / 96',
            'train_frozen_cost_scale': self.cost_scale,
            'best_checkpoint': str(self.r_checkpoint_path), 'w1_checkpoint': str(self.runtime.checkpoint_path),
            'prior_2025_model_suite': str(PRIOR), 'prior_2025_already_evaluated': True,
            'original_frozen_marker': str(self.marker_path),
            'all_actions_fixed_before_realized_target_read': True,
            'w1_packet_reused_across_variants': True, 'w1_encode_calls': batch_count,
            'training_performed': False, 'checkpoint_selection_performed': False,
            'threshold_fitting_performed': False, 'scaler_fitting_performed': False,
            'final_results_may_modify_selection': False, 'locked_2025_opened': True,
            'external_solver': False, 'candidate_search': False,
            'bootstrap': {'block_days': 7, 'resamples': 2000, 'seed': 20240813,
                          'method': 'circular_moving_block_over_daily_paired_differences'},
            'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
            'data_lineage': corpus.sealed_final_lineage,
        }
        evaluation._atomic_json(self.output_root / 'ablation_summary.json', {
            **common, 'schema': 'dcvpp.r_ecmoe.ablation_summary_2025.v1',
            'primary_variant': 'top2', 'variant_summaries': summaries, 'paired_comparisons': comparisons,
            'interpretation': 'negative top2-minus-comparator cost favors the registered Top-2 model',
        })
        evaluation._atomic_json(self.output_root / 'evaluation_2025_meta.json', common)
        return common


def _market_summary(output, world_config, dates):
    """Apply the existing 2024 threshold only as a post-decision diagnostic."""
    settlement_root = _locked_sibling(backend._configured_data_path(world_config, 'official_settlement_root'))
    state_by_date = {}
    for day in dates:
        with np.load(settlement_root / f'{day}.npz', allow_pickle=False) as source:
            dart = np.asarray(source['da_lbmp'], dtype=np.float64) - np.asarray(source['rt_lbmp'], dtype=np.float64)
        state_by_date[day] = float(dart.max() - dart.min()) > FIXED_RARE_THRESHOLD_USD_PER_MWH
    daily = backend._read_csv(output / 'per_day_variant.csv')
    rows = []
    for standard, native in backend.STANDARD_MECHANISM_TO_NATIVE.items():
        for stratum in ('all', 'rare', 'routine'):
            selected_dates = {day for day, rare in state_by_date.items()
                              if stratum == 'all' or rare == (stratum == 'rare')}
            values = [row for row in daily if row['variant'] == native and row['operating_date'] in selected_dates]
            if len(values) != len(selected_dates) or not values:
                raise RuntimeError(f'Incomplete 2025 market stratum: {native}/{stratum}')
            row = {'variant': standard, 'native_variant': native, 'stratum': stratum,
                   'year': 2025, 'split': FINAL_SPLIT, 'days': len(values),
                   'checkpoint_epoch': 130, 'active_top_k': 2,
                   'rare_threshold_usd_per_mwh': FIXED_RARE_THRESHOLD_USD_PER_MWH,
                   'threshold_fit_year': 2024, 'comparison_design': 'same_checkpoint_runtime_intervention'}
            for field in ('realized_operating_cost', 'normalized_operating_cost', 'deadline_expired_work',
                          'unserved_energy', 'battery_throughput', 'pv_curtailed_energy', 'grid_import_energy'):
                row[f'mean_{field}'] = sum(float(value[field]) for value in values) / len(values)
            rows.append(row)
    evaluation._atomic_csv(output / 'epoch130_mechanisms_by_market_state.csv', rows)


def _hashes(paths):
    result = {}
    for path in paths:
        with Path(path).open('rb') as source:
            result[str(Path(path).resolve())] = hashlib.file_digest(source, 'sha256').hexdigest()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob('*.json')) or any(output.glob('*.csv')):
        raise SystemExit('Use a new output directory for this supplemental evaluation.')
    started = time.monotonic()
    status = {'schema': 'dcvpp.frozen-mechanisms.progress.v1', 'pid': os.getpid(),
              'status': 'initializing', 'year': 2025, 'split': FINAL_SPLIT,
              'checkpoint_epoch': 130, 'seed': 20240813, 'expected_days': 362,
              'training_performed': False, 'checkpoint_selection_performed': False,
              'completed_replay_rows': 0, 'started_at_utc': datetime.now(timezone.utc).isoformat()}

    def record(**changes):
        status.update(changes, elapsed_seconds=round(time.monotonic()-started, 2),
                      updated_at_utc=datetime.now(timezone.utc).isoformat())
        evaluation._atomic_json(output / 'run_status.json', status)
        print(json.dumps(status), flush=True)

    record()
    try:
        os.chdir(REPO)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        random.seed(20240813)
        np.random.seed(20240813)
        torch.manual_seed(20240813)
        torch.cuda.manual_seed_all(20240813)
        main_config, world_config, world_path = backend._configuration()
        allow_tf32 = bool(main_config['training'].get('allow_tf32', True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        frozen_paths = [DEFAULT_CHECKPOINT, DEFAULT_NORMALIZER, MARKER,
                        backend._resolve(REPO, main_config['source']['w1_checkpoint'])]
        before = _hashes(frozen_paths)
        evaluation._atomic_json(output / 'launch_inputs.json', {
            'schema': 'dcvpp.frozen-mechanisms.2025-launch.v1',
            'authorization': 'User explicitly requested the 2025 test set in this task on 2026-09-10',
            'prior_2025_already_evaluated': True, 'evaluation_role': 'supplemental_mechanism_test',
            'input_sha256': {**before, **_hashes([Path(__file__), Path(evaluation.__file__),
                                              backend.DEFAULT_MAIN_CONFIG, world_path])},
            'main_config_resolved': main_config, 'world_config_resolved': world_config,
            'python': sys.executable, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0), 'allow_tf32': allow_tf32,
        })
        record(status='loading_data_and_checkpoints', gpu=torch.cuda.get_device_name(0),
               expected_replay_rows=362 * 6 * len(evaluation.REGISTERED_VARIANTS))
        runtime = backend._load_runtime(backend._lazy_native_api(), main_config, world_config, DEFAULT_CHECKPOINT, 'cuda')
        evaluator = TestMechanismEvaluator(
            runtime,
            lambda marker: build_sealed_final_corpus(config=world_config, repository_root=REPO, frozen_evaluation_marker=marker),
            MARKER, output, batch_size=16,
        )
        result = evaluator.run_mechanisms(record)
        _market_summary(output, world_config, result['dataset_dates'])
        after = _hashes(frozen_paths)
        if before != after:
            raise RuntimeError('A frozen input changed during evaluation')
        evaluation._atomic_json(output / 'frozen_evaluation_manifest.json', {
            **result, 'schema': 'dcvpp.frozen-mechanisms.2025-manifest.v1',
            'frozen_input_sha256_before': before, 'frozen_input_sha256_after': after,
            'frozen_inputs_unchanged': True,
        })
        record(status='completed', exit_code=0)
        return 0
    except BaseException as exc:
        record(status='failed', error=f'{type(exc).__name__}: {exc}', exit_code=1)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
