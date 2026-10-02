"""Run the prepared frozen mechanism study with persistent progress telemetry.

The original evaluator and its intervention registry remain unchanged. The
observer records completed replay rows only after native scientific work returns.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback


REPO = Path(__file__).resolve().parents[1]
PREPARED = (REPO.parent / 'NatureEnergy_R-ECMoE_投稿修订_20260829' /
            'round6_cpu_only_phase_20260831' / 'scripts')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'run_status.json'
    if status_path.exists():
        raise SystemExit('Output already has a run status; use a new output directory.')
    started = time.monotonic()
    status = {
        'schema': 'dcvpp.frozen-mechanisms.progress.v1',
        'pid': os.getpid(), 'status': 'initializing',
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.executable, 'checkpoint_epoch': 130, 'seed': 20240813,
        'year': 2024, 'expected_days': 364,
        'training_performed': False, 'checkpoint_selection_performed': False,
        'completed_replay_rows': 0,
    }

    def record(**changes):
        status.update(changes)
        status['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        status['elapsed_seconds'] = round(time.monotonic() - started, 2)
        temporary = status_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(status_path)
        print(json.dumps(status, ensure_ascii=False), flush=True)

    record()
    try:
        os.chdir(REPO)
        sys.path.insert(0, str(REPO / 'src'))
        sys.path.insert(0, str(PREPARED))
        import pyarrow.dataset  # noqa: F401; established Windows import order
        import numpy as np
        import torch
        import eval_epoch130_mechanisms as entry
        import r_ecmoe_frozen_backend as backend
        from frozen_eval_common import DEFAULT_NORMALIZER
        import dcvpp.vpp_world_moe.r_ecmoe_evaluation as evaluation

        if not torch.cuda.is_available():
            raise RuntimeError('The requested CUDA device is unavailable.')
        random.seed(20240813)
        np.random.seed(20240813)
        torch.manual_seed(20240813)
        torch.cuda.manual_seed_all(20240813)
        main_config, world_config, world_path = backend._configuration()
        torch.backends.cuda.matmul.allow_tf32 = bool(main_config['training'].get('allow_tf32', True))
        torch.backends.cudnn.allow_tf32 = bool(main_config['training'].get('allow_tf32', True))
        paths = [Path(__file__), PREPARED / 'eval_epoch130_mechanisms.py',
                 PREPARED / 'r_ecmoe_frozen_backend.py',
                 PREPARED / 'frozen_eval_common.py', backend.DEFAULT_MAIN_CONFIG,
                 world_path, DEFAULT_NORMALIZER,
                 backend._resolve(REPO, main_config['source']['w1_checkpoint']),
                 Path(evaluation.__file__)]
        hashes = {str(path): hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
                  for path in paths}
        (output / 'launch_inputs.json').write_text(json.dumps({
            'schema': 'dcvpp.frozen-mechanisms.launch.v1',
            'input_sha256': hashes, 'main_config_resolved': main_config,
            'world_config_resolved': world_config,
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0),
            'observer_changes_numerical_outputs': False,
        }, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')

        native_rows = evaluation._execution_rows
        variant_count = len(evaluation.REGISTERED_VARIANTS)
        total_rows = 364 * len(evaluation.EVALUATION_DECISIONS) * variant_count
        calls = 0
        row_count = 0

        def observed_rows(*positional, **kwargs):
            nonlocal calls, row_count
            rows = native_rows(*positional, **kwargs)
            calls += 1
            row_count += len(rows)
            if calls % variant_count == 0:
                record(status='running', completed_replay_rows=row_count,
                       expected_replay_rows=total_rows,
                       completed_batches=calls // variant_count,
                       stage_id=int(kwargs['stage_id']), origin=int(kwargs['origin']),
                       horizon=int(kwargs['horizon']), last_date=str(kwargs['dates'][-1]),
                       gpu_memory_allocated_mib=round(torch.cuda.memory_allocated() / 2**20, 1))
            return rows

        evaluation._execution_rows = observed_rows
        record(status='loading_data_and_checkpoints', gpu=torch.cuda.get_device_name(0),
               native_variants=[v.name for v in evaluation.REGISTERED_VARIANTS],
               expected_replay_rows=total_rows, batch_size=16)
        code = entry.main([
            '--execute', '--device', 'cuda',
            '--bundle-root', str(backend._configured_data_path(world_config, 'selection_bundle_root')),
            '--output-dir', str(output),
        ])
        if code != 0:
            raise RuntimeError(f'Native evaluation returned exit code {code}')
        record(status='completed', completed_replay_rows=row_count, exit_code=0)
        return 0
    except BaseException as exc:
        record(status='failed', error=f'{type(exc).__name__}: {exc}', exit_code=1)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
