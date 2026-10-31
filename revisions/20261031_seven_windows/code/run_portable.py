"""Run a selected frozen-checkpoint evaluation using only the included runtime.

Example:
python code/run_portable.py --year 2025 --window 48 --seed 20260920 \
  --cdf-epoch 6 --eta 0 --safety 0.9 --current-coupling --continuous-cost
Existing completed outputs are printed rather than overwritten. To run afresh,
use an extracted package copy with its generated result files kept separately.
"""
from pathlib import Path
import argparse,os,subprocess,sys,shutil
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--fresh-output',type=Path)
options,forward=p.parse_known_args()
output=options.fresh_output.resolve() if options.fresh_output else ROOT
(output/'results/trajectories').mkdir(parents=True,exist_ok=True)
(output/'runtime_cache').mkdir(parents=True,exist_ok=True)
runner=ROOT/'code/run_window_extension.py'
if output!=ROOT:
    (output/'code').mkdir(parents=True,exist_ok=True)
    destination=output/'code/run_window_extension.py'
    shutil.copy2(runner,destination);runner=destination
# Archived context modules create their empty result directories on import.
# All actual evaluation outputs are redirected by the extension runner.
for rel in ['ServiceCommitment_1to5_20260927/results/economic_trajectories',
            'ServiceCommitment_1to5_20260927/results/trajectories']:
    (ROOT/'frozen_runtime'/rel).mkdir(parents=True,exist_ok=True)
env=os.environ.copy()
env['WORLDMOE_ENGINE_ROOT']=str(ROOT/'frozen_runtime/ExecutableServiceCommitments_Revision_20260928/engine')
env.update(PYTHONUTF8='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
raise SystemExit(subprocess.call([sys.executable,str(runner),*forward],env=env,cwd=output))
