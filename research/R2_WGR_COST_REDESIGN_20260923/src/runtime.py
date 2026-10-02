"""Reuse the registered scientific packages without writing to the old project."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
REPO = WORKSPACE / 'power-dc-vpp'
PAPER = WORKSPACE / '新论文_市场可交付灵活性_20260917'
LEGACY = PAPER / 'revision_20260920_r2_executable'
FAILED = WORKSPACE / 'R2_WGR_DEEPFUSION_20260923'
sys.dont_write_bytecode = True
os.environ.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_LAUNCH_BLOCKING='0',
                  OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
for name in ('tmp', 'cache', 'logs', 'results'):
    (ROOT / name).mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(ROOT / 'tmp' / 'matplotlib')
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'tmp' / 'numba')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
if os.environ.get('R2WGR_RUNTIME') != 'base':
    sys.path.insert(0, str(REPO / '.venv' / 'Lib' / 'site-packages'))
sys.path.insert(0, str(REPO / 'src'))
