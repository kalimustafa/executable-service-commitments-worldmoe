from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent
REPO=WORKSPACE/'power-dc-vpp'
PAPER=WORKSPACE/'新论文_市场可交付灵活性_20260917'
LEGACY=PAPER/'revision_20260920_r2_executable'
FAILED=WORKSPACE/'R2_WGR_DEEPFUSION_20260923'
PREVIOUS=WORKSPACE/'R2_WGR_COST_REDESIGN_20260923'
ARCHIVE=WORKSPACE/'R2_PAPER_INTEGRATION_20260923'
EXTENSION=WORKSPACE/'ServiceCommitment_1to5_20260927'
sys.dont_write_bytecode=True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')

