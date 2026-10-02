"""Rebuild retained figures and apply the manuscript's Extended Data numbering."""
from pathlib import Path
import importlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'reproduced_figures'
sys.path.insert(0, str(ROOT / 'figure_source'))
base = importlib.import_module('final_figures')
base.F = OUTPUT
compact = importlib.import_module('compact_figures')
compact.F = OUTPUT
base.MANIFEST.clear()
(OUTPUT / 'main').mkdir(parents=True, exist_ok=True)
for ext in ['png', 'pdf', 'svg']:
    shutil.copy2(ROOT / 'figure_source/assets' / f'Fig1_original_v2.{ext}', OUTPUT / 'main' / f'Fig1.{ext}')
for fn in [compact.figure2, compact.figure3, compact.figure4, compact.figure5, compact.figure6, compact.figure7, compact.ed1, compact.ed2]:
    print('Building', fn.__name__, flush=True)
    fn()
target = OUTPUT / 'extended_data_submission'
target.mkdir(exist_ok=True)
for old, new in [(1, 2), (2, 1)]:
    for ext in ['png', 'pdf', 'svg']:
        shutil.copy2(OUTPUT / 'extended_data' / f'ExtendedDataFig{old}.{ext}', target / f'ExtendedDataFig{new}.{ext}')
(OUTPUT / 'figure_manifest.json').write_text(json.dumps(base.MANIFEST, indent=2), encoding='utf-8')
print('Figures written to', OUTPUT)
