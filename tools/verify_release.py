"""Check release integrity and summarize the included nominal records."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
errors = []
for row in manifest['files']:
    p = ROOT / row['path']
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
        errors.append(row['path'])
if errors:
    raise SystemExit('Missing or changed release files: ' + ', '.join(errors))
q = pd.read_csv(ROOT / 'source_data/worldmoe_three_seed_results.csv')
b = pd.read_csv(ROOT / 'source_data/current/extensions/certified_value_bounds.csv')
b = b[(b.year == 2025) & (b.window_h == 24) & (b.trace == 'Eagle')]
saving = float(q.saving_usd.mean())
ceiling = float(b.maximum_saving_upper_usd.iloc[0])
assert abs(saving - 105271.06) < 0.02
assert abs(ceiling - 132605.42) < 0.02
print(json.dumps({'verified_files': len(manifest['files']), 'refinements': len(q), 'mean_saving_usd': saving, 'certified_ceiling_usd': ceiling, 'captured_percent': 100 * saving / ceiling, 'scope': 'Included-result aggregation and file integrity only; no training or annual replay rerun'}, indent=2))
