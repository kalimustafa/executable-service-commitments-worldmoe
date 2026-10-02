from src.runtime import ROOT, REPO, PAPER, LEGACY
from src.data.access_guard import install_guard
install_guard()
import hashlib
import json
import yaml

def entry(path,role):
    return dict(path=str(path),role=role,access='READ ONLY',bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())

selected=json.loads((ROOT/'configs/selected_w1.json').read_text(encoding='utf-8'))
files=[
    (REPO/'src/dcvpp/vpp_world_moe/r2_market.py','exact A0/A1 architecture, loss primitive, copula and price identity'),
    (REPO/'src/dcvpp/vpp_world_moe/model.py','strict frozen W1 loading and public encode'),
    (REPO/'src/dcvpp/vpp_world_moe/backbone.py','real interaction blocks 3,6,10'),
    (LEGACY/'experiment/prepare_frozen_cache.py','causal world extraction helper source'),
    (LEGACY/'experiment/train_r2.py','formal loss and checkpoint score'),
    (LEGACY/'experiment/calibrate_r2.py','exact native affine calibration'),
    (LEGACY/'experiment/select_2024_economics.py','formal economic selector'),
    (LEGACY/'experiment/ef_forecast_provider.py','unchanged forecast to reservation boundary'),
    (PAPER/'revision_20260920_market_expansion/experiment/run_screen.py','unmodified accounting functions'),
    (PAPER/'revision_20260920_market_expansion/experiment/deepen_scheduling.py','unmodified reservation and late-work executor'),
    (PAPER/'experiments/prepare_gate_data.py','authoritative 2024 row labels and embargo'),
    (PAPER/'experiments/gate_hourly_panel.parquet','2021-2024 targets and issue-causal features'),
    (LEGACY/'cache/development_physical_reference.npz','2022-2024 physical forcing and training PUE envelope'),
    (LEGACY/'cache/development_cache_manifest.json','development-only shallow cache index'),
    (LEGACY/'cache/r2_normalization.json','training-only feature order and normalization'),
    (REPO/'artifacts/research/physical/run_id=paper-physical-v1/frozen_physical_artifact.json','original physical constraints; no economic outcomes')]
manifest=dict(experiment_root=str(ROOT),original_workspace_access='READ ONLY',selected_w1=dict(path=selected['checkpoint'],sha256=selected['recorded_sha256'],copy=False,trainable=False),
    sources=[entry(p,role) for p,role in files],
    shallow_cache=dict(directory=str(LEGACY/'cache'),manifest='development_cache_manifest.json',only_manifest_listed_development_shards=True,copy=False),
    rasc_development=dict(path=str(PAPER/'revision_20260918/forecast_2025/development_forecast_only_2024h2.parquet'),
        content_year=2024,guard='read date column before target/prediction columns; exact historical file registration',
        note='legacy folder name is not target year; no other forecast_2025 data file is permitted'),
    package_format='Markdown + CSV + PNG; user requested no site',locked_data_accessed=False)
(ROOT/'SOURCE_PATHS.yaml').write_text(yaml.safe_dump(manifest,allow_unicode=True,sort_keys=False),encoding='utf-8')
print('Source manifest written',len(files),flush=True)
