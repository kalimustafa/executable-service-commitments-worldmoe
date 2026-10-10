"""Build Figure 1 inputs from the existing, measured research archives."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
WORKSPACE = PACKAGE.parent
DATA = PACKAGE / 'source_data'
NS = {'k': 'http://www.opengis.net/kml/2.2'}
archive = HERE / 'census_states_2024.zip'
with ZipFile(archive) as z:
    kml = ET.fromstring(z.read('cb_2024_us_state_500k.kml'))
features = []
for pm in kml.findall('.//k:Placemark', NS):
    attributes = {n.attrib['name']: n.text for n in pm.findall('.//k:SimpleData', NS)}
    if attributes.get('STUSPS') not in ['NY', 'VT', 'NH', 'ME', 'MA', 'CT', 'RI', 'PA', 'NJ']:
        continue
    polygons = []
    for poly in pm.findall('.//k:Polygon', NS):
        rings = []
        for boundary in ['outerBoundaryIs', 'innerBoundaryIs']:
            for node in poly.findall(f'k:{boundary}/k:LinearRing/k:coordinates', NS):
                rings.append([[float(v) for v in point.split(',')[:2]] for point in node.text.split()])
        if rings:
            polygons.append(rings)
    features.append({'type': 'Feature', 'properties': {'STUSAB': attributes['STUSPS'], 'NAME': attributes['NAME']}, 'geometry': {'type': 'MultiPolygon', 'coordinates': polygons}})
assert len(features) == 9 and all(f['geometry']['coordinates'] for f in features)
(HERE / 'northeast_states.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': features}), encoding='utf-8')
(HERE / 'geography_source.json').write_text(json.dumps({
    'provider': 'U.S. Census Bureau', 'dataset': '2024 cartographic state boundaries, 1:500,000',
    'download_url': 'https://www2.census.gov/geo/tiger/GENZ2024/kml/cb_2024_us_state_500k.zip',
    'retrieved': '2026-10-10', 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    'projection': 'Albers equal-area conic; central meridian -74; latitude of origin 40; standard parallels 41 and 45 degrees',
    'meaning': 'State outlines; markers are configured modelled sites, not surveyed data-centre facilities. No grid lines or intermarket power flows are asserted.'
}, indent=2), encoding='utf-8')

site_path = WORKSPACE / 'power-dc-vpp/configs/sites.yaml'
config_path = WORKSPACE / 'R2_PAPER_INTEGRATION_20260923/configs/economic_constraints.json'
sites = yaml.safe_load(site_path.read_text(encoding='utf-8'))['sites']
constraints = json.loads(config_path.read_text(encoding='utf-8'))['constraints']
results = pd.read_csv(DATA / 'worldmoe_three_seed_results.csv')
bounds = pd.read_csv(DATA / 'current/extensions/certified_value_bounds.csv')
ceiling = float(bounds[(bounds.year == 2025) & (bounds.window_h == 24) & (bounds.trace == 'Eagle')].maximum_saving_upper_usd.iloc[0])
grid = pd.read_csv(DATA / 'grid_metrics_all_models.csv').set_index('model')
fields = ['portfolio_peak_hourly_mw', 'p95_abs_ramp_mw', 'max_upward_ramp_mw', 'max_downward_ramp_mw', 'top10_net_mwh', 'top10_pct']
with np.load(DATA / 'grid_metrics_inputs_2025.npz', allow_pickle=False) as z:
    times = pd.DatetimeIndex(pd.to_datetime(z['timestamp_local'], utc=True)).tz_convert('America/New_York')
    seeds = [20260920, 20260921, 20260922]
    rows = {'hour': times.hour, 'reference_mw': z['reference_grid_mw'].sum(axis=1)}
    for seed in seeds:
        rows[f'seed_{seed}_mw'] = z[f'grid_seed_{seed}'].sum(axis=1)
    profile = pd.DataFrame(rows).groupby('hour').mean()
    profile['worldmoe_mw'] = profile[[f'seed_{s}_mw' for s in seeds]].mean(axis=1)
    profile['seed_min_mw'] = profile[[f'seed_{s}_mw' for s in seeds]].min(axis=1)
    profile['seed_max_mw'] = profile[[f'seed_{s}_mw' for s in seeds]].max(axis=1)
    profile.reset_index().to_csv(HERE / 'annual_diurnal_imports.csv', index=False)

summary = {
    'year': 2025, 'window_h': 24, 'interval_minutes': 15,
    'seeds': seeds, 'sites': sites,
    'physical': {
        'variable_it_capacity_mw': constraints['work_power_per_unit'][0],
        'idle_it_mw': constraints['idle_power'][0],
        'import_cap_mw': constraints['grid_import_limit'][0],
        'export_cap_mw': constraints['grid_export_limit'][0],
        'pue_min': constraints['pue_min'][0], 'pue_max': constraints['pue_max'][0],
        'pv_capacity_mw': yaml.safe_load(site_path.read_text(encoding='utf-8'))['defaults']['pv_capacity_mw']
    },
    'reference_cost_usd': float(results.reference_cost_usd.mean()),
    'policy_cost_usd': float(results.adjusted_cost_usd.mean()),
    'saving_usd': float(results.saving_usd.mean()),
    'saving_pct': float(results.saving_pct.mean()),
    'savings_ceiling_usd': ceiling,
    'capture_pct': float(100 * results.saving_usd.mean() / ceiling),
    'annual_variable_it_mwh': float(results.IT_energy_mwh.mean()),
    'max_late_it_mwh': float(results.late_IT_mwh.max()),
    'max_closing_it_mwh': float(results.closing_it_mwh.max()),
    'grid_reference': grid.loc['No-flex', fields].to_dict(),
    'grid_worldmoe': grid.loc['WorldMoE', fields].to_dict(),
    'profile_definition': 'Full-year mean aggregate portfolio import for each local clock hour, using all 35,040 quarter-hours; WorldMoE is the mean of three fixed refinements.',
    'diagram_definition': 'Workload boxes are a conceptual service-ledger illustration without fitted or simulated numeric trajectories.',
    'result_scope': 'Modelled four-site portfolio; measured archived simulation/replay outcomes, not observed facility measurements.'
}
(HERE / 'figure_inputs.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
sources = [site_path, config_path, DATA / 'worldmoe_three_seed_results.csv', DATA / 'current/extensions/certified_value_bounds.csv', DATA / 'grid_metrics_all_models.csv', DATA / 'grid_metrics_inputs_2025.npz']
(HERE / 'input_manifest.json').write_text(json.dumps([
    {'path': str(p.relative_to(WORKSPACE)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources
], indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({k: summary[k] for k in ['saving_usd', 'saving_pct', 'capture_pct', 'max_late_it_mwh', 'max_closing_it_mwh']}))
