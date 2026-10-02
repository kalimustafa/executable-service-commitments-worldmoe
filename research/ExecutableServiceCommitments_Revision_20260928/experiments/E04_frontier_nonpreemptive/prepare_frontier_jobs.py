"""Retain the 2025 Frontier jobs and development-fitted node-power mapping."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[2]
DATA=WORKSPACE/'external_validation/data'
with np.load(DATA/'forcing_2024.npz') as f:
    dev_energy_mwh=float(.25*np.sum(f['eagle_arrivals']*f['k']))
with np.load(DATA/'forcing_2025.npz') as f:
    k=f['k'].astype(float)
    n=len(f['times'])
dev=pd.read_parquet(DATA/'frontier_2024/parsed_jobs.parquet',columns=['requested_nodes','runtime'])
denominator=float(np.dot(dev.requested_nodes.to_numpy(float),np.ceil(dev.runtime.to_numpy(float)/900)*.25))
node_mw=dev_energy_mwh/denominator
jobs=pd.read_parquet(DATA/'frontier_2025/parsed_jobs.parquet',columns=['job_id','submit_time','requested_nodes','runtime','status'])
jobs['release_step']=((jobs.submit_time.to_numpy(dtype=np.int64)+899)//900).astype(np.int32)
jobs['duration_steps']=((jobs.runtime.to_numpy(dtype=np.int64)+899)//900).astype(np.int32)
jobs['site']=(jobs.job_id.to_numpy(dtype=np.int64)%4).astype(np.int8)
jobs['power_mw']=jobs.requested_nodes.to_numpy(float)*node_mw
jobs['units']=jobs.power_mw.to_numpy(float)/k[jobs.site.to_numpy(dtype=np.int64)]
jobs=jobs.sort_values(['release_step','job_id'],kind='stable').reset_index(drop=True)
if np.any(jobs.release_step.to_numpy()>=n):raise RuntimeError('Job release outside 2025 calendar')
oversized=jobs.units.to_numpy()>1+1e-10
if oversized.any():raise RuntimeError(f'{oversized.sum()} oversized jobs require an explicit handling plan')
arr=np.zeros((n,4),dtype=np.float64)
np.add.at(arr,(jobs.release_step.to_numpy(dtype=np.intp),jobs.site.to_numpy(dtype=np.intp)),jobs.units.to_numpy()*jobs.duration_steps.to_numpy())
fluid_mwh=float(.25*np.sum(arr*k))
jobs_mwh=float(.25*np.dot(jobs.power_mw.to_numpy(),jobs.duration_steps.to_numpy()))
if abs(fluid_mwh-jobs_mwh)>1e-6:raise RuntimeError('Matched fluid energy differs from jobs')
jobs.to_parquet(HERE/'mapped_jobs_2025.parquet',index=False)
np.savez_compressed(HERE/'matched_job_energy_arrivals.npz',arrivals=arr)
meta=dict(year=2025,source_jobs=len(jobs),included_jobs=len(jobs),node_power_mw=node_mw,
    mapping_development_year=2024,development_eagle_it_mwh=dev_energy_mwh,
    development_slotted_node_hours=denominator,job_energy_mwh=jobs_mwh,
    matched_fluid_energy_mwh=fluid_mwh,site_assignment='job_id modulo 4',
    release_rounding='ceil to next 15-minute boundary',runtime_rounding='ceil to 15-minute slots',
    source='external_validation/data/frontier_2025/parsed_jobs.parquet')
(HERE/'mapping.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(json.dumps({k:meta[k] for k in ['source_jobs','node_power_mw','job_energy_mwh']},indent=2))
