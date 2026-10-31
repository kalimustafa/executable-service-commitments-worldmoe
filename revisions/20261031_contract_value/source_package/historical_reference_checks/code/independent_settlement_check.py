"""Re-settle the published execution arrays; this is not a policy/deadline replay."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np

def check(inputs):
    z=np.load(inputs,allow_pickle=False)
    ld=np.longdouble; dt=ld('0.25')
    rt=z['rt_lmp_usd_mwh'].astype(ld); pv=z['pv_mw'].astype(ld)
    pue=z['pue'].astype(ld); idle=z['idle_mw'].astype(ld)
    k=z['work_power_mw'].astype(ld); ref=z['reference_compute'].astype(ld)
    l0=pue*(idle+k*ref);g0=np.maximum(l0-pv,0);v0=np.maximum(pv-l0,0)
    c0=dt*np.sum(rt*g0+v0)+ld('2409.7842143056')
    rows=[]
    for seed in (20260920,20260921,20260922):
        x=z[f'compute_seed_{seed}'].astype(ld)
        load=pue*(idle+k*x);g=np.maximum(load-pv,0);v=np.maximum(pv-load,0)
        avoided=dt*np.sum(rt*np.maximum(g0-g,0))
        recovery=dt*np.sum(rt*np.maximum(g-g0,0))
        fee=dt*np.sum(np.abs(g-g0));spill=dt*np.sum(v0-v)
        saving=avoided-recovery-fee+spill
        rows.append(dict(seed=seed,reference_cost_usd=str(c0),
          observed_year_saving_before_terminal_difference_usd=str(saving),
          gross_avoided_energy_usd=str(avoided),recovery_energy_usd=str(recovery),
          import_deviation_fee_usd=str(fee),curtailment_benefit_usd=str(spill),
          import_reduction_mwh=str(dt*np.sum(g0-g)),
          curtailment_reduction_mwh=str(dt*np.sum(v0-v)),
          facility_demand_increase_mwh=str(dt*np.sum(load-l0)),
          compute_total_difference_inyear=str(np.sum(x-ref)),
          max_exported_import_reconstruction_error=str(np.max(np.abs(g-z[f'grid_seed_{seed}'])))))
    return dict(
      scope="Independent re-settlement of published annual arrays. Before terminal-cost difference. Not training, admission/deadline, first-release, tail feasibility or annual hindsight-witness validation.",
      source_commit="323c2084e7ee668ba512fb1e33dd56b8394b32d4",
      data_file=str(inputs),data_sha256=hashlib.sha256(Path(inputs).read_bytes()).hexdigest(),
      intervals=int(rt.shape[0]),sites=int(rt.shape[1]),rows=rows,
      means={key:str(np.mean([ld(row[key]) for row in rows])) for key in rows[0] if key!='seed'},
      published_max_abs_rt=str(np.max(np.abs(rt))),published_max_pue=str(np.max(pue)),
      uniform_lipschitz_usd_per_work=str(dt*np.max(k)*np.max(pue)*(np.max(np.abs(rt))+2)))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();r=check(a.inputs)
    a.output.write_text(json.dumps(r,indent=2),encoding="utf-8")
    print("Observed-year mean saving:",r["means"]["observed_year_saving_before_terminal_difference_usd"])
    print("Scope:",r["scope"])
