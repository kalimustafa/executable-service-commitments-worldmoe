from pathlib import Path
from fractions import Fraction as F
import numpy as np,json
root=Path(r'C:\Users\Kali\Desktop\电力市场');out=root/'EnergyEconomics_Revision_20261031/revision/proof_extension'
with np.load(root/'external_validation/data/forcing_2025.npz') as f:arr=f['eagle_arrivals'];a=arr[:,0];assert np.array_equal(arr,np.repeat(a[:,None],4,axis=1))
with np.load(root/'EnergyEconomics_Revision_20261031/source_package/source_snapshot/source_data/grid_metrics_inputs_2025.npz') as g:b=g['reference_compute'][:,0]
A=[F(0)];B=[F(0)]
for v,w in zip(a,b):A.append(A[-1]+F.from_float(float(v)));B.append(B[-1]+F.from_float(float(w)))
records=[]
for hours in [8,12,16,20,24,36,48]:
 D=4*hours
 gap=max(A[max(0,t-D+1)]-B[t] for t in range(1,len(A)))
 records.append({'hours':hours,'exact_earliest_max_deadline_violation':str(max(gap,F(0))),'source_arrivals_feasible_by_earliest':gap<=0})
r={'scope':'Read-only extension feasibility screen using exact prefixes; no new value optimisation or new WorldMoE policy replay','site_streams_identical':True,'windows':records,
 'uniform_envelope_8h_note':'A compatible stream can concentrate a 35-saturated-slot-plus-clearing busy block at its first slot, requiring more than 32 slots; the existing universal nonempty-corridor premise cannot cover every compatible admission array at 8 h. Use recovered actual admissions for 8 h evaluation.'}
(out/'additional_windows_scope.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))

