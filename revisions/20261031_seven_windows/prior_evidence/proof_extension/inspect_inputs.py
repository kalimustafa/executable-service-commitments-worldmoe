from pathlib import Path
import numpy as np, json, hashlib
root=Path(r'C:\Users\Kali\Desktop\电力市场')
p=root/'external_validation/data/forcing_2025.npz'
r=root/'EnergyEconomics_Revision_20261031/source_package/source_snapshot/source_data/grid_metrics_inputs_2025.npz'
with np.load(p) as f, np.load(r) as g:
 print('forcing', [(k,f[k].shape,str(f[k].dtype)) for k in f.files])
 print('released', [(k,g[k].shape,str(g[k].dtype)) for k in g.files])
 for fk,gk in [('rt','rt_lmp_usd_mwh'),('pv','pv_mw'),('pue','pue'),('k','work_power_mw'),('idle','idle_mw')]: print(fk,np.array_equal(f[fk],g[gk]),float(np.max(abs(f[fk]-g[gk]))))
 a=f['eagle_arrivals'];q=np.zeros(4);b=np.zeros_like(a)
 for t in range(len(a)):q+=a[t];b[t]=np.minimum(q,1.);q-=b[t]
 print('reference identical',np.array_equal(b,g['reference_compute']), 'max',np.max(abs(b-g['reference_compute'])), 'closing',q)
 print('forcing sha256',hashlib.sha256(p.read_bytes()).hexdigest())
