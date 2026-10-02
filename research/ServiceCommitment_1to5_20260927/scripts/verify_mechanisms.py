import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.context import *
from src.controllers import adaptive,project_grid

arr,reference,shared=configure(2024)
n=192;c.N=n;c.times=c.times[:n]
for key in ['da','rt','pv','pue']:setattr(c,key,getattr(c,key)[:n])
arr=arr[:n]
model=observer(2024,20260920,arr,12);engine.rank=lambda w,p:model
book,_,_=engine.schedule(arr,[(1.,12)],'robust')
model2=observer(2024,20260920,arr,12)
locked,late,extra=adaptive(arr,model2,12,.7,12)
equivalence=float(np.max(np.abs(book-locked)))
projected,_=project_grid(book,arr)
projection_error=float(np.max(np.abs(projected[:n]-book[:n])))
assert equivalence<1e-9 and projection_error<1e-8
grid=pd.read_csv(ROOT/'source_data/grid_execution_raw.csv')
max_import=float(grid.max_import_cap_excess_mw.max());max_ramp=float(grid.max_upward_ramp_excess_mw.max())
assert max_import<1e-8 and max_ramp<1e-8
manifest=json.loads((ROOT/'configs/source_manifest.json').read_text(encoding='utf-8'))
import hashlib
unchanged=all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'] for x in manifest)
assert unchanged
report=dict(fixed_lock_reproduces_reservations_max_error=equivalence,unconstrained_projection_max_error=projection_error,
    grid_import_limit_max_excess_mw=max_import,upward_ramp_max_excess_mw=max_ramp,
    original_source_hashes_unchanged=unchanged,validation_scope='Two-day identity controls plus all completed electrical constraint results; run-time uncertainty and discrete jobs excluded')
(RESULTS/'mechanism_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
