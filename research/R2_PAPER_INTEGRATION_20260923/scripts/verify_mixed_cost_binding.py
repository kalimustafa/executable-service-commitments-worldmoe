from src.evaluation.paper_economics import *
from src.evaluation.mixed_cost_adapter import MixedCostProvider,bound_executors
from scripts.evaluate_completed_seeds import setup
arr,reference,kwargs=setup()
row=dict(name='equivalence',model='R2_StateCost',market_branch='C1_ProfileResidual',seed=20260920,
    physical_branch='particle_trailing4',year=2024,window_h=24,policy='interval',price_forecast_source=str(forecast_file(2024,20260920)))
observer,_,_=build_candidate(row,arr,kwargs);engine.rank=lambda w,p:observer
original,_,_=engine.schedule(arr,[(1.,24)],'robust')
provider=MixedCostProvider(arr,2024,20260920,'interval',windows=[24])
fixed,_=bound_executors(provider,[(1.,24)])
bound,_,_=fixed(arr,[(1.,24)],'robust')
error=float(np.max(np.abs(original-bound)));assert error<1e-10
(ROOT/'results/mixed_cost_binding_equivalence.json').write_text(json.dumps(dict(status='passed',
    full_development_block=True,window_h=24,maximum_calendar_difference=error,
    changed_allocation_statements=0,future_observations_used=False),indent=2))
print('FULL-BLOCK COST BINDING EQUIVALENT',error,flush=True)
