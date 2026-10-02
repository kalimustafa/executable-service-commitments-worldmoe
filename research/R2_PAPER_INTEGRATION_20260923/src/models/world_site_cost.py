"""Translate market distributions into site costs using the original cash law.

Market DA/DART/RT forecasts stay untouched. Decision quantiles are separately
named capacity-average marginal costs in fixed grid-equivalent price units.
"""
import numpy as np

def marginal_cost_coefficients(pv,pue,idle,work_power,capacity,pue_reference):
    pv=np.maximum(np.asarray(pv,float),0.)
    pue=np.asarray(pue,float)
    load0=pue*np.asarray(idle)[None]
    increment=pue*np.asarray(work_power)[None]*np.asarray(capacity)[None]
    additional_pv=np.minimum(np.maximum(pv-load0,0.),increment)
    denominator=np.asarray(pue_reference)[None]*np.asarray(work_power)[None]*np.asarray(capacity)[None]
    a=(increment-additional_pv)/denominator
    b=-additional_pv/denominator
    assert np.isfinite(a).all() and np.isfinite(b).all() and (a>=0).all()
    return a,b

def transform_quantiles(market_quantiles,a,b):
    result=np.asarray(market_quantiles)*a[...,None]+b[...,None]
    assert np.isfinite(result).all() and (np.diff(result,axis=-1)>=-1e-7).all()
    return result
