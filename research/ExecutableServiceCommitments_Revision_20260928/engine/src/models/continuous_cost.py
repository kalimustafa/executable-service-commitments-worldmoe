"""Analytic expected grid and PV cross moments for a clipped Gaussian decoder.

The existing QPR2 PV head generates max(0, mu + sigma Z). Its upper two
quantiles identify mu/sigma. This module integrates that same distribution;
it neither changes the forecast weights nor reads realized future physics.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def _cdf(x):
    return .5*(1+math.erf(x/math.sqrt(2.)))


@njit(cache=True)
def _pdf(x):
    return math.exp(-.5*x*x)/math.sqrt(2*math.pi)


@njit(cache=True)
def normal_grid_moments(load,mu,sigma):
    """E[(load-PV)+], E[PV*(load-PV)+], for PV=max(0,Normal)."""
    z0=-mu/sigma;z=(load-mu)/sigma;f0=_cdf(z0);f=_cdf(z);p0=_pdf(z0);p=_pdf(z)
    m1=mu*(f-f0)+sigma*(p0-p)
    m2=(mu*mu+sigma*sigma)*(f-f0)+2*mu*sigma*(p0-p)+sigma*sigma*(z0*p0-z*p)
    return max(0.,load*f-m1),load*m1-m2


@njit(cache=True)
def expected_value(x,mu,sigma,price_mean,price_slope,pv_mean,pue,reference,idle,power,opportunity,pue_ref):
    load=pue*(idle+power*x);base=pue*idle;ref=pue*(idle+power*reference)
    grid,cross=normal_grid_moments(load,mu,sigma);grid0,cross0=normal_grid_moments(base,mu,sigma);refgrid,_=normal_grid_moments(ref,mu,sigma)
    value=(price_mean-price_slope*pv_mean+1)*(grid-grid0)+price_slope*(cross-cross0)-(load-base)
    value+=abs(grid-refgrid)-abs(grid0-refgrid)
    return .25*value+.25*opportunity*pue_ref*power*x


@njit(cache=True)
def continuous_curve(pv,prices,probabilities,pue,reference,idle,power,opportunity,pue_ref,capacity):
    sigma=max((pv[-1]-pv[-2])/(1.644854-1.281552),1e-6)
    mu=pv[-1]-1.644854*sigma
    a=mu/sigma;pv_mean=mu*_cdf(a)+sigma*_pdf(a)
    pv2=(mu*mu+sigma*sigma)*_cdf(a)+mu*sigma*_pdf(a)
    pv_std=math.sqrt(max(1e-12,pv2-pv_mean*pv_mean))
    price_mean=0.;old_mean=0.
    for j in range(len(pv)):
        price_mean+=probabilities[j]*prices[j];old_mean+=probabilities[j]*pv[j]
    old_var=0.;covariance=0.
    for j in range(len(pv)):
        old_var+=probabilities[j]*(pv[j]-old_mean)**2
        covariance+=probabilities[j]*(pv[j]-old_mean)*(prices[j]-price_mean)
    # Retain the learned conditional-price standard deviation, and recenter on
    # the continuous PV mean so the expected RT price remains unchanged.
    slope=covariance/max(math.sqrt(old_var),1e-6)/pv_std
    knots=np.empty(10+len(pv));knots[:9]=np.linspace(0.,capacity,9);knots[9]=min(capacity,max(0.,reference))
    for j in range(len(pv)):knots[j+10]=min(capacity,max(0.,(pv[j]/pue-idle)/power))
    knots.sort();costs=np.empty(len(knots)-1);widths=np.empty(len(knots)-1);count=0
    base_load=pue*idle;g0,h0=normal_grid_moments(base_load,mu,sigma)
    refgrid,_=normal_grid_moments(pue*(idle+power*reference),mu,sigma)
    coefficient=price_mean-slope*pv_mean+1.
    base_value=coefficient*g0+slope*h0-base_load+abs(g0-refgrid)
    values=np.empty(len(knots))
    for j in range(len(knots)):
        x=knots[j];load=pue*(idle+power*x);g,h=normal_grid_moments(load,mu,sigma)
        values[j]=.25*(coefficient*g+slope*h-load+abs(g-refgrid)-base_value)+.25*opportunity*pue_ref*power*x
    for j in range(len(knots)-1):
        left=knots[j];right=knots[j+1];width=right-left
        if width<1e-12:continue
        costs[count]=(values[j+1]-values[j])/width;widths[count]=width;count+=1
        # Match the existing convex planning proxy in nonconvex price regimes.
        # This is not a certified bound; realized settlement stays independent.
        while count>1 and costs[count-2]>costs[count-1]+1e-12:
            w=widths[count-2]+widths[count-1]
            costs[count-2]=(costs[count-2]*widths[count-2]+costs[count-1]*widths[count-1])/w
            widths[count-2]=w;count-=1
    return costs[:count],widths[:count]
