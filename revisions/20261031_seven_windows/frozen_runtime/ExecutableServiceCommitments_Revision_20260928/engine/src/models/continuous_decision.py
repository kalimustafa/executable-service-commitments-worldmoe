"""Smooth physical-cost decoding for R2, with gradients through Gaussian CDFs."""
import math
import torch
from src.models.coupled_rollout import ArchitectedR2QPHead


class ContinuousR2Head(ArchitectedR2QPHead):
    def decision(self,state,adapted,updated,pue_ref,shift):
        out=super().decision(state,adapted,updated,pue_ref,shift)
        pv=out['pv_quantiles'];pue=out['physical_pue'];cap=(1-state[:,:,0]).clamp_min(0)
        sigma=((pv[:,:,-1]-pv[:,:,-2])/(1.644854-1.281552)).clamp_min(1e-5)
        mu=pv[:,:,-1]-1.644854*sigma
        phi=lambda z:torch.exp(-.5*z.square())/math.sqrt(2*math.pi)
        cdf=lambda z:.5*(1+torch.erf(z/math.sqrt(2)))
        a=mu/sigma;mean=mu*cdf(a)+sigma*phi(a)
        second=(mu.square()+sigma.square())*cdf(a)+mu*sigma*phi(a)
        std=(second-mean.square()).clamp_min(1e-8).sqrt()
        rtmean=(updated*self.probability_weights).sum(-1)
        rtstd=((updated-rtmean[:,:,None]).square()*self.probability_weights).sum(-1).clamp_min(1e-8).sqrt()
        slope=out['cross_moment_rho']*rtstd/std
        reserved=state[:,:,0];load0=pue*(.23069976270198822+1.639059066772461*reserved)
        knots=torch.minimum(((pv-load0[:,:,None])/(pue[:,:,None]*1.639059066772461)).clamp_min(0),cap[:,:,None])
        uniform=cap[:,:,None]*torch.linspace(0,1,9,device=state.device)[None,None]
        boundaries=torch.sort(torch.cat([uniform,knots],-1),-1).values
        widths=(boundaries[:,:,1:]-boundaries[:,:,:-1]).clamp_min(0)
        middle=(boundaries[:,:,1:]+boundaries[:,:,:-1])*.5
        load=load0[:,:,None]+pue[:,:,None]*1.639059066772461*middle
        z=(load-mu[:,:,None])/sigma[:,:,None];z0=-mu/sigma;f=cdf(z)
        partial=mu[:,:,None]*(f-cdf(z0)[:,:,None])+sigma[:,:,None]*(phi(z0)[:,:,None]-phi(z))
        raw=(pue/pue_ref[:,None])[:,:,None]*((out['rt_decision'][:,:,None]+1)*f+slope[:,:,None]*(partial-mean[:,:,None]*f)-1)
        # Midpoint integration is used for the differentiable training QP;
        # deployed continuation curves use analytic interval integrals.
        nonconvex=(raw[:,:,1:]-raw[:,:,:-1]).amin(-1)<-1e-5
        average=(raw*widths).sum(-1)/cap.clamp_min(1e-8)
        costs=torch.where(nonconvex[:,:,None],average[:,:,None],raw)
        fees=torch.ones_like(cap)*2;fees[:,0]=0
        out.update(segment_scores=costs+out['opportunity'][:,:,None]+fees[:,:,None],segment_widths=widths,physical_segment_cost=raw)
        return out
