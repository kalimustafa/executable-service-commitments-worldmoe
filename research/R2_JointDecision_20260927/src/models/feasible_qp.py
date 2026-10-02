"""Analytic differentiable box/equality QP with a fixed work-conservation law."""
import torch

class _CappedQuadratic(torch.autograd.Function):
    @staticmethod
    def forward(ctx,cost,capacity,amount,tau):
        c=cost.double();cap=capacity.double().clamp_min(0);b=amount.double()
        b=torch.minimum(b.clamp_min(0),cap.sum(1))
        knots=torch.cat([c,c+float(tau)*cap],1);sign=torch.cat([torch.ones_like(c),-torch.ones_like(c)],1)
        order=knots.argsort(dim=1,stable=True);knots=knots.gather(1,order);sign=sign.gather(1,order)
        count=sign.cumsum(1);intercept=(sign*knots).cumsum(1)
        at=(count*knots-intercept)/float(tau)
        crossed=at>=b[:,None]-1e-10;index=crossed.long().argmax(1)
        index=torch.where(crossed.any(1),index,torch.full_like(index,knots.shape[1]-1))
        left=(index-1).clamp_min(0)[:,None];edge=knots.gather(1,left)[:,0];level=at.gather(1,left)[:,0];slope=count.gather(1,left)[:,0]
        lam=torch.where(slope>0,edge+(b-level)*float(tau)/slope.clamp_min(1),knots.gather(1,index[:,None])[:,0])
        lam=torch.where(b<=0,knots[:,0]-float(tau),lam)
        x=torch.minimum(((lam[:,None]-c)/float(tau)).clamp_min(0),cap)
        free=(x>1e-8)&(x<cap-1e-8);upper=(cap>1e-8)&(x>=cap-1e-8)
        ctx.save_for_backward(free,upper);ctx.tau=float(tau)
        return x.to(cost.dtype)

    @staticmethod
    def backward(ctx,grad):
        free,upper=ctx.saved_tensors
        count=free.sum(1,keepdim=True);mean=(grad*free).sum(1,keepdim=True)/count.clamp_min(1)
        dc=free*(mean-grad)/ctx.tau
        dcap=upper*(grad-mean)
        db=torch.where(count[:,0]>0,mean[:,0],torch.zeros_like(mean[:,0]))
        return dc,dcap,db,None

def qp_allocation(widths,scores,amount,tau=2.):
    """Minimize modeled cost + tau/2 * sum(slot_service**2).

    Convex slot segments are filled in prefix order. Adding tau times each
    segment's preceding width converts the separable slot QP into a capped
    quadratic resource-allocation problem. Implicit derivatives include cost,
    capacity and admitted work; no straight-through estimator is used.
    """
    prefix=widths.cumsum(-1)-widths
    effective=scores+float(tau)*prefix
    segment=_CappedQuadratic.apply(effective.reshape(len(widths),-1),widths.reshape(len(widths),-1),amount,float(tau))
    return segment.reshape(widths.shape).sum(-1)
