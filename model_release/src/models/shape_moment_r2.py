"""Low-capacity, quantile-shape-conditioned completion of the R2 cost mean."""
import torch
from torch import nn
import torch.nn.functional as F


def shape_features(q,zone,publication):
    upper=(q[:,-1]-q[:,-2]).clamp_min(0);lower=(q[:,1]-q[:,0]).clamp_min(0)
    width=(q[:,-1]-q[:,0]).clamp_min(0);level=torch.asinh(q[:,3]/40.)
    skew=(q[:,-1]+q[:,0]-2*q[:,3])/(width+1.)
    gate=publication.float();sites=F.one_hot(zone,4)[:,1:].float()
    return torch.cat([torch.stack([upper.log1p(),lower.log1p(),level,width.log1p(),skew,gate],-1),sites,torch.stack([gate*level,gate*skew],-1)],-1)


class ShapeMomentR2(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('feature_center',torch.zeros(11));self.register_buffer('feature_scale',torch.ones(11))
        self.coefficients=nn.Parameter(torch.zeros(3,12))
        self.register_buffer('tau_gaps',torch.tensor([.05,.15,.25,.25,.15,.05]))
        self.register_buffer('quadrature_weights',torch.tensor([.075,.1,.2,.25,.2,.1,.075]))

    def forward(self,adapted,quantiles,zone,publication):
        raw=shape_features(quantiles,zone,publication)
        normalized=((raw-self.feature_center)/self.feature_scale).clamp(-8,8)
        x=torch.cat([torch.ones_like(normalized[:,:1]),normalized],-1)
        scores=F.linear(x,self.coefficients)
        excess=F.softplus(scores[:,:2])
        width=((quantiles[:,1:]-quantiles[:,:-1])*self.tau_gaps).sum(-1)
        central_delta=scores[:,2].clamp(-.5,.5)*width
        return {'mean_correction':excess[:,0]-excess[:,1]+central_delta,'excess':excess,'central_delta':central_delta}
