"""Tail completion of the RT cost expectation without changing R2 quantiles.

E[Y] = E[clip(Y, q05, q95)] + E[(Y-q95)+] - E[(q05-Y)+].
The seven-quantile quadrature approximates the central clipped expectation;
two nonnegative, conditional first-moment heads estimate the omitted tails.
"""
import torch
from torch import nn


class TailMomentR2(nn.Module):
    def __init__(self, base_excess=None):
        super().__init__()
        self.register_buffer('base_excess',torch.ones(2,4,2) if base_excess is None else torch.as_tensor(base_excess,dtype=torch.float32))
        self.zone_embedding=nn.Embedding(4,8)
        self.encoder=nn.Sequential(nn.LayerNorm(144),nn.Linear(144,64),nn.GELU(),nn.LayerNorm(64),nn.Linear(64,32),nn.GELU())
        self.excess_head=nn.Linear(32,2)
        nn.init.zeros_(self.excess_head.weight);nn.init.zeros_(self.excess_head.bias)

    def forward(self, adapted, quantiles, zone, publication):
        x=torch.cat([adapted,quantiles/100.,publication[:,None].float(),self.zone_embedding(zone)],-1)
        encoded=self.encoder(x);raw=self.excess_head(encoded)
        ratio=3*torch.tanh(raw/3)
        excess=self.base_excess[publication.long(),zone]*torch.exp(ratio)
        return {'excess':excess,'mean_correction':excess[:,0]-excess[:,1],'log_ratio':ratio,'encoded':encoded}


class MomentCompletionR2(TailMomentR2):
    """Complete the central quantile integral and two tail first moments.

    Monotonicity bounds the integral between left/right quantile Riemann sums.
    A learned interpolation position stays inside that interval, so the seven
    original quantile anchors remain compatible with the completed expectation.
    """
    def __init__(self,base_excess=None):
        super().__init__(base_excess)
        self.curvature_head=nn.Linear(32,1)
        nn.init.zeros_(self.curvature_head.weight);nn.init.zeros_(self.curvature_head.bias)
        self.register_buffer('tau_gaps',torch.tensor([.05,.15,.25,.25,.15,.05]))
        self.register_buffer('quadrature_weights',torch.tensor([.075,.1,.2,.25,.2,.1,.075]))

    def forward(self,adapted,quantiles,zone,publication):
        out=super().forward(adapted,quantiles,zone,publication)
        width=((quantiles[:,1:]-quantiles[:,:-1])*self.tau_gaps).sum(-1)
        position=self.curvature_head(out['encoded']).squeeze(-1).sigmoid()
        central_delta=(position-.5)*width
        out['clipped_mean']=(quantiles*self.quadrature_weights).sum(-1)+central_delta
        out['mean_correction']=out['mean_correction']+central_delta
        out['central_delta']=central_delta
        return out
