"""BPSR: causal profile query and zero-initialized ordered-decoder residual."""
import math
import torch
from torch import nn
from vendor.r2_market import monotone_quantiles
from src.models.r2_repaired import R2WGR

class R2ProfileResidual(R2WGR):
    def __init__(self,base):
        super().__init__(base)
        n=base.config.numerical_features
        self.profile_encoder=nn.Sequential(nn.Linear(n+2,64),nn.GELU(),nn.LayerNorm(64))
        self.profile_query=nn.Linear(128,64,bias=False)
        self.profile_integrate=nn.Sequential(nn.Linear(128+64,128),nn.GELU(),nn.LayerNorm(128))
        self.residual=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.LayerNorm(128),nn.Linear(128,128),nn.GELU())
        # Zero only the final matrix, avoiding two serial zero projections.
        self.raw_projection=nn.Linear(128,14,bias=False)
        nn.init.zeros_(self.raw_projection.weight)

    def profile(self,z,batch,publication):
        mask=batch['profile_mask'];zone=batch['zone']
        anchor=(batch['profile_da'].float()-self.base.da_center[zone,None])/self.base.da_scale[zone,None]
        # Mask the full DA curve and its mean before PRE-DA query construction.
        anchor=torch.where(publication[:,None].bool(),anchor,torch.zeros_like(anchor))
        mean=(anchor*mask).sum(-1,keepdim=True)/mask.sum(-1,keepdim=True)
        relative=(anchor-mean)*mask
        inputs=torch.cat([batch['profile_numerical'].float(),relative[...,None],publication[:,None,None].expand(-1,25,1)],-1)
        tokens=self.profile_encoder(inputs)
        scores=(tokens*self.profile_query(z)[:,None]).sum(-1)/math.sqrt(64)
        weights=scores.float().masked_fill(~mask,-1e9).softmax(-1)
        return (weights[...,None]*tokens.float()).sum(1),weights

    def forward(self,batch,publication,anchor):
        z,h,g,alpha,entropy,available=self.context(batch,publication,anchor)
        profile,weights=self.profile(z,batch,publication)
        fused=self.profile_integrate(torch.cat([h,profile],-1))
        features=torch.tanh(self.residual(fused).float())
        # Residuals change median/increment logits BEFORE monotone decoding.
        # There is no free addition to final quantiles and no sorting repair.
        raw_delta=.5*torch.tanh(self.raw_projection(g*features).float())
        zone=batch['zone']
        da_raw=self.base.da_head(z).float()+raw_delta[:,:7]
        dart_raw=self.base.dart_head(z).float()+raw_delta[:,7:]
        da=monotone_quantiles(da_raw)*self.base.da_scale[zone,None]+self.base.da_center[zone,None]
        dart=monotone_quantiles(dart_raw)*self.base.dart_scale[zone,None]+self.base.dart_center[zone,None]
        out=self.from_latent(z,zone,publication,anchor,da_quantiles=da,dart_quantiles=dart)
        out.update(world_gate=g,world_residual=features,applied_residual=raw_delta,
                   expert_attention=alpha,router_entropy=entropy,world_available=available,
                   base_latent=z,profile_attention=weights,raw_decoder_residual=raw_delta)
        return out
