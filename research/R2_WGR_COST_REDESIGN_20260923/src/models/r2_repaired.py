"""Continuous Base-equivalent world residual, without rewriting its latent."""
import math
import torch
from torch import nn
from src.models.market_query import MarketQuery
from src.models.world_multiscale_adapter import WorldMultiscaleAdapter
from src.models.cross_attention import MarketWorldAttention
from src.models.world_gate import WorldGate

class R2WGR(nn.Module):
    def __init__(self,base,*,gated=True,world_dropout=.10):
        super().__init__();self.base=base;self.gated=gated;self.world_dropout=world_dropout
        self.market_query=MarketQuery();self.adapter=WorldMultiscaleAdapter();self.cross_attention=MarketWorldAttention()
        self.residual=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.LayerNorm(128),nn.Dropout(.05),nn.Linear(128,128))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
        self.gate=WorldGate() if gated else None
        self.ablation='normal';self.gate_override=None

    def train(self,mode=True):
        super().train(mode)
        self.base.eval()  # Base dropout must not corrupt its protected path.
        return self

    def from_latent(self,*args,**kwargs):return self.base.from_latent(*args,**kwargs)

    def context(self,batch,publication,anchor):
        z=self.base.encode(batch,publication,anchor)
        available=batch['world_available'].float()
        if self.training and self.world_dropout:
            available=available*(torch.rand_like(available)>=self.world_dropout).float()
        zone=batch['zone'];anchor_scaled=(anchor-self.base.da_center[zone])/self.base.da_scale[zone]
        query,context=self.market_query(z,batch,publication,anchor_scaled)
        tokens,bias,experts,_=self.adapter(batch,self.ablation)
        h,alpha=self.cross_attention(query,tokens,bias,experts,batch['expert_prior'])
        prior=batch['expert_prior'].float();p=prior/prior.sum(-1,keepdim=True).clamp_min(1e-8)
        entropy=-(p*p.clamp_min(1e-8).log()).sum(-1)/math.log(7)
        confidence=p.max(-1).values
        g=self.gate(z,h,context,available,entropy,confidence) if self.gated else available[:,None].expand(-1,128)
        if self.ablation=='force_gate_0':g=torch.zeros_like(g)
        if self.ablation=='force_gate_1':g=available[:,None].expand_as(g)
        if self.gate_override is not None:g=available[:,None].expand_as(g)*float(self.gate_override)
        return z,h,g,alpha*available[:,None],entropy,available

    def forward(self,batch,publication,anchor):
        z,h,g,alpha,entropy,available=self.context(batch,publication,anchor)
        delta=.5*torch.tanh(self.residual(h).float())
        latent=z.float()+g*delta
        out=self.from_latent(latent,batch['zone'],publication,anchor)
        out.update(world_gate=g,world_residual=delta,applied_residual=g*delta,expert_attention=alpha,
                   router_entropy=entropy,world_available=available,base_latent=z)
        return out
