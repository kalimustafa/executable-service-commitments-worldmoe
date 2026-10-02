"""Target-conditioned hourly world pooling, then scale/expert interaction."""
import math
import torch
from torch import nn
from src.models.r2_repaired import R2WGR

class HourlyWorldAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        keys=['early','middle','late','site','expert_bank','w1_global','tail_context']
        self.adapters=nn.ModuleDict({k:nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.LayerNorm(128)) for k in keys})
        self.entity=nn.Embedding(4,128);self.scale=nn.Embedding(7,128)
        self.time=nn.Sequential(nn.Linear(3,32),nn.GELU(),nn.Linear(32,128))
        self.query=nn.Linear(128,32,bias=False);self.key=nn.Linear(128,32,bias=False)
        self.last_temporal_attention=None

    def forward(self,batch,query,ablation='normal'):
        available=batch['hourly_mask'].flatten(1)
        hour=torch.arange(24,device=query.device).repeat_interleave(4).float()
        entity=torch.arange(4,device=query.device).repeat(24)
        angle=hour*2*math.pi/24
        timing=torch.stack([angle.sin()[None].expand(len(query),-1),angle.cos()[None].expand(len(query),-1),
                            (hour[None]-batch['target_utc_offset'][:,None])/24],-1)
        time_embedding=self.time(timing)+self.entity(entity)[None]
        tokens=[];bias=[];attention=[];experts=None
        for index,key in enumerate(['early','middle','late']):
            if ablation=='late_scale_only' and key!='late':continue
            values=batch['hourly_'+key].float().flatten(1,2)
            encoded=self.adapters[key](values)+time_embedding+self.scale.weight[index]
            logits=(self.key(encoded)*self.query(query)[:,None]).sum(-1)/math.sqrt(32)
            weights=logits.float().masked_fill(~available,-1e4).softmax(-1)*available
            weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
            pooled=(weights[...,None]*encoded.float()).sum(1)
            tokens.append(pooled[:,None]);bias.append(torch.zeros(len(query),1,device=query.device));attention.append(weights)
        self.last_temporal_attention=torch.stack(attention,1).detach()
        for index,key in enumerate(['site','expert_bank','w1_global','tail_context'],3):
            if key=='expert_bank' and ablation=='no_expert_tokens':continue
            values=batch[key].float()
            if values.ndim==2:values=values[:,None]
            encoded=self.adapters[key](values)+self.scale.weight[index]
            tokens.append(encoded)
            if key=='expert_bank':
                experts=encoded;prior=batch['expert_prior'].float()
                bias.append(prior.clamp_min(1e-8).log().masked_fill(prior<=1e-8,-1e4))
            else:bias.append(torch.zeros(encoded.shape[:2],device=query.device))
        return torch.cat(tokens,1),torch.cat(bias,1),experts

class R2TemporalWorld(R2WGR):
    def __init__(self,base):
        super().__init__(base);self.adapter=HourlyWorldAdapter()

    def context(self,batch,publication,anchor):
        z=self.base.encode(batch,publication,anchor)
        available=batch['world_available'].float()
        if self.training and self.world_dropout:available=available*(torch.rand_like(available)>=self.world_dropout).float()
        zone=batch['zone'];anchor_scaled=(anchor-self.base.da_center[zone])/self.base.da_scale[zone]
        query,context=self.market_query(z,batch,publication,anchor_scaled)
        tokens,bias,experts=self.adapter(batch,query,self.ablation)
        h,alpha=self.cross_attention(query,tokens,bias,experts,batch['expert_prior'])
        prior=batch['expert_prior'].float();p=prior/prior.sum(-1,keepdim=True).clamp_min(1e-8)
        entropy=-(p*p.clamp_min(1e-8).log()).sum(-1)/math.log(7)
        g=self.gate(z,h,context,available,entropy,p.max(-1).values)
        if self.ablation=='force_gate_0':g=torch.zeros_like(g)
        if self.ablation=='force_gate_1':g=available[:,None].expand_as(g)
        if self.gate_override is not None:g=available[:,None].expand_as(g)*float(self.gate_override)
        return z,h,g,alpha*available[:,None],entropy,available
