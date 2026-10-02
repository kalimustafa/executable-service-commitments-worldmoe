"""Small market-intelligence head over cached, frozen WorldMoE features.

No action or feasibility decoder is present. DA and DART have monotone marginal
quantiles; pre-publication RT is derived from paired joint draws. A published
DA anchor yields the exact complementary-quantile transformation for RT.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import math
import torch
from torch import nn,Tensor
import torch.nn.functional as F

QUANTILES=(.05,.10,.25,.50,.75,.90,.95)

@dataclass(frozen=True)
class R2Config:
    numerical_features:int
    width:int=128
    expert_width:int=64
    dropout:float=.05
    joint_samples:int=64
    use_world:bool=True
    seed:int=20260920
    slot_head_enabled:bool=False

def monotone_quantiles(raw:Tensor)->Tensor:
    """Seven ordered quantiles centred on the explicitly predicted median."""
    raw=raw.float();center=raw[...,3:4]
    left=F.softplus(raw[...,:3])+1e-4;right=F.softplus(raw[...,4:])+1e-4
    lower=center-torch.flip(torch.cumsum(torch.flip(left,[-1]),-1),[-1])
    upper=center+torch.cumsum(right,-1)
    return torch.cat((lower,center,upper),-1)

def inverse_piecewise_quantile(q:Tensor,u:Tensor)->Tensor:
    """Piecewise-linear inverse CDF, with linear outer-tail continuation."""
    tau=q.new_tensor(QUANTILES)
    idx=torch.searchsorted(tau,u.contiguous(),right=True).sub(1).clamp(0,5)
    q0=torch.gather(q,1,idx);q1=torch.gather(q,1,idx+1)
    return q0+(q1-q0)*(u-tau[idx])/(tau[idx+1]-tau[idx])

def pinball(prediction:Tensor,target:Tensor)->Tensor:
    residual=target.float()[...,None]-prediction.float()
    tau=prediction.new_tensor(QUANTILES,dtype=torch.float32)
    return torch.maximum(tau*residual,(tau-1.)*residual).mean(-1)

class R2MarketHead(nn.Module):
    def __init__(self,config:R2Config,*,da_center:Tensor,da_scale:Tensor,dart_center:Tensor,dart_scale:Tensor):
        super().__init__();self.config=config;h=config.width;e=config.expert_width
        self.numerical=nn.Sequential(nn.Linear(config.numerical_features+3,h),nn.LayerNorm(h),nn.GELU(),nn.Dropout(config.dropout),nn.Linear(h,h),nn.GELU())
        self.world_adapter=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,h),nn.GELU())
        self.expert_key=nn.Linear(256,e);self.expert_identity=nn.Embedding(7,e)
        self.expert_query=nn.Linear(h,e)
        self.shared=nn.Sequential(nn.Linear(2*h+e+41,h),nn.LayerNorm(h),nn.GELU(),nn.Dropout(config.dropout),nn.Linear(h,h),nn.GELU())
        self.da_head=nn.Linear(h,7);self.dart_head=nn.Linear(h,7);self.dependence=nn.Linear(h,1)
        self.regime_head=nn.Linear(h,3)
        self.slot_head=nn.Sequential(nn.Linear(h,64),nn.GELU(),nn.Linear(64,1))
        # Ranking is trained, if retained, in a separate explicit stage.
        self.slot_head.requires_grad_(False)
        for name,value in [('da_center',da_center),('da_scale',da_scale),('dart_center',dart_center),('dart_scale',dart_scale)]:
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32).reshape(4))
        sobol=torch.quasirandom.SobolEngine(2,scramble=True,seed=config.seed)
        uniforms=sobol.draw(config.joint_samples).clamp(1e-4,1-1e-4)
        self.register_buffer('joint_normal_draws',math.sqrt(2)*torch.erfinv(2*uniforms-1))
        self.register_buffer('quantile_levels',torch.tensor(QUANTILES,dtype=torch.float32))

    def encode(self,batch:dict[str,Tensor],gate:Tensor,anchor:Tensor)->Tensor:
        zone=batch['zone'].long();available=batch['world_available'].float()*float(self.config.use_world)
        published=gate.float();anchor_scaled=(anchor.float()-self.da_center[zone])/self.da_scale[zone]
        numerical=torch.cat((batch['numerical'].float(),published[:,None],(published*anchor_scaled)[:,None],available[:,None]),-1)
        numerical=self.numerical(numerical)
        world=torch.cat((batch['w0_local'],batch['w1_local'],batch['w1_global'],batch['tail_context']),-1).float()
        world=self.world_adapter(world)*available[:,None]
        keys=self.expert_key(batch['expert_bank'].float())+self.expert_identity.weight[None]
        prior=batch['expert_prior'].float();valid=prior>1e-8
        logits=(keys*self.expert_query(numerical)[:,None]).sum(-1)/math.sqrt(self.config.expert_width)
        logits=logits+torch.log(prior.clamp_min(1e-8))
        logits=logits.masked_fill(~valid,-1e4)
        expert=(torch.softmax(logits.float(),-1)[...,None]*keys.float()).sum(1)*available[:,None]
        regime=torch.cat((batch['regime'].float(),batch['log_scale'].float(),batch['tail_probability'].float()[:,None]),-1)*available[:,None]
        return self.shared(torch.cat((numerical,world,expert,regime),-1))

    def from_latent(self,latent:Tensor,zone:Tensor,gate:Tensor,anchor:Tensor,
                    *,da_quantiles:Tensor|None=None,dart_quantiles:Tensor|None=None)->dict[str,Tensor]:
        zone=zone.long();known=gate.bool()
        da_raw=monotone_quantiles(self.da_head(latent))*self.da_scale[zone,None]+self.da_center[zone,None] if da_quantiles is None else da_quantiles.float()
        dart=monotone_quantiles(self.dart_head(latent))*self.dart_scale[zone,None]+self.dart_center[zone,None] if dart_quantiles is None else dart_quantiles.float()
        rho=.95*torch.tanh(self.dependence(latent).float().squeeze(-1))
        normals=self.joint_normal_draws.float()
        first=normals[None,:,0].expand(len(zone),-1)
        second=rho[:,None]*first+torch.sqrt(1-rho.square())[:,None]*normals[None,:,1]
        u=(.5*(1+torch.erf(first/math.sqrt(2)))).clamp(1e-5,1-1e-5)
        v=(.5*(1+torch.erf(second/math.sqrt(2)))).clamp(1e-5,1-1e-5)
        da_samples=inverse_piecewise_quantile(da_raw,u)
        da_samples=torch.where(known[:,None],anchor.float()[:,None],da_samples)
        dart_samples=inverse_piecewise_quantile(dart,v)
        rt_samples=da_samples-dart_samples
        rt=torch.quantile(rt_samples,self.quantile_levels,dim=1).T
        rt=torch.where(known[:,None],anchor.float()[:,None]-torch.flip(dart,[-1]),rt)
        da=torch.where(known[:,None],anchor.float()[:,None].expand_as(da_raw),da_raw)
        da_mean=da_samples.mean(1);dart_mean=dart_samples.mean(1);rt_mean=da_mean-dart_mean
        return dict(da=da,dart=dart,rt=rt,da_raw=da_raw,rho=rho,da_samples=da_samples,dart_samples=dart_samples,
            rt_samples=rt_samples,da_mean=da_mean,dart_mean=dart_mean,rt_mean=rt_mean,
            regime_logits=self.regime_head(latent).float(),
            slot_score=self.slot_head(latent.detach()).float().squeeze(-1) if self.config.slot_head_enabled else rt[:,3],latent=latent)

    def forward(self,batch:dict[str,Tensor],gate:Tensor,anchor:Tensor)->dict[str,Tensor]:
        return self.from_latent(self.encode(batch,gate,anchor),batch['zone'],gate,anchor)

    def configuration(self):return asdict(self.config)

def identity_errors(output:dict[str,Tensor],gate:Tensor,anchor:Tensor)->dict[str,float]:
    known=gate.bool()
    errors=dict(sample=float((output['rt_samples']-(output['da_samples']-output['dart_samples'])).abs().max()),
        mean=float((output['rt_mean']-(output['da_mean']-output['dart_mean'])).abs().max()),
        quantile_crossing=float(torch.clamp_min(-torch.diff(output['rt'],dim=-1),0).max()))
    errors['post_DA_quantile']=float((output['rt'][known]-(anchor.float()[known,None]-output['dart'][known].flip(-1))).abs().max()) if known.any() else 0.
    return errors
