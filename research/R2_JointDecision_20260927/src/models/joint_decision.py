import torch
from torch import nn
import torch.nn.functional as F
from src.calibration.native_affine import NativeCalibration

class JointDecision(nn.Module):
    def __init__(self,r2,target_scales,historical_rt,pue_reference):
        super().__init__();self.r2=r2
        self.calibration=NativeCalibration()
        self.register_buffer('target_scales',target_scales.detach().clone())
        self.register_buffer('historical_rt',historical_rt.detach().clone())
        self.register_buffer('pue_reference',pue_reference.detach().clone())
        self.state_encoder=nn.Sequential(nn.Linear(16,64),nn.GELU(),nn.LayerNorm(64),nn.Linear(64,64),nn.GELU())
        self.value_head=nn.Sequential(nn.Linear(128+64+7,128),nn.GELU(),nn.LayerNorm(128),nn.Linear(128,64),nn.GELU(),nn.Linear(64,2))
        nn.init.zeros_(self.value_head[-1].weight);nn.init.zeros_(self.value_head[-1].bias)

    def forward(self,batch,publication,anchor,state=None):
        out=self.r2(batch,publication,anchor)
        da,dart=self.calibration(out['da_raw'],out['dart'],batch['zone'],publication,self.target_scales)
        price=self.r2.from_latent(out['latent'],batch['zone'],publication,anchor,da_quantiles=da,dart_quantiles=dart)
        for key in ['world_gate','applied_residual','regime_logits','decision_latent']:price[key]=out[key]
        if state is not None:
            # State: booked, delta, PV prior, PUE ratio, horizon fraction,
            # remaining-work log, backlog log, occupancy summary,
            # forecast available, physical available, published, deadline slack.
            booked=state[:,0];delta=state[:,1].clamp_min(1e-5);pv=state[:,2]
            ratio=state[:,3]*state[:,9]+(1-state[:,9]);pue=ratio*self.pue_reference[batch['zone']]
            load0=pue*(.23069976270198822+1.639059066772461*booked)
            increment=pue*1.639059066772461*delta
            covered=((pv-load0)/increment).clamp(0,1)*state[:,9]
            available=state[:,8,None]
            market=price['rt']*available+self.historical_rt[batch['zone'],None]*(1-available)
            latent=price['decision_latent']*available
            analytic=ratio*((1-covered)*market[:,3]-covered)
            encoded=self.state_encoder(state)
            heads=self.value_head(torch.cat([latent,encoded,market/100.],-1))
            residual=80.*torch.tanh(heads[:,0])
            opportunity=40.*torch.tanh(heads[:,1])
            price['decision_cost']=analytic+residual
            price['decision_score']=analytic+residual+opportunity
            price['opportunity_adjustment']=opportunity
            price['analytic_cost']=analytic
            price['decision_residual']=residual
        return price
