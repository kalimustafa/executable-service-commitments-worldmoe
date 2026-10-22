"""Physically structured, context-aware cost curves with a protected price task."""
import torch
from torch import nn
import torch.nn.functional as F
from src.models.joint_decision import JointDecision

class StructuredHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(22,64),nn.GELU(),nn.LayerNorm(64),nn.Linear(64,64),nn.GELU())
        self.head=nn.Sequential(nn.Linear(327,96),nn.GELU(),nn.LayerNorm(96),nn.Linear(96,64),nn.GELU(),nn.Linear(64,5))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('normal_scores',torch.tensor([-1.644854,-1.281552,-.674490,0.,.674490,1.281552,1.644854]))
        self.register_buffer('probability_weights',torch.tensor([.075,.1,.2,.25,.2,.1,.075]))

    def forward(self,state,latent,market,pue_ref):
        # Each batch element is one arriving cohort and its 96 candidate slots.
        available=state[:,:,8:9];physical=state[:,:,9]
        enc=self.encoder(state)
        mean=enc.mean(1,keepdim=True).expand_as(enc);maximum=enc.amax(1,keepdim=True).expand_as(enc)
        raw=self.head(torch.cat([enc,mean,maximum,latent*available,market/100.],-1)).float()
        return self.decode(state,market,pue_ref,raw)

    def decode(self,state,market,pue_ref,raw):
        available=state[:,:,8:9];physical=state[:,:,9]
        nowcast=state[:,:,2]*torch.exp(2*state[:,:,16])
        median=(nowcast*torch.exp(.6*torch.tanh(raw[:,:,0]))+.3*torch.tanh(raw[:,:,1])).clamp_min(0.)
        spread=(.03+.18*nowcast)*torch.exp(.8*torch.tanh(raw[:,:,2]))
        pv=(median[:,:,None]+spread[:,:,None]*self.normal_scores).clamp_min(0)*physical[:,:,None]
        # Expected price from the issued quantile distribution, with a small
        # bounded task residual. Price forecasts themselves retain APIC.
        expected=(market*self.probability_weights).sum(-1)
        rt=expected+5*torch.tanh(raw[:,:,4])*available.squeeze(-1)
        opportunity=5*torch.tanh(raw[:,:,3])*available.squeeze(-1)
        opportunity=opportunity-opportunity.mean(1,keepdim=True)
        ratio=torch.where(physical.bool(),state[:,:,3],torch.ones_like(state[:,:,3]))
        pue=ratio*pue_ref[:,None]
        cap=(1-state[:,:,0]).clamp_min(0)
        load=pue*(.23069976270198822+1.639059066772461*state[:,:,0])
        knots=((pv-load[:,:,None])/(pue[:,:,None]*1.639059066772461)).clamp_min(0)
        knots=torch.minimum(knots,cap[:,:,None])
        boundaries=torch.cat([torch.zeros_like(cap[:,:,None]),knots,cap[:,:,None]],-1)
        widths=(boundaries[:,:,1:]-boundaries[:,:,:-1]).clamp_min(0)
        mid=(boundaries[:,:,1:]+boundaries[:,:,:-1])*.5
        threshold=load[:,:,None]+pue[:,:,None]*1.639059066772461*mid
        coverage=((pv[:,:,:,None]>threshold[:,:,None,:]).float()*self.probability_weights[None,None,:,None]).sum(2)
        costs=ratio[:,:,None]*((1-coverage)*rt[:,:,None]-coverage)
        # For negative prices, the convex envelope supplies coherent slot-level
        # bundle scores. Execution remains a feasible allocation; its realized
        # bill is computed independently, without using the envelope as cash.
        average=(costs*widths).sum(-1)/cap.clamp_min(1e-8)
        costs=torch.where((rt < -1)[:,:,None],average[:,:,None],costs)
        fees=torch.ones_like(cap)*2.;fees[:,0]=0.
        scores=costs+opportunity[:,:,None]+fees[:,:,None]
        return dict(pv_quantiles=pv,rt_decision=rt,opportunity=opportunity,segment_widths=widths,segment_scores=scores,physical_pue=pue)

class CachedStructuredHead(StructuredHead):
    """Algebraic inference factorization; no added parameters or training."""
    def __init__(self,state_dict):
        super().__init__();self.load_state_dict(state_dict)
        first=self.head[0]
        self.register_buffer('local_weight',first.weight[:,:64].detach().contiguous())
        self.register_buffer('pool_weight',first.weight[:,64:192].detach().contiguous())
        self.register_buffer('context_weight',first.weight[:,192:].detach().contiguous())
        self.register_buffer('context_bias',first.bias.detach().clone())

    def project_context(self,latent,market):
        return F.linear(torch.cat([latent,market/100.],-1),self.context_weight,self.context_bias)

    def forward(self,state,context,market,pue_ref):
        enc=self.encoder(state)
        pool=torch.cat([enc.mean(1,keepdim=True),enc.amax(1,keepdim=True)],-1)
        raw=F.linear(enc,self.local_weight)+F.linear(pool,self.pool_weight)+context
        for layer in self.head[1:]:raw=layer(raw)
        return self.decode(state,market,pue_ref,raw.float())

class StructuredDecision(JointDecision):
    def __init__(self,r2,target_scales,historical_rt,pue_reference):
        super().__init__(r2,target_scales,historical_rt,pue_reference)
        del self.state_encoder;del self.value_head
        self.structured=StructuredHead()

    def forward(self,batch,publication,anchor,state=None):
        price=super().forward(batch,publication,anchor,None)
        if state is not None:
            shape=state.shape[:2]
            market=price['rt'].reshape(*shape,7)
            historical=self.historical_rt[batch['zone']].reshape(*shape,1)
            market=market*state[:,:,8:9]+historical*(1-state[:,:,8:9])
            latent=price['decision_latent'].reshape(*shape,128)
            zones=batch['zone'].reshape(shape)[:,0]
            price.update(self.structured(state,latent,market,self.pue_reference[zones]))
        return price

def structured_allocation(widths,scores,amount,straight_through=False,temperature=1.5):
    shape=widths.shape;capacity=widths.reshape(len(widths),-1);value=scores.reshape(len(widths),-1)
    order=torch.argsort(value,dim=1,stable=True);ordered=capacity.gather(1,order)
    take=torch.minimum(ordered,(amount[:,None]-(ordered.cumsum(1)-ordered)).clamp_min(0))
    hard=torch.zeros_like(capacity).scatter(1,order,take)
    if straight_through:
        soft=torch.zeros_like(capacity)
        for _ in range(8):
            left=(capacity-soft).clamp_min(0);need=(amount-soft.sum(1)).clamp_min(0)
            logits=-value/temperature+left.clamp_min(1e-12).log()
            probability=torch.softmax(logits.masked_fill(left<1e-9,-1e6),1)
            soft=soft+torch.minimum(need[:,None]*probability,left)
        left=(capacity-soft).clamp_min(0);need=(amount-soft.sum(1)).clamp_min(0)
        soft=soft+need[:,None]*left/left.sum(1,keepdim=True).clamp_min(1e-8)
        allocation=hard.detach()+soft-soft.detach()
    else:allocation=hard
    return allocation.reshape(shape).sum(-1)
