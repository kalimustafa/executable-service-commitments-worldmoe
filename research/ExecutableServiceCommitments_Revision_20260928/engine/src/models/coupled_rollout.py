"""Forecast-protected R2 adapter with price/PV cross moments and short rollouts."""
import torch
from torch import nn
from src.models.structured_decision import StructuredHead

class CoupledRolloutHead(StructuredHead):
    def __init__(self,initial_state=None):
        super().__init__()
        if initial_state is not None:self.load_state_dict(initial_state)
        self.adapter=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,128))
        nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)
        self.price_delta=nn.Linear(128,1);nn.init.zeros_(self.price_delta.weight);nn.init.zeros_(self.price_delta.bias)
        self.correlation=nn.Sequential(nn.Linear(157,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.correlation[-1].weight);nn.init.zeros_(self.correlation[-1].bias)

    def market_update(self,latent):
        adapted=latent+.1*torch.tanh(self.adapter(latent))
        shift=8*torch.tanh(self.price_delta(adapted)).squeeze(-1)
        return adapted,shift

    def forward(self,state,latent,market,pue_ref):
        adapted,shift=self.market_update(latent)
        updated=self.update_market_quantiles(state,market,adapted,shift)
        adapted=self.contextualize(state,adapted,updated)
        return self.decision(state,adapted,updated,pue_ref,shift)

    def update_market_quantiles(self,state,market,adapted,shift):
        return market+shift[:,:,None]*state[:,:,8:9]

    def decision(self,state,adapted,updated,pue_ref,shift):
        enc=self.encoder(state);mean=enc.mean(1,keepdim=True).expand_as(enc);maximum=enc.amax(1,keepdim=True).expand_as(enc)
        raw=self.head(torch.cat([enc,mean,maximum,adapted*state[:,:,8:9],updated/100.],-1)).float()
        out=self.decode(state,updated,pue_ref,raw)
        pv=out['pv_quantiles'];weights=self.probability_weights
        mu=(updated*weights).sum(-1)+5*torch.tanh(raw[:,:,4])*state[:,:,8]
        variance=((updated-(updated*weights).sum(-1,keepdim=True)).square()*weights).sum(-1)
        pvmean=(pv*weights).sum(-1,keepdim=True);pvstd=((pv-pvmean).square()*weights).sum(-1,keepdim=True).clamp_min(1e-10).sqrt()
        standardized=(pv-pvmean)/pvstd.clamp_min(1e-5)
        rho=.95*torch.tanh(self.correlation(torch.cat([state,adapted*state[:,:,8:9],updated/100.],-1)).float()).squeeze(-1)
        conditional_price=mu[:,:,None]+rho[:,:,None]*variance.clamp_min(1e-10).sqrt()[:,:,None]*standardized
        cap=(1-state[:,:,0]).clamp_min(0);pue=out['physical_pue'];ratio=pue/pue_ref[:,None]
        load=pue*(.23069976270198822+1.639059066772461*state[:,:,0])
        knots=torch.minimum(((pv-load[:,:,None])/(pue[:,:,None]*1.639059066772461)).clamp_min(0),cap[:,:,None])
        boundaries=torch.cat([torch.zeros_like(cap[:,:,None]),knots,cap[:,:,None]],-1)
        widths=(boundaries[:,:,1:]-boundaries[:,:,:-1]).clamp_min(0);mid=(boundaries[:,:,1:]+boundaries[:,:,:-1])*.5
        threshold=load[:,:,None]+pue[:,:,None]*1.639059066772461*mid
        covered=(pv[:,:,:,None]>threshold[:,:,None,:]).float()
        # E[price * grid] uses the conditional price/PV cross moment, rather
        # than multiplying their marginal expectations.
        solar_value=(covered*(conditional_price+1)[:,:,:,None]*weights[None,None,:,None]).sum(2)
        raw_cost=ratio[:,:,None]*(mu[:,:,None]-solar_value)
        # Negative conditional prices may yield nonconvex pieces. Their chord
        # is a linearized sorting proxy, not a certified bound or reported cash.
        nonconvex=(raw_cost[:,:,1:]-raw_cost[:,:,:-1]).amin(-1)<-1e-6
        average=(raw_cost*widths).sum(-1)/cap.clamp_min(1e-8)
        costs=torch.where(nonconvex[:,:,None],average[:,:,None],raw_cost)
        fees=torch.ones_like(cap)*2;fees[:,0]=0
        out.update(segment_scores=costs+out['opportunity'][:,:,None]+fees[:,:,None],segment_widths=widths,physical_segment_cost=raw_cost,cross_moment_rho=rho,rt_quantiles=updated,price_shift=shift,conditional_price=conditional_price,joint_mean_error=(conditional_price*weights).sum(-1)-mu)
        return out

    def contextualize(self,state,latent,market):
        return latent

class ArchitectedR2Head(CoupledRolloutHead):
    """Dual-stream cross attention, hourly temporal context, and queue queries.

    Only issue-legal forecasts and lagged observations enter the context stream.
    The market-only APIC shift is computed before queue-conditioned attention.
    """
    def __init__(self,initial_state=None):
        super().__init__(initial_state)
        self.market_encoder=nn.Sequential(nn.Linear(135,64),nn.GELU(),nn.LayerNorm(64))
        self.physical_encoder=nn.Sequential(nn.Linear(13,64),nn.GELU(),nn.LayerNorm(64))
        self.work_encoder=nn.Sequential(nn.Linear(9,64),nn.GELU(),nn.LayerNorm(64))
        self.market_to_physics=nn.MultiheadAttention(64,4,dropout=0,batch_first=True)
        self.physics_to_market=nn.MultiheadAttention(64,4,dropout=0,batch_first=True)
        self.fusion=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.LayerNorm(64))
        self.temporal=nn.ModuleList([nn.TransformerEncoderLayer(64,4,128,dropout=0,activation='gelu',batch_first=True,norm_first=True) for _ in range(2)])
        self.work_to_time=nn.MultiheadAttention(64,4,dropout=0,batch_first=True)
        self.slot_refinement=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.LayerNorm(64),nn.Linear(64,128))
        nn.init.zeros_(self.slot_refinement[-1].weight);nn.init.zeros_(self.slot_refinement[-1].bias)

    def encode_evidence(self,state,latent,market):
        adapted,shift=self.market_update(latent)
        updated=self.update_market_quantiles(state,market,adapted,shift)
        physical=torch.stack([state[:,:,j] for j in [2,3,4,8,9,10,11,16,17,18,19,20,21]],-1)
        m=self.market_encoder(torch.cat([adapted*state[:,:,8:9],updated/100.],-1))
        p=self.physical_encoder(physical)
        # Four quarter-hour tokens form one hourly context token. Local tokens
        # remain available for quarter-hour refinement after the temporal layers.
        mh=m.reshape(m.size(0),24,4,64).mean(2);ph=p.reshape(p.size(0),24,4,64).mean(2)
        mp,_=self.market_to_physics(mh,ph,ph,need_weights=False)
        pm,_=self.physics_to_market(ph,mh,mh,need_weights=False)
        context=self.fusion(torch.cat([mh+mp,ph+pm],-1))
        for layer in self.temporal:context=layer(context)
        return adapted,shift,updated,m,p,context

    def decode_evidence(self,state,evidence,pue_ref):
        adapted,shift,updated,m,p,context=evidence
        work=torch.stack([state[:,:,j] for j in [0,1,5,6,7,12,13,14,15]],-1)
        query=self.work_encoder(work)
        answer,_=self.work_to_time(query,context,context,need_weights=False)
        correction=self.slot_refinement(torch.cat([query+answer,m,p],-1))
        conditioned=adapted+.2*torch.tanh(correction)
        return self.decision(state,conditioned,updated,pue_ref,shift)

    def forward(self,state,latent,market,pue_ref):
        return self.decode_evidence(state,self.encode_evidence(state,latent,market),pue_ref)

class ArchitectedR2QPHead(ArchitectedR2Head):
    """Candidate-specific workload queries for the exact differentiable QP."""
    def decode_evidence(self,state,evidence,pue_ref):
        adapted,shift,updated,m,p,context=evidence
        work=torch.stack([state[:,:,j] for j in [0,1,5,6,7,12,13,14,15]],-1)
        query=self.work_encoder(work)+.5*(m+p)
        answer,_=self.work_to_time(query,context,context,need_weights=False)
        correction=self.slot_refinement(torch.cat([query+answer,m,p],-1))
        conditioned=adapted+.2*torch.tanh(correction)
        result=self.decision(state,conditioned,updated,pue_ref,shift)
        attended=query+answer
        result['continuation_features']=torch.cat([attended[:,0],attended.mean(1),context.mean(1)],-1)
        result['policy_tokens']=context
        return result
