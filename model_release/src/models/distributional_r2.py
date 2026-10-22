"""Explicit predictive PV distribution and task-specific price-quantile valuation."""
import torch
from torch import nn
from src.models.coupled_rollout import ArchitectedR2QPHead

class DistributionalR2Head(ArchitectedR2QPHead):
    def __init__(self):
        super().__init__()
        self.encoder[0]=nn.Linear(23,64)
        self.physical_encoder[0]=nn.Linear(14,64)
        self.pv_distribution=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,8))
        self.quantile_attention=nn.Sequential(nn.Linear(158,64),nn.GELU(),nn.Linear(64,7))
        self.distribution_embedding=nn.Linear(3,64)
        for layer in [self.pv_distribution[-1],self.quantile_attention[-1],self.distribution_embedding]:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.register_buffer('attenuation',torch.tensor([0.,.15,.35,.55,.75,1.,1.25,1.5]))

    def load_pretrained(self,source):
        state={k:v.clone() for k,v in source.items()}
        for key in ['encoder.0.weight','physical_encoder.0.weight']:
            current=self.state_dict()[key];padded=torch.zeros_like(current);padded[:,:state[key].shape[1]]=state[key];state[key]=padded
        missing,unexpected=self.load_state_dict(state,strict=False)
        assert not unexpected
        assert all(k.startswith(('pv_distribution.','quantile_attention.','distribution_embedding.')) or k=='attenuation' for k in missing)

    def encode_evidence(self,state,latent,market):
        adapted,shift=self.market_update(latent);updated=market+shift[:,:,None]*state[:,:,8:9]
        physical=torch.stack([state[:,:,j] for j in [2,3,4,8,9,10,11,16,17,18,19,20,21,22]],-1)
        m=self.market_encoder(torch.cat([adapted*state[:,:,8:9],updated/100.],-1));p=self.physical_encoder(physical)
        mh=m.reshape(m.size(0),24,4,64).mean(2);ph=p.reshape(p.size(0),24,4,64).mean(2)
        mp,_=self.market_to_physics(mh,ph,ph,need_weights=False);pm,_=self.physics_to_market(ph,mh,mh,need_weights=False)
        context=self.fusion(torch.cat([mh+mp,ph+pm],-1))
        for layer in self.temporal:context=layer(context)
        quarterly=context.repeat_interleave(4,dim=1)
        support=state[:,:,22,None].clamp_min(0)*self.attenuation
        mean=state[:,:,2].clamp_min(0);sigma=.03+.18*mean
        edges=(support[:,:,:-1]+support[:,:,1:])*.5
        cdf=.5*(1+torch.erf((edges-mean[:,:,None])/(sigma[:,:,None]*1.4142135623730951)))
        prior=torch.diff(torch.cat([torch.zeros_like(cdf[:,:,:1]),cdf,torch.ones_like(cdf[:,:,:1])],-1),dim=-1)
        prior=.9*prior+.1/8
        logits=self.pv_distribution(torch.cat([m,p,quarterly],-1)).float()
        probability=torch.softmax(prior.clamp_min(1e-8).log()+logits,-1)
        pvmean=(probability*support).sum(-1);pvstd=((support-pvmean[:,:,None]).square()*probability).sum(-1).clamp_min(1e-10).sqrt()
        upper=state[:,:,22].clamp_min(.02);light=(state[:,:,22]>.02).float()
        distribution=torch.stack([pvmean/upper,pvstd/upper*light,probability[:,:,0]*light],-1)
        embedded=self.distribution_embedding(distribution)
        p=p+embedded;context=context+embedded.reshape(embedded.size(0),24,4,64).mean(2)
        return adapted,shift,updated,m,p,context,support,probability

    def decode_evidence(self,state,evidence,pue_ref):
        adapted,shift,updated,m,p,context,support,probability=evidence
        work=torch.stack([state[:,:,j] for j in [0,1,5,6,7,12,13,14,15]],-1)
        query=self.work_encoder(work)+.5*(m+p);answer,_=self.work_to_time(query,context,context,need_weights=False)
        correction=self.slot_refinement(torch.cat([query+answer,m,p],-1));conditioned=adapted+.2*torch.tanh(correction)
        enc=self.encoder(state);raw=self.head(torch.cat([enc,enc.mean(1,keepdim=True).expand_as(enc),enc.amax(1,keepdim=True).expand_as(enc),conditioned*state[:,:,8:9],updated/100.],-1)).float()
        valuation_logits=2*torch.tanh(self.quantile_attention(torch.cat([state,conditioned,updated/100.],-1)).float())
        valuation=torch.softmax(self.probability_weights.log()+valuation_logits,-1)
        forecast_mean=(updated*valuation).sum(-1);mu=forecast_mean+5*torch.tanh(raw[:,:,4])*state[:,:,8]
        price_var=((updated-forecast_mean[:,:,None]).square()*valuation).sum(-1).clamp_min(1e-10)
        pvmean=(support*probability).sum(-1,keepdim=True);pvstd=((support-pvmean).square()*probability).sum(-1,keepdim=True).clamp_min(1e-10).sqrt()
        standardized=(support-pvmean)/pvstd
        rho=.95*torch.tanh(self.correlation(torch.cat([state[:,:,:22],conditioned*state[:,:,8:9],updated/100.],-1)).float()).squeeze(-1)
        conditional=mu[:,:,None]+rho[:,:,None]*price_var.sqrt()[:,:,None]*standardized
        ratio=torch.where(state[:,:,9].bool(),state[:,:,3],torch.ones_like(state[:,:,3]));pue=ratio*pue_ref[:,None];cap=(1-state[:,:,0]).clamp_min(0)
        load=pue*(.23069976270198822+1.639059066772461*state[:,:,0])
        knots=torch.minimum(((support-load[:,:,None])/(pue[:,:,None]*1.639059066772461)).clamp_min(0),cap[:,:,None])
        boundaries=torch.cat([torch.zeros_like(cap[:,:,None]),knots,cap[:,:,None]],-1);widths=(boundaries[:,:,1:]-boundaries[:,:,:-1]).clamp_min(0)
        mid=(boundaries[:,:,1:]+boundaries[:,:,:-1])*.5;threshold=load[:,:,None]+pue[:,:,None]*1.639059066772461*mid
        cover=(support[:,:,:,None]>threshold[:,:,None,:]).float();solar_value=(cover*(conditional+1)[:,:,:,None]*probability[:,:,:,None]).sum(2)
        raw_cost=ratio[:,:,None]*(mu[:,:,None]-solar_value)
        nonconvex=(raw_cost[:,:,1:]-raw_cost[:,:,:-1]).amin(-1)<-1e-6;average=(raw_cost*widths).sum(-1)/cap.clamp_min(1e-8)
        costs=torch.where(nonconvex[:,:,None],average[:,:,None],raw_cost)
        opportunity=5*torch.tanh(raw[:,:,3])*state[:,:,8];opportunity=opportunity-opportunity.mean(1,keepdim=True)
        fees=torch.ones_like(cap)*2;fees[:,0]=0
        kl=(valuation*(valuation.clamp_min(1e-8).log()-self.probability_weights.log())).sum(-1)
        return dict(segment_widths=widths,segment_scores=costs+opportunity[:,:,None]+fees[:,:,None],physical_segment_cost=raw_cost,pv_support=support,pv_probabilities=probability,rt_quantiles=updated,rt_decision=mu,physical_pue=pue,opportunity=opportunity,cross_moment_rho=rho,price_shift=shift,valuation_weights=valuation,valuation_kl=kl,joint_mean_error=(conditional*probability).sum(-1)-mu)

    def forward(self,state,latent,market,pue_ref):
        return self.decode_evidence(state,self.encode_evidence(state,latent,market),pue_ref)
