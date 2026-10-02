import math
import torch
from torch import nn

class CrossAttentionBlock(nn.Module):
    def __init__(self,width=128,heads=4,dropout=.05):
        super().__init__();self.heads=heads
        self.attention=nn.MultiheadAttention(width,heads,dropout=dropout,batch_first=True)
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*width,width))

    def forward(self,query,tokens,bias):
        mask=bias[:,None,None,:].expand(-1,self.heads,query.shape[1],-1).reshape(-1,query.shape[1],tokens.shape[1])
        out,_=self.attention(query,tokens,tokens,attn_mask=mask.to(query.dtype),need_weights=False)
        out=self.norm1(query+out);return self.norm2(out+self.ff(out))

class MarketWorldAttention(nn.Module):
    def __init__(self,width=128,heads=4,layers=2,dropout=.05):
        super().__init__()
        self.blocks=nn.ModuleList([CrossAttentionBlock(width,heads,dropout) for _ in range(layers)])
        self.expert_query=nn.Linear(width,width,bias=False)
        self.expert_key=nn.Linear(width,width,bias=False)
        self.integrate=nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.LayerNorm(width))

    def forward(self,query,tokens,bias,experts,prior):
        q=query[:,None]
        for block in self.blocks:q=block(q,tokens,bias)
        if experts is None:
            alpha=torch.zeros(len(query),7,device=query.device)
            selected=torch.zeros_like(query)
        else:
            logits=(self.expert_query(query)[:,None]*self.expert_key(experts)).sum(-1)/math.sqrt(query.shape[-1])
            logits=logits.float()+prior.float().clamp_min(1e-8).log()
            logits=logits.masked_fill(prior<=1e-8,-1e4)
            alpha=logits.softmax(-1)
            selected=(alpha[...,None]*experts.float()).sum(1)
        return self.integrate(torch.cat([q[:,0],selected],-1)),alpha

