import torch
from torch import nn

class MarketQuery(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.lead=nn.Embedding(97,16)
        self.zone=nn.Embedding(4,16)
        self.publication=nn.Embedding(2,8)
        self.project=nn.Sequential(nn.Linear(width+16+16+8+2,width),nn.GELU(),nn.LayerNorm(width))

    def forward(self,z_base,batch,publication,anchor_scaled):
        lead=self.lead(batch['lead_hours'].round().long().clamp(0,96))
        pub=self.publication(publication.long())
        query=self.project(torch.cat([z_base,lead,self.zone(batch['zone']),pub,
                                      publication[:,None],(publication*anchor_scaled)[:,None]],-1))
        return query,torch.cat([lead,pub],-1)

