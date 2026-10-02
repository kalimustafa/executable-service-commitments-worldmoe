"""Action queries attend to R2 time tokens to correct continuation valuations."""
import torch
from torch import nn


class ActionValueDecoder(nn.Module):
    def __init__(self, dollar_scale=1., bound=10.):
        super().__init__()
        self.register_buffer('dollar_scale',torch.tensor(float(dollar_scale)))
        self.register_buffer('bound',torch.tensor(float(bound)))
        self.state=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,64),nn.GELU())
        self.action=nn.Sequential(nn.Linear(3,64),nn.GELU(),nn.LayerNorm(64))
        self.attention=nn.MultiheadAttention(64,4,dropout=0,batch_first=True)
        self.value=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.LayerNorm(64),nn.Linear(64,1))
        nn.init.zeros_(self.value[-1].weight);nn.init.zeros_(self.value[-1].bias)

    def forward(self, features, tokens, actions, base, alternative):
        local=torch.stack([actions,base/self.dollar_scale,alternative/self.dollar_scale],-1)
        query=self.action(local)+self.state(features)[:,None]
        answer,_=self.attention(query,tokens,tokens,need_weights=False)
        raw=self.value(torch.cat([query,answer],-1)).squeeze(-1)
        correction=self.bound*torch.tanh(raw)*actions
        return base+correction,correction
