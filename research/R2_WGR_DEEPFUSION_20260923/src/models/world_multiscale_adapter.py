import torch
from torch import nn

class WorldMultiscaleAdapter(nn.Module):
    groups=('early','middle','late','site','expert_bank','w1_global','tail_context')
    def __init__(self,width=128):
        super().__init__()
        self.adapters=nn.ModuleDict({key:nn.Sequential(nn.Linear(256,width),nn.GELU(),nn.LayerNorm(width)) for key in self.groups})
        self.scale=nn.Embedding(len(self.groups),width)

    def forward(self,batch,ablation='normal'):
        tokens=[];bias=[];expert_tokens=None;expert_range=None
        for i,key in enumerate(self.groups):
            if ablation=='late_scale_only' and key in {'early','middle'}:continue
            if ablation=='no_expert_tokens' and key=='expert_bank':continue
            values=batch[key].float()
            if values.ndim==2:values=values[:,None]
            encoded=self.adapters[key](values)+self.scale.weight[i]
            start=sum(x.shape[1] for x in tokens)
            tokens.append(encoded)
            if key=='expert_bank':
                prior=batch['expert_prior'].float()
                b=prior.clamp_min(1e-8).log().masked_fill(prior<=1e-8,-1e4)
                expert_tokens=encoded;expert_range=(start,start+encoded.shape[1])
            else:b=torch.zeros(encoded.shape[:2],device=encoded.device)
            bias.append(b)
        return torch.cat(tokens,1),torch.cat(bias,1),expert_tokens,expert_range

