import torch
from torch import nn
import numpy as np
from vendor.r2_market import monotone_quantiles

class R2OrderedResidualTrees(nn.Module):
    def __init__(self,source,feature_count=16):
        super().__init__();self.source=source
        self.register_buffer('pca_center',torch.zeros(1024))
        self.register_buffer('pca_components',torch.zeros(feature_count,1024))
        self.trees=[];self.learning_rate=.2;self.ablation='normal'
        self.source.eval().requires_grad_(False)

    def train(self,mode=True):
        super().train(mode);self.source.eval();return self

    def from_latent(self,*args,**kwargs):return self.source.from_latent(*args,**kwargs)

    def features(self,batch,pub,anchor,out):
        zone=batch['zone'];base=self.source.base
        world=torch.cat([batch[k].float() for k in ['w0_local','w1_local','w1_global','tail_context']],-1)
        reduced=(world-self.pca_center)@self.pca_components.T
        reduced=reduced*batch['world_available'][:,None]
        anchor=(anchor-base.da_center[zone])/base.da_scale[zone]
        return torch.cat([batch['numerical'].float(),pub[:,None],(pub*anchor)[:,None],batch['world_available'][:,None],
            reduced,batch['expert_prior'].float(),batch['tail_probability'].float()[:,None],out['world_gate'].mean(-1,keepdim=True)],-1)

    def delta(self,features):
        x=np.ascontiguousarray(features.detach().cpu().numpy(),dtype=np.float32)
        delta=np.zeros((len(x),14),np.float32)
        for tree in self.trees:delta+=self.learning_rate*tree.predict(x).astype(np.float32)
        return torch.as_tensor(delta,device=features.device).clamp(-1,1)

    def decode(self,latent,raw,zone,pub,anchor):
        base=self.source.base
        da=monotone_quantiles(raw[:,:7])*base.da_scale[zone,None]+base.da_center[zone,None]
        dart=monotone_quantiles(raw[:,7:])*base.dart_scale[zone,None]+base.dart_center[zone,None]
        return self.from_latent(latent,zone,pub,anchor,da_quantiles=da,dart_quantiles=dart)

    def forward(self,batch,pub,anchor):
        source=self.source(batch,pub,anchor);z=source['latent'];base=self.source.base
        raw=torch.cat([base.da_head(z).float(),base.dart_head(z).float()],-1)
        delta=self.delta(self.features(batch,pub,anchor,source)) if self.trees and self.ablation!='disable_trees' else torch.zeros_like(raw)
        out=self.decode(z,raw+delta,batch['zone'],pub,anchor)
        for key in ['world_gate','world_residual','expert_attention','router_entropy','world_available','base_latent']:
            out[key]=source[key]
        out['applied_residual']=delta;return out
