import torch
from torch import nn
from vendor.r2_market import monotone_quantiles

class R2CausalErrorMemory(nn.Module):
    def __init__(self,source,memory_features=14):
        super().__init__();self.source=source;source.eval().requires_grad_(False)
        self.state_encoder=nn.Sequential(nn.Linear(memory_features,32),nn.GELU(),nn.LayerNorm(32))
        self.context=nn.Sequential(nn.Linear(128+32+2,64),nn.GELU(),nn.LayerNorm(64))
        self.output=nn.Linear(64,14,bias=False);self.linear_state=nn.Linear(memory_features,14,bias=False)
        self.gate=nn.Sequential(nn.Linear(128+32+2,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.linear_state.weight);nn.init.constant_(self.gate[-1].bias,-2.)
        self.ablation='normal'

    def train(self,mode=True):
        super().train(mode);self.source.eval();return self

    def from_latent(self,*args,**kwargs):return self.source.from_latent(*args,**kwargs)

    def forward(self,batch,publication,anchor):
        source=self.source(batch,publication,anchor);z=source['latent'];base=self.source.base
        row=torch.arange(len(z),device=z.device);state=batch['error_memory'][row,publication.long()]
        available=batch['memory_available'][row,publication.long()]
        encoded=self.state_encoder(state)
        inputs=torch.cat([z.float(),encoded,publication[:,None],available[:,None]],-1)
        gate=self.gate(inputs).float().sigmoid()*available[:,None]
        if self.ablation=='disable_memory':gate=torch.zeros_like(gate)
        delta=.5*torch.tanh(gate*(self.output(self.context(inputs)).float()+self.linear_state(state).float()))
        raw=torch.cat([base.da_head(z).float(),base.dart_head(z).float()],-1)+delta
        zone=batch['zone']
        da=monotone_quantiles(raw[:,:7])*base.da_scale[zone,None]+base.da_center[zone,None]
        dart=monotone_quantiles(raw[:,7:])*base.dart_scale[zone,None]+base.dart_center[zone,None]
        out=self.from_latent(z,zone,publication,anchor,da_quantiles=da,dart_quantiles=dart)
        for key in ['world_gate','world_residual','expert_attention','router_entropy','world_available','base_latent']:out[key]=source[key]
        out.update(applied_residual=delta,regularized_gate=gate,memory_gate=gate)
        return out
