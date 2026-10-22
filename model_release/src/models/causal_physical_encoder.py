"""A queue-independent, residual probabilistic observer for the R2 controller.

The temporal encoder sees only completed quarter-hour PV observations. Spatial
attention exchanges those same historical signals across the four sites. The
zero-initialized decoder preserves the registered observer at initialization.
"""
import torch
from torch import nn
import torch.nn.functional as F


class CausalTemporalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)
        self.mix = nn.Conv1d(width, width, 1)

    def forward(self, x):
        return x + .25 * self.mix(F.gelu(self.conv(F.pad(x, (self.padding, 0)))))


class CausalPhysicalEncoder(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.input = nn.Conv1d(5, width, 1)
        self.temporal = nn.ModuleList([CausalTemporalBlock(width, d) for d in (1, 2, 4)])
        self.site_embedding = nn.Embedding(4, width)
        self.spatial = nn.MultiheadAttention(width, 4, dropout=0, batch_first=True)
        self.context_norm = nn.LayerNorm(width)
        self.decoder = nn.Sequential(nn.Linear(width * 3 + 12, 64), nn.GELU(),
                                     nn.LayerNorm(64), nn.Linear(64, 32), nn.GELU(),
                                     nn.Linear(32, 2))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def encode_history(self, history):
        batch = history.size(0)
        x = self.input(history.reshape(batch * 4, 16, 5).transpose(1, 2))
        for block in self.temporal:
            x = block(x)
        last = x[:, :, -1].reshape(batch, 4, -1)
        recent = x[:, :, -4:].mean(-1).reshape(batch, 4, -1)
        tokens = last + self.site_embedding.weight[None]
        spatial, _ = self.spatial(tokens, tokens, tokens, need_weights=False)
        context = self.context_norm(tokens + spatial)
        return torch.cat([last, recent, context], -1)

    def forward(self, history, base_quantiles, upper, clock):
        # history: [batch, four sites, past 16 intervals, five features].
        encoded = self.encode_history(history)
        scale = upper.clamp_min(.1)
        features = torch.cat([encoded, base_quantiles / scale[:, :, None],
                              upper[:, :, None], clock[:, None].expand(-1, 4, -1)], -1)
        raw = self.decoder(features)
        # Positive affine scale preserves quantile ordering. No task/queue state
        # can enter this physical forecast branch.
        location = .15 * scale * torch.tanh(raw[:, :, 0])
        spread = torch.exp(1.5 * torch.tanh(raw[:, :, 1]))
        center = base_quantiles[:, :, 3:4]
        quantiles = (center + location[:, :, None] +
                     spread[:, :, None] * (base_quantiles - center)).clamp(0, 1.5)
        return {'quantiles': quantiles, 'location': location, 'spread': spread}


class HorizonPhysicalEncoder(CausalPhysicalEncoder):
    """Lead-conditioned, asymmetric ordered PV quantiles over all 96 slots.

    The history tower is shared across forecast leads; the decoder receives only
    issue-legal priors, deterministic calendar variables, and past observations.
    Workload features never enter this forecast branch.
    """
    def __init__(self, width=32):
        super().__init__(width)
        del self.decoder
        self.lead_embedding = nn.Embedding(24, 8)
        self.horizon_decoder = nn.Sequential(nn.Linear(width*3+9+8,64),nn.GELU(),
                                            nn.LayerNorm(64),nn.Linear(64,32),nn.GELU(),nn.Linear(32,7))
        nn.init.zeros_(self.horizon_decoder[-1].weight)
        nn.init.zeros_(self.horizon_decoder[-1].bias)
        self.register_buffer('normal_scores',torch.tensor([-1.644854,-1.281552,-.674490,0.,.674490,1.281552,1.644854]))

    def forward(self, history, future):
        encoded = self.encode_history(history)
        lead = self.lead_embedding(torch.arange(96,device=future.device)//4)
        inputs=torch.cat([encoded[:,:,None].expand(-1,-1,96,-1),future,
                          lead[None,None].expand(history.size(0),4,-1,-1)],-1)
        raw=self.horizon_decoder(inputs)
        center=future[:,:,:,0]+.35*future[:,:,:,1].clamp_min(.02)*torch.tanh(raw[:,:,:,0])
        native_spread=.03+.18*future[:,:,:,0]
        gaps=(self.normal_scores[1:]-self.normal_scores[:-1])*native_spread[:,:,:,None]
        gaps=gaps*torch.exp(1.8*torch.tanh(raw[:,:,:,1:]))
        lower=torch.flip(torch.cumsum(torch.flip(gaps[:,:,:,:3],[-1]),-1),[-1])
        upper=torch.cumsum(gaps[:,:,:,3:],-1)
        q=torch.cat([center[:,:,:,None]-lower,center[:,:,:,None],center[:,:,:,None]+upper],-1).clamp(0,1.5)
        return {'quantiles':q,'residual':raw}
