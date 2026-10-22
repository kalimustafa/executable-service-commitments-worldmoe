"""A trained continuation-value gate on the shared R2 state/forecast encoding."""
import math
import torch
from torch import nn
from src.models.coupled_rollout import ArchitectedR2QPHead

class ContinuationR2Head(ArchitectedR2QPHead):
    def __init__(self,base_eta=0.):
        super().__init__()
        self.register_buffer('eta_offset',torch.tensor(math.atanh(max(-.99,min(.99,float(base_eta))))))
        self.continuation=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,64),nn.GELU(),nn.Linear(64,16),nn.GELU(),nn.Linear(16,1))
        nn.init.zeros_(self.continuation[-1].weight);nn.init.zeros_(self.continuation[-1].bias)

    def load_forecast(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        assert not unexpected and all(k.startswith('continuation.') or k=='eta_offset' for k in missing)

    def continuation_adjustment(self,features):
        return torch.tanh(self.eta_offset+self.continuation(features).squeeze(-1))

    def decode_evidence(self,state,evidence,pue_ref):
        out=super().decode_evidence(state,evidence,pue_ref)
        out['continuation_eta']=self.continuation_adjustment(out['continuation_features'])
        return out
