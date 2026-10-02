"""APIC-compatible positive location/scale refinement for the R2 distribution."""
import torch
from torch import nn
from src.models.coupled_rollout import ArchitectedR2QPHead
from src.models.continuous_decision import ContinuousR2Head


class AffineMarketMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.price_scale=nn.Linear(128,1)
        nn.init.zeros_(self.price_scale.weight);nn.init.zeros_(self.price_scale.bias)

    def market_scale(self,adapted):
        # Keep small calibration changes around one resolvable under AMP.
        return torch.exp(.2*torch.tanh(self.price_scale(adapted).float().squeeze(-1)))

    def update_market_quantiles(self,state,market,adapted,shift):
        scale=1+(self.market_scale(adapted)-1)*state[:,:,8]
        return market*scale[:,:,None]+shift[:,:,None]*state[:,:,8:9]

    def load_parent(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        assert not unexpected and set(missing)=={'price_scale.weight','price_scale.bias'}


class AffineContinuousR2Head(AffineMarketMixin,ContinuousR2Head):
    pass


class AffineForecastR2Head(AffineMarketMixin,ArchitectedR2QPHead):
    """Same learned parameters; analytic continuation costs are evaluated externally."""
    pass
