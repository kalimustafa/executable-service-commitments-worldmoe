import torch
import torch.nn.functional as F
from vendor.r2_market import pinball

def price_losses(out,y,scales):
    loss=[pinball(out[key],y[:,j])/scales[:,j] for j,key in enumerate(['da','dart','rt'])]
    return loss,.5*loss[0]+loss[1]+.5*loss[2]

def training_loss(out,y,regime,scales,gated=False):
    _,price=price_losses(out,y,scales)
    total=(price+.05*F.cross_entropy(out['regime_logits'],regime,reduction='none')).mean()
    if gated:total=total+1e-4*out.get('regularized_gate',out['world_gate']).mean()+1e-5*out['applied_residual'].square().mean()
    return total
