import torch
from torch import nn

class WorldGate(nn.Module):
    def __init__(self,width=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2*width+24+3,width),nn.GELU(),nn.Linear(width,width))
        nn.init.constant_(self.net[-1].bias,-2.)

    def forward(self,base,world,context,available,entropy,confidence):
        x=torch.cat([base,world,context,available[:,None],entropy[:,None],confidence[:,None]],-1)
        return self.net(x).float().sigmoid()*available[:,None]

