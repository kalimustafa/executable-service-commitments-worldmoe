"""Exact formal 24-parameter affine calibrator."""
import torch
from torch import nn
class NativeCalibration(nn.Module):
    def __init__(self):
        super().__init__();self.da_shift=nn.Parameter(torch.zeros(4));self.da_width=nn.Parameter(torch.zeros(4))
        self.dart_shift=nn.Parameter(torch.zeros(2,4));self.dart_width=nn.Parameter(torch.zeros(2,4))
    def forward(self,da,dart,zone,gate,scales):
        z=zone.long();g=gate.long()
        a=da[:,3:4]+self.da_shift[z,None]*scales[z,0,None]+torch.exp(self.da_width[z,None].clamp(-2,2))*(da-da[:,3:4])
        d=dart[:,3:4]+self.dart_shift[g,z,None]*scales[z,1,None]+torch.exp(self.dart_width[g,z,None].clamp(-2,2))*(dart-dart[:,3:4])
        return a,d
