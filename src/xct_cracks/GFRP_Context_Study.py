"""Original local-contrast GFRP representation and BasicUNet backbone.

The pixel representation is retained from the archived research source.
Historical cache, fixture and training helpers are omitted from this portable
module; the separate command-line wrappers require Apple MPS.
"""
from pathlib import Path
from dataclasses import replace,asdict
import json,time
import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import BasicUNet

def local_channels(x):
    channels=[x]
    for size in (9,31):
        radius=size//2
        def mean(a):
            a=F.avg_pool2d(F.pad(a,(radius,radius,0,0),mode='reflect'),(1,size),stride=1)
            return F.avg_pool2d(F.pad(a,(0,0,radius,radius),mode='reflect'),(size,1),stride=1)
        mu=mean(x);variance=(mean(x*x)-mu*mu).clamp_min(0)
        z=(x-mu)/torch.sqrt(variance+.03**2)
        channels.append(z.clamp(-3,3)/3)
    return torch.cat(channels,dim=1)

class ContrastUNet(torch.nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.net=BasicUNet(spatial_dims=2,in_channels=3,out_channels=2,features=cfg.features,dropout=0.)
    def forward(self,x):return self.net(local_channels(x))

