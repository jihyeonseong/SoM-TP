import torch 
import torch.nn as nn
from .utils import construct_incr, construct_rho, clone_layer, keep_conservative

class BatchNorm2dLrp(nn.Module):
    def __init__(self, layer, rule):
        super().__init__()
        self.layer = layer
        self.kernel_sizes = 9

    def forward(self, Rj, Ai):
        Rj = Rj
        return Rj