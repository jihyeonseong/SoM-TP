import torch 
import torch.nn as nn
from .utils import construct_incr, construct_rho, clone_layer, keep_conservative

class Conv2dLrp(nn.Module):
    def __init__(self, layer, rule):
        super().__init__()
        self.layer = clone_layer(layer)
        self.rho = construct_rho(**rule)
        self.incr = construct_incr(**rule)

        self.layer.weight = self.rho(self.layer.weight)
        self.layer.bias = keep_conservative(self.layer.bias)
        self.mask = None  
        # ACtion -> Convolution -> [Activation]
        # Relevance   <- (weight) <- [Relevance]
        
    def forward(self, Rj, Ai):
        Ai = torch.autograd.Variable(Ai, requires_grad=True)
        Ai.retain_grad()
        Z = self.layer.forward(Ai)
        Z = self.incr(Z)
        S = (Rj / Z).data 
        (Z * S).sum().backward()
        Ci = Ai.grad
        Ri = (Ai * Ci).data
        Ri = Ri[:, :, :, int(self.layer.kernel_size[1]//2):-int(self.layer.kernel_size[1]//2)]
        
        Ri /= (Ri.sum() + 1e-7)
        relevance = Ri
        
        return  Ri
    
    def register_concept_mask(self, concept_ids):
        if concept_ids:
            self.mask = list(
                            set(range(self.layer.weight.size(1))) - set(concept_ids)
                        )

    def remove_concept_mask(self):
        self.mask = None

