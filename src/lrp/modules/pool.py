import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from .utils import construct_incr, clone_layer
import sys
sys.path.append('.../utils')
from utils.softdtw_cuda import SoftDTW

class AvgPoolLrp(nn.Module):
    def __init__(self, layer, rule):
        super().__init__()

        rule = {k: v for k,v in rule.items() if k=="epsilon"}  # only epsilont rule is possible
        self.layer = clone_layer(layer)
        self.incr = construct_incr(**rule)

    def forward(self, Rj, Ai):
        Ai = torch.autograd.Variable(Ai, requires_grad=True)
        Ai.retain_grad()
        Z = self.layer.forward(Ai)
        Z = self.incr(Z)
        S = (Rj / Z).data 
        (Z * S).sum().backward()
        Ci = Ai.grad 

        return  (Ai * Ci).data


class MaxPoolLrp(nn.Module):
    def __init__(self, layer, rule):
        super().__init__()

        rule = {k: v for k,v in rule.items() if k=="epsilon"}  # only epsilont rule is possible
        self.layer = torch.nn.AvgPool2d(kernel_size=layer.kernel_size)
        self.incr = construct_incr(**rule)

    def forward(self, Rj, Ai):
        Ai = torch.autograd.Variable(Ai, requires_grad=True)
        Ai.retain_grad()
        Z = self.layer.forward(Ai)
        Z = self.incr(Z)
        S = (Rj / Z).data 
        (Z * S).sum().backward()
        Ci = Ai.grad 

        return  (Ai * Ci).data
    

class SoftDTWLrp(nn.Module):
    def __init__(self, args, layer, rule):
        super().__init__()

        rule = {k: v for k,v in rule.items() if k=="epsilon"}  # only epsilont rule is possible
        self.softdtw = clone_layer(layer[0])
        self.protos = clone_layer(layer[1])
        if args.model == 'SoMTP':
            self.encoding = clone_layer(layer[2])
            self.switch = clone_layer(layer[3])
        self.incr = construct_incr(**rule)
        self.args = args
        
    def _gtp(self, Ai):
        n = self.protos.size(1)
        if self.args.gtp == 'MAX':
            out1 = torch.max(Ai, dim=2)[0]
        else:
            out1 = torch.mean(Ai, dim=2)
        if self.args.model=='SoMTP':
            return out1.unsqueeze(2).repeat(1,1,n)
        else:
            return out1
    
    def _stp(self, Ai):
        n = self.protos.size(1)
        segment_sizes = [int(Ai.shape[2]/n)] * n
        segment_sizes[-1] += Ai.shape[2] - sum(segment_sizes)

        hs = torch.split(Ai, segment_sizes, dim=2)
        if self.args.stp == 'MAX':
            hs = [h_.max(dim=2)[0].unsqueeze(dim=2) for h_ in hs]
        else:
            hs = [h_.mean(dim=2, keepdim=True) for h_ in hs]
        out2 = torch.cat(hs, dim=2)
        return out2
    
    def _dtp(self, Ai, A):
        if self.args.dtp == 'MAX':
            hs = Ai.unsqueeze(dim=2) * A.unsqueeze(dim=1)
            out3 = hs.max(dim=3)[0]
        else:
            A = A.clone()
            A /= A.sum(dim=2, keepdim=True)
            out3 = torch.bmm(Ai, A.transpose(1, 2))
        return out3

    def forward(self, Rj, Ai):
        Ai = Ai.squeeze(2)
        Ai = torch.autograd.Variable(Ai, requires_grad=True)
        Ai.retain_grad()
        
        if self.args.model == 'SoMTP':
            A = self.softdtw.align(self.protos.repeat(Ai.shape[0], 1, 1), Ai)
            n = self.protos.size(1)
            
            ### GTP ###
            out1 = self._gtp(Ai)
            
            ### STP ###
            out2 = self._stp(Ai)
            
            ### DTP ###
            out3 = self._dtp(Ai, A)

            concat_out = torch.cat([out1, out2, out3], dim=-1)
            
            raw_attn = self.switch.repeat(Ai.shape[0], 1, 1)
            encode_attn = concat_out * raw_attn
            attn = F.softmax(self.encoding(encode_attn.unsqueeze(2)), dim=-1).squeeze(1)
            
            if self.args.pool_op =='MAX':
                ind = torch.mean(torch.max(attn, dim=2)[1].squeeze(1).float())
                if ind.item() < n+1:
                    tmp = out1
                    op = 0
                elif ind.item() >= n+1 and ind.item()<= n*2+1:
                    tmp = out2
                    op = 1
                else:
                    tmp = out3
                    op=2
            else:
                ind = torch.cat([torch.mean(attn[:, :, :self.protos_num], dim=2), 
                             torch.mean(attn[:, :, self.protos_num:self.protos_num*2], dim=2), 
                             torch.mean(attn[:, :, self.protos_num*2:], dim=2)], dim=1)
                ind = torch.mean(torch.max(ind, dim=1)[1].float())

                if ind.item() < 0.6 :
                    tmp = out1
                    op = 0
                elif 0.6 <= ind.item() and ind.item() <1.6:
                    tmp = out2
                    op = 1
                else:
                    tmp = out3
                    op=2
            Z = tmp
        
        else:
            if self.args.pool=='DTP':
                A = self.softdtw.align(self.protos.repeat(Ai.shape[0], 1, 1), Ai)
                Z = self._dtp(Ai, A)
            elif self.args.pool=='STP':
                Z = self._stp(Ai)
            else:
                Z = self._gtp(Ai)

        Z = self.incr(Z)
        S = (Rj / Z).data 
        (Z * S).sum().backward()
        Ci = Ai.grad 

        return  (Ai.unsqueeze(2) * Ci.unsqueeze(2)).data

class AdaptiveAvgPoolLrp(nn.Module):
    def __init__(self, layer, rule):
        super().__init__()

        rule = {k: v for k,v in rule.items() if k=="epsilon"}  # only epsilont rule is possible
        self.layer = clone_layer(layer)
        self.incr = construct_incr(**rule)

    def forward(self, Rj, Ai):
        
        Ai = torch.autograd.Variable(Ai, requires_grad=True)
        Ai.retain_grad()
        Z = self.layer.forward(Ai)
        Z = self.incr(Z)
        S = (Rj / Z).data 
        (Z * S).sum().backward()
        Ci = Ai.grad 

        return  (Ai * Ci).data
