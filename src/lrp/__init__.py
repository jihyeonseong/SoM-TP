import torch.nn.functional as F
from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn 
from torch.autograd import Variable
from .modules import *


LookUpTable = {
    "Input" : InputLrp,
    "Linear" : LinearLrp,
    "ReLU" : ReluLrp,
    "Conv2d" : Conv2dLrp,
    "MaxPool2d": MaxPoolLrp,  # treat Max pool as Avg Pooling
    "AvgPool2d" : AvgPoolLrp,
    "AdaptiveAvgPool2d" : AdaptiveAvgPoolLrp,
    "Flatten":FlattenLrp,
    "BatchNorm2d" : BatchNorm2dLrp,
    "SoftDTW" : SoftDTWLrp,
    "softdtw": SoftDTWLrp,
    "protos" : SoftDTWLrp,
    "encoding" : SoftDTWLrp,
    "switch" : SoftDTWLrp,
    "Dropout" : DropoutLrp,
}

class LRP():
    def __init__(self, args, layers, rule_descriptions, device,  mean=None, std=None):
        super().__init__()
        self.device = device
        self.args = args
        self.rule_description = rule_descriptions
        self.original_layers = layers
        self.mean = mean 
        self.std = std
        self.kernel_sizes=[9, 5, 3]
        self.routings = 3
        self.lrp_modules = self.construct_lrp_modules(args, self.original_layers, rule_descriptions, device)
    
        assert len(layers) == len(rule_descriptions)

        
    def _gtp(self, Ai, n):
        if self.args.gtp == 'MAX' or self.args.pool_op == 'MAX':
            out1 = torch.max(Ai, dim=2)[0]
        else:
            out1 = torch.mean(Ai, dim=2)
        if self.args.model=='SoMTP':
            return out1.unsqueeze(2).repeat(1,1,n)
        else:
            return out1
    
    def _stp(self, Ai, n):
        segment_sizes = [int(Ai.shape[2]/n)] * n
        segment_sizes[-1] += Ai.shape[2] - sum(segment_sizes)

        hs = torch.split(Ai, segment_sizes, dim=2)
        if self.args.stp == 'MAX' or self.args.pool_op == 'MAX':
            hs = [h_.max(dim=2)[0].unsqueeze(dim=2) for h_ in hs]
        else:
            hs = [h_.mean(dim=2, keepdim=True) for h_ in hs]
        out2 = torch.cat(hs, dim=2)
        return out2
    
    def _dtp(self, Ai, A):
        if self.args.dtp == 'MAX' or self.args.pool_op == 'MAX':
            hs = Ai.unsqueeze(dim=2) * A.unsqueeze(dim=1)
            out3 = hs.max(dim=3)[0]
        else:
            A = A.clone()
            A /= A.sum(dim=2, keepdim=True)
            out3 = torch.bmm(Ai, A.transpose(1, 2))
        return out3

    def forward(self, a, y=None, class_specific=True):
        # store activations 
        activations = [torch.ones_like(a)] 
        cnt = 0
        for i, layer in enumerate(self.original_layers):
            try:
                if type(layer) is list:
                    a = a.squeeze(2)
                    A = layer[0].align(layer[1].repeat(a.shape[0], 1, 1), a)
                    if self.args.model == 'SoMTP':
                        encoding = layer[2]
                        switch = layer[3]
                        n = layer[1].size(1)
                        raw_attn = switch.repeat(a.shape[0], 1, 1)
                        
                        ### GTP ###
                        out1 = self._gtp(a, n)

                        ### STP ###
                        out2 = self._stp(a, n)

                        ### DTP ###
                        out3 = self._dtp(a, A)
                        
                        concat_out = torch.cat([out1, out2, out3], dim=-1)
                        
                        encode_attn = concat_out * raw_attn
                        attn = F.softmax(encoding(encode_attn.unsqueeze(2)), dim=-1).squeeze(1)
                        
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
                            ind = torch.cat([torch.mean(attn[:, :, :n], dim=2), 
                                         torch.mean(attn[:, :, n: n*2], dim=2), 
                                         torch.mean(attn[:, :, n*2:], dim=2)], dim=1)
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
                        a = tmp

                    else:
                        if self.args.pool=='DTP':
                            A = layer[0].align(layer[1].repeat(a.shape[0], 1, 1), a)
                            a = self._dtp(a, A)
                        elif self.args.pool=='STP':
                            n = layer[1].size(1)
                            a = self._stp(a, n)
                        else:
                            a = self._gtp(a, 0)
                elif i==0 or i==3 or i==6:
                    a = a.squeeze(2)
                    a = F.pad(a, (int(self.kernel_sizes[cnt]/2), int(self.kernel_sizes[cnt]/2)), "constant", 0)
                    a = a.unsqueeze(2)
                    a = layer(a)
                    cnt =cnt+ 1
                else:
                    a = layer(a)
            except Exception as e:
                print("Error:", layer)
                print("Error:", e)
                exit()
            activations.append(a)
        
        activations = activations[::-1]
        activations = [a.data.requires_grad_(True) for a in activations]
        
        #compute LRP
        prediction_outcome = activations.pop(0)
        score = torch.softmax(prediction_outcome, dim=-1)
        if class_specific:
            if y is None:
                class_index = score.argmax(axis=-1)
            else:
                class_index = y
            class_score = torch.FloatTensor(a.size(0), score.size()[-1]).zero_().to("cuda")
            class_score[:,class_index] = score[:,class_index]
        else:
            class_score = score
        modules = []
        class_score = class_score.data.fill_(1.)
        relevances = [class_score] 
        cnt = 2
        for i, (Ai, module) in enumerate(zip(activations, self.lrp_modules)):
            Rj = relevances[-1]
            if i == 9 or i==12 or i==15:             
                Ai = Ai.squeeze(2)
                Ai = F.pad(Ai, (int(self.kernel_sizes[cnt]/2), int(self.kernel_sizes[cnt]/2)), "constant", 0)
                Ai = Ai.unsqueeze(2)
                cnt = cnt-1
            Ri = module.forward(Rj, Ai)
            relevances.append(Ri)
        output = {
            "R" : relevances[-1],
            "all_relevnaces" : relevances,
            "activations" : activations,
            "prediction_outcome": prediction_outcome
        }
        return output 

    def construct_lrp_modules(self, args, original_layers, rule_descriptions, device):
        used_names = [] 
        modules = [] 

        for i, layer in enumerate(original_layers):
            rule = rule_descriptions[i]
            for k in rule:
                if k not in ['epsilon', 'gamma', "z_plus"]:
                    raise ValueError(f"Invalid LRP rule {k}")
            if type(layer) is list:
                name = layer[0].__class__.__name__
                assert name in LookUpTable, f"{name} is not in the LookupTable "
                if args.model == 'ConvPool':
                    lrp_module = LookUpTable[name](args, layer, rule)
                    lrp_module.softdtw.to(device)
                    lrp_module.protos.to(device)
                    modules.append(lrp_module)
                else:
                    lrp_module = LookUpTable[name](args, layer, rule)
                    lrp_module.softdtw.to(device)
                    lrp_module.protos.to(device)
                    lrp_module.switch.to(device)
                    modules.append(lrp_module)
            else:
                name  = layer.__class__.__name__
                assert name in LookUpTable, f"{name} is not in the LookupTable "
                lrp_module = LookUpTable[name](layer, rule)
                lrp_module.layer.to(device)
                modules.append(lrp_module)
            used_names.append(name)
        
        self.kind_warning(used_names)
        return modules[::-1]

    def kind_warning(self, used_names):
        if "ReLU" not in used_names:
            print(f'[Kind Warning] : ReLU is not in the layers. You should manually add activations.' )
            print(f'[Kind Warning] : Are you sure your model structure excludes ReLU : <{used_names}>?')
    
    

import numpy as np 
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt 
def process_lrp_before_imshow(R):
    power = 1.0
    b = 10*((np.abs(R)**power).mean()**(1.0/power))

    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:,0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    return (R, {"cmap":my_cmap, "vmin":-b, "vmax":b, "interpolation":'nearest'} )