# import torch.nn.functional as F 
import torch.nn as nn 
import torch 
import numpy  as np
from src.lrp import LRP

def construct_lrp(args, model, device):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225] 
    mean = torch.FloatTensor(mean).reshape(1,-1,1,1).to(device)
    std = torch.FloatTensor(std).reshape(1,-1,1,1).to(device)

    model.to(device)
    layers, rules = construct_lrp_layers_and_rules_for_CNN(args, model)
    
    lrp_model = LRP(args, layers, rules, device=device, mean=mean, std=std)
    return lrp_model
  

def construct_lrp_layers_and_rules_for_CNN(args, model):
    layers = [] 
    rules = [] 
    # Rule is z_plus
    for i, layer in enumerate(model.conv1.modules()): # Convolution 
        if i == 0:
            pass
        else:
            layers.append(layer)
            rules.append({"z_plus":True, "epsilon":1e-6})
    if args.model == 'ConvPool':
        layers.append([model.softdtw, model.protos])
    else:
        layers.append([model.softdtw, model.protos, model.encoding, model.switch])
    rules.append({"z_plus":True, "epsilon":1e-6})
    layers.append(nn.Flatten(start_dim=1))
    rules.append({"z_plus":True, "epsilon":1e-6})

    # Rule is epsilon 
    for i, layer in enumerate(model.decoder.modules()): # FCL # 3dense
        if i!=0:
            layers.append(layer)
            rules.append({"z_plus":True, "epsilon":1e-6})
    
    return layers, rules