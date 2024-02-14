import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

from utils.softdtw_cuda import SoftDTW
from model.deep_FeatureExtract import FCN, ResNet
    
class ConvPool(nn.Module):
    """
    1. init: initialize Network (Conv - FC)
    2. Temporal Poolings:
        1) gtpool: GTP
        2) stpool: STP
        3) dtpool: DTP
    3. get_htensor: convolutional feature extractor
    4. init_protos: intialize self.protos for DTP
    5. compute_aligncost: softDTW cost function for optimizing DTP
    6. compute_gradcam: GradCAM and DTP's align matrix
    """
    def __init__(self, input_size, time_length, classes, data_type, args):
        super(ConvPool, self).__init__()
        self.input_size = input_size #channel size
        self.time_length = time_length #sequence length
        self.classes = classes # class number
        self.data_type = data_type # uni / mul
        
        self.pool = args.pool # GTP / STP / DTP
        self.pool_op = args.pool_op # AVG / MAX
        
        self.deep_extract= args.deep_extract # shallow / FCN / ResNet
        
        ### Prototype number follows class number (STP and DTP) ###
        if classes < 4:
            protos = 4
        elif classes > 10:
            protos = 10
        else:
            protos = classes
            
        self.protos_num = protos
        
        ### for DTP ###
        self.dtp_distance = args.cost_type
        self.protos = nn.Parameter(torch.zeros(256, self.protos_num), requires_grad=True)
        self.softdtw = SoftDTW(use_cuda=True, gamma=1.0, cost_type=args.cost_type, normalize=False)
        
        ### Convolutional Feature Extractor ###
        if args.deep_extract=='shallow':
            self.conv1 = nn.Conv1d(in_channels=input_size,
                                 out_channels=256,
                                 kernel_size=(1))
        elif args.deep_extract=='FCN':
            self.conv1 = FCN(classes, 0, input_size) 
        elif args.deep_extract=='ResNet':
            self.conv1 = ResNet(classes, 0, input_size)
        else:
            raise Exception("No model!")

        ### Decoder network ###
        if self.pool == 'GTP':
            self.decoder = nn.Sequential(
                nn.Linear(256, 512), 
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, classes),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(256*self.protos_num, 512), 
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, classes),
            )

        self.relu = nn.ReLU()
    
    ### GTP: global temporal pooling ###
    def gtpool(self, h, op):
        if op == 'AVG':
            return torch.mean(h, dim=2)
        if op == 'SUM':
            return torch.sum(h, dim=2)
        elif op == 'MAX':
            return torch.max(h, dim=2)[0]
    
    ### STP: static temporal pooling ###
    def stpool(self, h, n, op):
        segment_sizes = [int(h.shape[2]/n)] * n
        segment_sizes[-1] += h.shape[2] - sum(segment_sizes)
       
        hs = torch.split(h, segment_sizes, dim=2)
        if op == 'AVG':
            hs = [h_.mean(dim=2, keepdim=True) for h_ in hs]
        elif op == 'SUM':
            hs = [h_.sum(dim=2, keepdim=True) for h_ in hs]
        elif op == 'MAX':
            hs = [h_.max(dim=2)[0].unsqueeze(dim=2) for h_ in hs]
        hs = torch.cat(hs, dim=2)
        return hs
    
    ### DTP: dynamic temporal pooling ###
    def dtpool(self, h, op):
        h_origin = h
        A = self.softdtw.align(self.protos.repeat(h.shape[0], 1, 1), h)

        if op == 'AVG':
            A = A.clone()
            A /= A.sum(dim=2, keepdim=True)
            h = torch.bmm(h, A.transpose(1, 2))
        elif op == 'SUM':
            h = h.unsqueeze(dim=2) * A.unsqueeze(dim=1)
            h = h.sum(dim=3)
        elif op == 'MAX':
            h = h.unsqueeze(dim=2) * A.unsqueeze(dim=1)
            h = h.max(dim=3)[0]
            
        return h
    
    ### Conv feature extraction ###
    def get_htensor(self, x):
        h = F.relu(self.conv1(x))
        return h
    
    ### initialize prototype parameter for DTP ###
    def init_protos(self, data_loader):
        for itr, batch in enumerate(data_loader):
            data = batch['data'].cuda()
            h = self.get_htensor(data).squeeze(2)
            self.protos.data += self.stpool(h, self.protos_num, 'AVG').mean(dim=0)
            
        self.protos.data /= len(data_loader)
            
    ### softDTW cost function for optimize DTP ###
    def compute_aligncost(self, h):
        cost = self.softdtw(self.protos.repeat(h.shape[0], 1, 1), h.detach())
        return cost.mean() / h.shape[2]
    
    ### GradCAM ###
    def compute_gradcam(self, x, labels):
        def hook_func(grad):
            self.h_grad = grad

        h, logits, _ = self.forward(x)
        h.register_hook(hook_func)

        self.zero_grad()
        scores = torch.gather(logits, 1, labels.unsqueeze(dim=1))
        scores.mean().backward()
        gradcam = (h * self.h_grad).sum(dim=1, keepdim=True)

        # min-max normalization
        gradcam_min = torch.min(gradcam, dim=2, keepdim=True)[0]
        gradcam_max = torch.max(gradcam, dim=2, keepdim=True)[0]
        gradcam = (gradcam - gradcam_min) / (gradcam_max - gradcam_min) 

        A = self.softdtw.align(self.protos.repeat(h.shape[0], 1, 1), h)
        
        return gradcam, A
        

    def forward(self, x):
        x = F.relu(self.conv1(x)).squeeze(2)

        if self.pool == 'GTP':
            out_p = self.gtpool(x, self.pool_op)
        elif self.pool == 'STP':
            out_p = self.stpool(x, self.protos_num, self.pool_op)
        else:
            out_p = self.dtpool(x, self.pool_op)

        out = out_p.reshape(out_p.shape[0], -1)
        out = self.decoder(out)

        return x, out, out_p
                    
                    
