import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

from utils.softdtw_cuda import SoftDTW
from model.deep_FeatureExtract import FCN, ResNet
   
class SoMTP(nn.Module):
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
    7. switch_pool: selection ensemble pooling
    """
    def __init__(self, input_size, time_length, classes, data_type, args):
        super(SoMTP, self).__init__()
        self.args = args
        self.input_size = input_size #channel size
        self.time_length = time_length #sequence length
        self.classes = classes # class number
        self.data_type = data_type # uni / mul
        
        self.pool_op = args.pool_op # AVG / MAX
        
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
        
        ### for selection ensemble ###
        self.switch = nn.Parameter(torch.ones(1, self.protos_num*3), requires_grad=True)
        self.encoding = nn.Conv2d(256, 1, 1)

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
        self.decoder = nn.Sequential(
            nn.Linear(256*self.protos_num, 512), 
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, classes),
        ) # CLS network: main decision network
        
        self.ensem_decoder = nn.Sequential(
            nn.Linear(256*(self.protos_num*3), 512), 
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, classes),
        ) # DPLN: subnetwork for selection ensemble learning

        self.relu = nn.ReLU()
        self.kl = nn.KLDivLoss(reduction="batchmean")
    
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
 
    ### Selection over Multiple Temporal Poolings by DPL Attention ###
    def switch_pool(self, h, op_):
        ### Get multiple pooling outputs ###
        out1 = self.gtpool(h, self.args.gtp).unsqueeze(2).repeat(1,1,self.protos_num)
        out2 = self.stpool(h, self.protos_num, self.args.stp)
        out3 = self.dtpool(h, self.args.dtp)
        
        op=0
        concat_out = torch.cat([out1, out2, out3], dim=-1)
        
        ### DPL Attention ###
        raw_attn = self.switch.repeat(h.shape[0], 1, 1)
        encode_attn = torch.matmul(concat_out.unsqueeze(3), raw_attn.unsqueeze(1))
        attn = F.softmax(self.encoding(encode_attn), dim=-1).squeeze(1)

        if op_ =='MAX':
            ind = torch.mean(torch.max(attn, dim=2)[1].squeeze(1).float())
            if ind.item() <self.protos_num+1:
                tmp = out1
                op = 0
            elif ind.item() >=self.protos_num+1 and ind.item()<=self.protos_num*2+1:
                tmp = out2
                op = 1
            else:
                tmp = out3
                op=2

        elif op_ == 'AVG':
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
                                
        out = tmp
        ensemble = torch.matmul(concat_out.unsqueeze(2), attn.unsqueeze(1))
        
        return ensemble, out, op, concat_out, attn
            
    ### softDTW cost function for optimize DTP ###
    def compute_aligncost(self, h):
        cost = self.softdtw(self.protos.repeat(h.shape[0], 1, 1), h.detach())
        return cost.mean() / h.shape[2]
    
    ### compute perspective loss ###
    def compute_perspectivecost(self, ensem, one):
        diverse = self.ensem_decoder(ensem.reshape(ensem.shape[0], -1))
        one = self.decoder(one.reshape(one.shape[0], -1))
        cost = self.kl(F.log_softmax(one), F.softmax(diverse))
        return diverse, cost
    
    ### compute L_attn ###
    def compute_attentioncost(self, concat, h):
        cost = torch.bmm(h.transpose(1,2),concat.squeeze(2))
        return cost.mean() / h.shape[2]
        

    def forward(self, x):
        x = F.relu(self.conv1(x)).squeeze(2)
        ensem, one, op, raw, attn = self.switch_pool(x, self.pool_op)

        out = one.reshape(one.shape[0], -1)            
        out = self.decoder(out)

        return x, out, ensem, one, op, attn