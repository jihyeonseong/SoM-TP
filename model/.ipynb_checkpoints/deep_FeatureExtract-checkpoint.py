import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init

class FCN(nn.Module):
    def __init__(self, num_classes, num_segments, input_size, hidden_sizes=[128, 256, 256], kernel_sizes=[9, 5, 3], cost_type='cosine', pooling_op='avg', gamma=1.0):
        super(FCN, self).__init__()

        self.num_classes = num_classes
        self.num_segments = num_segments
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.kernel_sizes = kernel_sizes
        self.pooling_op = pooling_op
        
        self._build_model(num_classes, num_segments, input_size, hidden_sizes, kernel_sizes)
        self._init_model()

    def _build_model(self, num_classes, num_segments, input_size, hidden_sizes, kernel_sizes):

        self.conv1 = nn.Conv2d(in_channels=input_size,
                                 out_channels=hidden_sizes[0],
                                 kernel_size=(1,kernel_sizes[0]))
        self.norm1 = nn.BatchNorm2d(num_features=hidden_sizes[0])
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=hidden_sizes[0],
                                 out_channels=hidden_sizes[1],
                                 kernel_size=(1,kernel_sizes[1]))
        self.norm2 = nn.BatchNorm2d(num_features=hidden_sizes[1])
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=hidden_sizes[1],
                                 out_channels=hidden_sizes[2],
                                 kernel_size=(1,kernel_sizes[2]))
        self.norm3 = nn.BatchNorm2d(num_features=hidden_sizes[2])
        self.relu3 = nn.ReLU()
     
    def _init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)

            if isinstance(m, nn.Conv1d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
    
    def get_htensor(self, x):
        h = x
        
        h = F.pad(h, (int(self.kernel_sizes[0]/2), int(self.kernel_sizes[0]/2)), "constant", 0)
        h = F.relu(self.norm1(self.conv1(h)))
        
        h = F.pad(h, (int(self.kernel_sizes[1]/2), int(self.kernel_sizes[1]/2)), "constant", 0)
        h = F.relu(self.norm2(self.conv2(h)))
        
        h = F.pad(h, (int(self.kernel_sizes[2]/2), int(self.kernel_sizes[2]/2)), "constant", 0)
        h = F.relu(self.norm3(self.conv3(h)))
        return h

    def forward(self, x):
        h = self.get_htensor(x.unsqueeze(2))
        
        return h
    
    
    
class ResidualBlock(nn.Module):
    """
    Args:
        hidden_size: input dimension (Channel) of 1d convolution
        output_size: output dimension (Channel) of 1d convolution
        kernel_size: kernel size
    """
    def __init__(self, input_size, output_size, kernel_sizes=[9, 5, 3]):
        super(ResidualBlock, self).__init__()
        self.kernel_sizes = kernel_sizes

        self.conv1 = nn.Conv2d(in_channels=input_size,
                                out_channels=output_size,
                                kernel_size=(1, kernel_sizes[0]))
        self.conv2 = nn.Conv2d(in_channels=output_size,
                                out_channels=output_size,
                                kernel_size=(1, kernel_sizes[1]))
        self.conv3 = nn.Conv2d(in_channels=output_size,
                                out_channels=output_size,
                                kernel_size=(1, kernel_sizes[2]))
        self.conv_skip = nn.Conv2d(in_channels=input_size,
                                out_channels=output_size,
                                kernel_size=1)
        
        self.norm1 = nn.BatchNorm2d(num_features=output_size)
        self.norm2 = nn.BatchNorm2d(num_features=output_size)
        self.norm3 = nn.BatchNorm2d(num_features=output_size)
        self.norm_skip = nn.BatchNorm2d(num_features=output_size)

    def forward(self, x):
        
        h = x
        h = F.pad(h, (int(self.kernel_sizes[0]/2), int(self.kernel_sizes[0]/2)), "constant", 0)
        h = F.relu(self.norm1(self.conv1(h)))
       
        h = F.pad(h, (int(self.kernel_sizes[1]/2), int(self.kernel_sizes[1]/2)), "constant", 0)
        h = F.relu(self.norm2(self.conv2(h)))
        
        h = F.pad(h, (int(self.kernel_sizes[2]/2), int(self.kernel_sizes[2]/2)), "constant", 0)
        h = self.norm3(self.conv3(h))
        
        s = self.norm_skip(self.conv_skip(x))
        h += s
        h = F.relu(h)

        return h

class ResNet(nn.Module):
    def __init__(self, num_classes, num_segments, input_size, hidden_sizes=[64, 128, 256], kernel_sizes=[9, 5, 3], cost_type='cosine', pooling_op='avg', gamma=1.0):
        super(ResNet, self).__init__()

        self.num_classes = num_classes
        self.num_segments = num_segments 
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.kernel_sizes = kernel_sizes
        self.pooling_op = pooling_op
        
        self._build_model(num_classes, num_segments, input_size, hidden_sizes, kernel_sizes)
        self._init_model()

    def _build_model(self, num_classes, num_segments, input_size, hidden_sizes, kernel_sizes):
            
        self.resblock1 = ResidualBlock(input_size=input_size,
                                        output_size=hidden_sizes[0],
                                        kernel_sizes=kernel_sizes)
        
        self.resblock2 = ResidualBlock(input_size=hidden_sizes[0],
                                        output_size=hidden_sizes[1],
                                        kernel_sizes=kernel_sizes)
        
        self.resblock3 = ResidualBlock(input_size=hidden_sizes[1],
                                        output_size=hidden_sizes[2],
                                        kernel_sizes=kernel_sizes)
        
    def _init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)

            if isinstance(m, nn.Conv1d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
    
    def get_htensor(self, x):
        h = self.resblock1(x)
        h = self.resblock2(h)
        h = self.resblock3(h)
        return h
    
    def forward(self, x):
        h = self.get_htensor(x.unsqueeze(2)) 
                
        return h
