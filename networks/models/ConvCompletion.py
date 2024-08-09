import numpy as np
import torch
from torch import nn

class ConvCompletion(nn.Module):
    def __init__(self, init_size):
        super(ConvCompletion, self).__init__()
        self.init_size = init_size
        ### Completion sub-network
        mybias = False 
        chs = [init_size, init_size*1, init_size*1, init_size*1]
        self.a_conv1 = nn.Sequential(nn.Conv3d(chs[1], chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU())
        self.a_conv2 = nn.Sequential(nn.Conv3d(chs[1], chs[1], 5, 1, padding=2, bias=mybias), nn.ReLU())
        self.a_conv3 = nn.Sequential(nn.Conv3d(chs[1], chs[1], 7, 1, padding=3, bias=mybias), nn.ReLU())
        self.a_conv4 = nn.Sequential(nn.Conv3d(chs[1]*2, chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU())
        self.a_conv5 = nn.Sequential(nn.Conv3d(chs[1]*2, chs[1], 5, 1, padding=2, bias=mybias), nn.ReLU())
        self.ch_conv1 = nn.Sequential(nn.Conv3d(chs[1]*5, chs[0], kernel_size=1, stride=1, bias=mybias), nn.ReLU())
        self.res_1 = nn.Sequential(nn.Conv3d(chs[0], chs[0], 3, 1, padding=1, bias=mybias), nn.ReLU())
        self.res_2 = nn.Sequential(nn.Conv3d(chs[0], chs[0], 5, 1, padding=2, bias=mybias), nn.ReLU())
   
    def forward(self, x_dense):
         ### Completion sub-network by dense convolution
        x1 = self.a_conv1(x_dense)
        x2 = self.a_conv2(x1)
        x3 = self.a_conv3(x1)
        t1 = torch.cat((x2, x3), 1)
        x4 = self.a_conv4(t1)
        x5 = self.a_conv5(t1)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        y0 = self.ch_conv1(x)
        y1 = self.res_1(x_dense)
        y2 = self.res_2(x_dense)
        x = x_dense + y0 + y1 + y2
        return x

        