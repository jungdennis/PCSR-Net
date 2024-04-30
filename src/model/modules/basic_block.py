import torch
import torch.nn as nn

from .function import *

class PA(nn.Module):
    def __init__(self, in_nc):
        super(PA, self).__init__()
        self.c1 = conv_block(in_nc, in_nc, kernel_size=1, norm_type=None, act_type='leakyrelu')

    def forward(self, x):
        x1 = self.c1(x)
        out = x.mul(x1) + x
        return out

class CA(nn.Module):
    def __init__(self, in_nc):
        super(CA, self).__init__()
        self.g = nn.AdaptiveAvgPool2d((1,1))
        self.c1 = conv_block(in_nc, 16, kernel_size=1, norm_type=None, act_type='prelu')
        self.c2 = conv_block(16, in_nc, kernel_size=1, norm_type=None, act_type='sigm')
        self.w1 = nn.Parameter(torch.FloatTensor([1]))

    def forward(self, x):
        g1 = self.g(x)
        b = torch.std(x.view(x.size(0), x.size(1), -1), 2)
        soc = b.view(b.size(0), b.size(1), 1, 1)
        x1 = self.w1 * g1 + (1 - self.w1) *soc
        atn = self.c2(self.c1(x1))
        out = x.mul(atn)
        return out
    
   
class ESA(nn.Module):
     def __init__(self, n_feats):
         super(ESA, self).__init__()
         f = n_feats // 4
         self.conv1 = conv_ESA(n_feats, f, kernel_size=1)
         self.conv_f = conv_ESA(f, f, kernel_size=1)
         self.conv_max = conv_ESA(f, f, kernel_size=3, padding=1)
         self.conv2 = conv_ESA(f, f, kernel_size=3, stride=2, padding=0)
         self.conv3 = conv_ESA(f, f, kernel_size=3, padding=1)
         self.conv3_ = conv_ESA(f, f, kernel_size=3, padding=1)
         self.conv4 = conv_ESA(f, n_feats,  kernel_size=1)
         self.sigmoid = nn.Sigmoid()
         self.relu = nn.ReLU(inplace=True)
  
     def forward(self, x):
         c1_ = self.conv1(x)
         c1 = self.conv2(c1_)
         v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
         v_range = self.relu(self.conv_max(v_max))
         c3 = self.relu(self.conv3(v_range))
         c3 = self.conv3_(c3)
         c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
         cf = self.conv_f(c1_)
         c4 = self.conv4(c3+cf)
         m = self.sigmoid(c4)
         
         return x * m
        
class CCA(nn.Module):
    def __init__(self, in_nc, reduction=16):
        super(CCA, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_nc, in_nc // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_nc // reduction, in_nc, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class Residual(nn.Module):
    def __init__(self, nf):
        super(Residual, self).__init__()
        self.conv1 = conv_block(nf, nf, kernel_size=3, norm_type='batch', act_type='leakyrelu')
        self.conv2 = conv_block(nf, nf, kernel_size=3, act_type='leakyrelu', dilation=2)
        self.pa = PA(nf)
        self.ca = CA(nf)

    def forward(self, x):
        x1 = self.ca(self.pa(self.conv2(self.conv1(x))))
        out = x + x1
        return out

class CSB(nn.Module):
    def __init__(self, in_nc, nc, stride=1, bias=True, kernel_size=3, pad_type='zero',
                 norm_type=None, act_type=None, mode='CNA'):
        super(CSB, self).__init__()
        self.nf = int(0.5*nc)
        self.conv0 = conv_block(in_nc, nc, kernel_size=1, norm_type='batch', act_type=act_type)
        self.conv1 = conv_block(self.nf, self.nf, kernel_size=kernel_size, norm_type=None, act_type=act_type)
        self.conv2 = conv_block(2*self.nf, self.nf, kernel_size=kernel_size, norm_type=None, act_type=act_type)
        self.conv3 = conv_block(3*self.nf, self.nf, kernel_size=kernel_size, norm_type=None, act_type=act_type)
        self.res1 = Residual(self.nf)
        self.res2 = Residual(self.nf)

    def forward(self, x):
        x0 = self.conv0(x)
        x11 = x0[:,0:self.nf,:,:]
        x12 = x0[:,self.nf:,:,:]
        x1 = self.conv1(x11)
        x2 = self.conv2(torch.cat((x1,x11),1))
        x3 = self.conv3(torch.cat((x2,x1,x11),1))
        x4 = self.res2(self.res1(x12))
        out = torch.cat((x3,x4),1)
        return out

class BB(nn.Module):
    def __init__(self ,nf):
        super(BB, self).__init__()
        self.b11=CSB(nf, nf, kernel_size=3, act_type='leakyrelu')
        self.b12=CSB(2*nf, nf, kernel_size=3, act_type='leakyrelu')
        self.b13=CSB(3*nf, nf, kernel_size=3, act_type='leakyrelu')
        self.c1 = conv_block(4*nf, 2*nf, kernel_size=1, act_type='leakyrelu')

    def forward(self, x):
        xa1 = self.b11(x)
        xa2 = self.b12(torch.cat((x,xa1),1))
        xa3 = self.b13(torch.cat((x,xa1,xa2),1))
        out = self.c1(torch.cat((x,xa1,xa2,xa3),1))
        return out