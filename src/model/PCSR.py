from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.ops as ops
from torch.nn import Flatten

from .modules.function import *
from .modules.upsample import *
from .modules.basic_block import PA, CA, ESA, CCA
from .modules.model_convnext_v2 import convnext_small
import timm
from torchvision.models.densenet import densenet169

'''
<SR Model>
'''

class model_sr(nn.Module):
    def __init__(self, upscale, in_nc=3, nf=128):
        super(model_sr, self).__init__()
        self.nf=nf

        if upscale != 2 and upscale != 4 :
            raise NotImplementedError('Only x2 or x4 is available. Choose other upscale factor.')

        self.cn1= conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None, mode='CNA')
        self.cn2= conv_block(nf, 2*nf, kernel_size=5, norm_type=None, act_type=None, mode='CNA')

        self.b1=BB(nf)
        self.b2=BB(nf)
        self.b3=BB(nf)
        self.b4=BB(nf)
        self.b5=BB(nf)
        self.b6=BB(nf)
        self.b7=BB(nf)
        self.b8=BB(nf)

        self.c1=conv_block(2*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        self.c2=conv_block(2*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        self.c3=conv_block(2*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        self.c4=conv_block(2*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        self.c5=conv_block(2*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        self.c6=conv_block(2*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        self.c7=conv_block(2*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        self.c8=conv_block(3*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')

        self.grl= nn.Upsample(scale_factor=upscale, mode='bicubic')
        
        self.ups1= OurUpSample(nf, nf, kernel_size=3, act_type='leakyrelu', upscale_factor=upscale)

    def forward(self, x):
        x1 = self.cn2(self.cn1(x))
        x11 = x1[:,0:self.nf,:,:]
        x12 = x1[:,self.nf:,:,:]
        
        xa1 = self.b1(x11)
        xa11 = xa1[:,0:self.nf,:,:]
        xa12 = xa1[:,self.nf:,:,:]

        xb1 = self.b2(xa11)
        xb11 = xb1[:,0:self.nf,:,:]
        xb12 = xb1[:,self.nf:,:,:]

        xc1 = self.b3(xb11)
        xc11 = xc1[:,0:self.nf,:,:]
        xc12 = xc1[:,self.nf:,:,:]

        xd1 = self.b4(xc11)
        xd11 = xd1[:,0:self.nf,:,:]
        xd12 = xd1[:,self.nf:,:,:]

        xe1 = self.b5(xd11)
        xe11 = xe1[:,0:self.nf,:,:]
        xe12 = xe1[:,self.nf:,:,:]

        xf1 = self.b6(xe11)
        xf11 = xf1[:,0:self.nf,:,:]
        xf12 = xf1[:,self.nf:,:,:]

        xg1 = self.b7(xf11)
        xg11 = xg1[:,0:self.nf,:,:]
        xg12 = xg1[:,self.nf:,:,:]

        xh1 = self.b8(xg11)
        xh11 = xh1[:,0:self.nf,:,:]
        xh12 = xh1[:,self.nf:,:,:]

        # Layer Attention
        xr1 = self.c1(torch.cat((xa12,xb12),1))
        xr2 = self.c2(torch.cat((xr1,xc12),1))
        xr3 = self.c3(torch.cat((xr2,xd12),1))
        xr4 = self.c4(torch.cat((xr3,xe12),1))
        xr5 = self.c5(torch.cat((xr4,xf12),1))
        xr6 = self.c6(torch.cat((xr5,xg12),1))
        xr7 = self.c7(torch.cat((xr6,xh12),1))
        xr = self.c8(torch.cat((xh11,xr7,x12),1))

        u1 = self.ups1(xr)
        u2 = self.grl(x)
        
        out = u1+u2
        return out

class Residual(nn.Module):
    def __init__(self, nf):
        super(Residual, self).__init__()
        self.conv1 = conv_block(nf, nf, kernel_size=3, norm_type='batch', act_type='leakyrelu')
        self.conv2 = conv_block(nf, nf, kernel_size=3, act_type='leakyrelu', dilation=2)
        self.esa = ESA(nf)
        self.ca = CA(nf)

    def forward(self, x):
        x1 = self.ca(self.esa(self.conv2(self.conv1(x))))
        out = x + x1
        return out

class CSB(nn.Module):
    def __init__(self, in_nc, nc, stride=1, bias=True, kernel_size=3, pad_type='zero', \
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
    def __init__(self, nf):
        super(BB, self).__init__()
        self.b11 = CSB(nf, nf, kernel_size=3, act_type='leakyrelu')
        self.b12 = CSB(2 * nf, nf, kernel_size=3, act_type='leakyrelu')
        self.b13 = CSB(3 * nf, nf, kernel_size=3, act_type='leakyrelu')
        self.c1 = conv_block(4 * nf, 2 * nf, kernel_size=1, act_type='leakyrelu')

    def forward(self, x):
        xa1 = self.b11(x)
        xa2 = self.b12(torch.cat((x,xa1),1))
        xa3 = self.b13(torch.cat((x,xa1,xa2),1))
        out = self.c1(torch.cat((x,xa1,xa2,xa3),1))
        return out

'''
<Re-ID Model>
'''

class model_reid(nn.Module):
    def __init__(self, database, base, fold):
        super(model_reid, self).__init__()
        self.database = database
        self.base = base

        self.model_path = f"C:/Users/syjung/Desktop/PCSR-Net/src/model/ckpt_reid/FE_{database}_{base}_{fold}_set.pt"

        if self.database == "Reg":
            num_classes = 185
        elif self.database == "SYSU":
            num_classes = 222

        if self.base == "convnext":
            self.model_base = convnext_small(num_classes=num_classes)

            self.model_base.load_state_dict(torch.load(self.model_path).state_dict(), strict=False)

            self.model_base.head = nn.Identity()
        elif self.base == "densenet":
            self.model_base = densenet169(weights="DEFAULT")
            self.model_base.classifier = nn.Linear(in_features=1664, out_features=num_classes, bias=True)

            self.model_base.load_state_dict(torch.load(self.model_path).state_dict(), strict=False)

            self.model_base.classifier = nn.Identity()
            self.flatten = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1, end_dim=-1)
            )
        elif self.base == "inception":
            self.model_base = timm.create_model('inception_v4', pretrained=True)
            self.model_base.last_linear = nn.Linear(in_features=1536, out_features=num_classes, bias=True)

            self.model_base.load_state_dict(torch.load(self.model_path).state_dict(), strict=False)

            self.model_base.global_pool = nn.Identity()
            self.model_base.last_linear = nn.Identity()
            self.flatten = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1, end_dim=-1)
            )
        self.part_attention = part_attention(model=self.base, dim=-1)

    def forward(self, x):
        # Make Feature Map
        x_whole = self.model_base(x)

        # Partitioning & Flatten
        if self.base == "convnext":
            x_whole = self.model_base(x)

            x_head, x_upper, x_lower = x_whole[:,:,0:2,:], x_whole[:,:,2:5,:], x_whole[:,:,5:8,:]

            v_whole = self.model_base.norm(x_whole.mean([-2, -1])).unsqueeze(-1).unsqueeze(-1)
            v_head = self.model_base.norm(x_head.mean([-2, -1])).unsqueeze(-1).unsqueeze(-1)
            v_upper = self.model_base.norm(x_upper.mean([-2, -1])).unsqueeze(-1).unsqueeze(-1)
            v_lower = self.model_base.norm(x_lower.mean([-2, -1])).unsqueeze(-1).unsqueeze(-1)
        elif self.base == "densenet":
            x_whole = self.model_base.features(x)
            x_whole = F.relu(x_whole, inplace=True)

            x_head, x_upper, x_lower = x_whole[:, :, 0:2, :], x_whole[:, :, 2:5, :], x_whole[:, :, 5:8, :]

            v_whole = self.flatten(x_whole).unsqueeze(-1).unsqueeze(-1)
            v_head = self.flatten(x_head).unsqueeze(-1).unsqueeze(-1)
            v_upper = self.flatten(x_upper).unsqueeze(-1).unsqueeze(-1)
            v_lower = self.flatten(x_lower).unsqueeze(-1).unsqueeze(-1)
        elif self.base == "inception":
            x_whole = self.model_base(x)

            x_head, x_upper, x_lower = x_whole[:, :, 0:2, :], x_whole[:, :, 2:4, :], x_whole[:, :, 4:6, :]

            v_whole = self.flatten(x_whole).unsqueeze(-1).unsqueeze(-1)
            v_head = self.flatten(x_head).unsqueeze(-1).unsqueeze(-1)
            v_upper = self.flatten(x_upper).unsqueeze(-1).unsqueeze(-1)
            v_lower = self.flatten(x_lower).unsqueeze(-1).unsqueeze(-1)


        # Part Attention
        att_score = self.part_attention(v_whole, v_head, v_upper, v_lower)
        _h = att_score[:, 0].unsqueeze(-1)
        _u = att_score[:, 1].unsqueeze(-1)
        _l = att_score[:, 2].unsqueeze(-1)

        if self.base == "convnext":
            flatten_feature = self.model_base.norm(x_whole.mean([-2, -1]))
        elif self.base == "inception" or self.base == "densenet":
            flatten_feature = self.flatten(x_whole)
        return flatten_feature, _h, _u, _l

class part_attention(nn.Module):
    def __init__(self, model, dim=-1):
        super(part_attention, self).__init__()
        if model == "convnext":
            c = 768
        elif model == "densenet":
            c = 1664
        elif model == "inception":
            c = 1536
        self.dim = dim
        self.conv_whole = nn.Conv2d(in_channels=c, out_channels=64, kernel_size=1)
        self.conv_head = nn.Conv2d(in_channels=c, out_channels=64, kernel_size=1)
        self.conv_upper = nn.Conv2d(in_channels=c, out_channels=64, kernel_size=1)
        self.conv_lower = nn.Conv2d(in_channels=c, out_channels=64, kernel_size=1)

    def forward(self, whole, head, upper, lower):
        v_whole = self.conv_whole(whole).squeeze()
        v_head = self.conv_head(head).squeeze()
        v_upper = self.conv_upper(upper).squeeze()
        v_lower = self.conv_lower(lower).squeeze()

        if v_whole.ndim == 1:
            v_whole = v_whole.unsqueeze(0)
        if v_head.ndim == 1:
            v_head = v_head.unsqueeze(0)
        if v_upper.ndim == 1:
            v_upper = v_upper.unsqueeze(0)
        if v_lower.ndim == 1:
            v_lower = v_lower.unsqueeze(0)

        _w = torch.transpose(v_whole, 0, 1)

        _hs = torch.matmul(v_head, _w)
        _h = torch.diagonal(_hs, 0)
        _us = torch.matmul(v_upper, _w)
        _u = torch.diagonal(_us, 0)
        _ls = torch.matmul(v_lower, _w)
        _l = torch.diagonal(_ls, 0)

        concat = torch.stack([_h, _u, _l], dim=self.dim)

        return torch.softmax(concat, dim=self.dim)