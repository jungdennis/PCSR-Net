import torch
import torch.nn as nn

from .function import *

def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                        pad_type='zero', norm_type=None, act_type='relu'):
    '''
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    '''
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=None, act_type=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)

class OurUpSample(nn.Module):
    def __init__(self, in_nc, nc, kernel_size=3, stride=1, bias=True, pad_type='zero', \
            act_type=None, mode='CNA',upscale_factor=2):
        super(OurUpSample, self).__init__()
        self.U1 = pixelshuffle_block(in_nc, nc, upscale_factor=upscale_factor, kernel_size=3, norm_type = 'batch')
        self.co1 = conv_block(nc, 16, kernel_size=1, norm_type=None, act_type='prelu', mode='CNA')
        self.co2 = conv_block(16, 3, kernel_size=3, norm_type=None, act_type='prelu', mode='CNA')

    def forward(self, x):
        out1 = self.U1(x)
        return self.co2(self.co1(out1))