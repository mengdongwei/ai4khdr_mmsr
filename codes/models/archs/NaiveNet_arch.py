import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util


class NaiveNetV1(nn.Module):
    '''
    interpolate upsample of 2 scale + height frequency residual information
    '''
    def __init__(self, in_nc=1, out_nc=1, nf=32, nb=10):
        super(NaiveNetV1, self).__init__()
        FeatBlock = functools.partial(arch_util.Conv3x3Relu, nf=nf)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.feat_trunk = arch_util.make_layer(FeatBlock, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upsample_x2 = nn.ConvTranspose2d(nf, nf, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

    def forward(self, x, x_up_x2):
        first_feat = self.conv_first(x)
        trunk = self.trunk_conv(self.feat_trunk(first_feat))
        feat = first_feat + trunk
        up_x2_feat = self.upsample_x2(feat)
        out = self.conv_last(up_x2_feat)

        return out + x_up_x2

class NaiveNetV2(nn.Module):
    '''
    based NaiveNetV1, but using deconv replace interpolate 2 scale upsample
    '''
    def __init__(self, in_nc=1, out_nc=1, nf=32, nb=10):
        super(NaiveNetV2, self).__init__()
        FeatBlock = functools.partial(arch_util.Conv3x3Relu, nf=nf)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.feat_trunk = arch_util.make_layer(FeatBlock, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        #### input upsample
        self.x_upsample_x2 = nn.ConvTranspose2d(nf, nf, 3, stride=2, padding=1, output_padding=1, bias=True)

        ####residual feat upsampling
        self.upsample_x2 = nn.ConvTranspose2d(nf, nf, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

    def forward(self, x, x_up_x2):
        first_feat = self.conv_first(x)
        trunk = self.trunk_conv(self.feat_trunk(first_feat))
        feat = first_feat + trunk

        up_x2_feat = self.upsample_x2(feat)
        out = self.conv_last(up_x2_feat)

        x_up_x2 = self.x_upsample_x2(x)

        return out + x_up_x2

class NaiveNetV3(nn.Module):
    '''
    based NaiveNetV2, but using groups(2) convolution with more channels
    '''
    def __init__(self, in_nc=1, out_nc=1, nf=48, nb=10):
        super(NaiveNetV3, self).__init__()
        FeatBlock = functools.partial(arch_util.Conv3x3ReluGroups2, nf=nf)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.feat_trunk = arch_util.make_layer(FeatBlock, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        #### input upsample
        self.x_upsample_x2 = nn.ConvTranspose2d(nf, nf//2, 6, stride=2, padding=1, output_padding=1, bias=True)

        ####residual feat upsampling
        self.upsample_x2 = nn.ConvTranspose2d(nf, nf//2, 6, stride=2, padding=1, output_padding=1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

    def forward(self, x, x_up_x2):
        first_feat = self.conv_first(x)
        trunk = self.trunk_conv(self.feat_trunk(first_feat))
        feat = first_feat + trunk

        up_x2_feat = self.upsample_x2(feat)
        out = self.conv_last(up_x2_feat)

        x_up_x2 = self.x_upsample_x2(x)

        return out + x_up_x2
