# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Hang Zhang
# ECE Department, Rutgers University
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
##
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Custermized NN Module"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

torch_ver = torch.__version__[:3]

__all__ = ['GlobalAvgPool2d', 'GramMatrix',
           'View', 'Sum', 'Mean', 'Normalize', 'ConcurrentModule',
           'PyramidPooling', 'StripPooling']


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return F.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class GramMatrix(nn.Module):
    r""" Gram Matrix for a 4D convolutional featuremaps as a mini-batch

    .. math::
        \mathcal{G} = \sum_{h=1}^{H_i}\sum_{w=1}^{W_i} \mathcal{F}_{h,w}\mathcal{F}_{h,w}^T
    """

    def forward(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram


class View(nn.Module):
    """Reshape the input into different size, an inplace operator, support
    SelfParallel mode.
    """

    def __init__(self, *args):
        super(View, self).__init__()
        if len(args) == 1 and isinstance(args[0], torch.Size):
            self.size = args[0]
        else:
            self.size = torch.Size(args)

    def forward(self, input):
        return input.view(self.size)


class Sum(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Sum, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.sum(self.dim, self.keep_dim)


class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class Normalize(nn.Module):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    Does:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm.

    With default arguments normalizes over the second dimension with Euclidean
    norm.

    Args:
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
    """

    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, self.p, self.dim, eps=1e-8)


class ConcurrentModule(nn.ModuleList):
    r"""Feed to a list of modules concurrently.
    The outputs of the layers are concatenated at channel dimension.

    Args:
        modules (iterable, optional): an iterable of modules to add
    """

    def __init__(self, modules=None):
        super(ConcurrentModule, self).__init__(modules)

    def forward(self, x):
        outputs = []
        for layer in self:
            outputs.append(layer(x))
        return torch.cat(outputs, 1)


class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU(True))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)


class PAM_Module(nn.Module):
    """ Position attention module"""
    # Ref from SAGAN

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class FA_Module(nn.Module):
    """ Position attention module"""
    # Ref from SAGAN

    def __init__(self, in_dim):
        super(FA_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W) -> ex: torch.Size([4, 256, 1, 60])
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()

        if height == 1:
            fc_dim = width
        elif width == 1:
            fc_dim = height

        else:
            raise (RuntimeError("Input tensor does not match. Either height or width should be 1"))

        x = x.view(m_batchsize, C, 1, fc_dim)

        proj_query = self.query_conv(x).view(m_batchsize, -1, fc_dim).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, fc_dim)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, fc_dim)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, 1, fc_dim)

        out = self.gamma*out + x
        return out


class FA_Full_Module(nn.Module):
    """ Position attention module"""
    # Ref from SAGAN

    def __init__(self, in_dim):
        super(FA_Full_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attention_with=None):
        """
            inputs :
                x : input feature maps( B X C X H X W) -> ex: torch.Size([4, 256, 1, 60])
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()

        if attention_with == "height":

            proj_query = self.query_conv(x).permute(0, 1, 3, 2).contiguous().view(
                m_batchsize, -1, height).permute(0, 2, 1)
            proj_key = self.key_conv(x).permute(
                0, 1, 3, 2).contiguous().view(m_batchsize, -1, height)
            energy = torch.bmm(proj_query, proj_key)
            attention = self.softmax(energy)
            # proj_value = self.value_conv(x).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, height)
            # out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            # out = out.view(m_batchsize, C, width, height).permute(0, 1, 3, 2)
        elif attention_with == "width":
            proj_query = self.query_conv(x).view(m_batchsize, -1, width).permute(0, 2, 1)
            proj_key = self.key_conv(x).view(m_batchsize, -1, width)
            energy = torch.bmm(proj_query, proj_key)
            attention = self.softmax(energy)
            # proj_value = self.value_conv(x).view(m_batchsize, -1, width)
            # out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            # out = out.view(m_batchsize, C, height, width)

        # out = self.gamma*out + x
        out = self.gamma*attention
        return out


class StripPooling(nn.Module):
    """
    Reference:
    """

    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                     norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                     norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))

        # bilinear interpolate options
        self._up_kwargs = up_kwargs

        # peter config:
        self.mode = 3
        self.PPM_ATTENTION = False
        self.CAM_ATTENTION = False
        self.CAM_ATTENTION_SCALE = False

        if self.mode == 0:
            self.fc_attention = FA_Module(in_dim=inter_channels)
        elif self.mode == 1 or self.mode == 2 or self.mode == 3:
            self.fc_full_attention = FA_Full_Module(in_dim=inter_channels)
        else:
            raise (RuntimeError("mode should be 0 or 1"))

        if self.PPM_ATTENTION == True:
            self.ppm_attention1 = PAM_Module(in_dim=inter_channels)
            self.ppm_attention2 = PAM_Module(in_dim=inter_channels)
            self.ppm_attention3 = PAM_Module(in_dim=inter_channels)

        if self.CAM_ATTENTION == True:
            self.cam_attention = CAM_Module(in_dim=in_channels)
            self.conv3 = nn.Sequential(nn.Conv2d(in_channels + inter_channels*2, in_channels, 1, bias=False),
                                       norm_layer(in_channels))
        else:
            self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                       norm_layer(in_channels))

        if self.CAM_ATTENTION_SCALE == True:
            self.cam_attention_scale = CAM_Module(in_dim=inter_channels*3)
            self.conv_cross = nn.Sequential(nn.Conv2d(inter_channels*3, inter_channels, 1, bias=False),
                                            norm_layer(inter_channels))

    def forward(self, x):
        bsize, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)

        # peter extension for strip pooling attention

        if self.mode == 0:  # peter -> low gpu memory : attention on pooling layer

            ## old version #######
            #fc_h = self.pool3(x2)
            #fc_w = self.pool4(x2)

            #fc_h_attention = self.fc_attention(fc_h)
            #fc_w_attention = self.fc_attention(fc_w)

            # x2_4 = F.interpolate(self.conv2_3(fc_h_attention), (h, w), **self._up_kwargs) # peter
            # x2_5 = F.interpolate(self.conv2_4(fc_w_attention), (h, w), **self._up_kwargs) # peter
            ## old version #######

            fc_h = self.conv2_3(self.pool3(x2))
            fc_w = self.conv2_4(self.pool4(x2))

            fc_h_attention = self.fc_attention(fc_h)
            fc_w_attention = self.fc_attention(fc_w)

            x2_4 = F.interpolate(fc_h_attention, (h, w), **self._up_kwargs)  # peter
            x2_5 = F.interpolate(fc_w_attention, (h, w), **self._up_kwargs)  # peter

        # jerry -> high gpu memory : attention on input feature maps (attention on the height dimenstion and then pooling on the "height" dimension)
        elif self.mode == 1:

            x2_height_attention = self.fc_full_attention(x2, attention_with="height")
            x2_width_attention = self.fc_full_attention(x2, attention_with="width")

            x2_4 = F.interpolate(self.conv2_3(self.pool3(
                x2_height_attention)), (h, w), **self._up_kwargs)
            x2_5 = F.interpolate(self.conv2_4(self.pool4(
                x2_width_attention)), (h, w), **self._up_kwargs)

        # jerry -> high gpu memory : attention on input feature maps (attention on the height dimenstion and then pooling on the "width" dimension)
        elif self.mode == 2:

            x2_height_attention = self.fc_full_attention(x2, attention_with="height")
            x2_width_attention = self.fc_full_attention(x2, attention_with="width")

            x2_4 = F.interpolate(self.conv2_3(self.pool3(
                x2_width_attention)), (h, w), **self._up_kwargs)
            x2_5 = F.interpolate(self.conv2_4(self.pool4(
                x2_height_attention)), (h, w), **self._up_kwargs)

        # jerry -> high gpu memory : attention on input feature maps (attention on the height dimenstion and then pooling on the "width" dimension)
        elif self.mode == 3:

            x2_height_attention = self.fc_full_attention(x2, attention_with="height")
            x2_width_attention = self.fc_full_attention(x2, attention_with="width")

            x2_4 = self.conv2_3(self.pool3(x2))
            x2_5 = self.conv2_4(self.pool4(x2))

            # print(x2_4.size(),x2_5.size(),x2_height_attention.size(),x2_width_attention.size())

            pooled_h_w = x2_5.matmul(x2_4)  # (h*1) x (1*w) become (b,c,h,w)
            # (b,c,h,w)->() x (b,w,w) become (b,c,h,w)
            pooled_h_w = pooled_h_w.view(
                bsize, -1, w).bmm(x2_width_attention).view(bsize, -1, h, w)
            pooled_h_w = pooled_h_w.permute(0, 1, 3, 2).contiguous().view(bsize, -1, h).bmm(
                x2_height_attention).view(bsize, -1, h, w)  # (b,c,h,w)->(b,c,w,h)->(b,-1,h) x (b,h,h) become (b,c,h,w)
            # print(pooled_H_W.size())
            # exit()
        else:
            raise (RuntimeError("mode should be 0 or 1"))

        # end peter extension for strip pooling attention

        # original
        #x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        #x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        # end original

        # peter: PPM_ATTENTION
        if self.PPM_ATTENTION == True:

            x2_1 = self.ppm_attention1(x2_1)
            x2_2 = self.ppm_attention2(x2_2)
            x2_3 = self.ppm_attention3(x2_3)

        # end peter: PPM_ATTENTION

        if self.CAM_ATTENTION == True:
            # print(x.shape)
            x_attention = self.cam_attention(x)

        if self.CAM_ATTENTION_SCALE == True:
            x2_123 = self.cam_attention_scale(torch.cat([x2_1, x2_2, x2_3], dim=1))
            x1 = self.conv2_5(F.relu_(self.conv_cross(x2_123)))

        else:
            x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))

        x2 = self.conv2_6(F.relu_(pooled_h_w))  # no softmax??

        if self.CAM_ATTENTION == True:
            out = self.conv3(torch.cat([x_attention, x1, x2], dim=1))
        else:
            out = self.conv3(torch.cat([x1, x2], dim=1))

        return F.relu_(x + out)

    def initialize(self):
        for n, m in self.named_children():
            print('initialize: '+n, ':', type(m))
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class SPBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None):
        super(SPBlock, self).__init__()
        midplanes = outplanes
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(
            3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(midplanes)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(
            1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        # x1 = x1.expand(-1, -1, h, w)
        #x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        # x2 = x2.expand(-1, -1, h, w)
        #x2 = F.interpolate(x2, (h, w))

        pooled_h_w = x1.matmul(x2)
        x = self.relu(pooled_h_w)
        x = self.conv3(x).sigmoid()
        return x


class SPBlock_attention_plus(nn.Module):
    def __init__(self, inplanes, pool_size, norm_layer=None, up_kwargs=None):
        super(SPBlock_attention_plus, self).__init__()

        # self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=(
        #     3, 1), padding=(1, 0), bias=False)
        # self.bn1 = norm_layer(inplanes)
        # self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=(
        #     1, 3), padding=(0, 1), bias=False)
        # self.bn2 = norm_layer(inplanes)

        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=True)
######################################################################################################
        self._up_kwargs = up_kwargs
        self.conv_h_w = nn.Sequential(nn.Conv2d(inplanes, inplanes,3, 1, 1, bias=False),
                                       norm_layer(inplanes))
        self.conv_n33n_h_w = nn.Sequential(nn.Conv2d(inplanes, inplanes,3, 1, 1, bias=False),
                                       norm_layer(inplanes))

        self.pooln3 = nn.AdaptiveAvgPool2d((None, 3))
        self.pool3n = nn.AdaptiveAvgPool2d((3, None))

        # self.conv_n3_1 = nn.Sequential(nn.Conv2d(inplanes, inplanes, (3, 1), 1, 1, bias=False),
        #                                norm_layer(inplanes))
        # self.conv_3n_1 = nn.Sequential(nn.Conv2d(inplanes, inplanes, (1, 3), 1, 1, bias=False),
        #                                norm_layer(inplanes))

        self.pool_nn = nn.AdaptiveAvgPool2d(pool_size)
        self.conv_nn = nn.Sequential(nn.Conv2d(inplanes, inplanes, 3, 1, 1, bias=False),
                                     norm_layer(inplanes))
        self.conv_out = nn.Conv2d(inplanes*3, inplanes, kernel_size=1, bias=False)

######################################################################################################
    def forward(self, x):
        bsize, _, h, w = x.size()

        attn_h_h = x.view(bsize, -1, w, h).matmul(x.view(bsize, -1, h, w))  # bchh
        attn_w_w = x.view(bsize, -1, h, w).matmul(x.view(bsize, -1, w, h))  # bcww

        x1 = self.pool1(x)
        # x1 = self.conv1(x1)
        # x1 = self.bn1(x1)
        # x1 = x1.expand(-1, -1, h, w)
        #x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        # x2 = self.conv2(x2)
        # x2 = self.bn2(x2)
        # x2 = x2.expand(-1, -1, h, w)
        #x2 = F.interpolate(x2, (h, w))

        pooled_h_w = x1.matmul(x2)
        pooled_h_w = attn_h_h.matmul(pooled_h_w).matmul(attn_w_w)
        pooled_h_w = self.conv_h_w(pooled_h_w)
######################################################################################################
        x_n3 = self.pooln3(x)
        x_3n = self.pool3n(x)
        pooled_n3_h_w = x_n3.matmul(x_3n)
        pooled_n3_h_w = attn_h_h.matmul(pooled_n3_h_w).matmul(attn_w_w)
        pooled_n3_h_w = self.conv_n33n_h_w(pooled_n3_h_w)

        pooled_kk = F.interpolate(self.conv_nn(self.pool_nn(x)),
                                  (h, w), **self._up_kwargs)
        pooled_kk = attn_h_h.matmul(pooled_kk).matmul(attn_w_w)

######################################################################################################
        x = torch.cat([self.relu(pooled_h_w),
            self.relu(pooled_n3_h_w),
             self.relu(pooled_kk)], dim=1)

        x = self.conv_out(x).sigmoid()
        return x


class SPBlock_attention(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None):
        super(SPBlock_attention, self).__init__()
        midplanes = outplanes
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.convhh = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.convww = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.bn1 = norm_layer(midplanes)
        self.bnhh = norm_layer(midplanes)
        self.bnww = norm_layer(midplanes)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # print(x.size())
        bsize, _, h, w = x.size()

        # attn_h_h = x.view(bsize,-1,w,h).matmul(x.view(bsize,-1,h,w))# bchh
        # attn_w_w = x.view(bsize,-1,h,w).matmul(x.view(bsize,-1,w,h))# bcww
        attn_h_h = self.bnhh(self.convhh(x.matmul(x.permute(0,1,3,2))))# bchh
        attn_w_w = self.bnww(self.convww(x.permute(0,1,3,2).matmul(x)))# bcww


        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = x1.expand(-1, -1, h, w)
        #x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        #x2 = F.interpolate(x2, (h, w))

        pooled_h_w = self.relu(x1 + x2)
        # pooled_h_w = x1.matmul(x2)
        # print(attn_h_h.shape, pooled_h_w.shape, attn_w_w.shape)

        attn_pool_h_w = attn_h_h.matmul(pooled_h_w).matmul(attn_w_w)
        x = self.relu(attn_pool_h_w)
        x = self.conv3(x).sigmoid()
        return x

class SPBlock_attention_two(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None):
        super(SPBlock_attention_two, self).__init__()
        midplanes = outplanes
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.convhh = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.convww = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.bn1 = norm_layer(midplanes)
        self.bnhh = norm_layer(midplanes)
        self.bnww = norm_layer(midplanes)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        self.conv4 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # print(x.size())
        bsize, _, h, w = x.size()

        # attn_h_h = x.view(bsize,-1,w,h).matmul(x.view(bsize,-1,h,w))# bchh
        # attn_w_w = x.view(bsize,-1,h,w).matmul(x.view(bsize,-1,w,h))# bcww
        attn_h_h = self.bnhh(self.convhh(x.matmul(x.permute(0,1,3,2))))# bchh
        attn_w_w = self.bnww(self.convww(x.permute(0,1,3,2).matmul(x)))# bcww


        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = x1.expand(-1, -1, h, w)
        #x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        #x2 = F.interpolate(x2, (h, w))

        pooled_h_w = self.relu(x1 + x2)
        # pooled_h_w = x1.matmul(x2)
        # print(attn_h_h.shape, pooled_h_w.shape, attn_w_w.shape)

        attn_pool_h_w = attn_h_h.matmul(pooled_h_w).matmul(attn_w_w)
        x = self.relu(attn_pool_h_w)
        x = self.conv3(x).sigmoid()
        x1 = self.conv4(x).sigmoid()
        return x, x1

class SPBlock_ori(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None):
        super(SPBlock_ori, self).__init__()
        midplanes = outplanes
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(midplanes)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = x1.expand(-1, -1, h, w)
        #x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        #x2 = F.interpolate(x2, (h, w))

        x = self.relu(x1 + x2)
        x = self.conv3(x).sigmoid()
        return x




if __name__ == "__main__":
    sp = SPBlock_attention(16,12,nn.BatchNorm2d,  {'mode': 'bilinear', 'align_corners': True})
    x = sp(torch.ones(1,16,256,256)).shape
    print(x)
