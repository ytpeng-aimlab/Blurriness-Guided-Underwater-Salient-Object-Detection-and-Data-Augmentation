#change line 243 for different backbone only for resnet50 resnet101

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from .resnet_model import *
# from resnet_model import *
from .customize import SPBlock_attention
from .BA import MKSP_dilated as MKSP
# from customize import SPBlock_attention

class SPADE(nn.Module):
    def __init__(self, param_free_norm_type, norm_nc, label_nc):
        super().__init__()

        # assert config_text.startswith('spade')
        # parsed = re.search('spade(\D+)(\d)x\d', config_text)
        # param_free_norm_type = str(parsed.group(1))
        ks = 3

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = norm_nc

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        # segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * gamma + beta

        return out


class DefocusNet(nn.Module):
    def __init__(self, n_channels, basic_channels):
        super(DefocusNet, self).__init__()
        # -------------Encoder--------------
        # resnet = models.resnet50(pretrained=True)
        self.inconv = nn.Conv2d(n_channels, 32, 3, padding=1)
        self.inbn = nn.BatchNorm2d(32)
        self.inrelu = nn.ReLU(inplace=True)
        # self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # self.encoder1 = nn.Sequential(nn.Conv2d(n_channels, 64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # #stage 1
        # self.encoder2 = nn.Sequential(*list(resnet.children())[3:5]) #256
        # #stage 2
        # self.encoder3 = resnet.layer2 #128
        # #stage 3
        # self.encoder4 = resnet.layer3 #64
        # #stage 4
        # self.encoder5 = resnet.layer4 #32
        # stage 1
        self.encoder1 = BasicBlock(32, basic_channels, 2, downsample=nn.Sequential(
            nn.Conv2d(32, basic_channels, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(basic_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))  # 512, 64, 64

        self.encoder2 = BasicBlock(basic_channels, basic_channels*4, 2, nn.Sequential(
            nn.Conv2d(basic_channels, basic_channels*4,
                      kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(basic_channels*4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))  # 64, 512, 512
        # stage 2
        self.encoder3 = BasicBlock(basic_channels*4, basic_channels*8, 2, downsample=nn.Sequential(
            nn.Conv2d(basic_channels*4, basic_channels*8,
                      kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(basic_channels*8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))  # 128, 256, 256
        # stage 3
        self.encoder4 = BasicBlock(basic_channels*8, basic_channels*16, 2, downsample=nn.Sequential(
            nn.Conv2d(basic_channels*8, basic_channels*16,
                      kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(basic_channels*16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))  # 256, 128, 128
        # stage 4
        self.encoder5 = BasicBlock(basic_channels*16, basic_channels*32, 2, downsample=nn.Sequential(
            nn.Conv2d(basic_channels*16, basic_channels*32,
                      kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(basic_channels*32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))  # 512, 64, 64


        self.squeeze5 = nn.Sequential(
            nn.Conv2d(2048, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.squeeze4 = nn.Sequential(
            nn.Conv2d(1024, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(
            nn.Conv2d(512, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(
            nn.Conv2d(256, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze1 = nn.Sequential(
            nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # del resnet

    def forward(self, x):
        x = self.inconv(x)
        x = self.inbn(x)
        x = self.inrelu(x)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)

        # print(enc1.shape, enc2.shape, enc3.shape, enc4.shape, enc5.shape)
        return x, self.squeeze1(enc1), self.squeeze2(enc2), self.squeeze3(enc3), self.squeeze4(enc4), self.squeeze5(enc5)


class RefUnet(nn.Module):
    def __init__(self, in_ch, inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)

        self.conv1 = nn.Conv2d(inc_ch, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64, 1, 3, padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx, hx4), 1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx, hx3), 1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))

        residual = self.conv_d0(d1)

        return x + residual


def make_layers(block, basic_channels):
    if block == 'resnet34' or 'resnet18':
        return BasicBlock(basic_channels*8, basic_channels*8)

    elif block == 'resnet50':
        return Bottleneck(basic_channels*8, basic_channels*2, downsample=nn.Sequential(
            nn.Conv2d(basic_channels*8, basic_channels*8,
                      kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(basic_channels*8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))
    elif block == 'vgg16':
        return nn.Conv2d(basic_channels*8, basic_channels*8, 3, padding=1)


class BASNet(nn.Module):
    def __init__(self, basic_channels=64):
        super(BASNet, self).__init__()

        net = models.resnet50(pretrained=True)
        print('Backbone:','resnet50')
        self.basic_channels = basic_channels

        self.div_2 = nn.Sequential(*list(net.children())[:3])
        self.div_4 = nn.Sequential(*list(net.children())[3:5])
        self.div_8 = net.layer2
        self.div_16 = net.layer3
        self.div_32 = net.layer4

        del net
        self.squeeze5 = nn.Sequential(
            nn.Conv2d(basic_channels*32, basic_channels*4, 1), nn.BatchNorm2d(basic_channels*4), nn.ReLU(inplace=True))
        self.squeeze4 = nn.Sequential(
            nn.Conv2d(basic_channels*16, basic_channels*2, 1), nn.BatchNorm2d(basic_channels*2), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(
            nn.Conv2d(basic_channels*8, basic_channels*2, 1), nn.BatchNorm2d(basic_channels*2), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(
            nn.Conv2d(basic_channels*4, basic_channels*2, 1), nn.BatchNorm2d(basic_channels*2), nn.ReLU(inplace=True))
        self.squeeze1 = nn.Sequential(
            nn.Conv2d(basic_channels, basic_channels, 1), nn.BatchNorm2d(basic_channels), nn.ReLU(inplace=True))


        # self.refunet = RefUnet(1, 64)


        self.spblock1 = MKSP(basic_channels, basic_channels)#, nn.BatchNorm2d)
        self.spblock2 = MKSP(basic_channels, basic_channels)#, nn.BatchNorm2d)
        self.spblock3 = MKSP(basic_channels*2, basic_channels*2)#, nn.BatchNorm2d)
        self.spblock4 = MKSP(basic_channels*2, basic_channels*2)#, nn.BatchNorm2d)
        self.spblock5 = MKSP(basic_channels*2, basic_channels*2)#, nn.BatchNorm2d)

        self.spade1 = SPADE('batch', basic_channels, basic_channels)
        self.spade2 = SPADE('batch', basic_channels*2, basic_channels*2)
        self.spade3 = SPADE('batch', basic_channels*2, basic_channels*2)
        self.spade4 = SPADE('batch', basic_channels*2, basic_channels*2)
        self.spade5 = SPADE('batch', basic_channels*4, basic_channels*4)
        # self.strip_pool1 = StripPooling(basic_channels, (20, 12), nn.BatchNorm2d, {'mode': 'bilinear', 'align_corners': True})
        # self.strip_pool2 = StripPooling(basic_channels*2, (20, 12), nn.BatchNorm2d, {'mode': 'bilinear', 'align_corners': True})
        # self.strip_pool3 = StripPooling(basic_channels*4, (20, 12), nn.BatchNorm2d, {'mode': 'bilinear', 'align_corners': True})
        # self.strip_pool4 = StripPooling(basic_channels*8, (20, 12), nn.BatchNorm2d, {'mode': 'bilinear', 'align_corners': True})
        self.mpm1 = BasicBlock(basic_channels*2, basic_channels*2)
        self.mpm2 = BasicBlock(basic_channels*2, basic_channels*2)
        self.mpm3 = BasicBlock(basic_channels*2, basic_channels*2)
        self.mpm4 = BasicBlock(basic_channels, basic_channels)
        self.mpm5 = BasicBlock(basic_channels, basic_channels)
        self.mpm12 = BasicBlock(basic_channels*2, basic_channels*2)
        self.mpm22 = BasicBlock(basic_channels*2, basic_channels*2)
        self.mpm32 = BasicBlock(basic_channels*2, basic_channels*2)
        self.mpm42 = BasicBlock(basic_channels, basic_channels)
        self.mpm52 = BasicBlock(basic_channels, basic_channels)

        self.conv1 = nn.Conv2d(basic_channels*4, basic_channels*2, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(basic_channels*2)
        self.conv2 = nn.Conv2d(basic_channels*4, basic_channels*2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(basic_channels*2)
        self.conv3 = nn.Conv2d(basic_channels*4, basic_channels*2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(basic_channels*2)
        self.conv4 = nn.Conv2d(basic_channels*4, basic_channels, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(basic_channels)
        self.conv5 = nn.Conv2d(basic_channels*2, basic_channels, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(basic_channels)

        self.conv_out1 = nn.Conv2d(basic_channels*2, 1, 3, padding=1)
        self.conv_out2 = nn.Conv2d(basic_channels*2, 1, 3, padding=1)
        self.conv_out3 = nn.Conv2d(basic_channels*2, 1, 3, padding=1)
        self.conv_out4 = nn.Conv2d(basic_channels, 1, 3, padding=1)
        self.conv_out5 = nn.Conv2d(basic_channels, 1, 3, padding=1)
        self.conv_final= nn.Sequential(
            nn.Conv2d(basic_channels, basic_channels, 3, padding=1), nn.BatchNorm2d(basic_channels), nn.ReLU(inplace=True),
            nn.Conv2d(basic_channels, 1, 3, padding=1))

        # self.squeeze_8 = nn.Sequential(
        #     nn.Conv2d(basic_channels*2, basic_channels, 1), nn.BatchNorm2d(basic_channels), nn.ReLU(inplace=True))
        # self.squeeze_4 = nn.Sequential(
        #     nn.Conv2d(basic_channels*2, basic_channels, 1), nn.BatchNorm2d(basic_channels), nn.ReLU(inplace=True))
        # self.squeeze_2 = nn.Sequential(
        #     nn.Conv2d(basic_channels*2, basic_channels//2, 1), nn.BatchNorm2d(basic_channels//2), nn.ReLU(inplace=True))
        # self.squeeze1 = nn.Sequential(
        #     nn.Conv2d(basic_channels*2, basic_channels//2, 1), nn.BatchNorm2d(basic_channels//2), nn.ReLU(inplace=True))


        # self.defocusnet = DefocusNet(n_channels=1, basic_channels=64)
    def forward(self, x, defocus):

        hsize, wsize = x.size()[2:]
        # defocus0, defocus1, defocus2, defocus3, defocus4, defocus5 = self.defocusnet(defocus)  # channel: 64, 64 128 256 512
        in_data_2 = self.div_2(x)
        skip_2 = self.squeeze1(in_data_2)

        in_data_4 = self.div_4(in_data_2)
        skip_4 = self.squeeze2(in_data_4)

        in_data_8 = self.div_8(in_data_4)
        skip_8 = self.squeeze3(in_data_8)

        in_data_16 = self.div_16(in_data_8)
        skip_16 = self.squeeze4(in_data_16)

        in_data_32 = self.div_32(in_data_16)
        skip_32 = self.squeeze5(in_data_32)
##########################################################################################

        in_data_2defocus = self.div_2(torch.cat([defocus, defocus, defocus],1))
        defocus1 = self.squeeze1(in_data_2defocus)

        in_data_4defocus = self.div_4(in_data_2defocus)
        defocus2 = self.squeeze2(in_data_4defocus)

        in_data_8defocus = self.div_8(in_data_4defocus)
        defocus3 = self.squeeze3(in_data_8defocus)

        in_data_16defocus = self.div_16(in_data_8defocus)
        defocus4 = self.squeeze4(in_data_16defocus)

        in_data_32defocus = self.div_32(in_data_16defocus)
        defocus5 = self.squeeze5(in_data_32defocus)

        # print(skip_2.shape, skip_4.shape, skip_8.shape, skip_16.shape, skip_32.shape)

##########################################decoder##########################################
        dec_32 = self.spade5(skip_32, defocus5)
        dec_32 = self.mpm12(self.mpm1(F.relu(self.bn1(self.conv1(dec_32)),inplace=True)))# 128 256 128
        dec_32, dec_32_m = self.spblock5(dec_32)
        # before_32 = dec_32
        # dec_32 = dec_32* (1+ d32gamma)  +d32beta
        # after_32 = dec_32
        dec_16 = F.interpolate(dec_32, size=(hsize//16, wsize//16),
                               mode="bilinear", align_corners=False)

        skip_16 = self.spade4(skip_16, defocus4)
        dec_16 = self.mpm22(self.mpm2(F.relu(self.bn2(self.conv2(torch.cat((dec_16,skip_16),1))),inplace=True))) # 64 128 128 128
        dec_16, dec_16_m = self.spblock4(dec_16)

        # before_16 = dec_16
        # dec_16 = dec_16* (1+ d16gamma) + d16beta
        # after_16 = dec_16
        dec_8 = F.interpolate(dec_16, size=(hsize//8, wsize//8),
                              mode="bilinear", align_corners=False)

        skip_8 = self.spade3(skip_8, defocus3)
        dec_8 = self.mpm32(self.mpm3(F.relu(self.bn3(self.conv3(torch.cat((dec_8,skip_8),1))),inplace=True))) # 64 64 64 64
        dec_8, dec_8_m = self.spblock3(dec_8)

        # before_8 = dec_8
        # dec_8 = dec_8* (1+d8gamma) + d8beta
        # after_8 = dec_8
        dec_4 = F.interpolate(dec_8, size=(hsize//4, wsize//4),
                              mode="bilinear", align_corners=False)

        skip_4 = self.spade2(skip_4, defocus2)
        dec_4 = self.mpm42(self.mpm4(F.relu(self.bn4(self.conv4(torch.cat((dec_4,skip_4),1))),inplace=True)))
        dec_4, dec_4_m = self.spblock2(dec_4)

        # before_4 = dec_4
        # dec_4 = dec_4* (1+d4gamma) + d4beta
        # after_4 = dec_4
        dec_2=F.interpolate(dec_4, size=(hsize//2, wsize//2),
                            mode="bilinear", align_corners=False)

        skip_2 = self.spade1(skip_2, defocus1)
        dec_2=self.mpm52(self.mpm5(F.relu(self.bn5(self.conv5(torch.cat((dec_2,skip_2),1))),inplace=True)))
        dec_2, dec_2_m = self.spblock1(dec_2)

        # before_2 = dec_2
        # dec_2 = dec_2* (1+d2gamma) +d2beta
        # after_2 = dec_2

        out32=self.conv_out1(dec_32)
        out32=F.interpolate(out32, size=(hsize, wsize),
                            mode="bilinear", align_corners=False)
        out16=self.conv_out2(dec_16)
        out16=F.interpolate(out16, size=(hsize, wsize),
                            mode="bilinear", align_corners=False)
        out8=self.conv_out3(dec_8)
        out8=F.interpolate(out8, size=(hsize, wsize),
                           mode="bilinear", align_corners=False)
        out4=self.conv_out4(dec_4)
        out4=F.interpolate(out4, size=(hsize, wsize),
                           mode="bilinear", align_corners=False)
        out2=self.conv_out5(dec_2)
        out2=F.interpolate(out2, size=(hsize, wsize),
                           mode="bilinear", align_corners=False)

        dec_1=F.interpolate(dec_2, size=(hsize, wsize),
                            mode="bilinear", align_corners=False)
        out_final=self.conv_final(dec_1)



#         print(dec_32.shape,
# dec_16.shape,
# dec_8.shape,
# dec_4.shape,
# dec_2.shape,
# dec_1.shape
# )
        # feature_list = [before_32 , after_32, before_16, after_16, before_8, after_8, before_4, after_4, before_2, after_2]
        return torch.sigmoid(out_final), torch.sigmoid(out2), torch.sigmoid(out4), torch.sigmoid(out8), torch.sigmoid(out16), torch.sigmoid(out32) #,feature_list

if __name__ == "__main__":
    net=BASNet(64).cuda()
    # print(net)
    a,b,c,d,e, f = net(torch.ones(14, 3, 224, 224).cuda(), torch.ones(14, 1, 224, 224).cuda())
    print(a.size(),b.size(),c.size(),d.size(),e.size() )