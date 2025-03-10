import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from .resnet_model import *

class DefocusNet(nn.Module):
    def __init__(self,n_channels,basic_channels):
        super(DefocusNet,self).__init__()
                ## -------------Encoder--------------
        resnet = models.resnet18(pretrained=True)
        self.inconv = nn.Conv2d(n_channels,basic_channels,3,padding=1)
        self.inbn = nn.BatchNorm2d(basic_channels)
        self.inrelu = nn.ReLU(inplace=True)

        self.encoder1 = resnet.layer1 #256
        #stage 2
        self.encoder2 = resnet.layer2 #128
        #stage 3
        self.encoder3 = resnet.layer3 #64
        #stage 4
        self.encoder4 = resnet.layer4 #32
        # #stage 1
        # self.encoder1 = BasicBlock(basic_channels,basic_channels) # 64, 512, 512
        # #stage 2
        # self.encoder2 = BasicBlock(basic_channels,basic_channels*2,2,downsample = nn.Sequential(
        # nn.Conv2d(basic_channels, basic_channels*2, kernel_size=(1, 1), stride=(2, 2), bias=False),
        # nn.BatchNorm2d(basic_channels*2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))  #128, 256, 256
        # #stage 3
        # self.encoder3 = BasicBlock(basic_channels*2,basic_channels*4,2,downsample = nn.Sequential(
        # nn.Conv2d(basic_channels*2, basic_channels*4, kernel_size=(1, 1), stride=(2, 2), bias=False),
        # nn.BatchNorm2d(basic_channels*4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))  #256, 128, 128
        # #stage 4
        # self.encoder4 = BasicBlock(basic_channels*4,basic_channels*8,2,downsample = nn.Sequential(
        # nn.Conv2d(basic_channels*4, basic_channels*8, kernel_size=(1, 1), stride=(2, 2), bias=False),
        # nn.BatchNorm2d(basic_channels*8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))  #512, 64, 64

    def forward(self, x):
        x = self.inconv(x)
        x = self.inbn(x)
        x = self.inrelu(x)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        return x, enc1, enc2, enc3, enc4

class RefUnet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x):

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

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)

        return x + residual

def make_layers(block,basic_channels):
    if block == 'resnet34' or 'resnet18':
        return BasicBlock(basic_channels*8,basic_channels*8)

    elif block == 'resnet50':
        return Bottleneck(basic_channels*8,basic_channels*2,downsample=nn.Sequential(
        nn.Conv2d(basic_channels*8, basic_channels*8, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.BatchNorm2d(basic_channels*8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))
    elif block == 'vgg16':
        return nn.Conv2d(basic_channels*8,basic_channels*8,3,padding=1)

class BASNet(nn.Module):
    def __init__(self,n_channels,n_classes,Add_defocus = True,backbone = 'resnet34'):
        super(BASNet,self).__init__()
        self.Add_defocus = Add_defocus
        self.backbone = backbone

        if backbone == 'resnet18' or backbone == 'resnet34' or backbone == 'vgg16':
            basic_channels = 64
        elif backbone == 'resnet50':
            basic_channels = 256

        if backbone == 'resnet34':
            resnet = models.resnet34(pretrained=True)

        elif backbone == 'resnet18':
            resnet = models.resnet18(pretrained=True)

        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)

        elif backbone == 'vgg16':
            vgg = models.vgg16(pretrained=True)
            resnet = vgg
            resnet.features[0] = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            resnet.layer1 = resnet.features[0:4]
            resnet.layer2 = resnet.features[4:9]
            resnet.layer3 = resnet.features[9:16]
            resnet.layer4 = resnet.features[16:23]

        else:
            raise ValueError("only 'resnet18', 'resnet34', 'resnet50', and 'vgg16' available")



        self.inconv = nn.Conv2d(n_channels,64,3,padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)

        #stage 1
        self.encoder1 = resnet.layer1 #256
        #stage 2
        self.encoder2 = resnet.layer2 #128
        #stage 3
        self.encoder3 = resnet.layer3 #64
        #stage 4
        self.encoder4 = resnet.layer4 #32

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #stage 5
        self.resb5_1 = make_layers(backbone,basic_channels)
        # self.resb5_1 = BasicBlock(1024,512,downsample = nn.Sequential(
        # nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
        # nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))

        self.resb5_2 = make_layers(backbone,basic_channels)
        self.resb5_3 = make_layers(backbone,basic_channels) #16

        self.pool5 = nn.MaxPool2d(2,2,ceil_mode=True)

        #stage 6
        self.resb6_1 = make_layers(backbone,basic_channels)
        self.resb6_2 = make_layers(backbone,basic_channels)
        self.resb6_3 = make_layers(backbone,basic_channels)#8

        ## -------------Bridge--------------

        #stage Bridge
        self.convbg_1 = nn.Conv2d(basic_channels*8,basic_channels*8,3,dilation=2, padding=2) # 8
        self.bnbg_1 = nn.BatchNorm2d(basic_channels*8)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(basic_channels*8,basic_channels*8,3,dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(basic_channels*8)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(basic_channels*8,basic_channels*8,3,dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(basic_channels*8)
        self.relubg_2 = nn.ReLU(inplace=True)

        ## -------------Decoder--------------

        #stage 6d
        self.conv6d_1 = nn.Conv2d(basic_channels*16,basic_channels*8,3,padding=1) # 16
        self.bn6d_1 = nn.BatchNorm2d(basic_channels*8)
        self.relu6d_1 = nn.ReLU(inplace=True)

        self.conv6d_m = nn.Conv2d(basic_channels*8,basic_channels*8,3,dilation=2, padding=2)###
        self.bn6d_m = nn.BatchNorm2d(basic_channels*8)
        self.relu6d_m = nn.ReLU(inplace=True)

        self.conv6d_2 = nn.Conv2d(basic_channels*8,basic_channels*8,3,dilation=2, padding=2)
        self.bn6d_2 = nn.BatchNorm2d(basic_channels*8)
        self.relu6d_2 = nn.ReLU(inplace=True)

        #stage 5d
        self.conv5d_1 = nn.Conv2d(basic_channels*16,basic_channels*8,3,padding=1) # 16
        self.bn5d_1 = nn.BatchNorm2d(basic_channels*8)
        self.relu5d_1 = nn.ReLU(inplace=True)

        self.conv5d_m = nn.Conv2d(basic_channels*8,basic_channels*8,3,padding=1)###
        self.bn5d_m = nn.BatchNorm2d(basic_channels*8)
        self.relu5d_m = nn.ReLU(inplace=True)

        self.conv5d_2 = nn.Conv2d(basic_channels*8,basic_channels*8,3,padding=1)
        self.bn5d_2 = nn.BatchNorm2d(basic_channels*8)
        self.relu5d_2 = nn.ReLU(inplace=True)

        #stage 4d
        #self.conv4d_1 = nn.Conv2d(1536,512,3,padding=1) # 32 ########
        self.conv4d_1 = nn.Conv2d(basic_channels*16,basic_channels*8,3,padding=1)
        self.bn4d_1 = nn.BatchNorm2d(basic_channels*8)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_m = nn.Conv2d(basic_channels*8,basic_channels*8,3,padding=1)###
        self.bn4d_m = nn.BatchNorm2d(basic_channels*8)
        self.relu4d_m = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(basic_channels*8,basic_channels*4,3,padding=1)
        self.bn4d_2 = nn.BatchNorm2d(basic_channels*4)
        self.relu4d_2 = nn.ReLU(inplace=True)

        #stage 3d
        #self.conv3d_1 = nn.Conv2d(768,256,3,padding=1) # 64 #####
        self.conv3d_1 = nn.Conv2d(basic_channels*8,basic_channels*4,3,padding=1)
        self.bn3d_1 = nn.BatchNorm2d(basic_channels*4)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_m = nn.Conv2d(basic_channels*4,basic_channels*4,3,padding=1)###
        self.bn3d_m = nn.BatchNorm2d(basic_channels*4)
        self.relu3d_m = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(basic_channels*4,basic_channels*2,3,padding=1)
        self.bn3d_2 = nn.BatchNorm2d(basic_channels*2)
        self.relu3d_2 = nn.ReLU(inplace=True)

        #stage 2d

        #self.conv2d_1 = nn.Conv2d(384,128,3,padding=1) # 128#####
        self.conv2d_1 = nn.Conv2d(basic_channels*4,basic_channels*2,3,padding=1)
        self.bn2d_1 = nn.BatchNorm2d(basic_channels*2)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.conv2d_m = nn.Conv2d(basic_channels*2,basic_channels*2,3,padding=1)###
        self.bn2d_m = nn.BatchNorm2d(basic_channels*2)
        self.relu2d_m = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(basic_channels*2,basic_channels,3,padding=1)
        self.bn2d_2 = nn.BatchNorm2d(basic_channels)
        self.relu2d_2 = nn.ReLU(inplace=True)

        #stage 1d
        #self.conv1d_1 = nn.Conv2d(192,64,3,padding=1) # 256 #####
        self.conv1d_1 = nn.Conv2d(basic_channels*2,basic_channels,3,padding=1)
        self.bn1d_1 = nn.BatchNorm2d(basic_channels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.conv1d_m = nn.Conv2d(basic_channels,basic_channels,3,padding=1)###
        self.bn1d_m = nn.BatchNorm2d(basic_channels)
        self.relu1d_m = nn.ReLU(inplace=True)

        self.conv1d_2 = nn.Conv2d(basic_channels,basic_channels,3,padding=1)
        self.bn1d_2 = nn.BatchNorm2d(basic_channels)
        self.relu1d_2 = nn.ReLU(inplace=True)

        ## -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32,mode='bilinear',align_corners=False)### 0.4.0 align_corners=None
        self.upscore5 = nn.Upsample(scale_factor=16,mode='bilinear',align_corners=False)
        self.upscore4 = nn.Upsample(scale_factor=8,mode='bilinear',align_corners=False)
        self.upscore3 = nn.Upsample(scale_factor=4,mode='bilinear',align_corners=False)
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)

        ## -------------Side Output--------------
        self.outconvb = nn.Conv2d(basic_channels*8,1,3,padding=1)
        self.outconv6 = nn.Conv2d(basic_channels*8,1,3,padding=1)
        self.outconv5 = nn.Conv2d(basic_channels*8,1,3,padding=1)
        self.outconv4 = nn.Conv2d(basic_channels*4,1,3,padding=1)
        self.outconv3 = nn.Conv2d(basic_channels*2,1,3,padding=1)
        self.outconv2 = nn.Conv2d(basic_channels,1,3,padding=1)
        self.outconv1 = nn.Conv2d(basic_channels,1,3,padding=1)

        ## -------------Refine Module-------------
        self.refunet = RefUnet(1,64)

        if self.Add_defocus:
            self.defocusnet = DefocusNet(3,basic_channels)
        # else:
            # self.defocusnet = None

        ## -------Feature Fusion--------------
        # if self.Add_defocus:
            # self.ffsigm = nn.Sigmoid()

            # self.ffconv1_1 = nn.Conv2d(basic_channels*2,basic_channels,3,padding=1)
            # self.ffconv1_2 = nn.Conv2d(basic_channels,basic_channels,1,padding=0)
            # self.ffrelu1_3 = nn.ReLU(inplace=True)
            # self.ffconv1_4 = nn.Conv2d(basic_channels,basic_channels,1,padding=0)

            # self.ffconv2_1 = nn.Conv2d(basic_channels*4,basic_channels*2,3,padding=1)
            # self.ffconv2_2 = nn.Conv2d(basic_channels*2,basic_channels*2,1,padding=0)
            # self.ffrelu2_3 = nn.ReLU(inplace=True)
            # self.ffconv2_4 = nn.Conv2d(basic_channels*2,basic_channels*2,1,padding=0)

            # self.ffconv3_1 = nn.Conv2d(basic_channels*8,basic_channels*4,3,padding=1)
            # self.ffconv3_2 = nn.Conv2d(basic_channels*4,basic_channels*4,1,padding=0)
            # self.ffrelu3_3 = nn.ReLU(inplace=True)
            # self.ffconv3_4 = nn.Conv2d(basic_channels*4,basic_channels*4,1,padding=0)

            # self.ffconv4_1 = nn.Conv2d(basic_channels*16,basic_channels*8,3,padding=1)
            # self.ffconv4_2 = nn.Conv2d(basic_channels*8,basic_channels*8,1,padding=0)
            # self.ffrelu4_3 = nn.ReLU(inplace=True)
            # self.ffconv4_4 = nn.Conv2d(basic_channels*8,basic_channels*8,1,padding=0)

        ## -----------------------------------

    def forward(self,x,defocus):

        if self.Add_defocus:
            defocus0, defocus1, defocus2, defocus3, defocus4 = self.defocusnet(defocus)#channel: 64, 64 128 256 512

        hx = x
        ## -------------Encoder-------------
        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)

        ho1= self.encoder1(hx)
        h1 = ho1

        ho2= self.encoder2(ho1)
        h2 = ho2

        ho3= self.encoder3(ho2)
        h3 = ho3

        ho4= self.encoder4(ho3)
        h4 = ho4


        hx = self.pool4(ho4) # 512, 14, 14

        hx = self.resb5_1(hx) ####512, 14, 14

        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx) #512, 14, 14
        hx = self.pool5(h5) # 8

        hx = self.resb6_1(hx)
        hx = self.resb6_2(hx)
        h6 = self.resb6_3(hx) # 512, 7, 7

        ## -------------Bridge-------------
        hx = self.relubg_1(self.bnbg_1(self.convbg_1(h6))) # 8
        hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))

        ## -------------Decoder-------------

        hx = self.relu6d_1(self.bn6d_1(self.conv6d_1(torch.cat((hbg,h6),1)))) #1024->512
        hx = self.relu6d_m(self.bn6d_m(self.conv6d_m(hx))) #512
        hd6 = self.relu6d_2(self.bn5d_2(self.conv6d_2(hx))) #512

        hx = self.upscore2(hd6) # 8 -> 16 #512

        hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((hx,h5),1)))) #cat(512,512) #1024->512
        hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx))) #512
        hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx))) #512

        hx = self.upscore2(hd5) # 16 -> 32

        # if self.Add_defocus: #--------------------------------------------------------------------------------
        hx= hx+ defocus4
            # hi4= self.ffconv4_1(hi4)  # 512 ch
            # h4 = torch.mean(hi4, (2,3), keepdim=True, out=None)
            # h4 = self.ffconv4_2(h4)
            # h4 = self.ffrelu4_3(h4)
            # h4 = self.ffconv4_4(h4)
            # h4 = self.ffsigm(h4)
            # h4 = torch.mul(hi4, h4)
            # hx = torch.add(hi4, h4)
                              #--------------------------------------------------------------------------------

        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hx,h4),1)))) ###cat(512,1024)
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

        hx = self.upscore2(hd4) # 32 -> 64

        # if self.Add_defocus:
            # hi3= torch.cat((hx, defocus3),1)
            # hi3= self.ffconv3_1(hi3)  # 256 ch
            # h3 = torch.mean(hi3, (2,3), keepdim=True, out=None)
            # h3 = self.ffconv3_2(h3)
            # h3 = self.ffrelu3_3(h3)
            # h3 = self.ffconv3_4(h3)
            # h3 = self.ffsigm(h3)
            # h3 = torch.mul(hi3, h3)
            # h3 = torch.add(hi3, h3)
        hx= hx+ defocus3

        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((hx,h3),1))))###cat(256,512) #512->256
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx))) #256->256
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx))) #256->128

        hx = self.upscore2(hd3) # 64 -> 128

        # if self.Add_defocus:
            # hi2= torch.cat((hx, defocus2),1)
            # hi2= self.ffconv2_1(hi2)  # 128 ch
            # h2 = torch.mean(hi2, (2,3), keepdim=True, out=None)
            # h2 = self.ffconv2_2(h2)
            # h2 = self.ffrelu2_3(h2)
            # h2 = self.ffconv2_4(h2)
            # h2 = self.ffsigm(h2)
            # h2 = torch.mul(hi2, h2)
            # hx = torch.add(hi2, h2)
        hx= hx+ defocus2



        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hx,h2),1)))) ###cat(128,256) #256->128
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx))) #128->128
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx))) #128->64

        hx = self.upscore2(hd2) # 128 -> 256

        # if self.Add_defocus:

            # hi1= torch.cat((hx, defocus1),1) #cat(64,64)
            # hi1= self.ffconv1_1(hi1)  # 128->64 ch
            # h1 = torch.mean(hi1, (2,3), keepdim=True, out=None)
            # h1 = self.ffconv1_2(h1)
            # h1 = self.ffrelu1_3(h1)
            # h1 = self.ffconv1_4(h1)
            # h1 = self.ffsigm(h1)
            # h1 = torch.mul(hi1, h1)
            # hx = torch.add(hi1, h1)
        hx= hx+ defocus1

        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx,h1),1))))  ###cat(64,128)
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

        ## -------------Side Output-------------
        db = self.outconvb(hbg)
        db = self.upscore6(db) # 8->256

        d6 = self.outconv6(hd6)
        d6 = self.upscore6(d6) # 8->256

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5) # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4) # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3) # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2) # 128->256

        d1 = self.outconv1(hd1)  # 256

        ## -------------Refine Module-------------



        dout = self.refunet(d1) # 256

        return torch.sigmoid(dout), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6), torch.sigmoid(db)

