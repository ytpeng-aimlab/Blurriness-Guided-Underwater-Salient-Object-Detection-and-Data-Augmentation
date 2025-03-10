""" Full assembly of the parts to form the complete network """
import torch
import torch.nn.functional as F

from .unet_parts import *
from torchvision import models


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


        net = models.resnet50(pretrained=True)
        self.inconv = nn.Sequential(nn.Conv2d(n_channels,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(*list(net.children())[3:5])
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.bridge = nn.Sequential(
                nn.MaxPool2d(2),
                  nn.Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                  nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                  nn.ReLU(inplace=True)
                  )
        # self.inc = DoubleConv(n_channels, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(4096, 1024, bilinear)
        self.up2 = Up(2048, 512, bilinear)
        self.up3 = Up(1024, 256, bilinear)
        self.up4 = Up(512, 64, bilinear)
        self.up5 = Up(128, 32, bilinear)
        self.outc = OutConv(32, n_classes)

        self.upscore1 = nn.Upsample(scale_factor=16,mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=8,mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=2, mode='bilinear')

        ## -------------Side Output--------------
        self.outconv1 = nn.Conv2d(1024,1,3,padding=1)
        self.outconv2 = nn.Conv2d(512,1,3,padding=1)
        self.outconv3 = nn.Conv2d(256,1,3,padding=1)
        self.outconv4 = nn.Conv2d(64,1,3,padding=1)
        self.outconv5 = nn.Conv2d(32,1,3,padding=1)
    def forward(self, x):
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        x1 = self.inconv(x)
        # print('x1',x1.shape)#x1 torch.Size([3, 64, 256, 256])

        x2= self.layer1(x1)
        # print('x2',x2.shape)#x2 torch.Size([3, 256, 128, 128])

        x3= self.layer2(x2)
        # print('x3',x3.shape)#x3 torch.Size([3, 512, 64, 64])

        x4 = self.layer3(x3)
        # print('x4',x4.shape)#x4 torch.Size([3, 1024, 32, 32])

        x5 = self.layer4(x4)
        # print('x5',x5.shape)#x5 torch.Size([3, 2048, 16, 16])

        x6 = self.bridge(x5)
        # print('x6',x6.shape)#x6 torch.Size([3, 2048, 8, 8])




        out1 = self.up1(x6, x5) #
        # print('up1', x.shape) # up1 torch.Size([3, 1024, 16, 16])

        out2 = self.up2(out1, x4) # 512 -> 128
        # print('up2', x.shape)# up2 torch.Size([3, 512, 32, 32])
        out3 = self.up3(out2, x3) # 256 -> 64
        # print('up3', x.shape) #up3 up3 torch.Size([3, 256, 64, 64])
        out4 = self.up4(out3, x2) # 128 -> 64
        # print('up4', x.shape) #up4  torch.Size([3, 64, 128, 128])
        out5 = self.up5(out4, x1) # 128 -> 64
        # print('up5', x.shape)# up5 torch.Size([3, 32, 256, 256])
        logits = self.outc(out5)
        # print('logits',logits.shape)

        out1 = self.upscore1(self.outconv1(out1))
        out2 = self.upscore2(self.outconv2(out2))
        out3 = self.upscore3(self.outconv3(out3))
        out4 = self.upscore4(self.outconv4(out4))
        out5 = self.outconv5(out5)
        return torch.sigmoid(logits), torch.sigmoid(out5), torch.sigmoid(out4), torch.sigmoid(out3), torch.sigmoid(out2), torch.sigmoid(out1)


if __name__ == "__main__":
    net = UNet(3, 1)
    net(torch.ones(3,3,256,256))
    # net = models.resnet34(pretrained=True)
    # inconv = nn.Sequential(*list(net.children())[:5])
    # # exit()
    # print(inconv)
    # x = inconv(torch.ones(1,3,64,64))
    # print(x.shape)
    # # x = net.layer1(x)
    # # print(x.shape)
    # x = net.layer2(x)
    # print(x.shape)
    # x = net.layer3(x)
    # print(x.shape)
    # x = net.layer4(x)
    # print(x.shape)

