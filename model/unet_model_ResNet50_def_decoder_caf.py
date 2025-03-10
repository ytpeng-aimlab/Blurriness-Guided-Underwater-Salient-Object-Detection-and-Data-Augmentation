""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
from torchvision import models
class SKConv3(nn.Module):
    def __init__(self, channel, reduction=4):

        super(SKConv3, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)#,
            # nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)#,
            # nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, blur):

        b, c, _, _ = x.size()
        y_blur = self.avg_pool(blur).view(b, c)
        y_x = self.avg_pool(x).view(b, c)

        y_x = self.fc(y_x).view(b, c, 1, 1)
        y_blur = self.fc2(y_blur).view(b, c, 1, 1)

        attention_vectors = torch.cat([y_x, y_blur], dim=1)
        attention_vectors = attention_vectors.view(b, 2, c, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats = torch.cat([x.unsqueeze(1), blur.unsqueeze(1)], dim=1)

        return torch.sum(feats*attention_vectors, dim=1)

class defNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(defNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = nn.Sequential(
                # nn.MaxPool2d(2),
                DoubleConv(n_channels, 32)
                )
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # self.down4 = Down(512, 1024 // factor)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x1, x2, x3, x4 ,x5


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
        # factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(4096, 1024, bilinear)
        self.up2 = Up(2048, 512, bilinear)
        self.up3 = Up(1024, 256, bilinear)
        self.up4 = Up(512, 64, bilinear)
        self.up5 = Up(128, 32, bilinear)
        self.outc = OutConv(32, n_classes)

        self.defNet = defNet(3)
        self.se1 = SKConv3(32)
        self.se2 = SKConv3(64)
        self.se3 = SKConv3(256)
        self.se4 = SKConv3(512)
        self.se5 = SKConv3(1024)
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

    def forward(self, x, defocus):
        x1_def, x2_def, x3_def, x4_def ,x5_def = self.defNet(defocus)
        # print(x1_def.shape, x2_def.shape, x3_def.shape, x4_def.shape ,x5_def.shape)
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




        # out1 = self.up1(x6, x5)+ x5_def #
        upx1 = self.up1(x6, x5)
        out1 = self.se5(upx1,  x5_def) #
        # print('up1', x.shape) # up1 torch.Size([3, 1024, 16, 16])

        # out2 = self.up2(out1, x4) + x4_def# 512 -> 128
        upx2 = self.up2(out1, x4)
        out2 = self.se4(upx2, x4_def)# 512 -> 128
        # print('up2', x.shape)# up2 torch.Size([3, 512, 32, 32])
        # out3 = self.up3(out2, x3) +x3_def# 256 -> 64
        ups3 = self.up3(out2, x3)
        out3 = self.se3(ups3, x3_def)# 256 -> 64
        # print('up3', x.shape) #up3 up3 torch.Size([3, 256, 64, 64])
        # out4 = self.up4(out3, x2)+x2_def # 128 -> 64
        upx4 = self.up4(out3, x2)
        out4 = self.se2(upx4, x2_def) # 128 -> 64
        # print('up4', x.shape) #up4  torch.Size([3, 64, 128, 128])
        # out5 = self.up5(out4, x1)+x1_def # 128 -> 64
        upx5 = self.up5(out4, x1)
        out5 = self.se1(upx5, x1_def) # 128 -> 64
        # print('up5', x.shape)#_ up5 torch.Size([3, 64, 256, 256])
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
    net(torch.ones(3,3,256,256),torch.ones(3,1,256,256))
