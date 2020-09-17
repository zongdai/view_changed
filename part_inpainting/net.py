import torch
import torch.nn as nn
import numpy as np


class InpaintNet1(nn.Module):
    def __init__(self):
        super(InpaintNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 100, 4),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.ConvTranspose2d(100, 512, 4),
            nn.BatchNorm2d(512),
            nn.ReLU(True), 

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, input):
        res = self.net(input)
        tmp = torch.detach(res)
        if np.any(np.isnan(tmp.cpu().numpy()) == True):
            exit()
        return res

class InpaintNet(nn.Module):
    def __init__(self):
        super(InpaintNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 100, 4),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.channel_fc1_8 = nn.Sequential(
            nn.Conv2d(29, 29, 1),
            nn.BatchNorm2d(29),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.channel_fc2_8 = nn.Sequential(
            nn.Conv2d(5, 5, 1),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.channel_fc1_4 = nn.Sequential(
            nn.Conv2d(5, 5, 1),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.channel_fc2_4 = nn.Sequential(
            nn.Conv2d(13, 13, 1),
            nn.BatchNorm2d(13),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.channel_fc1_0 = nn.Sequential(
            nn.Conv2d(5, 5, 1),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.channel_fc2_0 = nn.Sequential(
            nn.Conv2d(17, 17, 1),
            nn.BatchNorm2d(17),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.channel_fc1_1 = nn.Sequential(
            nn.Conv2d(7, 7, 1),
            nn.BatchNorm2d(7),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.channel_fc2_1 = nn.Sequential(
            nn.Conv2d(12, 12, 1),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.channel_fc1_2 = nn.Sequential(
            nn.Conv2d(5, 5, 1),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.channel_fc2_2 = nn.Sequential(
            nn.Conv2d(12, 12, 1),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.channel_fc1_3 = nn.Sequential(
            nn.Conv2d(7, 7, 1),
            nn.BatchNorm2d(7),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.channel_fc2_3 = nn.Sequential(
            nn.Conv2d(7, 7, 1),
            nn.BatchNorm2d(7),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.channel_fc1 = nn.Sequential(
            nn.Conv2d(5, 5, 1),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.channel_fc2 = nn.Sequential(
            nn.Conv2d(13, 13, 1),
            nn.BatchNorm2d(13),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4),
            nn.BatchNorm2d(512),
            nn.ReLU(True), 

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, input):
        en_res = self.encoder(input)

        en_res = en_res.permute(0, 2, 3, 1)
        # print(en_res.shape)
        fc_res1 = self.channel_fc1(en_res)

        fc_res1 = fc_res1.permute(0, 2, 1, 3)
        # print(fc_res1.shape)
        fc_res2 = self.channel_fc2(fc_res1)
        
        fc_res = fc_res2.permute(0, 3, 2, 1)
        de_res = self.decoder(fc_res)
        tmp = torch.detach(de_res)
        if np.any(np.isnan(tmp.cpu().numpy()) == True):
            exit()
        return de_res

if __name__ == "__main__":
    x = torch.randn(8, 3, 320, 320)
    # x = torch.randn(8, 3, 1024, 256)
    net = InpaintNet()
    res = net(x)
    print(res.shape)

# 0: (190, 595) (256, 640)
# 1: (296, 431) (320, 480)
# 2: (254, 386) (320, 480)(SAME AS 1)
# 3: (359, 309) (320, 480)(SAME AS 1)
# 4: (161, 356) (256, 512)
# 5: (235, 533) (256, 512)
# 8: (916, 223) (1024, 256)