import torch.nn as nn
import torch.nn.init as init
import torch

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        # x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        x = self.interp(x, size=self.size, mode=self.mode)
        return x

class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(7, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)

        self.layers_out_01 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Interpolate(size=500, mode='nearest'),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.layers_out_02 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Interpolate(size=3000, mode='nearest'),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.layers_out_03 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Interpolate(size=3000, mode='nearest'),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.layers_out_04 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Interpolate(size=3000, mode='nearest'),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.layers_out_05 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Interpolate(size=1500, mode='nearest'),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.layers_out_06 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Interpolate(size=1500, mode='nearest'),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.layers_out_07 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Interpolate(size=1500, mode='nearest'),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.layers_out_08 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Interpolate(size=3000, mode='nearest'),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.layers_out_8A = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Interpolate(size=1500, mode='nearest'),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.layers_out_09 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Interpolate(size=500, mode='nearest'),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.layers_out_10 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Interpolate(size=500, mode='nearest'),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.layers_out_11 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Interpolate(size=1500, mode='nearest'),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.layers_out_12 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Interpolate(size=1500, mode='nearest'),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        )

        # self.conv4 = nn.Conv2d(32, upscale_factor ** 2, kernel_size=3, stride=1, padding=1)
        # self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        # init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        
        init.orthogonal_(self.layers_out_01[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.layers_out_02[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.layers_out_03[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.layers_out_04[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.layers_out_05[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.layers_out_06[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.layers_out_07[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.layers_out_08[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.layers_out_8A[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.layers_out_09[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.layers_out_10[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.layers_out_11[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.layers_out_12[0].weight, init.calculate_gain('relu'))
        
        init.orthogonal_(self.layers_out_01[3].weight)
        init.orthogonal_(self.layers_out_02[3].weight)
        init.orthogonal_(self.layers_out_03[3].weight)
        init.orthogonal_(self.layers_out_04[3].weight)
        init.orthogonal_(self.layers_out_05[3].weight)
        init.orthogonal_(self.layers_out_06[3].weight)
        init.orthogonal_(self.layers_out_07[3].weight)
        init.orthogonal_(self.layers_out_08[3].weight)
        init.orthogonal_(self.layers_out_8A[3].weight)
        init.orthogonal_(self.layers_out_09[3].weight)
        init.orthogonal_(self.layers_out_10[3].weight)
        init.orthogonal_(self.layers_out_11[3].weight)
        init.orthogonal_(self.layers_out_12[3].weight)

        # init.orthogonal_(self.conv4.weight)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = self.conv3(x)
        x = self.relu(x)

        out01 = self.layers_out_01(x)
        out02 = self.layers_out_02(x)
        out03 = self.layers_out_03(x)
        out04 = self.layers_out_04(x)
        out05 = self.layers_out_05(x)
        out06 = self.layers_out_06(x)
        out07 = self.layers_out_07(x)
        out08 = self.layers_out_08(x)
        out8A = self.layers_out_8A(x)
        out09 = self.layers_out_09(x)
        out10 = self.layers_out_10(x)
        out11 = self.layers_out_11(x)
        out12 = self.layers_out_12(x)

        # im60 = Interpolate(size=500, mode='bilinear')(x)
        # im10 = Interpolate(size=3000, mode='bilinear')(x)
        # im20 = Interpolate(size=1500, mode='bilinear')(x)

        # x60 = self.conv1(im60)
        # x60 = self.relu(x60)

        # x10 = self.conv1(im10)
        # x10 = self.relu(x10)

        # x20 = self.conv1(im20)
        # x20 = self.relu(x20)

        # out01 = self.layers_out_01(x60)
        # out02 = self.layers_out_02(x10)
        # out03 = self.layers_out_03(x10)
        # out04 = self.layers_out_04(x10)
        # out05 = self.layers_out_05(x20)
        # out06 = self.layers_out_06(x20)
        # out07 = self.layers_out_07(x20)
        # out08 = self.layers_out_08(x10)
        # out8A = self.layers_out_8A(x20)
        # out09 = self.layers_out_09(x60)
        # out10 = self.layers_out_10(x60)
        # out11 = self.layers_out_11(x20)
        # out12 = self.layers_out_12(x20)

        # x = self.conv4(x)
        # x = self.pixel_shuffle(x)
        return out01, out02, out03, out04, out05, out06, out07, out08, out8A, out09, out10, out11, out12
