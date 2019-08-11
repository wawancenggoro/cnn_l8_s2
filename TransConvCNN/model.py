import torch.nn as nn
import torch.nn.init as init
import torch


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
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1, bias=True)
        )
        self.layers_out_02 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 3, stride = 3, padding = 0, bias=True)
        )
        self.layers_out_03 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 3, stride = 3, padding = 0, bias=True)
        )
        self.layers_out_04 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 3, stride = 3, padding = 0, bias=True)
        )
        self.layers_out_05 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 3, stride = 3, padding = 0, bias=True)
        )
        self.layers_out_06 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 3, stride = 3, padding = 0, bias=True)
        )
        self.layers_out_07 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 3, stride = 3, padding = 0, bias=True)
        )
        self.layers_out_08 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 3, stride = 3, padding = 0, bias=True)
        )
        self.layers_out_8A = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 3, stride = 3, padding = 0, bias=True)
        )
        self.layers_out_09 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1, bias=True)
        )
        self.layers_out_10 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1, bias=True)
        )
        self.layers_out_11 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 3, stride = 3, padding = 0, bias=True)
        )
        self.layers_out_12 = torch.nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 3, stride = 3, padding = 0, bias=True)
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
        
        init.orthogonal_(self.layers_out_01[2].weight)
        init.orthogonal_(self.layers_out_02[2].weight)
        init.orthogonal_(self.layers_out_03[2].weight)
        init.orthogonal_(self.layers_out_04[2].weight)
        init.orthogonal_(self.layers_out_05[2].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.layers_out_06[2].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.layers_out_07[2].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.layers_out_08[2].weight)
        init.orthogonal_(self.layers_out_8A[2].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.layers_out_09[2].weight)
        init.orthogonal_(self.layers_out_10[2].weight)
        init.orthogonal_(self.layers_out_11[2].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.layers_out_12[2].weight, init.calculate_gain('relu'))

        init.orthogonal_(self.layers_out_05[4].weight)
        init.orthogonal_(self.layers_out_06[4].weight)
        init.orthogonal_(self.layers_out_07[4].weight)
        init.orthogonal_(self.layers_out_8A[4].weight)
        init.orthogonal_(self.layers_out_11[4].weight)
        init.orthogonal_(self.layers_out_12[4].weight)
        # init.orthogonal_(self.conv4.weight)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = self.conv3(x)
        out_base = self.relu(x)
        out01 = self.layers_out_01(out_base)
        out02 = self.layers_out_02(out_base)
        out03 = self.layers_out_03(out_base)
        out04 = self.layers_out_04(out_base)
        out05 = self.layers_out_05(out_base)
        out06 = self.layers_out_06(out_base)
        out07 = self.layers_out_07(out_base)
        out08 = self.layers_out_08(out_base)
        out8A = self.layers_out_8A(out_base)
        out09 = self.layers_out_09(out_base)
        out10 = self.layers_out_10(out_base)
        out11 = self.layers_out_11(out_base)
        out12 = self.layers_out_12(out_base)

        # x = self.conv4(x)
        # x = self.pixel_shuffle(x)
        return out01, out02, out03, out04, out05, out06, out07, out08, out8A, out09, out10, out11, out12
