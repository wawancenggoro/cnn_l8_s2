import torch
import torch.nn as nn


class Net(torch.nn.Module):
    def __init__(self, num_channels, base_filter):
        super(Net, self).__init__()

        self.base_layers = torch.nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=base_filter, kernel_size=9, stride=1, padding=4, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_filter, out_channels=base_filter // 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            # nn.PixelShuffle(upscale_factor)
        )
        self.layers_out_01 = torch.nn.Sequential(
            nn.Conv2d(in_channels=base_filter // 2, out_channels=1, kernel_size=5, stride=2, padding=2, bias=True)
        )
        self.layers_out_02 = torch.nn.Sequential(
            nn.Conv2d(in_channels=base_filter // 2, out_channels=(3 ** 2), kernel_size=5, stride=1, padding=2, bias=True),
            nn.PixelShuffle(3)
        )
        self.layers_out_03 = torch.nn.Sequential(
            nn.Conv2d(in_channels=base_filter // 2, out_channels=(3 ** 2), kernel_size=5, stride=1, padding=2, bias=True),
            nn.PixelShuffle(3)
        )
        self.layers_out_04 = torch.nn.Sequential(
            nn.Conv2d(in_channels=base_filter // 2, out_channels=(3 ** 2), kernel_size=5, stride=1, padding=2, bias=True),
            nn.PixelShuffle(3)
        )
        self.layers_out_05 = torch.nn.Sequential(
            nn.Conv2d(in_channels=base_filter // 2, out_channels=(3 ** 2), kernel_size=5, stride=2, padding=2, bias=True),
            nn.PixelShuffle(3)
        )
        self.layers_out_06 = torch.nn.Sequential(
            nn.Conv2d(in_channels=base_filter // 2, out_channels=(3 ** 2), kernel_size=5, stride=2, padding=2, bias=True),
            nn.PixelShuffle(3)
        )
        self.layers_out_07 = torch.nn.Sequential(
            nn.Conv2d(in_channels=base_filter // 2, out_channels=(3 ** 2), kernel_size=5, stride=2, padding=2, bias=True),
            nn.PixelShuffle(3)
        )
        self.layers_out_08 = torch.nn.Sequential(
            nn.Conv2d(in_channels=base_filter // 2, out_channels=(3 ** 2), kernel_size=5, stride=1, padding=2, bias=True),
            nn.PixelShuffle(3)
        )
        self.layers_out_8A = torch.nn.Sequential(
            nn.Conv2d(in_channels=base_filter // 2, out_channels=(3 ** 2), kernel_size=5, stride=2, padding=2, bias=True),
            nn.PixelShuffle(3)
        )
        self.layers_out_09 = torch.nn.Sequential(
            nn.Conv2d(in_channels=base_filter // 2, out_channels=1, kernel_size=5, stride=2, padding=2, bias=True)
        )
        self.layers_out_10 = torch.nn.Sequential(
            nn.Conv2d(in_channels=base_filter // 2, out_channels=1, kernel_size=5, stride=2, padding=2, bias=True)
        )
        self.layers_out_11 = torch.nn.Sequential(
            nn.Conv2d(in_channels=base_filter // 2, out_channels=(3 ** 2), kernel_size=5, stride=2, padding=2, bias=True),
            nn.PixelShuffle(3)
        )
        self.layers_out_12 = torch.nn.Sequential(
            nn.Conv2d(in_channels=base_filter // 2, out_channels=(3 ** 2), kernel_size=5, stride=2, padding=2, bias=True),
            nn.PixelShuffle(3)
        )

        self.out = []

    def forward(self, x):
        out_base = self.base_layers(x)
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
        
        return out01, out02, out03, out04, out05, out06, out07, out08, out8A, out09, out10, out11, out12

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
