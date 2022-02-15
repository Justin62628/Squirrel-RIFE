import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from RIFE.warplayer import warp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv_wo_act(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
    )


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class IFBlock(nn.Module):
    def __init__(self, in_planes, scale=1, c=64):
        super(IFBlock, self).__init__()
        self.scale = scale
        self.conv0 = nn.Sequential(
            conv(in_planes, c, 3, 2, 1),
            conv(c, 2 * c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
        )
        self.conv1 = nn.ConvTranspose2d(2 * c, 4, 4, 2, 1)

    def forward(self, x):
        if self.scale != 1:
            x = F.interpolate(x, scale_factor=1. / self.scale, mode="bilinear",
                              align_corners=False)
        x = self.conv0(x)
        x = self.convblock(x)
        x = self.conv1(x)
        flow = x
        if self.scale != 1:
            flow = F.interpolate(flow, scale_factor=self.scale, mode="bilinear",
                                 align_corners=False)
        return flow


class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(6, scale=8, c=192)
        self.block1 = IFBlock(10, scale=4, c=128)
        self.block2 = IFBlock(10, scale=2, c=96)
        self.block3 = IFBlock(10, scale=1, c=48)

    def get_auto_scale(self, x):
        scale = 1.0
        size = x.shape
        threshold = np.sqrt(size[2] * size[3] / 2088960.) * 24  # 1920x1080, 24 is magic
        flow0 = self.block0(x)
        if flow0[:, :2].abs().max() > threshold and flow0[:, 2:4].abs().max() > threshold:
            scale = 0.5
            xt = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
            flow0 = self.block0(xt)
            if flow0[:, :2].abs().max() > threshold and flow0[:, 2:4].abs().max() > threshold:
                scale = 0.25
        return scale

    def forward(self, x, scale=1.0, ensemble=False, ada=True):
        if scale != 1.0:
            x = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
        flow0 = self.block0(torch.cat((x[:, :3], x[:, 3:]), 1))
        if ensemble:
            flow01 = self.block0(torch.cat((x[:, 3:], x[:, :3]), 1))
            flow0 = (flow0 + torch.cat((flow01[:, 2:4], flow01[:, :2]), 1)) / 2
        F1 = flow0
        F1_large = F.interpolate(F1, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        warped_img0 = warp(x[:, :3], F1_large[:, :2])
        warped_img1 = warp(x[:, 3:], F1_large[:, 2:4])
        flow1 = self.block1(torch.cat((warped_img0, warped_img1, F1_large), 1))
        if ensemble:
            F1_large = torch.cat((F1_large[:, 2:4], F1_large[:, :2]), 1)
            flow11 = self.block1(torch.cat((warped_img1, warped_img0, F1_large), 1))
            flow1 = (flow1 + torch.cat((flow11[:, 2:4], flow11[:, :2]), 1)) / 2
        F2 = (flow0 + flow1)
        F2_large = F.interpolate(F2, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        warped_img0 = warp(x[:, :3], F2_large[:, :2])
        warped_img1 = warp(x[:, 3:], F2_large[:, 2:4])
        flow2 = self.block2(torch.cat((warped_img0, warped_img1, F2_large), 1))
        if ensemble:
            F2_large = torch.cat((F2_large[:, 2:4], F2_large[:, :2]), 1)
            flow21 = self.block2(torch.cat((warped_img1, warped_img0, F2_large), 1))
            flow2 = (flow2 + torch.cat((flow21[:, 2:4], flow21[:, :2]), 1)) / 2
        F3 = (flow0 + flow1 + flow2)
        F3_large = F.interpolate(F3, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        warped_img0 = warp(x[:, :3], F3_large[:, :2])
        warped_img1 = warp(x[:, 3:], F3_large[:, 2:4])
        flow3 = self.block3(torch.cat((warped_img0, warped_img1, F3_large), 1))
        if ensemble:
            F3_large = torch.cat((F3_large[:, 2:4], F3_large[:, :2]), 1)
            flow31 = self.block3(torch.cat((warped_img1, warped_img0, F3_large), 1))
            flow3 = (flow3 + torch.cat((flow31[:, 2:4], flow31[:, :2]), 1)) / 2
        F4 = (flow0 + flow1 + flow2 + flow3)
        if scale != 1.0:
            F4 = F.interpolate(F4, scale_factor=1 / scale, mode="bilinear", align_corners=False) / scale
        return F4, [F1, F2, F3, F4]


if __name__ == '__main__':
    img0 = torch.zeros(3, 3, 256, 256).float().to(device)
    img1 = torch.tensor(np.random.normal(
        0, 1, (3, 3, 256, 256))).float().to(device)
    imgs = torch.cat((img0, img1), 1)
    flownet = IFNet()
    flow, _ = flownet(imgs)
    print(flow.shape)
