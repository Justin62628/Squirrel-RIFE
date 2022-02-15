import numpy as np

from RIFE.refine_v4 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes)
    )


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale=1):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask


class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(6 + 1, c=240)
        self.block1 = IFBlock(13 + 4 + 1, c=150)
        self.block2 = IFBlock(13 + 4 + 1, c=90)
        self.block_tea = IFBlock(16 + 4 + 1, c=90)
        self.contextnet = Contextnet()
        self.unet = Unet()

    def get_auto_scale(self, x):
        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]
        timestep = (x[:, :1].clone() * 0 + 1) * 0.5  # fix timestep = 0.5 for auto_scale

        block = [self.block0, self.block1, self.block2]

        scale = 1.0
        size = x.shape
        threshold = np.sqrt(size[2] * size[3] / 2088960.) * 64
        flow, mask = block[0](torch.cat((img0, img1, timestep), 1), None, scale=4)
        if flow[:, :2].abs().max() > threshold and flow[:, 2:4].abs().max() > threshold:
            scale = 0.5
            flow, mask = block[0](torch.cat((img0, img1, timestep), 1), None, scale=8)
            if flow[:, :2].abs().max() > threshold and flow[:, 2:4].abs().max() > threshold:
                scale = 0.25
        return scale

    def forward(self, x, timestep=0.5, scale_list=[4, 2, 1], training=False, fastmode=True, ensemble=False,
                ada=True):
        # if training == False:
        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]
        if not torch.is_tensor(timestep):  # float
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        block = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                flow_d, mask_d = block[i](torch.cat((img0, img1, timestep, warped_img0, warped_img1, mask), 1), flow,
                                        scale=scale_list[i])
                if ensemble:
                    f1, m1 = block[i](torch.cat((img1, img0, 1 - timestep, warped_img1, warped_img0, -mask), 1),
                                    torch.cat((flow[:, 2:4], flow[:, :2]), 1), scale=scale_list[i])
                    flow_d = (flow_d + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    mask_d = (mask_d + (-m1)) / 2
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = block[i](torch.cat((img0, img1, timestep), 1), None, scale=scale_list[i])
                if ensemble:
                    f1, m1 = block[i](torch.cat((img1, img0, 1 - timestep), 1), None, scale=scale_list[i])
                    flow = (flow + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    mask = (mask + (-m1)) / 2
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        return flow, flow_list, mask_list[2], merged, res
