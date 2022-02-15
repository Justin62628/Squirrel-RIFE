import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from RIFE.IFNet_HD_Mu_2 import conv, deconv, IFNet
from RIFE.warplayer import warp
from Utils.StaticParameters import RGB_TYPE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


c = 32


class ContextNet(nn.Module):
    def __init__(self):
        super(ContextNet, self).__init__()
        self.conv0 = Conv2(3, c)
        self.conv1 = Conv2(c, c)
        self.conv2 = Conv2(c, 2 * c)
        self.conv3 = Conv2(2 * c, 4 * c)
        self.conv4 = Conv2(4 * c, 8 * c)

    def forward(self, x, flow):
        x = self.conv0(x)
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f1 = warp(x, flow)
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                             align_corners=False) * 0.5
        f2 = warp(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                             align_corners=False) * 0.5
        f3 = warp(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                             align_corners=False) * 0.5
        f4 = warp(x, flow)
        return [f1, f2, f3, f4]


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.conv0 = Conv2(10, c)
        self.down0 = Conv2(c, 2 * c)
        self.down1 = Conv2(4 * c, 4 * c)
        self.down2 = Conv2(8 * c, 8 * c)
        self.down3 = Conv2(16 * c, 16 * c)
        self.up0 = deconv(32 * c, 8 * c)
        self.up1 = deconv(16 * c, 4 * c)
        self.up2 = deconv(8 * c, 2 * c)
        self.up3 = deconv(4 * c, c)
        self.conv = nn.ConvTranspose2d(c, 4, 4, 2, 1)

    def forward(self, img0, img1, flow, c0, c1, flow_gt):
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        if flow_gt == None:
            warped_img0_gt, warped_img1_gt = None, None
        else:
            warped_img0_gt = warp(img0, flow_gt[:, :2])
            warped_img1_gt = warp(img1, flow_gt[:, 2:4])
        x = self.conv0(torch.cat((warped_img0, warped_img1, flow), 1))
        s0 = self.down0(x)
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        return x, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt


class Model:
    def __init__(self, use_multi_cards=False, forward_ensemble=False, tta=0, ada=False, output_mode=0, local_rank=-1):
        self.tta = tta
        self.ada = ada
        self.forward_ensemble = forward_ensemble
        self.use_multi_cards = use_multi_cards
        self.device_count = torch.cuda.device_count()
        self.flownet = IFNet()
        self.contextnet = ContextNet()
        self.fusionnet = FusionNet()
        self.output_mode = output_mode
        self.device()
        self.version = 4.0
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()
        self.contextnet.train()
        self.fusionnet.train()

    def eval(self):
        self.flownet.eval()
        self.contextnet.eval()
        self.fusionnet.eval()

    def device(self):
        self.flownet.to(device)
        self.contextnet.to(device)
        self.fusionnet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param

        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))), False)
                self.contextnet.load_state_dict(convert(torch.load('{}/contextnet.pkl'.format(path))), False)
                self.fusionnet.load_state_dict(convert(torch.load('{}/unet.pkl'.format(path))), False)
            else:
                self.contextnet.load_state_dict(
                    convert(torch.load('{}/contextnet.pkl'.format(path), map_location='cpu')), False)
                self.fusionnet.load_state_dict(convert(torch.load('{}/unet.pkl'.format(path), map_location='cpu')),
                                               False)

                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path), map_location='cpu')),
                                             False)

    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(), '{}/flownet.pkl'.format(path))

    def get_auto_scale(self, img0, img1):
        imgs = torch.cat((img0, img1), 1)
        scale = self.flownet.get_auto_scale(imgs)
        return scale

    def calculate_prediction(self, img0, img1, scale, timestep=0.5):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [4 / scale, 2 / scale, 1 / scale]
        flow, flow_list, mask, merged, res = self.flownet(imgs, timestep, scale_list, ada=self.ada)
        # note that in this module, res is returned with flownet. Other models wont
        if self.output_mode == 1:  # residual
            merged[2] = torch.clamp(merged[2] + res, 0, 1)
            return merged[2]
        if self.output_mode == 2:  # direct
            return merged[2]

        # new part
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        flow = F.interpolate(flow, scale_factor=2.0, mode="bilinear",
                             align_corners=False) * 2.0
        refine_output, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt = self.fusionnet(
            img0, img1, flow, c0, c1, None)
        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        mask = torch.sigmoid(refine_output[:, 3:4])
        merged_img = warped_img0 * mask + warped_img1 * (1 - mask)
        pred = merged_img + res
        pred = torch.clamp(pred, 0, 1)
        return pred

    def inference(self, img0, img1, scale=1.0, n=1):
        multiple = n + 1
        inferenced = list()
        t = np.linspace((1 / multiple), (1 - (1 / multiple)), (multiple - 1))
        for frame_index, t_value in enumerate(t):
            prediction = self.calculate_prediction(img0, img1, scale, t_value)
            prediction = (prediction[0] * RGB_TYPE.SIZE).float().cpu().numpy()
            inferenced.append(prediction)
            del prediction
        return inferenced


if __name__ == '__main__':
    _img0 = torch.zeros(1, 3, 256, 256).float().to(device)
    _img1 = torch.tensor(np.random.normal(
        0, 1, (1, 3, 256, 256))).float().to(device)
    _imgs = torch.cat((_img0, _img1), 1)
    model = Model(True, True, 3)
    model.load_model(r"D:\60-fps-Project\Projects\RIFE GUI\train_log\official_4.0", -1)
    model.eval()
    _t = time.time()
    _inferenced = model.inference(_img0, _img1, n=2)
    print(_inferenced[0].shape)
    print(round(time.time() - _t, 2))
