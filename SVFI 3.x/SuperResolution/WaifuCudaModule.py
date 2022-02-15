import math
import os
import time

import cv2
import torch
from basicsr.utils.registry import ARCH_REGISTRY
# from line_profiler_pycharm import profile
from torch import nn as nn
from torch.nn import functional as F

from SuperResolution.CudaResolutionModule import CudaSuperResolutionBase, LicenseCudaModel
from Utils.utils import overtime_reminder_deco, Tools

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = Tools.get_logger("WaifuCuda", '')


class DIM:
    BATCH = 0
    CHANNEL = 1
    WIDTH = 2
    HEIGHT = 3


@ARCH_REGISTRY.register()
class UpCunet(nn.Module):
    def __init__(self, channels=3):
        super(UpCunet, self).__init__()
        self.cunet_unet1 = CunetUnet1(channels, deconv=True)  # True:2x-16#False:1x-8
        self.cunet_unet2 = CunetUnet2(channels, deconv=False)  # -20
        self.spatial_zero_padding = SpatialZeroPadding(-20)

    # @profile
    def forward(self, x):
        try:
            x = F.pad(x, (18, 18, 18, 18), 'reflect')
        except:
            x = F.pad(x, (18, 18, 18, 18), 'constant')
        x = self.cunet_unet1.forward(x)
        x0 = self.cunet_unet2.forward(x)
        x1 = self.spatial_zero_padding(x)
        x = torch.add(x0, x1)
        # x = torch.clamp(x, min=0, max=1)
        return x


class CunetUnet1(nn.Module):
    def __init__(self, channels: int, deconv: bool):
        super().__init__()
        self.unet_conv = UnetConv(channels, 32, 64, se=False)
        block1 = UnetConv(64, 128, 64, se=True)
        self.unet_branch = UnetBranch(block1, 64, 64, depad=-4)
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3)
        self.lrelu = nn.LeakyReLU(0.1)
        if deconv:
            # Uncertain
            self.conv1 = nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=3)
        else:
            self.conv1 = nn.Conv2d(64, channels, kernel_size=3)

    def forward(self, x):
        x = self.unet_conv(x)
        x = self.unet_branch(x)
        x = self.conv0(x)
        x = self.lrelu(x)
        x = self.conv1(x)
        return x


class CunetUnet2(nn.Module):
    def __init__(self, channels: int, deconv: bool):
        super().__init__()
        self.unet_conv = UnetConv(channels, 32, 64, se=False)
        block1 = UnetConv(128, 256, 128, se=True)
        block2 = nn.Sequential(
            UnetConv(64, 64, 128, se=True),
            UnetBranch(block1, 128, 128, depad=-4),
            UnetConv(128, 64, 64, se=True),
        )
        self.unet_branch = UnetBranch(block2, 64, 64, depad=-16)
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3)
        self.lrelu = nn.LeakyReLU(0.1)
        if deconv:
            # Uncertain
            self.conv1 = nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=3)
        else:
            self.conv1 = nn.Conv2d(64, channels, kernel_size=3)

    def forward(self, x):
        x = self.unet_conv(x)
        x = self.unet_branch(x)
        x = self.conv0(x)
        x = self.lrelu(x)
        x = self.conv1(x)
        return x


class UnetConv(nn.Module):
    def __init__(self, channels_in: int, channels_mid: int, channels_out: int, se: bool):
        super().__init__()
        self.conv0 = nn.Conv2d(channels_in, channels_mid, 3)
        self.lrelu0 = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv2d(channels_mid, channels_out, 3)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.se = se
        if se:
            self.se_block = SEBlock(channels_out, r=8)

    def forward(self, x):
        x = self.conv0(x)
        x = self.lrelu0(x)
        x = self.conv1(x)
        x = self.lrelu1(x)
        if self.se:
            x = self.se_block(x)
        return x


class UnetBranch(nn.Module):
    def __init__(self, insert: nn.Module, channels_in: int, channels_out: int, depad: int):
        super().__init__()
        self.conv0 = nn.Conv2d(channels_in, channels_in, kernel_size=2, stride=2)
        self.lrelu0 = nn.LeakyReLU(0.1)
        self.insert = insert
        self.conv1 = nn.ConvTranspose2d(channels_out, channels_out, kernel_size=2, stride=2)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.spatial_zero_padding = SpatialZeroPadding(depad)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.lrelu0(x0)
        x0 = self.insert(x0)
        x0 = self.conv1(x0)
        x0 = self.lrelu1(x0)
        x1 = self.spatial_zero_padding(x)
        x = torch.add(x0, x1)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels_out: int, r: int):
        super().__init__()
        channels_mid = math.floor(channels_out / r)
        self.conv0 = nn.Conv2d(channels_out, channels_mid, kernel_size=1, stride=1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels_mid, channels_out, kernel_size=1, stride=1)
        self.sigmoid0 = nn.Sigmoid()

    def forward(self, x):
        x0 = torch.mean(x, dim=(DIM.WIDTH, DIM.HEIGHT), keepdim=True)
        x0 = self.conv0(x0)
        x0 = self.relu0(x0)
        x0 = self.conv1(x0)
        x0 = self.sigmoid0(x0)
        x = torch.mul(x, x0)
        return x


class SpatialZeroPadding(nn.Module):
    def __init__(self, padding: int):
        super().__init__()
        if padding > 0:
            raise NotImplementedError("I don't know how to actually pad 0s")
        self.slice = [slice(None) for _ in range(4)]
        self.slice[DIM.HEIGHT] = slice(-padding, padding)
        self.slice[DIM.WIDTH] = slice(-padding, padding)

    def forward(self, x):
        return x[self.slice]


class WaifuCudaer(CudaSuperResolutionBase):
    def __init__(self, scale, model_path, tile=0, tile_pad=10, pre_pad=10, half=False):
        super().__init__(scale, model_path, tile, tile_pad, pre_pad, half)
        model = UpCunet()
        self.LCM = LicenseCudaModel()
        loadnet = torch.load(self.LCM.load_decrypted_model(model_path), map_location='cpu')
        model.load_state_dict(loadnet, strict=True)
        model.eval()
        if self.half:
            self.model = model.half().to(self.device)  # compulsory switch to half mode
        else:
            self.model = model.to(self.device)  # compulsory switch to half mode


class SvfiWaifuCuda:
    def __init__(self, model="", gpu_id=0, precent=90, scale=2, tile=200, resize=(0, 0), half=False):
        app_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.resize_param = resize
        self.scale_exp = 2
        self.scale = scale
        # self.scale = 4
        self.available_scales = [2, 4]
        self.alpha = None
        model_path = os.path.join(app_dir, "ncnn", "sr", "waifuCuda", "models", model)
        self.upscaler = WaifuCudaer(scale=self.scale_exp, model_path=model_path, tile=tile, half=half)
        pass

    # @profile
    @overtime_reminder_deco(100, "WaifuCuda",
                            "Low Super-Resolution speed (>100s per image) detected, Please Consider tweak tilesize or lower output resolution to enhance speed")
    def svfi_process(self, img):
        if self.scale > 1:
            cur_scale = 1
            while cur_scale < self.scale:
                img = self.process(img)
                cur_scale *= self.scale_exp
        return img

    def process(self, img):
        output = self.upscaler.enhance(img)
        return output


if __name__ == '__main__':
    test = SvfiWaifuCuda(model="waifu2x-cunet2x-305k.pth", tile=0, half=True)
    # test.svfi_process(cv2.imread(r"D:\60-fps-Project\Projects\RIFE GUI\Utils\RealESRGAN\input\used\input.png"))
    for i in range(8):
        t = time.time()
        o = test.svfi_process(
            cv2.imread(r"../test/images/0.png", cv2.IMREAD_UNCHANGED))
        cv2.imwrite("../test/images/0-waifu.png", o)
        print(time.time() - t)
    # cv2.imshow('test', o)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
