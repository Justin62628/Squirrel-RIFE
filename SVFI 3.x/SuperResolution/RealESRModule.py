import os

import cv2
import torch

from SuperResolution.CudaResolutionModule import CudaSuperResolutionBase
from Utils.utils import overtime_reminder_deco, Tools

# from line_profiler_pycharm import profile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = Tools.get_logger("RealESR", '')


class RealESRGANer(CudaSuperResolutionBase):
    def __init__(self, scale, model_path, tile=0, tile_pad=10, pre_pad=10, half=False):
        super().__init__(scale, model_path, tile, tile_pad, pre_pad, half)
        num_block = 23
        net_scale = scale
        is_change_RRDB = False
        is_RFDN = False
        if 'RealESRGAN_x4plus_anime_6B.pth' in model_path:
            num_block = 6
        elif 'RealESRGAN_x2plus.pth' in model_path:
            """Double Check"""
            net_scale = 2
        elif 'RealESRGAN_x2plus_anime110k_6B.pth' in model_path:
            is_change_RRDB = True
            num_block = 6
            net_scale = 2
        elif 'RFDN' in model_path:
            is_RFDN = True
        # debug
        # num_block = 23
        if is_RFDN:
            from basicsr.archs.rfdn_arch import RFDN
            model = RFDN()
        else:
            if is_change_RRDB:
                from basicsr.archs.svfi_rrdbnet_arch import MyRRDBNet as RRDBNet
            else:
                from basicsr.archs.rrdbnet_arch import RRDBNet as RRDBNet
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_grow_ch=32,
                            num_block=num_block, scale=net_scale)

        loadnet = torch.load(model_path, map_location='cpu')
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)
        model.eval()
        if self.half:
            self.model = model.half().to(self.device)  # compulsory switch to half mode
        else:
            self.model = model.to(self.device)  # compulsory switch to half mode


class SvfiRealESR:
    def __init__(self, model="", gpu_id=0, precent=90, scale=2, tile=100, resize=(0, 0), half=False):
        # TODO optimize auto detect tilesize
        # const_model_memory_usage = 0.6
        # const_pixel_memory_usage = 0.9 / 65536
        # total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 3) * 0.8 * (
        #             precent / 100)
        # available_memory = (total_memory - const_model_memory_usage)
        # tile_size = int(math.sqrt(available_memory / const_pixel_memory_usage)) / scale * 2
        # padding = (scale ** 2) * tile_size
        app_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.resize_param = resize
        self.scale_exp = 4
        self.scale = scale

        # Clarify Model Scale
        if "x2plus" in model:
            self.scale_exp = 2
        else:  # "x4plus":
            self.scale_exp = 4

        self.available_scales = [2, 4]
        self.alpha = None
        model_path = os.path.join(app_dir, "ncnn", "sr", "realESR", "models", model)
        self.upscaler = RealESRGANer(scale=self.scale_exp, model_path=model_path, tile=tile, half=half)
        pass

    # @profile
    @overtime_reminder_deco(100, "RealESR",
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
    test = SvfiRealESR(model="RealESR_RFDN_x2plus_anime110k-160k.pth", )
    # test.svfi_process(cv2.imread(r"D:\60-fps-Project\Projects\RIFE GUI\Utils\RealESRGAN\input\used\input.png"))
    o = test.svfi_process(
        cv2.imread(r"/test/images/0.png", cv2.IMREAD_UNCHANGED))
    cv2.imwrite("../Utils/out3.png", o)
    # cv2.imshow('test', o)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
