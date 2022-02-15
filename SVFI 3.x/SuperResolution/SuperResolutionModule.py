# coding: utf-8

import cv2
import numpy as np
from PIL import Image

from Utils.StaticParameters import appDir, RGB_TYPE
from Utils.utils import overtime_reminder_deco, Tools
from ncnn.sr.waifu2x.waifu2x_ncnn_vulkan import Waifu2x

logger = Tools.get_logger('SWIG-SR', appDir)


class SvfiWaifu(Waifu2x):
    def __init__(self, model="", scale=1, num_threads=4, resize=(0, 0), **kwargs):
        self.available_scales = [2, 4, 8, 16]
        super().__init__(gpuid=0,
                         model=model,
                         tta_mode=False,
                         num_threads=num_threads,
                         scale=scale,
                         noise=0,
                         tilesize=0, )
        self.resize_param = resize

    @overtime_reminder_deco(100, "RealSR",
                            "Low Super-Resolution speed detected, Please Consider lower your output resolution to enhance speed")
    def svfi_process(self, img):
        image = Image.fromarray(cv2.cvtColor(img.astype(RGB_TYPE.DTYPE), cv2.COLOR_BGR2RGB))
        image = self.process(image)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return image
