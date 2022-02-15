import io
import math
import os

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from Utils.LicenseModule import AESCipher
from Utils.StaticParameters import RGB_TYPE


class LicenseCudaModel(AESCipher):
    def __init__(self):
        super().__init__()

    def load_decrypted_model(self, path):
        content = open(path, 'rb').read()
        decrpyted_stream = io.BytesIO(self._decrypt(content))
        return decrpyted_stream

    def encrypt_model(self, path):
        with open(path, 'rb') as f1:
            encrypted = self._encrypt(f1.read())
            model_name, model_ext = os.path.splitext(path)
            encrpted_model_path = model_name+'_svfi'+model_ext
            with open(encrpted_model_path, 'wb') as f2:
                f2.write(encrypted)


class CudaSuperResolutionBase:
    def __init__(self, scale, model_path, tile=0, tile_pad=10, pre_pad=10, half=False):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half

        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = model_path

    def pre_process(self, img):
        self.img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)
        if self.half:
            self.img = self.img.astype(np.float16)

        # pre_pad
        if self.pre_pad != 0:
            self.img = np.pad(self.img, ((0,0), (0,0), (0,self.pre_pad), (0,self.pre_pad)), 'reflect')
        # mod pad
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.shape
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = np.pad(self.img, ((0, 0), (0, 0), (0, self.mod_pad_h), (0, self.mod_pad_w)), 'reflect')

    def model_process(self, tile):
        """
        Custom model process of different CUDA SR module, needing overwrite
        :return:
        """
        return self.model(tile)

    @torch.no_grad()
    def process(self):
        self.img = torch.from_numpy(self.img).to(self.device).float()
        if self.half:
            self.img = self.img.half()
        self.output = self.model_process(self.img)
        self.output = self.output.data.float().clamp_(0, 1) * RGB_TYPE.SIZE
        self.output = self.output.cpu().numpy()

    def tile_process(self):
        """Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = np.zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                with torch.no_grad():
                    input_tile = torch.from_numpy(input_tile).to(self.device).float()
                    if self.half:
                        input_tile = input_tile.half()
                    output_tile = self.model_process(input_tile)
                    output_tile = output_tile.data.float().clamp_(0, 1) * RGB_TYPE.SIZE
                    output_tile = output_tile.cpu().numpy()
                # print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]

    def post_process(self):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.shape
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.shape
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        self.output = self.output.squeeze().transpose(1,2,0)
        return self.output

    @torch.no_grad()
    # @profile
    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        # img: numpy
        img = img / RGB_TYPE.SIZE

        # ------------------- process image (without the alpha channel) ------------------- #
        self.pre_process(img)
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()
        output = self.post_process()
        return output
