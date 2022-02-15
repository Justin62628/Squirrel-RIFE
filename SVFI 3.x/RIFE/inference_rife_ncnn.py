import os
import random
import warnings

from Utils.utils import ArgumentManager, VideoFrameInterpolationBase
from ncnn.rife import rife_ncnn_vulkan

warnings.filterwarnings("ignore")
raw = rife_ncnn_vulkan.raw


class RifeInterpolation(VideoFrameInterpolationBase):
    def __init__(self, __args: ArgumentManager, logger):
        super().__init__(__args, logger)
        self.ARGS = __args
        uhd_mode = True if self.ARGS.rife_exp < 1 else False
        self.initiated = False
        self.use_multi_cards = self.ARGS.use_rife_multi_cards
        self.device = []
        if self.use_multi_cards:
            for nvidia_card_id in range(self.ARGS.ncnn_gpu):
                self.device.append(rife_ncnn_vulkan.RIFE(
                    nvidia_card_id, os.path.basename(self.ARGS.rife_model),
                    uhd_mode=uhd_mode, num_threads=self.ARGS.ncnn_thread))
        else:
            self.device.append(rife_ncnn_vulkan.RIFE(
                self.ARGS.ncnn_gpu, os.path.basename(self.ARGS.rife_model),
                uhd_mode=uhd_mode, num_threads=self.ARGS.ncnn_thread))
        self.model = None
        self.tta = self.ARGS.rife_tta_mode
        self.model_path = ""

    def initiate_algorithm(self):
        if self.initiated:
            return
        self.initiated = True

    def generate_input_img(self, img):
        """
        :param img: cv2.imread [:, :, ::-1]
        :return:
        """
        return img

    def calculate_prediction(self, i1, i2, scale=1.0):
        rife_instance = self.device[random.randrange(0, len(self.device))]
        if self.ARGS.is_rife_reverse:
            mid = rife_instance.process(i2, i1)[0]
        else:
            mid = rife_instance.process(i1, i2)[0]
        return mid

    def TTA_FRAME(self, img0, img1, iter_time=2, scale=1.0):
        if iter_time != 0:
            img0 = self.calculate_prediction(img0, img1, scale)
            return self.TTA_FRAME(img0, img1, iter_time=iter_time - 1, scale=scale)
        else:
            return img0

    def inference(self, img0, img1, scale=1.0, iter_time=2):
        if self.tta == 0:
            return self.TTA_FRAME(img0, img1, 1, scale)
        elif self.tta == 1:  # side_vector
            LX = self.TTA_FRAME(img0, img1, iter_time, scale)
            RX = self.TTA_FRAME(img1, img0, iter_time, scale)
            return self.TTA_FRAME(LX, RX, 1, scale)
        elif self.tta == 2:  # mid_vector
            mid = self.TTA_FRAME(img0, img1, 1, scale)
            LX = self.TTA_FRAME(img0, mid, iter_time, scale)
            RX = self.TTA_FRAME(img1, mid, iter_time, scale)
            return self.TTA_FRAME(LX, RX, 1, scale)
        else:  # mix_vector
            mid = self.TTA_FRAME(img0, img1, 1, scale)
            LX = self.TTA_FRAME(img0, mid, iter_time, scale)
            RX = self.TTA_FRAME(img1, mid, iter_time, scale)
            m1 = self.TTA_FRAME(LX, RX, 1, scale)
            LX = self.TTA_FRAME(img0, img1, iter_time, scale)
            RX = self.TTA_FRAME(img1, img0, iter_time, scale)
            m2 = self.TTA_FRAME(LX, RX, 1, scale)
            return self.TTA_FRAME(m1, m2, 1, scale)

    def _make_n_inference(self, i1, i2, scale, n):
        mid = self.inference(i1, i2, iter_time=self.ARGS.rife_tta_iter)
        if n == 1:
            return [mid]
        first_half = self._make_n_inference(i1, mid, scale, n=n // 2)
        second_half = self._make_n_inference(mid, i2, scale, n=n // 2)
        if n % 2:
            return [*first_half, mid, *second_half]
        else:
            return [*first_half, *second_half]

    # def get_auto_scale(self, img0, img1):
    #     return self.ARGS.rife_exp

    def generate_n_interp(self, img1, img2, n, scale, debug=False):
        if debug:
            output_gen = list()
            for i in range(n):
                output_gen.append(img1)
            return output_gen
        img1 = self.generate_input_img(img1)
        img2 = self.generate_input_img(img2)
        interp_gen = self._make_n_inference(img1, img2, scale, n)
        return interp_gen

    def run(self):
        pass


if __name__ == "__main__":
    pass
