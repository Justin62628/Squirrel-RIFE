import os
import traceback
import warnings

from ncnn.rife import rife_ncnn_vulkan

warnings.filterwarnings("ignore")
# Utils = Utils()
raw = rife_ncnn_vulkan.raw


class RifeInterpolation(rife_ncnn_vulkan.RIFE):
    def __init__(self, __args: dict):
        uhd_mode = True if __args["exp"] < 1 else False
        super().__init__(__args["ncnn_gpu"], os.path.basename(__args["selected_model"]),
                         tta_mode=__args.get("tta_mode", False), uhd_mode=uhd_mode, num_threads=4)
        self.initiated = False
        self.args = {}
        if __args is not None:
            """Update Args"""
            self.args = __args
        else:
            raise NotImplementedError("Args not sent in")

        self.device = None
        self.model = None
        self.model_path = ""

    def initiate_rife(self, __args=None):
        if self.initiated:
            return
        self.initiated = True

    def __make_inference(self, i1, i2, scale, exp):
        # i1 = self.generate_input_img(img1)
        # i2 = self.generate_input_img(img2)
        if self.args["reverse"]:
            mid = self.process(i1, i2)[0]
        else:
            mid = self.process(i2, i1)[0]
        # del i1, i2
        # mid = cv2.cvtColor(numpy.asarray(mid), cv2.COLOR_RGB2BGR)
        if exp == 1:
            return [mid]
        first_half = self.__make_inference(i1, mid, scale, exp=exp - 1)
        second_half = self.__make_inference(mid, i2, scale, exp=exp - 1)
        return [*first_half, mid, *second_half]

    def __make_n_inference(self, i1, i2, scale, n):
        if self.args["reverse"]:
            mid = self.process(i1, i2)[0]
        else:
            mid = self.process(i2, i1)[0]
        # del i1, i2
        # mid = cv2.cvtColor(numpy.asarray(mid), cv2.COLOR_RGB2BGR)
        if n == 1:
            return [mid]
        first_half = self.__make_n_inference(i1, mid, scale, n=n // 2)
        second_half = self.__make_n_inference(mid, i2, scale, n=n // 2)
        if n % 2:
            return [*first_half, mid, *second_half]
        else:
            return [*first_half, *second_half]

    def generate_input_img(self, img):
        """
        :param img: cv2.imread [:, :, ::-1]
        :return:
        """
        return img
        try:
            pass
            # image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(img)
            traceback.print_exc()
            raise e

    def generate_interp(self, img1, img2, exp, scale, n=None, debug=False, test=False):
        """

        :param test:
        :param img1: cv2.imread
        :param img2:
        :param exp:
        :param scale:
        :param n:
        :param debug:
        :return: list of interp cv2 image
        """
        if debug:
            output_gen = list()
            if n is not None:
                dup = n
            else:
                dup = 2 ** exp - 1
            for i in range(dup):
                output_gen.append(img1)
            return output_gen
        img1 = self.generate_input_img(img1)
        img2 = self.generate_input_img(img2)
        if n is not None:
            interp_gen = self.__make_n_inference(img1, img2, scale, n=n)
        else:
            interp_gen = self.__make_inference(img1, img2, scale, exp=exp)

        return interp_gen

    def generate_n_interp(self, img1, img2, n, scale, debug=False, test=False):
        if debug:
            output_gen = list()
            for i in range(n):
                output_gen.append(img1)
            return output_gen
        img1 = self.generate_input_img(img1)
        img2 = self.generate_input_img(img2)
        interp_gen = self.__make_n_inference(img1, img2, scale, n)
        return interp_gen

    def run(self):
        pass


if __name__ == "__main__":
    pass
