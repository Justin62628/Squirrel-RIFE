# coding: utf-8
import datetime
import logging
import os
import re
import shutil
import threading
import time
from configparser import ConfigParser, NoOptionError, NoSectionError
from queue import Queue

import cv2
import numpy as np


class EncodePresetAssemply:
    encoder = {
        "CPU": {
            "H264, 8bit": ["slow", "ultrafast", "fast", "medium", "veryslow", "placebo", ],
            "H264, 10bit": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "H265, 8bit": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "H265, 10bit": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "ProRes, 422": ["hq", "4444", "4444xq"],
            "ProRes, 444": ["hq", "4444", "4444xq"],
        },
        "NVENC": {"H264, 8bit": ["slow", "medium", "fast", "hq", "bd", "llhq", "loseless"],
                  "H265, 8bit": ["slow", "medium", "fast", "hq", "bd", "llhq", "loseless"],
                  "H265, 10bit": ["slow", "medium", "fast", "hq", "bd", "llhq", "loseless"], },
        "QSV": {"H264, 8bit": ["slow", "fast", "medium", "veryslow", ],
                "H265, 8bit": ["slow", "fast", "medium", "veryslow", ],
                "H265, 10bit": ["slow", "fast", "medium", "veryslow", ], },

    }
    preset = {
        "HEVC": {
            "x265": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "NVENC": ["slow", "medium", "fast", "hq", "bd", "llhq", "loseless"],
            "QSV": ["slow", "fast", "medium", "veryslow", ],
        },
        "H264": {
            "x264": ["slow", "ultrafast", "fast", "medium", "veryslow", "placebo", ],
            "NVENC": ["slow", "medium", "fast", "hq", "bd", "llhq", "loseless"],
            "QSV": ["slow", "fast", "medium", "veryslow", ],
        },
        "ProRes": ["hq", "4444", "4444xq"]
    }
    pixfmt = {
        "HEVC": {
            "x265": ["yuv420p10le", "yuv420p", "yuv422p", "yuv444p", "yuv422p10le", "yuv444p10le", "yuv420p12le",
                     "yuv422p12le", "yuv444p12le"],
            "NVENC": ["p010le", "yuv420p", "yuv444p", ],
            "QSV": ["yuv420p", "p010le", ],
        },
        "H264": {
            "x264": ["yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p10le", "yuv444p10le", ],
            "NVENC": ["yuv420p", "yuv444p"],
            "QSV": ["yuv420p", ],  # TODO Seriously? QSV Not supporting p010le?
        },
        "ProRes": ["yuv422p10le", "yuv444p10le"]
    }


class CommandResult:
    def __init__(self, command, output_path="output.txt"):
        self.command = command
        self.output_path = output_path
        pass

    def execute(self, ):
        os.system(f"{self.command} > {Utils().fillQuotation(self.output_path)} 2>&1")
        with open(self.output_path, "r", encoding="UTF-8") as tool_read:
            content = tool_read.read()
        return content


class DefaultConfigParser(ConfigParser):
    """
    自定义参数提取
    """
    def get(self, section, option, fallback=None, raw=False):
        try:
            d = self._unify_values(section, None)
        except NoSectionError:
            if fallback is None:
                raise
            else:
                return fallback
        option = self.optionxform(option)
        try:
            value = d[option]
        except KeyError:
            if fallback is None:
                raise NoOptionError(option, section)
            else:
                return fallback

        if type(value) == str and not len(str(value)):
            return fallback

        if type(value) == str and value in ["false", "true"]:
            if value == "false":
                return False
            return True

        return value


class Utils:
    def __init__(self):
        self.resize_param = (480, 270)
        self.crop_param = (0, 0, 0, 0)
        pass

    def fillQuotation(self, string):
        if string[0] != '"':
            return f'"{string}"'

    def get_logger(self, name, log_path, debug=False):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger_formatter = logging.Formatter(f'%(asctime)s - %(module)s - %(lineno)s - %(levelname)s - %(message)s')
        log_path = os.path.join(log_path, "log")  # private dir for logs
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        logger_path = os.path.join(log_path,
                                   f"{datetime.datetime.now().date()}.txt")
        txt_handler = logging.FileHandler(logger_path)

        txt_handler.setFormatter(logger_formatter)

        console_handler = logging.StreamHandler()
        if debug:
            txt_handler.setLevel(level=logging.DEBUG)
            console_handler.setLevel(level=logging.DEBUG)
        else:
            txt_handler.setLevel(level=logging.INFO)
            console_handler.setLevel(level=logging.INFO)
        console_handler.setFormatter(logger_formatter)

        logger.addHandler(console_handler)
        logger.addHandler(txt_handler)
        return logger

    def make_dirs(self, dir_lists, rm=False):
        for d in dir_lists:
            if rm and os.path.exists(d):
                shutil.rmtree(d)
                continue
            if not os.path.exists(d):
                os.mkdir(d)
        pass

    def gen_next(self, gen: iter):
        try:
            return next(gen)
        except StopIteration:
            return None

    def generate_prebuild_map(self, exp, req):
        """
        For Inference duplicate frames removal
        :return:
        """
        I_step = 1 / (2 ** exp)
        IL = [x * I_step for x in range(1, 2 ** exp)]
        N_step = 1 / (req + 1)
        NL = [x * N_step for x in range(1, req + 1)]
        KPL = []
        for x1 in NL:
            min = 1
            kpt = 0
            for x2 in IL:
                value = abs(x1 - x2)
                if value < min:
                    min = value
                    kpt = x2
            KPL.append(IL.index(kpt))
        return KPL

    def clean_parsed_config(self, args: dict) -> dict:
        for a in args:
            if args[a] in ["false", "true"]:
                if args[a] == "false":
                    args[a] = False
                else:
                    args[a] = True
                continue
            try:
                tmp = float(args[a])
                try:
                    if not tmp - int(args[a]):
                        tmp = int(args[a])
                except ValueError:
                    pass
                args[a] = tmp
                continue
            except ValueError:
                pass
            if not len(args[a]):
                print(f"Warning: Find Empty Args at '{a}'")
                args[a] = ""
        return args
        pass

    def check_pure_img(self, img1):
        if np.var(img1) == 0:
            return True
        return False

    def get_norm_img(self, img1, resize=True):
        if resize:
            img1 = cv2.resize(img1, self.resize_param, interpolation=cv2.INTER_LINEAR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img1 = cv2.equalizeHist(img1)  # 进行直方图均衡化
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # _, img1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img1

    def get_norm_img_diff(self, img1, img2, resize=True) -> float:
        """
        Normalize Difference
        :param resize:
        :param img1: cv2
        :param img2: cv2
        :return: float
        """
        img1 = self.get_norm_img(img1, resize)
        img2 = self.get_norm_img(img2, resize)
        # h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        diff = cv2.absdiff(img1, img2).mean()
        return diff

    def get_filename(self, path):
        if not os.path.isfile(path):
            return os.path.basename(path)
        return os.path.splitext(os.path.basename(path))[0]

    def rm_edge(self, img):
        """
        return img info of edges
        :param img:
        :return:
        """
        gray = img
        x = gray.shape[1]
        y = gray.shape[0]

        if np.var(gray) == 0:
            """pure image, like white or black"""
            return 0, y, 0, x

        # if np.mean(self.crop_param) != 0:
        #     return self.crop_param

        edges_x = []
        edges_y = []
        edges_x_up = []
        edges_y_up = []
        edges_x_down = []
        edges_y_down = []
        edges_x_left = []
        edges_y_left = []
        edges_x_right = []
        edges_y_right = []

        for i in range(x):
            for j in range(y):
                if int(gray[j][i]) > 10:
                    edges_x_left.append(i)
                    edges_y_left.append(j)
            if len(edges_x_left) != 0 or len(edges_y_left) != 0:
                break

        for i in range(x):
            for j in range(y):
                if int(gray[j][x - i - 1]) > 10:
                    edges_x_right.append(i)
                    edges_y_right.append(j)
            if len(edges_x_right) != 0 or len(edges_y_right) != 0:
                break

        for j in range(y):
            for i in range(x):
                if int(gray[j][i]) > 10:
                    edges_x_up.append(i)
                    edges_y_up.append(j)
            if len(edges_x_up) != 0 or len(edges_y_up) != 0:
                break

        for j in range(y):
            for i in range(x):
                if int(gray[y - j - 1][i]) > 10:
                    edges_x_down.append(i)
                    edges_y_down.append(j)
            if len(edges_x_down) != 0 or len(edges_y_down) != 0:
                break

        edges_x.extend(edges_x_left)
        edges_x.extend(edges_x_right)
        edges_x.extend(edges_x_up)
        edges_x.extend(edges_x_down)
        edges_y.extend(edges_y_left)
        edges_y.extend(edges_y_right)
        edges_y.extend(edges_y_up)
        edges_y.extend(edges_y_down)

        left = min(edges_x) if len(edges_x) else 0  # 左边界
        right = max(edges_x) if len(edges_x) else x  # 右边界
        bottom = min(edges_y) if len(edges_y) else 0  # 底部
        top = max(edges_y) if len(edges_y) else y  # 顶部

        # image2 = img[bottom:top, left:right]
        self.crop_param = (bottom, top, left, right)
        return bottom, top, left, right

    def get_exp_edge(self, num):
        b = 2
        scale = 0
        while num > b ** scale:
            scale += 1
        return scale


class ImgSeqIO:
    def __init__(self, folder=None, is_read=True, thread=4, is_tool=False, start_frame=0, **kwargs):
        if folder is None or os.path.isfile(folder):
            print(f"ERROR - [IMG.IO] Invalid ImgSeq Folder: {folder}")
            return
        if start_frame in [-1, 0]:
            start_frame = 0
        self.seq_folder = folder  # + "/tmp"  # weird situation, cannot write to target dir, father dir instead
        if not os.path.exists(self.seq_folder):
            os.mkdir(self.seq_folder)
        self.frame_cnt = 0
        self.img_list = list()
        self.write_queue = Queue(maxsize=1000)
        self.thread_cnt = thread
        self.thread_pool = list()
        self.use_imdecode = False
        self.resize = (0, 0)
        self.resize_flag = False
        if "resize" in kwargs and len(kwargs["resize"]):
            self.resize = list(map(lambda x: int(x), kwargs["resize"].split("x")))
            self.resize_flag = True

        if is_tool:
            return
        if is_read:
            tmp = os.listdir(folder)
            for p in tmp:
                if os.path.splitext(p)[-1] in [".jpg", ".png", ".jpeg"]:
                    if self.frame_cnt < start_frame:
                        self.frame_cnt += 1
                        continue
                    self.img_list.append(os.path.join(self.seq_folder, p))
            print(f"INFO - [IMG.IO] Load {len(self.img_list)} frames at {start_frame}")
        else:
            png_re = re.compile("\d+\.png")
            write_png = sorted([i for i in os.listdir(self.seq_folder) if png_re.match(i)],
                               key=lambda x: int(x[:-4]), reverse=True)
            if len(write_png):
                self.frame_cnt = int(os.path.splitext(write_png[0])[0]) + 1
                print(f"INFO - update Img Cnt to {self.frame_cnt}")
            for t in range(self.thread_cnt):
                _t = threading.Thread(target=self.write_buffer, name=f"[IMG.IO] Write Buffer No.{t + 1}")
                self.thread_pool.append(_t)
            for _t in self.thread_pool:
                _t.start()
            # print(f"INFO - [IMG.IO] Set {self.seq_folder} As output Folder")

    def get_frames_cnt(self):
        return len(self.img_list)

    def read_frame(self, path):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)[:, :, ::-1].copy()
        if self.resize_flag:
            img = cv2.resize(img, (self.resize[0], self.resize[1]))
        return img

    def write_frame(self, img, path):
        if self.resize_flag:
            img = cv2.resize(img, (self.resize[0], self.resize[1]))
        cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(path)

    def nextFrame(self):
        for p in self.img_list:
            img = self.read_frame(p)
            yield img

    def write_buffer(self):
        while True:
            img_data = self.write_queue.get()
            if img_data[1] is None:
                print(f"INFO - [IMG.IO] {threading.current_thread().name}: get None, break")
                break
            self.write_frame(img_data[1], img_data[0])

    def writeFrame(self, img):
        img_path = os.path.join(self.seq_folder, f"{self.frame_cnt:0>8d}.png")
        img_path = img_path.replace("\\", "/")
        if img is None:
            for t in range(self.thread_cnt):
                self.write_queue.put((img_path, None))
            return
        self.write_queue.put((img_path, img))
        self.frame_cnt += 1
        return

    def close(self):
        for t in range(self.thread_cnt):
            self.write_queue.put(("", None))
        for _t in self.thread_pool:
            while _t.is_alive():
                time.sleep(0.2)
        # if os.path.exists(self.seq_folder):
        #     shutil.rmtree(self.seq_folder)
        return


if __name__ == "__main__":
    u = Utils()
    cp = DefaultConfigParser(allow_no_value=True)
    cp.read(r"D:\60-fps-Project\arXiv2020-RIFE-main\release\SVFI.Ft.RIFE_GUI.release.v6.2.2.A\RIFE_GUI.ini",
            encoding='utf-8')
    print(cp.get("General", "UseCUDAButton=true", 6))
    print(u.clean_parsed_config(dict(cp.items("General"))))
    #
    # check = VideoInfo("L:\Frozen\Remux\Frozen.Fever.2015.1080p.BluRay.REMUX.AVC.DTS-HD.MA.5.1-RARBG.mkv", False, img_input=True)
    # check.update_info()
    # pprint(check.get_info())
