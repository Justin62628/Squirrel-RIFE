# coding: utf-8
import datetime
import logging
import math
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
        self.Utils = Utils()
        pass

    def execute(self, ):
        os.system(f"{self.command} > {self.Utils.fillQuotation(self.output_path)} 2>&1")
        with open(self.output_path, "r") as tool_read:
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
    resize_param = (480, 270)
    crop_param = (0, 0, 0, 0)

    def __init__(self):
        self.resize_param = (480, 270)
        self.crop_param = (0, 0, 0, 0)
        pass

    @staticmethod
    def fillQuotation(string):
        if string[0] != '"':
            return f'"{string}"'

    @staticmethod
    def get_logger(name, log_path, debug=False):
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

    @staticmethod
    def make_dirs(dir_lists, rm=False):
        for d in dir_lists:
            if rm and os.path.exists(d):
                shutil.rmtree(d)
                continue
            if not os.path.exists(d):
                os.mkdir(d)
        pass

    @staticmethod
    def gen_next(gen: iter):
        try:
            return next(gen)
        except StopIteration:
            return None

    @staticmethod
    def clean_parsed_config(args: dict) -> dict:
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

    @staticmethod
    def check_pure_img(img1):
        if np.var(img1) < 10:
            return True
        return False

    @staticmethod
    def get_norm_img(img1, resize=True):
        if resize:
            img1 = cv2.resize(img1, Utils.resize_param, interpolation=cv2.INTER_LINEAR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img1 = cv2.equalizeHist(img1)  # 进行直方图均衡化
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # _, img1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img1

    @staticmethod
    def get_norm_img_diff(img1, img2, resize=True) -> float:
        """
        Normalize Difference
        :param resize:
        :param img1: cv2
        :param img2: cv2
        :return: float
        """
        img1 = Utils.get_norm_img(img1, resize)
        img2 = Utils.get_norm_img(img2, resize)
        # h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        diff = cv2.absdiff(img1, img2).mean()
        return diff

    @staticmethod
    def get_norm_img_flow(img1, img2, resize=True, flow_thres=1) -> (int, np.array):
        """
        Normalize Difference
        :param flow_thres: 光流移动像素长
        :param resize:
        :param img1: cv2
        :param img2: cv2
        :return:  (int, np.array)
        """
        prevgray = Utils.get_norm_img(img1, resize)
        gray = Utils.get_norm_img(img2, resize)
        # h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        # prevgray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # 使用Gunnar Farneback算法计算密集光流
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # 绘制线
        step = 10
        h, w = gray.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)
        line = []
        flow_cnt = 0

        for l in lines:
            if math.sqrt(math.pow(l[0][0] - l[1][0], 2) + math.pow(l[0][1] - l[1][1], 2)) > flow_thres:
                flow_cnt += 1
                line.append(l)

        cv2.polylines(prevgray, line, 0, (0, 255, 255))
        comp_stack = np.hstack((prevgray, gray))
        return flow_cnt, comp_stack

    @staticmethod
    def get_filename(path):
        if not os.path.isfile(path):
            return os.path.basename(path)
        return os.path.splitext(os.path.basename(path))[0]

    @staticmethod
    def get_exp_edge(num):
        b = 2
        scale = 0
        while num > b ** scale:
            scale += 1
        return scale

    @staticmethod
    def get_mixed_scenes(img0, img1, n):
        """
        return n-1 images
        :param img0:
        :param img1:
        :param n:
        :return:
        """
        step = 1 / n
        beta = 0
        output = list()
        for _ in range(n - 1):
            beta += step
            alpha = 1 - beta
            mix = cv2.addWeighted(img0[:, :, ::-1], alpha, img1[:, :, ::-1], beta, 0)[:, :, ::-1].copy()
            output.append(mix)
        return output


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
        if "exp" in kwargs:
            self.exp = kwargs["exp"]
        else:
            self.exp = 0
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
        return len(self.img_list) * 2 ** self.exp

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
            for e in range(2 ** self.exp):
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
