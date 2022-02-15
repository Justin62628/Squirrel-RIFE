# coding: utf-8
import datetime
import functools
import glob
import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import signal
import string
import subprocess
import threading
import time
import traceback
from collections import deque
from configparser import ConfigParser, NoOptionError, NoSectionError
from queue import Queue

import cv2
import numpy as np
import psutil
from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from sklearn import linear_model

from Utils.StaticParameters import appDir, SupportFormat, HDR_STATE, RGB_TYPE, RT_RATIO, SR_TILESIZE_STATE, LUTS_TYPE, \
    RIFE_TYPE
from skvideo.utils import check_output


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


class ArgumentManager:
    """
    For OLS's arguments input management
    """
    app_id = 1692080
    pro_dlc_id = [1718750]

    community_qq = 264023742
    professional_qq = 1054016374

    """Release Version Control"""
    is_steam = True
    is_free = False
    is_release = False
    traceback_limit = 0 if is_release else None
    gui_version = "3.10.21"
    version_tag = f"{gui_version}-alpha " \
                  f"{'Professional' if not is_free else 'Community'} - {'Steam' if is_steam else 'Retail'}"
    ols_version = "7.4.24"
    """ 发布前改动以上参数即可 """

    update_log = f"""
    {version_tag}
    Update Log
    - Update RIFE model output mode in output path
    """

    path_len_limit = 230
    overtime_reminder_queue = Queue()
    overtime_reminder_ids = dict()

    screen_w = 1920
    screen_h = 1080

    def __init__(self, args: dict):
        self.app_dir = args.get("app_dir", appDir)

        self.config = args.get("config", "")
        self.input = args.get("input", "")
        self.output_dir = args.get("output_dir", "")
        self.task_id = args.get("task_id", "")
        self.gui_inputs = args.get("gui_inputs", "")
        self.input_fps = args.get("input_fps", 0)
        self.target_fps = args.get("target_fps", 0)
        self.input_ext = ".mp4"
        self.output_ext = args.get("output_ext", ".mp4")
        self.is_img_input = False
        self.is_img_output = args.get("is_img_output", False)
        self.is_output_only = args.get("is_output_only", True)
        self.is_save_audio = args.get("is_save_audio", True)
        self.input_start_point = args.get("input_start_point", None)
        self.input_end_point = args.get("input_end_point", None)
        if self.input_start_point == "00:00:00":
            self.input_start_point = None
        if self.input_end_point == "00:00:00":
            self.input_end_point = None
        self.output_chunk_cnt = args.get("output_chunk_cnt", 0)
        self.interp_start = args.get("interp_start", 0)
        self.risk_resume_mode = args.get("risk_resume_mode", False)

        self.is_no_scdet = args.get("is_no_scdet", False)
        self.is_scdet_mix = args.get("is_scdet_mix", False)
        self.use_scdet_fixed = args.get("use_scdet_fixed", False)
        self.is_scdet_output = args.get("is_scdet_output", False)
        self.scdet_threshold = args.get("scdet_threshold", 12)
        self.scdet_fixed_max = args.get("scdet_fixed_max", 40)
        self.scdet_flow_cnt = args.get("scdet_flow_cnt", 4)
        self.scdet_mode = args.get("scdet_mode", 0)
        self.remove_dup_mode = args.get("remove_dup_mode", 0)
        self.remove_dup_threshold = args.get("remove_dup_threshold", 0.1)
        self.use_dedup_sobel = args.get("use_dedup_sobel", False)

        self.use_manual_buffer = args.get("use_manual_buffer", False)
        self.manual_buffer_size = args.get("manual_buffer_size", 1)

        resize_width = Tools.get_plural(args.get("resize_width", 0))
        resize_height = Tools.get_plural(args.get("resize_height", 0))
        self.resize_param = [resize_width, resize_height]  # resize parameter, 输出分辨率参数
        self.resize_exp = args.get("resize_exp", 1)

        self.transfer_ratio = RT_RATIO(args.get("transfer_ratio_index", 0))

        crop_width = args.get("crop_width", 0)
        crop_height = args.get("crop_height", 0)
        self.crop_param = [crop_width, crop_height]  # crop parameter, 裁切参数

        self.use_sr = args.get("use_sr", False)
        self.use_sr_algo = args.get("use_sr_algo", "")
        self.use_sr_model = args.get("use_sr_model", "")
        self.use_sr_mode = args.get("use_sr_mode", "")
        self.sr_tilesize_mode = SR_TILESIZE_STATE(args.get("sr_tilesize_mode", 0))
        self.sr_tilesize = args.get("sr_tilesize", 200)
        if self.sr_tilesize_mode != SR_TILESIZE_STATE.CUSTOM:
            self.sr_tilesize = SR_TILESIZE_STATE.get_tilesize(self.sr_tilesize_mode)
        self.sr_realCUGAN_tilemode = args.get("sr_realcugan_tilemode", 2)  # default: h, w both /2
        self.sr_module_exp = args.get("sr_module_exp", 0)
        self.use_realesr_fp16 = args.get("use_realesr_fp16", False)

        self.render_gap = args.get("render_gap", 1000)
        self.use_crf = args.get("use_crf", True)
        self.use_bitrate = args.get("use_bitrate", False)
        self.render_crf = args.get("render_crf", 16)
        self.render_bitrate = args.get("render_bitrate", 90)
        self.render_encode_format = args.get("render_encoder", "")
        self.render_encoder = args.get("render_hwaccel_mode", "")
        self.render_encoder_preset = args.get("render_encoder_preset", "fast")
        self.use_render_avx512 = args.get("use_render_avx512", False)
        self.use_render_zld = args.get("use_render_zld", False)  # enables zero latency decode
        self.render_nvenc_preset = args.get("render_hwaccel_preset", "")
        self.use_hwaccel_decode = args.get("use_hwaccel_decode", True)
        self.use_manual_encode_thread = args.get("use_manual_encode_thread", False)
        self.render_encode_thread = args.get("render_encode_thread", 16)
        self.use_render_encoder_default_preset = args.get("use_render_encoder_default_preset", False)
        self.is_encode_audio = args.get("is_encode_audio", False)
        self.is_quick_extract = args.get("is_quick_extract", True)
        self.hdr_cube_mode = LUTS_TYPE(args.get("hdr_cube_index", 0))
        self.is_16bit_workflow = args.get("is_16bit_workflow", False)
        self.hdr_mode = args.get("hdr_mode", 0)
        if self.hdr_mode == 0:  # AUTO
            self.hdr_mode = HDR_STATE(-2)
        else:
            self.hdr_mode = HDR_STATE(self.hdr_mode)

        if not self.is_16bit_workflow:  # change to 8bit
            RGB_TYPE.change_8bit(True)

        self.render_ffmpeg_customized = args.get("render_ffmpeg_customized", "").strip('"').strip("'")
        self.is_no_concat = args.get("is_no_concat", False)
        self.use_fast_denoise = args.get("use_fast_denoise", False)
        self.gif_loop = args.get("gif_loop", True)
        self.is_render_slow_motion = args.get("is_render_slow_motion", False)
        self.render_slow_motion_fps = args.get("render_slow_motion_fps", 0)
        self.use_deinterlace = args.get("use_deinterlace", False)
        self.is_keep_head = args.get("is_keep_head", False)

        self.use_ncnn = args.get("use_ncnn", False)
        self.ncnn_thread = args.get("ncnn_thread", 4)
        self.ncnn_gpu = args.get("ncnn_gpu", 0)
        self.rife_tta_mode = args.get("rife_tta_mode", 0)
        self.rife_tta_iter = args.get("rife_tta_iter", 1)
        self.use_evict_flicker = args.get("use_evict_flicker", False)
        self.use_rife_fp16 = args.get("use_rife_fp16", False)
        self.rife_scale = args.get("rife_scale", 1.0)
        self.rife_model_dir = args.get("rife_model_dir", "")
        self.rife_model = args.get("rife_model", "")
        self.rife_model_name = args.get("rife_model_name", "")
        self.rife_exp = args.get("rife_exp", 1.0)
        self.rife_cuda_cnt = args.get("rife_cuda_cnt", 0)
        self.is_rife_reverse = args.get("is_rife_reverse", False)
        self.use_specific_gpu = args.get("use_specific_gpu", 0)  # !
        self.use_rife_auto_scale = args.get("use_rife_auto_scale", False)
        self.rife_interp_before_resize = args.get("rife_interp_before_resize", 0)
        self.use_rife_forward_ensemble = args.get("use_rife_forward_ensemble", False)
        self.use_rife_multi_cards = args.get("use_rife_multi_cards", False)
        self.rife_interlace_inference = args.get("rife_interlace_inference", 0)
        self.rife_layer_connect_mode = args.get("rife_layer_connect_mode", 0)

        self.debug = args.get("debug", False)
        self.multi_task_rest = args.get("multi_task_rest", False)
        self.multi_task_rest_interval = args.get("multi_task_rest_interval", 1)
        self.after_mission = args.get("after_mission", False)
        self.force_cpu = args.get("force_cpu", False)
        self.expert_mode = args.get("expert_mode", True)
        self.preview_args = args.get("preview_args", False)
        self.is_rude_exit = args.get("is_rude_exit", True)
        self.is_no_dedup_render = args.get("is_no_dedup_render", True)
        self.pos = args.get("pos", "")
        self.size = args.get("size", "")

        """OLS Params"""
        self.concat_only = args.get("concat_only", False)
        self.extract_only = args.get("extract_only", False)
        self.render_only = args.get("render_only", False)
        self.version = args.get("version", "0.0.0 beta")

        """Preview Imgs"""
        self.is_preview_imgs = args.get("is_preview_imgs", True)

    @staticmethod
    def is_empty_overtime_task_queue():
        return ArgumentManager.overtime_reminder_queue.empty()

    @staticmethod
    def put_overtime_task(_over_time_reminder_task):
        ArgumentManager.overtime_reminder_queue.put(_over_time_reminder_task)

    @staticmethod
    def get_overtime_task():
        return ArgumentManager.overtime_reminder_queue.get()

    @staticmethod
    def update_screen_size(w: int, h: int):
        ArgumentManager.screen_h = h
        ArgumentManager.screen_w = w

    @staticmethod
    def get_screen_size():
        """

        :return: h, w
        """
        return ArgumentManager.screen_h, ArgumentManager.screen_w


class Tools:
    resize_param = (300, 300)
    crop_param = (0, 0, 0, 0)

    def __init__(self):
        pass

    @staticmethod
    def fillQuotation(string):
        if string[0] != '"':
            return f'"{string}"'
        else:
            return string

    @staticmethod
    def get_logger(name, log_path, debug=False):
        logger = logging.getLogger(name)
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        logger_formatter = logging.Formatter(f'%(asctime)s - %(module)s - %(lineno)s - %(levelname)s - %(message)s')
        if ArgumentManager.is_release:
            logger_formatter = logging.Formatter(f'%(asctime)s - %(module)s - %(levelname)s - %(message)s')

        log_path = os.path.join(log_path, "log")  # private dir for logs
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        logger_path = os.path.join(log_path,
                                   f"{datetime.datetime.now().date()}.log")

        txt_handler = logging.FileHandler(logger_path, encoding='utf-8')
        txt_handler.setFormatter(logger_formatter)
        console_handler = logging.StreamHandler()
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
    def dict2Args(d: dict):
        args = []
        for key in d.keys():
            args.append(key)
            if len(d[key]):
                args.append(d[key])
        return args

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
                print(f"Warning: Find Empty Arguments at '{a}'")
                args[a] = ""
        return args
        pass

    @staticmethod
    def check_pure_img(img1):
        try:
            if np.var(img1[::4, ::4, 0]) < 10:
                return True
            return False
        except:
            return False

    @staticmethod
    def check_non_ascii(s: str):
        ascii_set = set(string.printable)
        _s = ''.join(filter(lambda x: x in ascii_set, s))
        if s != _s:
            return True
        else:
            return False

    @staticmethod
    def get_u1_from_u2_img(img: np.ndarray):
        if img.dtype in (np.uint16, np.dtype('>u2'), np.dtype('<u2')):
            img = img.view(np.uint8)[:, :, ::2]  # default to uint8
        return img

    @staticmethod
    def get_norm_img(img1, resize=True):
        img1 = Tools.get_u1_from_u2_img(img1)
        if img1.shape[0] > 1000:
            img1 = img1[::4, ::4, 0]
        else:
            img1 = img1[::2, ::2, 0]
        if resize and img1.shape[0] > Tools.resize_param[0]:
            img1 = cv2.resize(img1, Tools.resize_param)
        img1 = cv2.equalizeHist(img1)  # 进行直方图均衡化
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
        if (img1[::4, ::4, 0] == img2[::4, ::4, 0]).all():
            return 0
        img1 = Tools.get_norm_img(img1, resize)
        img2 = Tools.get_norm_img(img2, resize)
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
        prevgray = Tools.get_norm_img(img1, resize)
        gray = Tools.get_norm_img(img2, resize)
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

        def normalize_img(img):
            if img.dtype in (np.uint16, np.dtype('>u2'), np.dtype('<u2')):
                img = img.astype(np.uint16)
            return img

        img0 = normalize_img(img0)
        img1 = normalize_img(img1)
        for _ in range(n - 1):
            beta += step
            alpha = 1 - beta
            mix = cv2.addWeighted(img0[:, :, ::-1], alpha, img1[:, :, ::-1], beta, 0)[:, :, ::-1].copy()
            output.append(mix)
        return output

    @staticmethod
    def get_fps(path: str):
        """
        Get Fps from path
        :param path:
        :return: fps float
        """
        if not os.path.isfile(path):
            return 0
        try:
            if not os.path.isfile(path):
                input_fps = 0
            else:
                input_stream = cv2.VideoCapture(path)
                input_fps = input_stream.get(cv2.CAP_PROP_FPS)
            return input_fps
        except Exception:
            return 0

    @staticmethod
    def get_existed_chunks(project_dir: str):
        chunk_paths = []
        for chunk_p in os.listdir(project_dir):
            if re.match("chunk-\d+-\d+-\d+\.\w+", chunk_p):
                chunk_paths.append(chunk_p)

        if not len(chunk_paths):
            return chunk_paths, -1, -1

        chunk_paths.sort()
        last_chunk = chunk_paths[-1]
        chunk_cnt, last_frame = re.findall('chunk-(\d+)-\d+-(\d+).*?', last_chunk)[0]
        return chunk_paths, int(chunk_cnt), int(last_frame)

    @staticmethod
    def get_custom_cli_params(_command: str):
        command_params = _command.split('||')
        command_dict = dict()
        param = ""
        for command in command_params:
            command = command.strip().replace("\\'", "'").replace('\\"', '"').strip('\\')
            if command.startswith("-"):
                if param != "":
                    command_dict.update({param: ""})
                param = command
            else:
                command_dict.update({param: command})
                param = ""
        if param != "":  # final note
            command_dict.update({param: ""})
        return command_dict

    @staticmethod
    def popen(args: str):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags = subprocess.CREATE_NEW_CONSOLE | subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        p = subprocess.Popen(args, startupinfo=startupinfo)
        return p

    @staticmethod
    def md5(d: str):
        m = hashlib.md5(d.encode(encoding='utf-8'))
        return m.hexdigest()

    @staticmethod
    def get_pids():
        """
        get key-value of pids
        :return: dict {pid: pid-name}
        """
        pid_dict = {}
        pids = psutil.pids()
        for pid in pids:
            try:
                p = psutil.Process(pid)
                pid_dict[pid] = p.name()
            except psutil.NoSuchProcess:
                pass
            # print("pid-%d,pname-%s" %(pid,p.name()))
        return pid_dict

    @staticmethod
    def kill_svfi_related():
        pids = Tools.get_pids()
        for pid, pname in pids.items():
            if pname in ['ffmpeg.exe', 'ffprobe.exe', 'one_line_shot_args.exe', 'QSVEncC64.exe', 'NVEncC64.exe',
                         'SvtHevcEncApp.exe', 'SvtVp9EncApp.exe', 'SvtAv1EncApp.exe']:
                try:
                    os.kill(pid, signal.SIGABRT)
                except Exception as e:
                    traceback.print_exc()
                print(f"Warning: Kill Process before exit: {pname}")

    @staticmethod
    def get_plural(i: int):
        if i > 0:
            if i % 2 != 0:
                return i + 1
        return i


class ImageIO:
    def __init__(self, logger, folder, start_frame=0, exp=2, **kwargs):
        """
        Image I/O Operation Base Class
        :param logger:
        :param folder:
        :param start_frame:
        :param exp:
        :param kwargs:
        """
        self.logger = logger
        if folder is None or os.path.isfile(folder):
            raise OSError(f"Invalid Image Sequence Folder: {folder}")
        self.folder = folder  # + "/tmp"  # weird situation, cannot write to target dir, father dir instead
        os.makedirs(self.folder, exist_ok=True)
        self.start_frame = start_frame
        self.exp = exp
        self.frame_cnt = 0
        self.img_list = list()

    def get_write_start_frame(self):
        raise NotImplementedError()

    def get_frames_cnt(self):
        """
        Get Frames Cnt with EXP
        :return:
        """
        return len(self.img_list) * 2 ** self.exp

    def read_frame(self, path):
        raise NotImplementedError()

    def write_frame(self, img, path):
        raise NotImplementedError()

    def nextFrame(self):
        raise NotImplementedError()

    def write_buffer(self):
        raise NotImplementedError()

    def writeFrame(self, img):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()


class ImageRead(ImageIO):
    def __init__(self, logger, folder, start_frame=0, exp=2, resize=(0, 0), **kwargs):
        super().__init__(logger, folder, start_frame, exp)
        self.resize = resize
        self.resize_flag = all(self.resize)

        if self.start_frame != -1:
            self.start_frame = int(self.start_frame / 2 ** self.exp)

        img_list = os.listdir(self.folder)
        img_list.sort()
        for p in img_list:
            fn, ext = os.path.splitext(p)
            if ext.lower() in SupportFormat.img_inputs:
                if self.frame_cnt < start_frame:
                    self.frame_cnt += 1  # update frame_cnt
                    continue  # do not read frame until reach start_frame img
                self.img_list.append(os.path.join(self.folder, p))
        self.logger.debug(f"Load {len(self.img_list)} frames at {self.frame_cnt}")

    def get_frames_cnt(self):
        """
        Get Frames Cnt with EXP
        :return:
        """
        return len(self.img_list) * 2 ** self.exp

    def read_frame(self, path):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)[:, :, ::-1].copy()
        if self.resize_flag:
            img = cv2.resize(img, (self.resize[0], self.resize[1]), interpolation=cv2.INTER_AREA)
        return img

    def nextFrame(self):
        for p in self.img_list:
            img = self.read_frame(p)
            for e in range(2 ** self.exp):
                yield img

    def close(self):
        return


class ImageWrite(ImageIO):
    def __init__(self, logger, folder, start_frame=0, exp=2, resize=(0, 0), output_ext='.png', thread_cnt=4,
                 is_tool=False, **kwargs):
        super().__init__(logger, folder, start_frame, exp)
        self.resize = resize
        self.resize_flag = all(self.resize)
        self.output_ext = output_ext
        self.thread_cnt = thread_cnt
        self.thread_pool = list()
        if self.start_frame != -1:
            self.start_frame = int(self.start_frame / (2 ** self.exp))
        self.write_queue = Queue()
        self.frame_cnt = start_frame
        if not is_tool:
            self.logger.debug(f"Start Writing {self.output_ext} at No. {self.frame_cnt}")
            for t in range(self.thread_cnt):
                _t = threading.Thread(target=self.write_buffer, name=f"IMG.IO Write Buffer No.{t + 1}")
                self.thread_pool.append(_t)
            for _t in self.thread_pool:
                _t.start()

    def get_write_start_frame(self):
        """
        Get Start Frame when start_frame is at its default value
        :return:
        """
        img_list = list()
        for f in os.listdir(self.folder):  # output folder
            fn, ext = os.path.splitext(f)
            if ext in SupportFormat.img_inputs:
                img_list.append(fn)
        if not len(img_list):
            return 0
        return len(img_list)

    def write_buffer(self):
        while True:
            img_data = self.write_queue.get()
            if img_data[1] is None:
                self.logger.debug(f"{threading.current_thread().name}: get None, break")
                break
            self.write_frame(img_data[1], img_data[0])

    def write_frame(self, img, path):
        if self.resize_flag:
            if img.shape[1] != self.resize[0] or img.shape[0] != self.resize[1]:
                img = cv2.resize(img, (self.resize[0], self.resize[1]), interpolation=cv2.INTER_AREA)
        cv2.imencode(self.output_ext, cv2.cvtColor(img.astype(RGB_TYPE.DTYPE), cv2.COLOR_RGB2BGR))[1].tofile(path)

    def writeFrame(self, img):
        img_path = os.path.join(self.folder, f"{self.frame_cnt:0>8d}{self.output_ext}")
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
        return


class SuperResolutionBase:
    """
    超分抽象类
    """

    def __init__(
            self,
            gpuid=0,
            model="",
            tta_mode=False,
            num_threads=1,
            scale: float = 2,
            noise=0,
            tilesize=0,
    ):
        self.tilesize = tilesize
        self.noise = noise
        self.scale = scale
        self.num_threads = num_threads
        self.tta_mode = tta_mode
        self.model = model
        self.gpuid = gpuid

    def process(self, im):
        return im

    def svfi_process(self, img):
        """
        SVFI 用于超分的接口
        :param img:
        :return:
        """
        return img


class VideoFrameInterpolationBase:
    def __init__(self, __args: ArgumentManager, logger: logging.Logger):
        self.initiated = False
        self.logger = logger
        self.args = {}
        if __args is not None:
            """Update Args"""
            self.args = __args
        else:
            raise NotImplementedError("Args not sent in")

        self.split_w = 1
        self.split_h = 1
        self.is_interlace_inference = self.args.rife_interlace_inference > 0
        if self.args.rife_interlace_inference == 1:  # w split to 2
            self.split_w, self.split_h = 2, 1
        elif self.args.rife_interlace_inference == 2:  # w split to 2, h split to 2
            self.split_w, self.split_h = 2, 2
        elif self.args.rife_interlace_inference == 3:  # w split to 4, h split to 2
            self.split_w, self.split_h = 4, 2
        elif self.args.rife_interlace_inference == 4:  # w split to 4, h split to 4
            self.split_w, self.split_h = 4, 4
        elif self.args.rife_interlace_inference == 5:  # w split to 8, h split to 4
            self.split_w, self.split_h = 8, 4
        elif self.args.rife_interlace_inference == 6:  # w split to 8, h split to 8
            self.split_w, self.split_h = 8, 8

        if self.args.use_rife_multi_cards:
            self.split_w, self.split_h = 2, 1  # override previous settings

    def initiate_algorithm(self):
        raise NotImplementedError()

    def split_input_image(self, img0):
        pieces = list()
        for x in range(self.split_w):
            for y in range(self.split_h):
                pieces.append(img0[y::self.split_h, x::self.split_w, :])
        return pieces

    def sew_input_pieces(self, pieces, h, w, c):
        """

        :param pieces: list or tuple
        :param h:
        :param w:
        :param c:
        :return:
        """
        background = np.zeros((h, w, c))
        p_i = 0
        for x in range(self.split_w):
            for y in range(self.split_h):
                background[y::self.split_h, x::self.split_w, :] = pieces[p_i]
                p_i += 1
        return background

    def generate_n_interp(self, img0, img1, n, scale, debug=False) -> list:
        interp_list = list()
        for i in range(n):
            interp_list.append(img0)
        return interp_list

    def get_auto_scale(self, img1, img2) -> float:
        def mean_scale(HR, targetHeight, targetWidth):
            h, w, c = HR.shape
            BChannel = HR[:, :, 0]
            GChannel = HR[:, :, 1]
            RChannel = HR[:, :, 2]
            ystep = h / targetHeight
            xstep = w / targetWidth
            bgr_map = np.zeros((targetHeight, targetWidth, c), np.float32)
            for y in range(targetHeight - 1):
                for x in range(targetWidth - 1):
                    B = BChannel[int(y * ystep):int(ystep * (y + 1)) - 1,
                        int(x * xstep):int(xstep * (x + 1)) - 1].mean()
                    G = GChannel[int(y * ystep):int(ystep * (y + 1)) - 1,
                        int(x * xstep):int(xstep * (x + 1)) - 1].mean()
                    R = RChannel[int(y * ystep):int(ystep * (y + 1)) - 1,
                        int(x * xstep):int(xstep * (x + 1)) - 1].mean()
                    bgr_map[y, x] = [B, G, R]
            return bgr_map.astype(np.uint8)

        img1 = Tools.get_u1_from_u2_img(img1)
        img2 = Tools.get_u1_from_u2_img(img2)
        i0 = mean_scale(img1, 8, 8)
        i1 = mean_scale(img2, 8, 8)
        scale_list = [1.0, 0.5, 0.25]
        dis_ssim = (1 - compare_ssim(i0, i1, multichannel=True)) * 100
        scale_max = len(scale_list)
        if dis_ssim >= scale_max:
            return scale_list[scale_max - 1]
        else:
            return scale_list[int(dis_ssim)]

    def _make_n_inference(self, img1, img2, scale, n):
        raise NotImplementedError("Abstract")

    @staticmethod
    def get_model_version(model_path: str) -> RIFE_TYPE:
        """
        CUDA Model Only
        :param model_path:
        :return:
        """
        model_path = model_path.lower()
        if 'abme_best' in model_path:
            current_model_index = RIFE_TYPE.ABME
        elif 'anytime' in model_path:  # prior than anime
            current_model_index = RIFE_TYPE.RIFEvAnyTime  # RIFEv New from Master Zhe
        elif 'anime' in model_path:
            if any([i in model_path for i in ['sharp', 'smooth']]):
                current_model_index = RIFE_TYPE.RIFEv2  # RIFEv2
            else:  # RIFEv6, anime_training
                current_model_index = RIFE_TYPE.RIFEv6
        elif 'official' in model_path:
            if '2.' in model_path:
                current_model_index = RIFE_TYPE.RIFEv2  # RIFEv2.x
            elif '3.' in model_path:
                current_model_index = RIFE_TYPE.RIFEv3
            elif 'v6' in model_path:
                current_model_index = RIFE_TYPE.RIFEv6
            elif '4.' in model_path:
                current_model_index = RIFE_TYPE.RIFEv4
            else:  # RIFEv7
                current_model_index = RIFE_TYPE.RIFEv7
        elif 'xvfi' in model_path:
            current_model_index = RIFE_TYPE.XVFI
        else:
            current_model_index = RIFE_TYPE.RIFEv2  # default RIFEv2
        return current_model_index

    def run(self):
        raise NotImplementedError("Abstract")


class Hdr10PlusProcessor:
    def __init__(self, logger: logging, project_dir: str, render_gap: int,
                 interp_times: int, hdr10_metadata: dict, **kwargs):
        """

        :param logger:
        :param project_dir:
        :param args:
        :param kwargs:
        """
        self.logger = logger
        self.project_dir = project_dir
        self.interp_times = interp_times
        self.render_gap = render_gap
        self.hdr10_metadata_dict = hdr10_metadata
        self.hdr10plus_metadata_4interp = []
        self._initialize()

    def _initialize(self):
        if not len(self.hdr10_metadata_dict):
            return
        hdr10plus_metadata = self.hdr10_metadata_dict.copy()
        hdr10plus_metadata = hdr10plus_metadata['SceneInfo']
        hdr10plus_metadata.sort(key=lambda x: int(x['SceneFrameIndex']))
        current_index = -1
        for m in hdr10plus_metadata:
            for j in range(int(self.interp_times)):
                current_index += 1
                _m = m.copy()
                _m['SceneFrameIndex'] = current_index
                self.hdr10plus_metadata_4interp.append(_m)
        return

    def get_hdr10plus_metadata_path_at_point(self, start_frame: 0):
        """

        :return: path of metadata json to use immediately
        """
        if not len(self.hdr10plus_metadata_4interp) or start_frame < 0 or start_frame > len(
                self.hdr10plus_metadata_4interp):
            return ""
        if start_frame + self.render_gap < len(self.hdr10plus_metadata_4interp):
            hdr10plus_metadata = self.hdr10plus_metadata_4interp[start_frame:start_frame + self.render_gap]
        else:
            hdr10plus_metadata = self.hdr10plus_metadata_4interp[start_frame:]
        hdr10plus_metadata_path = os.path.join(self.project_dir,
                                               f'hdr10plus_metadata_{start_frame}_{start_frame + self.render_gap}.json')
        json.dump(hdr10plus_metadata, open(hdr10plus_metadata_path, 'w'))
        return hdr10plus_metadata_path.replace('/', '\\')


class DoviProcessor:
    def __init__(self, concat_input: str, original_input: str, project_dir: str, interp_times: int, logger: logging,
                 **kwargs):
        """

        :param concat_input:
        :param original_input:
        :param project_dir:
        :param interp_times:
        :param logger:
        :param kwargs:
        """
        self.concat_input = concat_input
        self.input = original_input
        self.project_dir = project_dir
        self.interp_times = interp_times
        self.logger = logger
        self.ffmpeg = "ffmpeg"
        self.ffprobe = "ffprobe"
        self.dovi_tool = "dovi_tool"
        self.dovi_muxer = "dovi_muxer"
        self.video_info, self.audio_info = {}, {}
        self.dovi_profile = 8
        self.get_input_info()
        self.concat_video_stream = Tools.fillQuotation(
            os.path.join(self.project_dir, f"concat_video.{self.video_info['codec_name']}"))
        self.dv_video_stream = Tools.fillQuotation(
            os.path.join(self.project_dir, f"dv_video.{self.video_info['codec_name']}"))
        self.dv_audio_stream = ""
        self.dv_before_rpu = Tools.fillQuotation(os.path.join(self.project_dir, f"dv_before_rpu.rpu"))
        self.rpu_edit_json = os.path.join(self.project_dir, 'rpu_duplicate_edit.json')
        self.dv_after_rpu = Tools.fillQuotation(os.path.join(self.project_dir, f"dv_after_rpu.rpu"))
        self.dv_injected_video_stream = Tools.fillQuotation(
            os.path.join(self.project_dir, f"dv_injected_video.{self.video_info['codec_name']}"))
        self.dv_concat_output_path = Tools.fillQuotation(f'{os.path.splitext(self.concat_input)[0]}_dovi.mp4')

    def get_input_info(self):
        check_command = (f'{self.ffprobe} -v error '
                         f'-show_streams -print_format json '
                         f'{Tools.fillQuotation(self.input)}')
        result = check_output(shlex.split(check_command))
        try:
            stream_info = json.loads(result)['streams']  # select first video stream as input
        except Exception as e:
            self.logger.warning(f"Parse Video Info Failed: {result}")
            raise e
        for stream in stream_info:
            if stream['codec_type'] == 'video':
                self.video_info = stream
                break
        for stream in stream_info:
            if stream['codec_type'] == 'audio':
                self.audio_info = stream
                break
        self.logger.info(f"DV Processing [0] - Information gathered, Start Extracting")
        pass

    def run(self):
        try:
            self.split_video2va()
            self.extract_rpu()
            self.modify_rpu()
            self.inject_rpu()
            result = self.mux_va()
            return result
        except Exception:
            self.logger.error("Dovi Conversion Failed")
            raise Exception

    def split_video2va(self):
        audio_map = {'eac3': 'ec3'}
        command_line = (
            f"{self.ffmpeg} -i {Tools.fillQuotation(self.concat_input)} -c:v copy -an -f {self.video_info['codec_name']} {self.concat_video_stream} -y")
        check_output(command_line)
        if len(self.audio_info):
            audio_ext = self.audio_info['codec_name']
            if self.audio_info['codec_name'] in audio_map:
                audio_ext = audio_map[self.audio_info['codec_name']]
            self.dv_audio_stream = Tools.fillQuotation(os.path.join(self.project_dir, f"dv_audio.{audio_ext}"))
            command_line = (
                f"{self.ffmpeg} -i {Tools.fillQuotation(self.input)} -c:a copy -vn -f {self.audio_info['codec_name']} {self.dv_audio_stream} -y")
            check_output(command_line)
        self.logger.info(f"DV Processing [1] - Video and Audio track Extracted, start RPU Extracting")

        pass

    def extract_rpu(self):
        command_line = (
            f"{self.ffmpeg} -loglevel panic -i {Tools.fillQuotation(self.input)} -c:v copy "
            f'-vbsf {self.video_info["codec_name"]}_mp4toannexb -f {self.video_info["codec_name"]} - | {self.dovi_tool} extract-rpu --rpu-out {self.dv_before_rpu} -')
        check_output(command_line, shell=True)
        self.logger.info(f"DV Processing [2] - Dolby Vision RPU layer extracted, start RPU Modifying")
        pass

    def modify_rpu(self):
        command_line = (
            f"{self.dovi_tool} info -i {self.dv_before_rpu} -f 0")
        rpu_info = check_output(command_line)
        try:
            rpu_info = re.findall('dovi_profile: (.*?),\s.*?offset: (\d+), len: (\d+)', rpu_info.decode())[0]
        except Exception as e:
            self.logger.warning(f"Parse Video Info Failed: {rpu_info}")
            raise e
        self.dovi_profile, dovi_offset, dovi_len = map(lambda x: int(x), rpu_info)
        if 'nb_frames' in self.video_info:
            dovi_len = int(self.video_info['nb_frames'])
        elif 'r_frame_rate' in self.video_info and 'duration' in self.video_info:
            frame_rate = self.video_info['r_frame_rate'].split('/')
            frame_rate = int(frame_rate[0]) / int(frame_rate[1])
            dovi_len = int(frame_rate * float(self.video_info['duration']))

        duplicate_list = []
        for frame in range(dovi_len):
            duplicate_list.append({'source': frame, 'offset': frame, 'length': self.interp_times - 1})
        edit_dict = {'duplicate': duplicate_list}
        with open(self.rpu_edit_json, 'w') as w:
            json.dump(edit_dict, w)
        command_line = (
            f"{self.dovi_tool} editor -i {self.dv_before_rpu} -j {Tools.fillQuotation(self.rpu_edit_json)} -o {self.dv_after_rpu}")
        check_output(command_line)
        self.logger.info(
            f"DV Processing [3] - RPU layer modified with duplication {self.interp_times - 1} at length {dovi_len}, start RPU Injecting")

        pass

    def inject_rpu(self):
        command_line = (
            f"{self.dovi_tool} inject-rpu -i {self.concat_video_stream} --rpu-in {self.dv_after_rpu} -o {self.dv_injected_video_stream}")
        check_output(command_line)
        self.logger.info(f"DV Processing [4] - RPU layer Injected to interpolated stream, start muxing")

        pass

    def mux_va(self):
        audio_path = ''
        if len(self.audio_info):
            audio_path = f"-i {Tools.fillQuotation(self.dv_audio_stream)}"
        command_line = f"{self.dovi_muxer} -i {self.dv_injected_video_stream} {audio_path} -o {self.dv_concat_output_path} " \
                       f"--dv-profile {self.dovi_profile} --mpeg4-comp-brand mp42,iso6,isom,msdh,dby1 --overwrite --dv-bl-compatible-id 1"
        check_output(command_line)
        self.logger.info(
            f"DV Processing [5] - interpolated stream muxed to destination: {Tools.get_filename(self.dv_concat_output_path)}")
        self.logger.info(f"DV Processing FINISHED")
        return True


class VideoInfoProcessor:
    StandardBt2020ColorData = {'color_range': 'tv', 'color_transfer': 'smpte2084',
                               'color_space': 'bt2020nc', 'color_primaries': 'bt2020'}

    def __init__(self, input_file: str, logger, project_dir: str, interp_exp=0, hdr_cube_mode=LUTS_TYPE.NONE, **kwargs):
        """

        :param input_file:
        :param logger:
        :param project_dir:
        :param interp_exp:
        :param kwargs:
        """
        self.input_file = input_file
        self.logger = logger
        self.is_img_input = not os.path.isfile(self.input_file)
        self.ffmpeg = "ffmpeg"
        self.ffprobe = "ffprobe"
        self.hdr10_parser = "hdr10plus_parser"
        self.hdr_mode = HDR_STATE.NOT_CHECKED
        self.project_dir = project_dir
        self.color_data_tag = [('color_range', ''),
                               ('color_space', ''),
                               ('color_transfer', ''),
                               ('color_primaries', '')]

        self.interp_exp = interp_exp
        self.fps = 0  # float
        self.frame_size = (0, 0)  # width, height, float
        self.first_img_frame_size = (0, 0)
        self.frames_cnt = 0  # int
        self.duration = 0
        self.video_info = {'color_range': '', 'color_transfer': '',
                           'color_space': '', 'color_primaries': ''}
        self.hdr_cube_mode = hdr_cube_mode
        self.audio_info = dict()
        self.hdr10plus_metadata_path = None
        self.check_single_image_input()
        self.update_info()

    def check_single_image_input(self):
        """
        This is quite amending actually
        By replacing input image into a folder
        :return:
        """
        ext_split = os.path.splitext(self.input_file)
        if ext_split[1].lower() not in SupportFormat.img_inputs:
            return
        self.is_img_input = True
        os.makedirs(ext_split[0], exist_ok=True)
        shutil.copy(self.input_file, os.path.join(ext_split[0], os.path.split(ext_split[0])[1] + ext_split[1]))
        self.input_file = ext_split[0]

    def update_hdr_mode(self):
        self.hdr_mode = HDR_STATE.NONE  # default to be bt709
        if any([i in str(self.video_info) for i in ['dv_profile', 'DOVI']]):
            self.hdr_mode = HDR_STATE.DOLBY_VISION  # Dolby Vision
            self.logger.warning("Dolby Vision Content Detected")
            return
        if "color_transfer" not in self.video_info:
            self.logger.warning("Not Find Color Transfer Characteristics")
            return

        color_trc = self.video_info["color_transfer"]
        if "smpte2084" in color_trc or "bt2020" in color_trc:
            self.hdr_mode = HDR_STATE.CUSTOM_HDR  # hdr(normal)
            self.logger.warning("HDR Content Detected")
            if any([i in str(self.video_info).lower()]
                   for i in ['mastering-display', "mastering display", "content light level metadata"]):
                """Could be HDR10+"""
                self.hdr_mode = HDR_STATE.HDR10
                self.hdr10plus_metadata_path = os.path.join(self.project_dir, "hdr10plus_metadata.json")
                check_command = (f'{self.ffmpeg} -loglevel panic -i {Tools.fillQuotation(self.input_file)} -c:v copy '
                                 f'-vbsf hevc_mp4toannexb -f hevc - | '
                                 f'{self.hdr10_parser} -o {Tools.fillQuotation(self.hdr10plus_metadata_path)} -')
                try:
                    check_output(shlex.split(check_command), shell=True)
                except Exception:
                    self.logger.warning("Failed to extract HDR10+ data")
                    self.logger.error(traceback.format_exc(limit=ArgumentManager.traceback_limit))
                if len(self.getInputHdr10PlusMetadata()):
                    self.logger.warning("HDR10+ Content Detected")
                    self.hdr_mode = HDR_STATE.HDR10_PLUS  # hdr10+

        elif "arib-std-b67" in color_trc:
            self.hdr_mode = HDR_STATE.HLG  # HLG
            self.logger.warning("HLG Content Detected")

        if self.hdr_cube_mode != LUTS_TYPE.NONE:
            if self.hdr_mode == HDR_STATE.NONE:  # BT709
                self.logger.warning(f"One Click HDR Applying: {self.hdr_cube_mode.name}")
                self.hdr_mode = HDR_STATE.HDR10
                self.video_info.update(self.StandardBt2020ColorData)
            else:
                self.logger.warning(f"HDR Content Detected, Neglect Applying One Click HDR")

        pass

    def update_frames_info_ffprobe(self):
        check_command = (f'{self.ffprobe} -v error '
                         f'-show_streams -print_format json '
                         f'{Tools.fillQuotation(self.input_file)}')
        result = check_output(shlex.split(check_command))
        try:
            stream_info = json.loads(result)['streams']  # select first video stream as input
        except Exception as e:
            self.logger.warning(f"Parse Video Info Failed: {result}")
            raise e
        """Select first stream"""
        for stream in stream_info:
            if stream['codec_type'] == 'video':
                self.video_info = stream
                break
        for stream in stream_info:
            if stream['codec_type'] == 'audio':
                self.audio_info = stream
                break

        for cdt in self.color_data_tag:
            if cdt[0] not in self.video_info:
                self.video_info[cdt[0]] = cdt[1]

        self.update_hdr_mode()

        # update frame size info
        if 'width' in self.video_info and 'height' in self.video_info:
            self.frame_size = (int(self.video_info['width']), int(self.video_info['height']))

        if "r_frame_rate" in self.video_info:
            fps_info = self.video_info["r_frame_rate"].split('/')
            self.fps = int(fps_info[0]) / int(fps_info[1])
            self.logger.info(f"Auto Find FPS in r_frame_rate: {self.fps}")
        else:
            self.logger.warning("Auto Find FPS Failed")
            return False

        if "nb_frames" in self.video_info:
            self.frames_cnt = int(self.video_info["nb_frames"])
            self.logger.info(f"Auto Find frames cnt in nb_frames: {self.frames_cnt}")
        elif "duration" in self.video_info:
            self.duration = float(self.video_info["duration"])
            self.frames_cnt = round(float(self.duration * self.fps))
            self.logger.info(f"Auto Find Frames Cnt by duration deduction: {self.frames_cnt}")
        else:
            self.logger.warning("FFprobe Not Find Frames Cnt")
            return False
        return True

    def update_frames_info_cv2(self):
        if self.is_img_input:
            return
        video_input = cv2.VideoCapture(self.input_file)
        try:
            if not self.fps:
                self.fps = video_input.get(cv2.CAP_PROP_FPS)
            if not self.frames_cnt:
                self.frames_cnt = video_input.get(cv2.CAP_PROP_FRAME_COUNT)
            if not self.duration:
                self.duration = self.frames_cnt / self.fps
            if self.frame_size == (0, 0):
                self.frame_size = (
                    round(video_input.get(cv2.CAP_PROP_FRAME_WIDTH)), round(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        except Exception:
            self.logger.error(traceback.format_exc(limit=ArgumentManager.traceback_limit))

    def update_info(self):
        if self.is_img_input:
            seq_list = []
            for ext in SupportFormat.img_inputs:
                glob_expression = os.path.join(self.input_file, f"*{ext}")
                seq_list.extend(glob.glob(glob_expression))
            if not len(seq_list):
                raise OSError("Input Dir does not contain any valid images(png, jpg, tiff only)")
            self.frames_cnt = len(seq_list) * 2 ** self.interp_exp
            img = cv2.imdecode(np.fromfile(os.path.join(self.input_file, seq_list[0]), dtype=np.uint8), 1)[:, :,
                  ::-1].copy()
            h, w, _ = img.shape
            self.first_img_frame_size = (
                w, h)  # for img input, do not question their size for non-monotonous resolution input
            return
        self.update_frames_info_ffprobe()
        self.update_frames_info_cv2()

    def getInputColorInfo(self) -> dict:
        return dict(map(lambda x: (x[0], self.video_info.get(x[0], x[1])), self.color_data_tag))
        pass

    def getInputHdr10PlusMetadata(self) -> dict:
        """

        :return: dict
        """
        if self.hdr10plus_metadata_path is not None and os.path.exists(self.hdr10plus_metadata_path):
            try:
                hdr10plus_metadata = json.load(open(self.hdr10plus_metadata_path, 'r'))
                return hdr10plus_metadata
            except json.JSONDecodeError:
                self.logger.error("Unable to Decode HDR10Plus Metadata")
        return {}


class TransitionDetection_ST:
    def __init__(self, project_dir, scene_queue_length, scdet_threshold=50, no_scdet=False,
                 use_fixed_scdet=False, fixed_max_scdet=50, scdet_output=False):
        """

        :param project_dir: 项目所在文件夹
        :param scene_queue_length:
        :param scdet_threshold:
        :param no_scdet: 无转场检测
        :param use_fixed_scdet: 使用固定转场识别
        :param fixed_max_scdet: 固定转场识别模式下的阈值
        :param scdet_output:
        """
        self.scdet_output = scdet_output
        self.scdet_threshold = scdet_threshold
        self.use_fixed_scdet = use_fixed_scdet
        if self.use_fixed_scdet:
            self.scdet_threshold = fixed_max_scdet
        self.scdet_cnt = 0
        self.scene_stack_len = scene_queue_length
        self.absdiff_queue = deque(maxlen=self.scene_stack_len)  # absdiff队列
        self.black_scene_queue = deque(maxlen=self.scene_stack_len)  # 黑场开场特判队列
        self.scene_checked_queue = deque(maxlen=self.scene_stack_len // 2)  # 已判断的转场absdiff特判队列
        self.utils = Tools
        self.dead_thres = 80
        self.born_thres = 2
        self.img1 = None
        self.img2 = None
        self.scdet_cnt = 0
        self.scene_dir = os.path.join(project_dir, "scene")
        if not os.path.exists(self.scene_dir):
            os.mkdir(self.scene_dir)
        self.scene_stack = Queue(maxsize=scene_queue_length)
        self.no_scdet = no_scdet
        self.scedet_info = {"scene": 0, "normal": 0, "dup": 0, "recent_scene": -1}

    def __check_coef(self):
        reg = linear_model.LinearRegression()
        reg.fit(np.array(range(len(self.absdiff_queue))).reshape(-1, 1), np.array(self.absdiff_queue).reshape(-1, 1))
        return reg.coef_, reg.intercept_

    def __check_var(self):
        coef, intercept = self.__check_coef()
        coef_array = coef * np.array(range(len(self.absdiff_queue))).reshape(-1, 1) + intercept
        diff_array = np.array(self.absdiff_queue)
        sub_array = diff_array - coef_array
        return sub_array.var() ** 0.65

    def __judge_mean(self, diff):
        var_before = self.__check_var()
        self.absdiff_queue.append(diff)
        var_after = self.__check_var()
        if var_after - var_before > self.scdet_threshold and diff > self.born_thres:
            """Detect new scene"""
            self.scdet_cnt += 1
            self.save_scene(
                f"diff: {diff:.3f}, var_a: {var_before:.3f}, var_b: {var_after:.3f}, cnt: {self.scdet_cnt}")
            self.absdiff_queue.clear()
            self.scene_checked_queue.append(diff)
            return True
        else:
            if diff > self.dead_thres:
                self.absdiff_queue.clear()
                self.scdet_cnt += 1
                self.save_scene(f"diff: {diff:.3f}, Dead Scene, cnt: {self.scdet_cnt}")
                self.scene_checked_queue.append(diff)
                return True
            return False

    def end_view(self):
        self.scene_stack.put(None)
        while True:
            scene_data = self.scene_stack.get()
            if scene_data is None:
                return
            title = scene_data[0]
            scene = scene_data[1]
            self.save_scene(title)

    def save_scene(self, title):
        if not self.scdet_output:
            return
        try:
            comp_stack = np.hstack((self.img1, self.img2))
            comp_stack = cv2.resize(comp_stack, (960, int(960 * comp_stack.shape[0] / comp_stack.shape[1])),
                                    interpolation=cv2.INTER_AREA)
            cv2.putText(comp_stack,
                        title,
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (int(RGB_TYPE.SIZE), 0, 0))
            if "pure" in title.lower():
                path = f"{self.scdet_cnt:08d}_pure.png"
            elif "band" in title.lower():
                path = f"{self.scdet_cnt:08d}_band.png"
            else:
                path = f"{self.scdet_cnt:08d}.png"
            path = os.path.join(self.scene_dir, path)
            if os.path.exists(path):
                os.remove(path)
            cv2.imencode('.png', cv2.cvtColor(comp_stack, cv2.COLOR_RGB2BGR))[1].tofile(path)
            return
            # TODO Preview Add Scene Preview
            cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow(title, cv2.cvtColor(comp_stack, cv2.COLOR_BGR2RGB))
            cv2.moveWindow(title, 500, 500)
            cv2.resizeWindow(title, 1920, 540)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            traceback.print_exc()

    def check_scene(self, _img1, _img2, add_diff=False, no_diff=False, use_diff=-1, **kwargs) -> bool:
        """
        Check if current scene is scene
        :param use_diff:
        :param _img2:
        :param _img1:
        :param add_diff:
        :param no_diff: check after "add_diff" mode
        :return: 是转场则返回真
        """

        if self.no_scdet:
            return False

        img1 = _img1.copy()
        img2 = _img2.copy()
        self.img1 = img1
        self.img2 = img2

        if use_diff != -1:
            diff = use_diff
        else:
            diff = self.utils.get_norm_img_diff(self.img1, self.img2)

        if self.use_fixed_scdet:
            if diff < self.scdet_threshold:
                return False
            else:
                self.scdet_cnt += 1
                self.save_scene(f"diff: {diff:.3f}, Fix Scdet, cnt: {self.scdet_cnt}")
                return True

        """检测开头黑场"""
        if diff < 0.001:
            """000000"""
            if self.utils.check_pure_img(img1):
                self.black_scene_queue.append(0)
            return False
        elif len(self.black_scene_queue) and np.mean(self.black_scene_queue) == 0:
            """检测到00000001"""
            self.black_scene_queue.clear()
            self.scdet_cnt += 1
            self.save_scene(f"diff: {diff:.3f}, Pure Scene, cnt: {self.scdet_cnt}")
            # self.save_flow()
            return True

        # Check really hard scene at the beginning
        if diff > self.dead_thres:
            self.absdiff_queue.clear()
            self.scdet_cnt += 1
            self.save_scene(f"diff: {diff:.3f}, Dead Scene, cnt: {self.scdet_cnt}")
            self.scene_checked_queue.append(diff)
            return True

        if len(self.absdiff_queue) < self.scene_stack_len or add_diff:
            if diff not in self.absdiff_queue:
                self.absdiff_queue.append(diff)
            return False

        """Duplicate Frames Special Judge"""
        if no_diff and len(self.absdiff_queue):
            self.absdiff_queue.pop()
            if not len(self.absdiff_queue):
                return False

        """Judge"""
        return self.__judge_mean(diff)

    def update_scene_status(self, recent_scene, scene_type: str):
        """更新转场检测状态"""
        self.scedet_info[scene_type] += 1
        if scene_type == "scene":
            self.scedet_info["recent_scene"] = recent_scene

    def get_scene_status(self):
        return self.scedet_info


class TransitionDetection:
    def __init__(self, scene_queue_length, scdet_threshold=50, project_dir="", no_scdet=False,
                 use_fixed_scdet=False, fixed_max_scdet=50, remove_dup_mode=0, scdet_output=False, scdet_flow=0,
                 **kwargs):
        """
        转场检测类
        :param scdet_flow: 输入光流模式：0：2D 1：3D
        :param scene_queue_length: 转场判定队列长度
        :param fixed_scdet:
        :param scdet_threshold: （标准输入）转场阈值
        :param output: 输出
        :param no_scdet: 不进行转场识别
        :param use_fixed_scdet: 使用固定转场阈值
        :param fixed_max_scdet: 使用的最大转场阈值
        :param kwargs:
        """
        self.view = False
        self.utils = Tools
        self.scdet_cnt = 0
        self.scdet_threshold = scdet_threshold
        self.scene_dir = os.path.join(project_dir, "scene")  # 存储转场图片的文件夹路径
        if not os.path.exists(self.scene_dir):
            os.mkdir(self.scene_dir)

        self.dead_thres = 80  # 写死最高的absdiff
        self.born_thres = 3  # 写死判定为非转场的最低阈值

        self.scene_queue_len = scene_queue_length
        if remove_dup_mode in [1, 2]:
            """去除重复帧一拍二或N"""
            self.scene_queue_len = 8  # 写死

        self.flow_queue = deque(maxlen=self.scene_queue_len)  # flow_cnt队列
        self.black_scene_queue = deque(maxlen=self.scene_queue_len)  # 黑场景特判队列
        self.absdiff_queue = deque(maxlen=self.scene_queue_len)  # absdiff队列
        self.scene_stack = Queue(maxsize=self.scene_queue_len)  # 转场识别队列

        self.no_scdet = no_scdet
        self.use_fixed_scdet = use_fixed_scdet
        self.scedet_info = {"scene": 0, "normal": 0, "dup": 0, "recent_scene": -1}
        # 帧种类，scene为转场，normal为正常帧，dup为重复帧，即两帧之间的计数关系

        self.img1 = None
        self.img2 = None
        self.flow_img = None
        self.before_img = None
        if self.use_fixed_scdet:
            self.dead_thres = fixed_max_scdet

        self.scene_output = scdet_output
        if scdet_flow == 0:
            self.scdet_flow = 3
        else:
            self.scdet_flow = 1

        self.now_absdiff = -1
        self.now_vardiff = -1
        self.now_flow_cnt = -1

    def __check_coef(self):
        reg = linear_model.LinearRegression()
        reg.fit(np.array(range(len(self.flow_queue))).reshape(-1, 1), np.array(self.flow_queue).reshape(-1, 1))
        return reg.coef_, reg.intercept_

    def __check_var(self):
        """
        计算“转场”方差
        :return:
        """
        coef, intercept = self.__check_coef()
        coef_array = coef * np.array(range(len(self.flow_queue))).reshape(-1, 1) + intercept
        diff_array = np.array(self.flow_queue)
        sub_array = np.abs(diff_array - coef_array)
        return sub_array.var() ** 0.65

    def __judge_mean(self, flow_cnt, diff, flow):
        # absdiff_mean = 0
        # if len(self.absdiff_queue) > 1:
        #     self.absdiff_queue.pop()
        #     absdiff_mean = np.mean(self.absdiff_queue)

        var_before = self.__check_var()
        self.flow_queue.append(flow_cnt)
        var_after = self.__check_var()
        self.now_absdiff = diff
        self.now_vardiff = var_after - var_before
        self.now_flow_cnt = flow_cnt
        if var_after - var_before > self.scdet_threshold and diff > self.born_thres and flow_cnt > np.mean(
                self.flow_queue):
            """Detect new scene"""
            self.see_flow(
                f"flow_cnt: {flow_cnt:.3f}, diff: {diff:.3f}, before: {var_before:.3f}, after: {var_after:.3f}, "
                f"cnt: {self.scdet_cnt + 1}", flow)
            self.flow_queue.clear()
            self.scdet_cnt += 1
            self.save_flow()
            return True
        else:
            if diff > self.dead_thres:
                """不漏掉死差转场"""
                self.flow_queue.clear()
                self.see_result(f"diff: {diff:.3f}, False Alarm, cnt: {self.scdet_cnt + 1}")
                self.scdet_cnt += 1
                self.save_flow()
                return True
            # see_result(f"compare: False, diff: {diff}, bm: {before_measure}")
            self.absdiff_queue.append(diff)
            return False

    def end_view(self):
        self.scene_stack.put(None)
        while True:
            scene_data = self.scene_stack.get()
            if scene_data is None:
                return
            title = scene_data[0]
            scene = scene_data[1]
            self.see_result(title)

    def see_result(self, title):
        """捕捉转场帧预览"""
        if not self.view:
            return
        comp_stack = np.hstack((self.img1, self.img2))
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, cv2.cvtColor(comp_stack, cv2.COLOR_BGR2RGB))
        cv2.moveWindow(title, 0, 0)
        cv2.resizeWindow(title, 960, 270)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_flow(self):
        if not self.scene_output:
            return
        try:
            cv2.putText(self.flow_img,
                        f"diff: {self.now_absdiff:.2f}, vardiff: {self.now_vardiff:.2f}, flow: {self.now_flow_cnt:.2f}",
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0))
            cv2.imencode('.png', cv2.cvtColor(self.flow_img, cv2.COLOR_RGB2BGR))[1].tofile(
                os.path.join(self.scene_dir, f"{self.scdet_cnt:08d}.png"))
        except Exception:
            traceback.print_exc()
        pass

    def see_flow(self, title, img):
        """捕捉转场帧光流"""
        if not self.view:
            return
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, img)
        cv2.moveWindow(title, 0, 0)
        cv2.resizeWindow(title, 960, 270)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def check_scene(self, _img1, _img2, add_diff=False, use_diff=-1.0) -> bool:
        """
                检查当前img1是否是转场
                :param use_diff: 使用已计算出的absdiff
                :param _img2:
                :param _img1:
                :param add_diff: 仅添加absdiff到计算队列中
                :return: 是转场则返回真
                """
        img1 = _img1.copy()
        img2 = _img2.copy()

        if self.no_scdet:
            return False

        if use_diff != -1:
            diff = use_diff
        else:
            diff = self.utils.get_norm_img_diff(img1, img2)

        if self.use_fixed_scdet:
            if diff < self.dead_thres:
                return False
            else:
                self.scdet_cnt += 1
                return True

        self.img1 = img1
        self.img2 = img2

        """检测开头转场"""
        if diff < 0.001:
            """000000"""
            if self.utils.check_pure_img(img1):
                self.black_scene_queue.append(0)
            return False
        elif np.mean(self.black_scene_queue) == 0:
            """检测到00000001"""
            self.black_scene_queue.clear()
            self.scdet_cnt += 1
            self.see_result(f"absdiff: {diff:.3f}, Pure Scene Alarm, cnt: {self.scdet_cnt}")
            self.flow_img = img1
            self.save_flow()
            return True

        flow_cnt, flow = self.utils.get_norm_img_flow(img1, img2, flow_thres=self.scdet_flow)

        self.absdiff_queue.append(diff)
        self.flow_img = flow

        if len(self.flow_queue) < self.scene_queue_len or add_diff or self.utils.check_pure_img(img1):
            """检测到纯色图片，那么下一帧大概率可以被识别为转场"""
            if flow_cnt > 0:
                self.flow_queue.append(flow_cnt)
            return False

        if flow_cnt == 0:
            return False

        """Judge"""
        return self.__judge_mean(flow_cnt, diff, flow)

    def update_scene_status(self, recent_scene, scene_type: str):
        """更新转场检测状态"""
        self.scedet_info[scene_type] += 1
        if scene_type == "scene":
            self.scedet_info["recent_scene"] = recent_scene

    def get_scene_status(self):
        return self.scedet_info


class OverTimeReminderTask:
    def __init__(self, interval: float, function_name, function_warning):
        self.start_time = time.time()
        self.interval = interval
        self.function_name = function_name
        self.function_warning = function_warning
        self._is_active = True

    def is_overdue(self):
        return time.time() - self.start_time > self.interval

    def is_active(self):
        return self._is_active

    def get_msgs(self):
        return self.function_name, self.interval, self.function_warning

    def deactive(self):
        self._is_active = False


def overtime_reminder_deco(interval: int, msg_1="Function Type", msg_2="Function Warning"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _over_time_reminder_task = OverTimeReminderTask(interval, msg_1, msg_2)
            ArgumentManager.put_overtime_task(_over_time_reminder_task)
            result = func(*args, **kwargs)
            _over_time_reminder_task.deactive()
            return result

        return wrapper

    return decorator


if __name__ == "__main__":
    # u = Tools()
    # cp = DefaultConfigParser(allow_no_value=True)
    # cp.read(r"D:\60-fps-Project\arXiv2020-RIFE-main\release\SVFI.Ft.RIFE_GUI.release.v6.2.2.A\RIFE_GUI.ini",
    #         encoding='utf-8')
    # print(cp.get("General", "UseCUDAButton=true", 6))
    # print(u.clean_parsed_config(dict(cp.items("General"))))
    # dm = DoviMaker(r"D:\60-fps-Project\input_or_ref\Test\output\dolby vision-blocks_71fps_[S-0.5]_[offical_3.8]_963577.mp4", Tools.get_logger('', ''),
    #                r"D:\60-fps-Project\input_or_ref\Test\output\dolby vision-blocks_ec4c18_963577",
    #                ArgumentManager(
    #                    {'ffmpeg': r'D:\60-fps-Project\ffmpeg',
    #                     'input': r"E:\Library\Downloads\Video\dolby vision-blocks.mp4"}),
    #                int(72 / 24),
    #                )
    # dm.run()
    u = Tools()
    # print(u.get_custom_cli_params("-t -d x=\" t\":p=6 -p g='p ':z=1 -qf 3 --dd-e 233"))
    print(u.get_custom_cli_params("-x265-params loseless=1 -preset:v placebo"))
    pass
