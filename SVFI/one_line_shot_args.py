# coding: utf-8
import argparse
import datetime
import json
import math
import os
import re
import shlex
import sys
import threading
import time
import traceback
from collections import deque
from queue import Queue

import cv2
import numpy as np
import psutil
import tqdm
from PIL import Image
from sklearn import linear_model
from skvideo.io import FFmpegWriter, FFmpegReader

from Utils.utils import Utils, ImgSeqIO, DefaultConfigParser, CommandResult
from ncnn.sr.realSR.realsr_ncnn_vulkan import RealSR
from ncnn.sr.waifu2x.waifu2x_ncnn_vulkan import Waifu2x

print("INFO - ONE LINE SHOT ARGS 6.3.7 2021/5/18")
Utils = Utils()

"""设置环境路径"""
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print("Changing working dir to {0}".format(dname))
os.chdir(os.path.dirname(dname))
sys.path.append(dname)

"""输入命令行参数"""
parser = argparse.ArgumentParser(prog="#### RIFE CLI tool/补帧分步设置命令行工具 by Jeanna ####",
                                 description='Interpolation for sequences of images')
basic_parser = parser.add_argument_group(title="Basic Settings, Necessary")
basic_parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                          help="原视频/图片序列文件夹路径")
basic_parser.add_argument('-o', '--output', dest='output', type=str, required=True,
                          help="成品输出的路径，注意默认在项目文件夹")
basic_parser.add_argument("-c", '--config', dest='config', type=str, required=True, help="配置文件路径")
basic_parser.add_argument('--concat-only', dest='concat_only', action='store_true', help='只执行合并已有区块操作')
basic_parser.add_argument('--extract-only', dest='extract_only', action='store_true', help='只执行拆帧操作')

args_read = parser.parse_args()
cp = DefaultConfigParser(allow_no_value=True)  # 把SVFI GUI传来的参数格式化
cp.read(args_read.config, encoding='utf-8')
cp_items = dict(cp.items("General"))
args = Utils.clean_parsed_config(cp_items)
args.update(vars(args_read))  # update -i -o -c，将命令行参数更新到config生成的字典

"""设置可见的gpu"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if int(args["use_specific_gpu"]) != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args['use_specific_gpu']}"

"""强制使用CPU"""
if args["force_cpu"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = f""


class RifeInterpolation:
    """Rife 补帧 抽象类"""

    def __init__(self, __args):
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
        pass

    def initiate_rife(self, __args=None):
        raise NotImplementedError("Abstract")

    def __make_inference(self, img1, img2, scale, exp):
        raise NotImplementedError("Abstract")

    def __make_n_inference(self, img1, img2, scale, n):
        raise NotImplementedError("Abstract")

    def generate_padding(self, img, scale):
        raise NotImplementedError("Abstract")

    def generate_torch_img(self, img, padding):
        """
        :param img: cv2.imread [:, :, ::-1]
        :param padding:
        :return:
        """
        raise NotImplementedError("Abstract")

    def pad_image(self, img, padding):
        raise NotImplementedError("Abstract")

    def generate_interp(self, img1, img2, exp, scale, n=None, debug=False, test=False):
        """

        :param img1: cv2.imread
        :param img2:
        :param exp:
        :param scale:
        :param n:
        :param debug:
        :return: list of interp cv2 image
        """
        raise NotImplementedError("Abstract")

    def generate_n_interp(self, img1, img2, n, scale, debug=False):
        raise NotImplementedError("Abstract")

    def get_auto_scale(self, img1, img2, scale):
        raise NotImplementedError("Abstract")

    def run(self):
        raise NotImplementedError("Abstract")


class SuperResolution:
    """
    超分抽象类
    """

    def __init__(
            self,
            gpuid=0,
            model="models-cunet",
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


class SvfiWaifu(Waifu2x):
    def __init__(self, **kwargs):
        super().__init__(gpuid=0,
                         model=kwargs["model"],
                         tta_mode=False,
                         num_threads=1,
                         scale=kwargs["scale"],
                         noise=0,
                         tilesize=0, )

    def svfi_process(self, img):
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image = self.process(image)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return image


class SvfiRealSR(RealSR):
    def __init__(self, **kwargs):
        super().__init__(gpuid=0,
                         model=kwargs["model"],
                         tta_mode=False,
                         scale=kwargs["scale"],
                         tilesize=0, )

    def svfi_process(self, img):
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image = self.process(image)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return image


class PathManager:
    """
    路径管理器
    """

    def __init__(self):
        pass


class VideoInfo:
    def __init__(self, input, HDR=False, ffmpeg=None, img_input=False, strict_mode=False, **kwargs):
        self.filepath = input
        self.img_input = img_input
        self.strict_mode = strict_mode
        self.ffmpeg = "ffmpeg"
        self.ffprobe = "ffprobe"
        if ffmpeg is not None:
            self.ffmpeg = os.path.join(ffmpeg, "ffmpeg.exe")
            self.ffprobe = os.path.join(ffmpeg, "ffprobe.exe")
        if not os.path.exists(self.ffmpeg):
            self.ffmpeg = "ffmpeg"
            self.ffprobe = "ffprobe"
        self.color_info = dict()
        if HDR:  # this should not be used
            self.color_info.update({"-colorspace": "bt2020nc",
                                    "-color_trc": "smpte2084",
                                    "-color_primaries": "bt2020",
                                    "-color_range": "tv"})
        else:
            self.color_info.update({"-colorspace": "bt709",
                                    "-color_trc": "bt709",
                                    "-color_primaries": "bt709",
                                    "-color_range": "tv"})
        self.frames_cnt = 0
        self.frames_size = (0, 0)  # width, height
        self.fps = 0
        self.duration = 0
        self.video_info = dict()
        self.update_info()

    def update_frames_info_ffprobe(self):
        result = CommandResult(
            f'{self.ffprobe} -v error -show_streams -select_streams v:0 -v error '
            f'-show_entries stream=index,width,height,r_frame_rate,nb_frames,duration,'
            f'color_primaries,color_range,color_space,color_transfer -print_format json '
            f'{Utils.fillQuotation(self.filepath)}').execute()
        try:
            video_info = json.loads(result)["streams"][0]  # select first video stream as input
        except Exception as e:
            print(f"Error: Parse Video Info Failed: {result}")
            raise e
        print("\nInput Video Info:")
        self.video_info = video_info
        print(video_info)
        # update color info
        if "color_range" in video_info:
            self.color_info["-color_range"] = video_info["color_range"]
        if "color_space" in video_info:
            self.color_info["-colorspace"] = video_info["color_space"]
        if "color_transfer" in video_info:
            self.color_info["-color_trc"] = video_info["color_transfer"]
        if "color_primaries" in video_info:
            self.color_info["-color_primaries"] = video_info["color_primaries"]

        # update frame size info
        if 'width' in video_info and 'height' in video_info:
            self.frames_size = (video_info['width'], video_info['height'])

        if "r_frame_rate" in video_info:
            fps_info = video_info["r_frame_rate"].split('/')
            self.fps = int(fps_info[0]) / int(fps_info[1])
            print(f"INFO - Auto Find FPS in r_frame_rate: {self.fps}")
        else:
            print("WARNING - Auto Find FPS Failed")
            return False

        if "nb_frames" in video_info:
            self.frames_cnt = int(video_info["nb_frames"])
            print(f"INFO - Auto Find frames cnt in nb_frames: {self.frames_cnt}")
        elif "duration" in video_info:
            self.duration = float(video_info["duration"])
            self.frames_cnt = round(float(self.duration * self.fps))
            print(f"INFO - Auto Find Frames Cnt by duration deduction: {self.frames_cnt}")
        else:
            print("WARNING - FFprobe Not Find Frames Cnt")
            return False
        return True

    def update_frames_info_cv2(self):
        video_input = cv2.VideoCapture(self.filepath)
        if not self.fps:
            self.fps = video_input.get(cv2.CAP_PROP_FPS)
        if not self.frames_cnt:
            self.frames_cnt = video_input.get(cv2.CAP_PROP_FRAME_COUNT)
        if not self.duration:
            self.duration = self.frames_cnt / self.fps
        self.frames_size = (video_input.get(cv2.CAP_PROP_FRAME_WIDTH), video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def update_info(self):
        if self.img_input:
            if os.path.isfile(self.filepath):
                self.filepath = os.path.dirname(self.filepath)
            seqlist = os.listdir(self.filepath)
            self.frames_cnt = len(seqlist)
            img = cv2.imdecode(np.fromfile(os.path.join(self.filepath, seqlist[0]), dtype=np.uint8), 1)[:, :,
                  ::-1].copy()
            self.frames_size = (img.shape[1], img.shape[0])
            return
        self.update_frames_info_ffprobe()
        self.update_frames_info_cv2()

    def get_info(self):
        get_dict = {}
        get_dict.update(self.color_info)
        get_dict.update({"video_info": self.video_info})
        get_dict["fps"] = self.fps
        get_dict["size"] = self.frames_size
        get_dict["cnt"] = self.frames_cnt
        get_dict["duration"] = self.duration
        return get_dict


class TransitionDetection:
    def __init__(self, scene_stack_length, scdet_threshold=50, output="", no_scdet=False,
                 use_fixed_scdet=False, fixed_max_scdet=50, **kwargs):
        """
        转场检测类
        :param scene_stack_length: 转场判定队列长度
        :param fixed_scdet:
        :param scdet_threshold: （标准输入）转场阈值
        :param output: 输出
        :param no_scdet: 不进行转场识别
        :param use_fixed_scdet: 使用固定转场阈值
        :param fixed_max_scdet: 使用的最大转场阈值
        :param kwargs:
        """
        self.scdet_threshold = scdet_threshold
        self.scdet_cnt = 0
        self.scene_stack_len = scene_stack_length
        self.absdiff_queue = deque(maxlen=self.scene_stack_len)  # absdiff队列
        self.utils = Utils
        self.dead_thres = 80  # 写死最高的absdiff
        self.born_thres = 1  # 写死判定为非转场的最低阈值
        self.img1 = None
        self.img2 = None
        self.scdet_cnt = 0  # 识别到的转场计数
        self.scene_dir = os.path.join(os.path.dirname(output), "scene")  # 存储转场图片的文件夹路径
        self.scene_stack = Queue(maxsize=self.scene_stack_len)  # 转场识别队列
        self.no_scdet = no_scdet
        self.use_fixed_scdet = use_fixed_scdet
        self.scedet_info = {"scene": 0, "normal": 0, "dup": 0, "recent_scene": -1}
        # 帧种类，scene为转场，normal为正常帧，dup为重复帧，即两帧之间的计数关系

        if kwargs.get("remove_dup_mode", 0) in [1, 2]:
            """去除重复帧一拍二或N"""
            self.scene_stack_len = 6  # 写死

        if self.use_fixed_scdet:
            self.scdet_threshold = fixed_max_scdet

    def __check_coef(self):
        reg = linear_model.LinearRegression()
        reg.fit(np.array(range(len(self.absdiff_queue))).reshape(-1, 1), np.array(self.absdiff_queue).reshape(-1, 1))
        return reg.coef_, reg.intercept_

    def __check_var(self):
        """
        计算“转场”方差
        :return:
        """
        coef, intercept = self.__check_coef()
        coef_array = coef * np.array(range(len(self.absdiff_queue))).reshape(-1, 1) + intercept
        diff_array = np.array(self.absdiff_queue)
        sub_array = diff_array - coef_array
        return math.sqrt(sub_array.var())

    def __judge_mean(self, diff):
        var_before = self.__check_var()
        self.absdiff_queue.append(diff)
        var_after = self.__check_var()
        if abs(var_after - var_before) > self.scdet_threshold and diff > self.born_thres:
            """Detect new scene"""
            self.see_result(
                f"diff: {diff:.3f}, before: {var_before:.3f}, after: {var_after:.3f}, cnt: {self.scdet_cnt + 1}")
            self.absdiff_queue.clear()
            self.scdet_cnt += 1
            return True
        else:
            if diff > self.dead_thres:
                """不漏掉死差转场"""
                self.absdiff_queue.clear()
                self.see_result(f"diff: {diff:.3f}, False Alarm, cnt: {self.scdet_cnt + 1}")
                self.scdet_cnt += 1
                return True
            # see_result(f"compare: False, diff: {diff}, bm: {before_measure}")
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
        return
        comp_stack = np.hstack((self.img1, self.img2))
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, cv2.cvtColor(comp_stack, cv2.COLOR_BGR2RGB))
        cv2.moveWindow(title, 500, 500)
        cv2.resizeWindow(title, 1920, 540)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def check_scene(self, img1, img2, add_diff=False, no_diff=False, use_diff=-1.0) -> bool:
        """
        检查当前img1是否是转场
        :param use_diff: 使用已计算出的absdiff
        :param img2:
        :param img1:
        :param add_diff: 仅添加absdiff到计算队列中
        :param no_diff: 和add_diff配合使用，使用即时算出的diff判断img1是否是转场
        :return: 是转场则返回真
        """

        if self.no_scdet:
            return False

        if use_diff != -1:
            diff = use_diff
        else:
            diff = self.utils.get_norm_img_diff(img1, img2)

        if self.use_fixed_scdet:
            if diff < self.scdet_threshold:
                return False
            else:
                self.scdet_cnt += 1
                return True

        if diff < 0.001:
            """重复帧之间不可能是转场"""
            return False

        self.img1 = img1
        self.img2 = img2

        if len(self.absdiff_queue) < self.scene_stack_len or add_diff or self.utils.check_pure_img(img1):
            """检测到纯色图片，那么下一帧大概率可以被识别为转场"""
            self.absdiff_queue.append(diff)
            return False

        """重复帧特判"""
        if no_diff and len(self.absdiff_queue):
            self.absdiff_queue.pop()  # 弹出上一个帧
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


class InterpWorkFlow:
    def __init__(self, __args: dict, **kwargs):
        self.args = __args

        """获得补帧输出路径"""
        if os.path.isfile(self.args["output"]):
            self.project_dir = os.path.join(os.path.dirname(self.args["output"]), os.path.basename(self.args["input"]))
        else:
            self.project_dir = os.path.join(self.args["output"], Utils.get_filename(self.args["input"]))

        if not os.path.exists(self.project_dir):
            os.mkdir(self.project_dir)

        """Set Logger"""
        sys.path.append(self.project_dir)
        self.logger = Utils.get_logger("[ARGS]", self.project_dir, debug=self.args["debug"])

        self.logger.info(f"Initial New Interpolation Project: project_dir: %s, INPUT_FILEPATH: %s", self.project_dir,
                         self.args["input"])

        """Set FFmpeg"""
        self.ffmpeg = os.path.join(self.args["ffmpeg"], "ffmpeg.exe")
        self.ffplay = os.path.join(self.args["ffmpeg"], "ffplay.exe")
        if not os.path.exists(self.ffmpeg):
            self.ffmpeg = "ffmpeg"
            self.logger.warning("Not find selected ffmpeg, use default")

        """Set input output and initiate environment"""
        self.input = self.args["input"]
        self.output = self.args["output"]
        self.input_dir = os.path.join(self.project_dir, 'frames')
        self.interp_dir = os.path.join(self.project_dir, 'interp')
        self.scene_dir = os.path.join(self.project_dir, 'scenes')
        self.env = [self.input_dir, self.interp_dir, self.scene_dir]

        self.args["img_input"] = not os.path.isfile(self.input)

        """Load Interpolation Exp"""
        self.exp = self.args["exp"]

        """Get input's info"""
        self.video_info_instance = VideoInfo(**self.args)
        self.video_info = self.video_info_instance.get_info()
        if self.args["batch"] and not self.args["img_input"]:  # 检测到批处理，且输入不是文件夹，使用检测到的帧率
            self.fps = self.video_info["fps"]
        elif self.args["fps"]:
            self.fps = self.args["fps"]
        else:  # 用户有毒，未发现有效的输入帧率，用检测到的帧率
            if self.video_info["fps"] is None or not self.video_info["fps"]:
                raise OSError("Input File not valid")
            self.fps = self.video_info["fps"]

        if self.args["img_input"]:
            self.target_fps = self.args["target_fps"]
            self.args["save_audio"] = False
            # but assigned output fps will be not touched
        else:
            if self.args["target_fps"]:
                self.target_fps = self.args["target_fps"]
            else:
                self.target_fps = (2 ** self.exp) * self.fps

        """Update All Frames Count"""
        self.all_frames_cnt = int(self.video_info["duration"] * self.target_fps)

        """Crop Video"""
        self.crop_param = [0, 0]  # crop parameter, 裁切参数
        crop_param = self.args["crop"].replace("：", ":")
        if crop_param not in ["", "0", None]:
            width_black, height_black = crop_param.split(":")
            width_black = int(width_black)
            height_black = int(height_black)
            self.crop_param = [width_black, height_black]
            self.logger.info(f"Update Crop Parameters to {self.crop_param}")

        """initiation almost ready"""
        self.logger.info(
            f"Check Interpolation Source, FPS: {self.fps}, TARGET FPS: {self.target_fps}, "
            f"FRAMES_CNT: {self.all_frames_cnt}, EXP: {self.exp}")

        """RIFE Core"""
        self.rife_core = RifeInterpolation(self.args)  # 用于补帧的模块

        """Guess Memory and Render System"""
        if self.args["use_manual_buffer"]:
            # 手动指定内存占用量
            free_mem = self.args["manual_buffer_size"] * 1024
        else:
            mem = psutil.virtual_memory()
            free_mem = round(mem.free / 1024 / 1024)
        self.frames_output_size = round(free_mem / (sys.getsizeof(
            np.random.rand(3, round(self.video_info["size"][0]),
                           round(self.video_info["size"][1]))) / 1024 / 1024) * 0.8)
        if self.frames_output_size < 100:
            self.frames_output_size = 100
        self.logger.info(f"Buffer Size to {self.frames_output_size}")

        self.frames_output = Queue(maxsize=self.frames_output_size)  # 补出来的帧序列队列（消费者）
        self.rife_task_queue = Queue(maxsize=self.frames_output_size)  # 补帧任务队列（生产者）
        self.rife_thread = None  # 帧插值预处理线程（生产者）
        self.sr_module = SuperResolution()  # 超分类
        self.frame_reader = None  # 读帧的迭代器／帧生成器
        self.render_gap = self.args["render_gap"]  # 每个chunk的帧数
        self.render_thread = None  # 帧渲染器
        self.task_info = {"chunk_cnt": -1, "render": -1, "now_frame": -1}  # 有关渲染的实时信息

        """Scene Detection"""
        self.scene_detection = TransitionDetection(int(0.5 * self.fps), **self.args)

        """Duplicate Frames Removal"""
        self.dup_skip_limit = int(0.5 * self.fps) + 1  # 当前跳过的帧计数超过这个值，将结束当前判断循环

        """Main Thread Lock"""
        self.main_event = threading.Event()
        self.render_lock = threading.Event()  # 渲染锁，没有用
        self.main_event.set()

        """Set output's color info"""
        self.color_info = {}
        for k in self.video_info:
            if k.startswith("-"):
                self.color_info[k] = self.video_info[k]

        """maintain output extension"""
        self.output_ext = "." + self.args["output_ext"]
        if self.args["encoder"] == "ProRes":
            self.output_ext = ".mov"

        self.main_error = None
        self.first_hdr_check_report = True
        pass

    def generate_frame_reader(self, start_frame=-1, frame_check=False):
        """
        输入帧迭代器
        :param frame_check:
        :param start_frame:
        :return:
        """
        """If input is sequence of frames"""
        if self.args["img_input"]:
            img_io = ImgSeqIO(folder=self.input, is_read=True, start_frame=start_frame, **self.args)
            self.all_frames_cnt = img_io.get_frames_cnt()
            self.logger.info(f"Img Input, update frames count to {self.all_frames_cnt}")
            return img_io

        """If input is a video"""
        input_dict = {"-vsync": "cfr", "-hwaccel": "auto"}
        if self.args.get("start_point", None) is not None or self.args.get("end_point", None) is not None:
            """任意时段补帧"""
            time_fmt = "%H:%M:%S"
            start_point = datetime.datetime.strptime(self.args["start_point"], time_fmt)
            end_point = datetime.datetime.strptime(self.args["end_point"], time_fmt)
            if end_point > start_point:
                input_dict.update({"-ss": self.args['start_point'], "-to": self.args['end_point']})
                start_frame = -1
                clip_duration = end_point - start_point
                clip_fps = self.target_fps
                self.all_frames_cnt = round(clip_duration.total_seconds() * clip_fps)
                self.logger.info(
                    f"Update Input Range: in {self.args['start_point']} -> out {self.args['end_point']}, all_frames_cnt -> {self.all_frames_cnt}")
            else:
                self.logger.warning(f"Input Time Section change to origianl course")

        output_dict = {
            "-vframes": str(int(abs(self.all_frames_cnt * 100))),
            "-r": f"{self.target_fps}"}  # use read frames cnt to avoid ffprobe, fuck

        output_dict.update(self.color_info)

        if frame_check:
            """用以一拍二一拍N除重模式的预处理"""
            output_dict.update({"-sws_flags": "lanczos+full_chroma_inp",
                                "-s": "256x256"})
        elif len(self.args["resize"]) and not self.args["use_sr"]:
            output_dict.update({"-sws_flags": "lanczos+full_chroma_inp",
                                "-s": self.args["resize"].replace(":", "x").replace("*", "x")})

        vf_args = f"copy"
        if start_frame not in [-1, 0]:
            vf_args += f",trim=start_frame={start_frame}"

        """Quick Extraction"""
        if not self.args["quick_extract"]:
            vf_args += f",format=yuv444p10le,zscale=matrixin=input:chromal=input:cin=input,format=rgb48be,format=rgb24"

        """Update video filters"""
        output_dict["-vf"] = vf_args
        self.logger.debug(f"reader: {input_dict} {output_dict}")
        return FFmpegReader(filename=self.input, inputdict=input_dict, outputdict=output_dict)

    def generate_frame_renderer(self, output_path):
        """
        渲染帧
        :param output_path:
        :return:
        """
        params_265 = ("ref=4:rd=3:no-rect=1:no-amp=1:b-intra=1:rdoq-level=2:limit-tu=4:me=3:subme=5:"
                      "weightb=1:no-strong-intra-smoothing=1:psy-rd=2.0:psy-rdoq=1.0:no-open-gop=1:"
                      f"keyint={int(self.target_fps * 3)}:min-keyint=1:rc-lookahead=120:bframes=6:"
                      f"aq-mode=1:aq-strength=0.8:qg-size=8:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:"
                      f"deblock=-1:no-sao=1")

        def HDRChecker():
            nonlocal params_265
            if self.args["img_input"]:
                return

            if self.args.get("strict_mode", False):
                self.logger.warning("Strict Mode, Skip HDR Check")
                return

            if "color_transfer" not in self.video_info["video_info"]:
                if self.first_hdr_check_report:
                    self.logger.warning("Not Find Color Transfer\n%s" % str(self.video_info["video_info"]))
                    self.first_hdr_check_report = False
                return

            color_trc = self.video_info["video_info"]["color_transfer"]

            if "smpte2084" in color_trc or "bt2020" in color_trc:
                hdr = True
                self.args["encoder"] = "H265, 10bit"
                self.args["hwaccel_mode"] = "CPU"
                if "master-display" in str(self.video_info["video_info"]):
                    self.args["hwaccel_mode"] = "CPU"
                    params_265 += ":hdr10-opt=1:repeat-headers=1"
                    self.logger.warning("\nWARNING - Detect HDR10+ Content, Switch to CPU Render Compulsorily")
                else:
                    self.logger.warning("\nWARNING - PQ or BT2020 Content Detected, Switch to CPU Render Compulsorily")

            elif "arib-std-b67" in color_trc:
                hdr = True
                self.args["encoder"] = "H265, 10bit"
                self.args["hwaccel_mode"] = "CPU"
                self.logger.warning("\nWARNING - HLG Content Detected, Switch to CPU Render Compulsorily")
            pass

        """If output is sequence of frames"""
        if self.args["img_output"]:
            return ImgSeqIO(folder=self.output, is_read=False, **self.args)

        """HDR Check"""
        if self.first_hdr_check_report:
            HDRChecker()

        """Output Video"""
        input_dict = {"-vsync": "cfr"}

        output_dict = {"-r": f"{self.target_fps}", "-preset": self.args["preset"], }

        output_dict.update(self.color_info)

        if not self.args["img_input"]:  # TODO: Img Seq supports any fps
            input_dict.update({"-r": f"{self.target_fps}"})
        else:
            input_dict.update({"-r": f"{self.fps}"})

        """Slow motion design"""
        if self.args["slow_motion"]:
            if self.args.get("slow_motion_fps", 0):
                input_dict.update({"-r": f"{self.args['slow_motion_fps']}"})
            else:
                input_dict.update({"-r": f"{self.target_fps}"})
            output_dict.pop("-r")

        vf_args = "copy"  # debug
        output_dict.update({"-vf": vf_args})

        if self.args["use_sr"]:
            output_dict.update({"-sws_flags": "lanczos+full_chroma_inp",
                                "-s": self.args["resize"].replace(":", "x").replace("*", "x")})

        """Assign Render Codec"""
        """CRF / Bitrate Controll"""
        if self.args["hwaccel_mode"] == "CPU":
            if "H264" in self.args["encoder"]:
                output_dict.update({"-c:v": "libx264", "-preset:v": self.args["preset"]})
                if "8bit" in self.args["encoder"]:
                    output_dict.update({"-pix_fmt": "yuv420p", "-profile:v": "high",
                                        "-weightb": "1", "-weightp": "2", "-mbtree": "1", "-forced-idr": "1",
                                        "-coder": "1",
                                        "-x264-params": "keyint=200:min-keyint=1:bframes=6:b-adapt=2:no-open-gop=1:"
                                                        "ref=8:deblock='-1:-1':rc-lookahead=50:chroma-qp-offset=-2:"
                                                        "aq-mode=1:aq-strength=0.8:qcomp=0.75:me=umh:merange=24:"
                                                        "subme=10:psy-rd='1:0.1'",
                                        })
                else:
                    """10bit"""
                    output_dict.update({"-pix_fmt": "yuv420p10", "-profile:v": "high10",
                                        "-weightb": "1", "-weightp": "2", "-mbtree": "1", "-forced-idr": "1",
                                        "-coder": "1",
                                        "-x264-params": "keyint=200:min-keyint=1:bframes=6:b-adapt=2:no-open-gop=1:"
                                                        "ref=8:deblock='-1:-1':rc-lookahead=50:chroma-qp-offset=-2:"
                                                        "aq-mode=1:aq-strength=0.8:qcomp=0.75:me=umh:merange=24:"
                                                        "subme=10:psy-rd='1:0.1'",
                                        })
            elif "H265" in self.args["encoder"]:
                output_dict.update({"-c:v": "libx265", "-preset:v": self.args["preset"]})
                if "8bit" in self.args["encoder"]:
                    output_dict.update({"-pix_fmt": "yuv420p", "-profile:v": "main",
                                        "-x265-params": params_265})
                else:
                    """10bit"""
                    output_dict.update({"-pix_fmt": "yuv420p10", "-profile:v": "main10",
                                        "-x265-params": params_265})
            else:
                """ProRes"""
                output_dict.update({"-c:v": "prores_ks", "-profile:v": self.args["preset"], "-quant_mat": "hq"})
        elif self.args["hwaccel_mode"] == "NVENC":
            if "H264" in self.args["encoder"]:
                output_dict.update({f"-g": f"{int(self.target_fps * 3)}", "-c:v": "h264_nvenc", "-rc:v": "vbr_hq", })
                hwacccel_preset = self.args["hwaccel_preset"]
                if hwacccel_preset != "None":
                    output_dict.update({"-i_qfactor": "0.71", "-b_qfactor": "1.3", "-bf": "4", "-keyint_min": "1",
                                        f"-rc-lookahead": "120", "-forced-idr": "1",
                                        f"-spatial-aq": "1", "-temporal-aq": "1", "-strict_gop": "1", "-coder": "1",
                                        "-b_ref_mode": "2", })
            elif "H265" in self.args["encoder"]:
                output_dict.update({"-c:v": "hevc_nvenc", "-rc:v": "vbr_hq",
                                    f"-g": f"{int(self.target_fps * 3)}", })
                hwacccel_preset = self.args["hwaccel_preset"]
                if hwacccel_preset != "None":
                    output_dict.update({"-i_qfactor": "0.71", "-b_qfactor": "1.3", "-keyint_min": "1",
                                        f"-rc-lookahead": "120", "-forced-idr": "1", "-nonref_p": "1",
                                        "-strict_gop": "1", })
                    if hwacccel_preset == "5th":
                        output_dict.update({"-bf": "0"})
                    elif hwacccel_preset == "6th":
                        output_dict.update({"-bf": "0", "-weighted_pred": "1"})
                    elif hwacccel_preset == "7th+":
                        output_dict.update({"-bf": "4", "-temporal-aq": "1", "-b_ref_mode": "2"})
        else:
            """QSV"""
            if "H264" in self.args["encoder"]:
                output_dict.update({"-c:v": "h264_qsv",
                                    "-i_qfactor": "0.75", "-b_qfactor": "1.1",
                                    f"-rc-lookahead": "120", })
            elif "H265" in self.args["encoder"]:
                output_dict.update({"-c:v": "hevc_qsv",
                                    f"-g": f"{int(self.target_fps * 3)}", "-i_qfactor": "0.75", "-b_qfactor": "1.1",
                                    f"-look_ahead": "120", })

        if "ProRes" not in self.args["encoder"]:

            if self.args["crf"] and self.args["use_crf"]:
                if self.args["hwaccel_mode"] != "CPU":
                    hwaccel_mode = self.args["hwaccel_mode"]
                    if hwaccel_mode == "NVENC":
                        output_dict.update({"-cq:v": str(self.args["crf"])})
                    elif hwaccel_mode == "QSV":
                        output_dict.update({"-q": str(self.args["crf"])})
                else:  # CPU
                    output_dict.update({"-crf": str(self.args["crf"])})

            if self.args["bitrate"] and self.args["use_bitrate"]:
                output_dict.update({"-b:v": f'{self.args["bitrate"]}M'})
                if self.args["hwaccel_mode"] == "QSV":
                    output_dict.update({"-maxrate": "200M"})

        self.logger.debug(f"writer: {output_dict}, {input_dict}")

        """Customize FFmpeg Render Command"""
        ffmpeg_customized_command = {}
        if len(self.args["ffmpeg_customized"]):
            shlex_out = shlex.split(self.args["ffmpeg_customized"])
            if len(shlex_out) % 2 != 0:
                self.logger.warning(f"Customized FFmpeg is invalid: {self.args['ffmpeg_customized_command']}")
            else:
                for i in range(int(len(shlex_out) / 2)):
                    ffmpeg_customized_command.update({shlex_out[i * 2]: shlex_out[i * 2 + 1]})
        self.logger.debug(ffmpeg_customized_command)
        output_dict.update(ffmpeg_customized_command)

        return FFmpegWriter(filename=output_path, inputdict=input_dict, outputdict=output_dict)

    def check_chunk(self, del_chunk=False):
        """
        Get Chunk Start
        :param: del_chunk: delete all chunks existed
        :return:
        """
        chunk_list = list()
        chunk_regex = rf"chunk-[\d+].*?\{self.output_ext}"

        """获得现有区块"""
        for f in os.listdir(self.project_dir):
            if re.match(chunk_regex, f):
                if del_chunk:
                    os.remove(os.path.join(self.project_dir, f))
                else:
                    chunk_list.append(f)

        """If remove only"""
        if del_chunk:
            return 1, 0

        """Manually Prioritized"""
        if self.args["interp_start"] not in [1, 0] or self.args["chunk"] not in [1, 0]:
            return int(self.args["chunk"]), int(self.args["interp_start"])

        """Not find previous chunk"""
        if not len(chunk_list):
            return 1, 0

        """Remove last chunk(high possibility of dilapidation)"""
        chunk_list.sort(key=lambda x: int(x.split('-')[2]))

        self.logger.info("Found Previous Chunks")
        last_chunk = chunk_list[-1]  # select last chunk to assign start frames
        chunk_regex = rf"chunk-(\d+)-(\d+)-(\d+)\{self.output_ext}"
        match_result = re.findall(chunk_regex, last_chunk)[0]

        chunk = int(match_result[0])
        last_frame = int(match_result[2])
        return chunk + 1, last_frame + 1

    def render(self, chunk_cnt, render):
        """
        Render thread
        :param chunk_cnt:
        :param render:
        :return:
        """

        def rename_chunk():
            chunk_desc_path = "chunk-{:0>3d}-{:0>8d}-{:0>8d}{}".format(chunk_cnt, render, now_frame,
                                                                       self.output_ext)
            chunk_desc_path = os.path.join(self.project_dir, chunk_desc_path)
            if os.path.exists(chunk_desc_path):
                os.remove(chunk_desc_path)
            os.rename(chunk_tmp_path, chunk_desc_path)

        def check_audio_concat():
            if not self.args["save_audio"]:
                return
            """Check Input file ext"""
            output_ext = os.path.splitext(self.input)[-1]
            if output_ext not in [".mp4", ".mov", ".mkv"]:
                output_ext = self.output_ext
            if self.args["encoder"] == "ProRes":
                output_ext = ".mov"

            concat_filepath = f"{os.path.join(self.output, 'concat_test')}" + output_ext
            map_audio = f'-i "{self.input}" -map 0:v:0 -map 1:a? -map 1:s? -c:a copy -c:s copy -shortest '
            ffmpeg_command = f'{self.ffmpeg} -hide_banner -i "{chunk_tmp_path}" {map_audio} -c:v copy ' \
                             f'{Utils.fillQuotation(concat_filepath)} -y'

            self.logger.info("Start Audio Concat Test")
            os.system(ffmpeg_command)
            if not os.path.exists(concat_filepath) or not os.path.getsize(concat_filepath):
                self.logger.error(f"Concat Test Error, {output_ext}, empty output")
                self.main_error = FileExistsError("Concat Test Error, empty output, Check Output Extension!!!")
                raise FileExistsError("Concat Test Error, empty output, Check Output Extension!!!")
            self.logger.info("Audio Concat Test Success")
            os.remove(concat_filepath)

        concat_test_flag = True

        chunk_frame_cnt = 1  # number of frames of current output chunk
        chunk_tmp_path = os.path.join(self.project_dir, f"chunk-tmp{self.output_ext}")
        frame_writer = self.generate_frame_renderer(chunk_tmp_path)  # get frame renderer

        now_frame = render
        while True:
            if not self.main_event.is_set():
                self.logger.warning("Main interpolation thread Dead, break")  # 主线程已结束，这里的锁其实没用，调试用的
                frame_writer.close()
                rename_chunk()
                break

            frame_data = self.frames_output.get()
            if frame_data is None:
                frame_writer.close()
                if not self.args["img_output"]:
                    rename_chunk()
                break

            frame = frame_data[1]
            now_frame = frame_data[0]
            frame_writer.writeFrame(frame)

            chunk_frame_cnt += 1
            self.task_info.update({"chunk_cnt": chunk_cnt, "render": now_frame})  # update render info

            if not chunk_frame_cnt % self.render_gap:
                frame_writer.close()
                if concat_test_flag:
                    check_audio_concat()
                    concat_test_flag = False
                rename_chunk()
                chunk_cnt += 1
                render = now_frame + 1
                frame_writer = self.generate_frame_renderer(chunk_tmp_path)
        return

    def feed_to_render(self, frames_list: list, is_end=False):
        """
        维护输出帧数组的输入（往输出渲染线程喂帧
        :param frames_list:
        :param is_end: 是否是视频结尾
        :return:
        """
        frames_list_len = len(frames_list)

        for frame_i in range(frames_list_len):
            if frames_list[frame_i] is None:
                self.frames_output.put(None)
                self.logger.info("Put None to write_buffer in advance")
                return
            self.frames_output.put(frames_list[frame_i])  # 往输出队列（消费者）喂正常的帧
            if frame_i == frames_list_len - 1:
                if is_end:
                    self.frames_output.put(None)
                    self.logger.info("Put None to write_buffer")
                    return
        pass

    def feed_to_rife(self, now_frame: int, img0, img1, n=0, exp=0, is_end=False, add_scene=False, ):
        """
        创建任务，输出到补帧任务队列消费者
        :param now_frame:当前帧数
        :param add_scene:加入转场的前一帧（避免音画不同步和转场鬼畜）
        :param img0:
        :param img1:
        :param n:要补的帧数
        :param exp:使用指定的补帧倍率（2**exp）
        :param is_end:是否是任务结束
        :return:
        """

        scale = self.args["scale"]
        if self.args.get("auto_scale", False):
            """使用动态光流"""
            img0_ = np.float64(cv2.resize(img0, (256, 256)))
            img1_ = np.float64(cv2.resize(img1, (256, 256)))
            mse = np.mean((img0_ - img1_) ** 2)
            if mse == 0:
                return 1.0
            p = 20 * math.log10(255.0 / math.sqrt(mse))
            if p > 20:
                scale = 1.0
            elif 15 < p <= 20:
                scale = 0.5
            else:
                scale = 0.25

        self.rife_task_queue.put(
            {"now_frame": now_frame, "img0": img0, "img1": img1, "n": n, "exp": exp, "scale": scale,
             "is_end": is_end, "add_scene": add_scene})

    def rife_task_consumer(self, start_frame):
        """
        补帧队列消费者
        :param start_frame: 起始帧
        :return:
        """

        """Build SR"""
        if self.args["use_sr"]:
            input_resolution = self.video_info["size"][0] * self.video_info["size"][1]
            output_resolution = self.args.get("resize_width", 0) * self.args.get("resize_height", 0)
            resolution_rate = math.sqrt(output_resolution / input_resolution)
            if input_resolution and resolution_rate > 1:
                sr_scale = Utils.get_exp_edge(resolution_rate)
                if self.args["use_sr_algo"] == "waifu2x":
                    self.sr_module = SvfiWaifu(model=self.args["use_sr_model"], scale=sr_scale)
                elif self.args["use_sr_algo"] == "realSR":
                    self.sr_module = SvfiRealSR(model=self.args["use_sr_model"], scale=sr_scale)
                self.logger.info(
                    f"Load AI SR at {self.args['use_sr_algo']}, {self.args['use_sr_model']}, scale = {sr_scale}")
            else:
                self.logger.warning("Abort to load AI SR since Resolution Rate < 1")

        previous_cnt = start_frame
        now_frame = start_frame

        def update_progress():
            nonlocal previous_cnt, task_acquire_time, process_time
            scene_status = self.scene_detection.get_scene_status()

            render_status = self.task_info  # render status quo
            """(chunk_cnt, start_frame, end_frame, frame_cnt)"""

            pbar.set_description(
                f"Process at Chunk {render_status['chunk_cnt']:0>3d}")
            pbar.set_postfix({"R": f"{render_status['render']}", "C": f"{now_frame}",
                              "S": f"{scene_status['recent_scene']}",
                              "SC": f"{self.scene_detection.scdet_cnt}", "TAT": f"{task_acquire_time:.2f}s",
                              "PT": f"{process_time:.2f}s", "QL": f"{self.rife_task_queue.qsize()}"})
            pbar.update(now_frame - previous_cnt)
            previous_cnt = now_frame
            pass

        pbar = tqdm.tqdm(total=self.all_frames_cnt, unit="it")
        pbar.update(n=start_frame)
        pbar.unpause()
        task_acquire_time = time.time()
        process_time = time.time()
        while True:
            task_acquire_time = time.time()
            task = self.rife_task_queue.get()
            task_acquire_time = time.time() - task_acquire_time
            if task is None:
                self.feed_to_render([None], is_end=True)
                break
            """
            task = {"now_frame", "img0", "img1", "n", "exp","scale", "is_end", "is_scene", "add_scene"}
            """
            process_time = time.time()
            now_frame = task["now_frame"]
            img0 = task["img0"]
            img1 = task["img1"]
            n = task["n"]
            exp = task["exp"]
            scale = task["scale"]
            is_end = task["is_end"]
            add_scene = task["add_scene"]
            frames_list = [img0]

            debug = False

            if self.args["use_sr"] and self.args.get("use_sr_mode", 0) == 0:
                """先超后补"""
                img0, img1 = self.sr_module.svfi_process(img0), self.sr_module.svfi_process(img1)

            if n > 0:
                interp_list = self.rife_core.generate_n_interp(img0, img1, n=n, scale=scale, debug=debug)
                frames_list.extend(interp_list)
            elif exp > 0:
                interp_list = self.rife_core.generate_interp(img0, img1, exp=exp, scale=scale, debug=debug)
                frames_list.extend(interp_list)

            if add_scene:
                frames_list.append(img1)

            if self.args["use_sr"] and self.args.get("use_sr_mode", 0) == 1:
                """先补后超"""
                for i in range(len(frames_list)):
                    frames_list[i] = self.sr_module.svfi_process(frames_list[i])

            feed_list = [[now_frame, i] for i in frames_list]

            self.feed_to_render(feed_list, is_end=is_end)
            process_time = time.time() - process_time
            update_progress()
            if is_end:
                break

        while self.render_thread is not None and self.render_thread.is_alive():
            """等待渲染线程结束"""
            update_progress()
            time.sleep(0.1)

        pbar.update(abs(self.all_frames_cnt - now_frame))
        pbar.close()
        self.logger.info(f"Scedet Status Quo: {self.scene_detection.get_scene_status()}")

        pass

    def crop_read_img(self, img):
        """
        Crop using self.crop parameters
        :param img:
        :return:
        """
        if img is None:
            return img

        h, w, _ = img.shape
        if self.crop_param[0] > w or self.crop_param[1] > h:
            """奇怪的黑边参数，不予以处理"""
            return img
        return img[self.crop_param[1]:h - self.crop_param[1], self.crop_param[0]:w - self.crop_param[0]]

    def nvidia_vram_test(self):
        """
        显存测试
        :param img:
        :return:
        """
        try:
            if len(self.args["resize"]):
                w, h = list(map(lambda x: int(x), self.args["resize"].split("x")))
            else:

                w, h = list(map(lambda x: round(x), self.video_info["size"]))

            if w * h > 1920 * 1080:
                if self.args["scale"] > 0.5:
                    """超过1080p锁光流尺度为0.5"""
                    self.args["scale"] = 0.5
                self.logger.warning(f"Big Resolution (>1080p) Input found")
            # else:
            self.logger.info(f"Start VRAM Test: {w}x{h} with scale {self.args['scale']}")

            test_img0, test_img1 = np.random.randint(0, 255, size=(w, h, 3)).astype(np.uint8), \
                                   np.random.randint(0, 255, size=(w, h, 3)).astype(np.uint8)
            self.rife_core.generate_interp(test_img0, test_img1, 1, self.args["scale"], test=True)
            self.logger.info(f"VRAM Test Success")
            del test_img0, test_img1
        except Exception as e:
            self.logger.error("VRAM Check Failed, PLS Lower your presets\n" + traceback.format_exc())
            raise e

    def get_check_frames_list(self, videogen_check: FFmpegReader.nextFrame):
        """
        获得新除重预处理帧数序列
        :param videogen_check:
        :return:
        """
        check_queue_size = 1000  # 预处理长度
        side_vec = self.args["dup_threshold"]  # 双边最小运动幅度
        check_frame_list = list()  # 采样图片帧数序列
        check_frame_data = dict()  # 采样图片数据
        """
            check_frame_list contains key, check_frame_data contains (key, frame_data)
        """
        check_frame_cnt = 0

        while check_frame_cnt < check_queue_size:
            check_frame = Utils.gen_next(videogen_check)
            if check_frame is None:
                break
            check_frame_list.append(check_frame_cnt)  # key list
            check_frame_data[check_frame_cnt] = check_frame
            check_frame_cnt += 1
        if not len(check_frame_list):
            return []

        duplicate_data = list()  # key list
        last_frame = check_frame_list[0]
        for i in range(1, len(check_frame_list)):
            f = check_frame_list[i]  # key
            # 两两对比，diff值=0，辨别为重复帧
            if Utils.get_norm_img_diff(check_frame_data[last_frame], check_frame_data[f], resize=False) == 0:
                duplicate_data.append(f)
            last_frame = f

        for x in duplicate_data:  # key
            if x in check_frame_list:
                check_frame_list.remove(x)
                check_frame_data.pop(x)
            else:
                print(f'pass at {x} removing duplicated')

        # 去除一拍二
        duplicate_12_data = list()  # 用于存放表示一拍二的多组四帧列表, indexes
        I0 = check_frame_list[0]  # key
        i = 1
        while i < len(check_frame_list) - 2:
            # i0,i1,i2,i3为输入帧
            I1 = check_frame_list[i]
            I2 = check_frame_list[i + 1]
            I3 = check_frame_list[i + 2]
            #   i0,i1  i1,i2   i2,i3   分别对比的到diff值，i1,i2最为一个整体

            x1 = Utils.get_norm_img_diff(check_frame_data[I0], check_frame_data[I1], resize=True)
            x2 = Utils.get_norm_img_diff(check_frame_data[I1], check_frame_data[I2], resize=True)
            x3 = Utils.get_norm_img_diff(check_frame_data[I2], check_frame_data[I3], resize=True)
            #   中间值 - 左侧diff (i1,i2) > 最小运动幅度     右侧diff - 中间值 > 最小运动幅度
            if x1 - x2 > side_vec and x3 - x2 > side_vec:
                # duplicate_12.append([i-1,i,i+1,i+2])
                duplicate_12_data.append(
                    i - 1)  # check_frame_list = [pos1, pos3, pos5], pos3 could be dead when i - 1 = 1, index
            I0 = I1
            i += 1

        # 去除一拍三
        frames_edge = dict()
        for i in check_frame_list:  # key
            frames_edge[i] = cv2.Canny(check_frame_data[i], 100, 200)
        duplicate_data = list()  # indexes
        i0 = check_frame_list[0]
        i = 1
        while i < len(check_frame_list) - 3:
            i1 = check_frame_list[i]
            i2 = check_frame_list[i + 1]
            i3 = check_frame_list[i + 2]
            i4 = check_frame_list[i + 3]
            l = cv2.absdiff(frames_edge[i0], frames_edge[i1]).mean()
            m = (cv2.absdiff(frames_edge[i1], frames_edge[i2]).mean() + cv2.absdiff(frames_edge[i2],
                                                                                    frames_edge[i3]).mean()) / 2
            r = cv2.absdiff(frames_edge[i3], frames_edge[i4]).mean()
            # diff(i0,i1) - 估计三帧diff值(i1,i2,i3) > side_vec AND diff(i3,i4) - 估计三帧diff值 > min_vec
            if l - m > side_vec and r - m > side_vec:
                duplicate_data.append(i - 1)
            i0 = i1
            i += 1

        delgen = list()
        for x in duplicate_data:  # index of check_frame_list
            if x in duplicate_12_data or x + 1 in duplicate_12_data or x - 1 in duplicate_12_data:
                delgen.append(x)
        for x in delgen:
            duplicate_data.remove(x)

        check_len = len(check_frame_list)
        for x in duplicate_12_data:
            if x + 1 < check_len:
                check_frame_list[x + 1] = -1
            else:
                print(f'pass at {x + 1} > len')

        """去除一拍二完成"""

        if self.args.get("remove_dup_mode", 1) == 2:
            """去除一拍三"""
            for x in duplicate_data:
                if x + 1 < check_len:
                    check_frame_list[x + 1] = -1
                    pass
                else:
                    print(f'pass at {x + 1} > len')
                if x + 3 < check_len:
                    pass
                    check_frame_list[x + 3] = -1
                else:
                    print(f'pass at {x + 3} > len')

        return [i for i in check_frame_list if i > 0]

    def rife_run(self):
        """
        Go through all procedures to produce interpolation result
        :return:
        """

        self.logger.info("Activate Remove Duplicate Frames Mode")

        """Get Start Info"""
        _debug = False
        chunk_cnt, start_frame = self.check_chunk()  # start_frame = 0
        self.logger.info("Resuming Video Frames...")
        frame_reader = self.generate_frame_reader(start_frame)
        frame_check_reader = self.generate_frame_reader(start_frame, frame_check=True)

        """Get Frames to interpolate"""
        videogen = frame_reader.nextFrame()
        videogen_check = frame_check_reader.nextFrame()

        img1 = self.crop_read_img(Utils.gen_next(videogen))
        now_frame = start_frame
        if img1 is None:
            raise OSError(f"Input file not valid: {self.input}, img_input: {self.args['img_input']}")

        is_end = False

        """Start Process"""
        run_time = time.time()
        while True:
            if is_end or self.main_error:
                break

            if not self.render_thread.is_alive():
                self.logger.critical("Render Thread Dead Unexpectedly")
                break

            if self.args["multi_task_rest"] and self.args["multi_task_rest_interval"] and \
                    time.time() - run_time > self.args["multi_task_rest_interval"] * 3600:
                self.logger.info(
                    f"\n\n INFO - Exceed Run Interval {self.args['multi_task_rest_interval']} hour. Time to Rest for 5 minutes!")
                time.sleep(600)
                run_time = time.time()

            frame_cnt = 0
            last_frame_cnt = 0
            check_frame_list = self.get_check_frames_list(videogen_check)
            img0 = img1
            if not len(check_frame_list):
                skip = 0
                while True:
                    before_img = img1
                    img1 = self.crop_read_img(Utils.gen_next(videogen))
                    if img1 is None:
                        is_end = True
                        img1 = before_img
                        self.feed_to_rife(now_frame, img1, img1, n=skip - 1,
                                          is_end=is_end)  # TODO check redundant last frame added?
                        break
                    skip += 1
                break

            else:
                for f in check_frame_list:
                    before_img = img1
                    while frame_cnt < f:
                        img1 = self.crop_read_img(Utils.gen_next(videogen))
                        frame_cnt += 1
                        self.scene_detection.update_scene_status(now_frame + frame_cnt, "dup")
                        self.scene_detection.check_scene(img0, img1, add_diff=True)
                    is_scene = self.scene_detection.check_scene(img0, img1, no_diff=True)
                    if is_scene:
                        self.feed_to_rife(now_frame + frame_cnt, img0, before_img, n=frame_cnt - last_frame_cnt - 2,
                                          add_scene=True,
                                          is_end=is_end)
                        self.scene_detection.update_scene_status(now_frame + frame_cnt, "scene")
                    else:
                        before_img = img1
                        self.feed_to_rife(now_frame + frame_cnt, img0, before_img, n=frame_cnt - last_frame_cnt - 1,
                                          add_scene=False,
                                          is_end=is_end)
                        self.scene_detection.update_scene_status(now_frame + frame_cnt, "normal")
                    last_frame_cnt = frame_cnt
                    img0 = img1

                now_frame += frame_cnt
                self.task_info.update({"now_frame": now_frame})

        pass
        self.rife_task_queue.put(None)  # bad way to end # TODO need optimize
        """Wait for Rife and Render Thread to finish"""

    def rife_run_any_fps(self):
        """
        Go through all procedures to produce interpolation result
        :return:
        """

        self.logger.info("Activate Any FPS Mode")

        """Get Start Info"""
        _debug = False
        chunk_cnt, start_frame = self.check_chunk()  # start_frame = 0
        self.logger.info("Resuming Video Frames...")
        self.frame_reader = self.generate_frame_reader(start_frame)

        """Get Frames to interpolate"""
        videogen = self.frame_reader.nextFrame()
        img1 = self.crop_read_img(Utils.gen_next(videogen))
        now_frame = start_frame
        if img1 is None:
            raise OSError(f"Input file not valid: {self.input}")

        is_end = False

        """Update Interp Mode Info"""
        if self.args.get("remove_dup_mode", 0) == 3:  # 朴素模式
            self.args["dup_threshold"] = self.args["dup_threshold"] if self.args["dup_threshold"] > 0.01 else 0.01
        else:  # 0， 不去除重复帧
            self.args["dup_threshold"] = 0.01

        """Start Process"""
        run_time = time.time()
        while True:
            if is_end or self.main_error:
                break

            if not self.render_thread.is_alive():
                self.logger.critical("Render Thread Dead Unexpectedly")
                break

            if self.args["multi_task_rest"] and self.args["multi_task_rest_interval"] and \
                    time.time() - run_time > self.args["multi_task_rest_interval"] * 3600:
                self.logger.info(
                    f"\n\n INFO - Exceed Run Interval {self.args['multi_task_rest_interval']} hour. Time to Rest for 5 minutes!")
                time.sleep(600)
                run_time = time.time()

            img0 = img1
            img1 = self.crop_read_img(Utils.gen_next(videogen))

            now_frame += 1

            if img1 is None:
                is_end = True
                self.feed_to_rife(now_frame, img0, img0, is_end=is_end)
                break

            diff = Utils.get_norm_img_diff(img0, img1)
            skip = 0  # 用于记录跳过的帧数

            """Find Scene"""
            if self.scene_detection.check_scene(img0, img1, use_diff=diff):
                self.feed_to_rife(now_frame, img0, img1, is_end=is_end)
                self.scene_detection.update_scene_status(now_frame, "scene")
                continue
            else:
                if diff < self.args["dup_threshold"]:
                    before_img = img1
                    while diff < self.args["dup_threshold"]:
                        skip += 1
                        self.scene_detection.update_scene_status(now_frame, "dup")

                        img1 = self.crop_read_img(Utils.gen_next(videogen))

                        if img1 is None:
                            img1 = before_img
                            is_end = True
                            break

                        diff = Utils.get_norm_img_diff(img0, img1)

                        self.scene_detection.check_scene(img0, img1, add_diff=True, use_diff=diff)  # update scene stack
                        if skip == self.dup_skip_limit * self.target_fps // self.fps:
                            """超过重复帧计数限额，直接跳出"""
                            break

                    # 除去重复帧后可能im0，im1依然为转场，因为转场或大幅度运动的前一帧可以为重复帧
                    if self.scene_detection.check_scene(img0, img1, no_diff=True):
                        skip -= 1  # 两帧间隔计数器-1
                        self.feed_to_rife(now_frame, img0, before_img, n=skip, add_scene=True,
                                          is_end=is_end)
                        self.scene_detection.update_scene_status(now_frame, "scene")

                    elif skip != 0:
                        """Not Scene"""
                        self.feed_to_rife(now_frame, img0, img1, n=skip, is_end=is_end)
                        self.scene_detection.update_scene_status(now_frame, "normal")
                    now_frame += skip
                else:
                    """normal frames"""
                    self.feed_to_rife(now_frame, img0, img1, is_end=is_end)
                    self.scene_detection.update_scene_status(now_frame, "normal")
                self.task_info.update({"now_frame": now_frame})
            pass

        self.rife_task_queue.put(None)  # bad way to end # TODO need optimize

    def run(self):
        if self.args["concat_only"]:
            self.concat_all()
        elif self.args["extract_only"]:
            self.extract_only()
            pass
        else:
            def update_progress():
                nonlocal previous_cnt, task_acquire_time, process_time
                scene_status = self.scene_detection.get_scene_status()

                render_status = self.task_info  # render status quo
                """(chunk_cnt, start_frame, end_frame, frame_cnt)"""

                pbar.set_description(
                    f"Process at Chunk {render_status['chunk_cnt']:0>3d}")
                pbar.set_postfix({"R": f"{render_status['render']}", "C": f"{now_frame}",
                                  "S": f"{scene_status['recent_scene']}",
                                  "SC": f"{self.scene_detection.scdet_cnt}", "TAT": f"{task_acquire_time:.2f}s",
                                  "PT": f"{process_time:.2f}s", "QL": f"{self.rife_task_queue.qsize()}"})
                pbar.update(now_frame - previous_cnt)
                previous_cnt = now_frame
                pass

            """Load RIFE Model"""
            if self.args["ncnn"]:
                self.args["selected_model"] = os.path.basename(self.args["selected_model"])
                import inference_A as inference
            else:
                try:
                    import inference  # 导入补帧模块
                except Exception:
                    self.logger.warning("Import Torch Failed, use NCNN-RIFE instead")
                    traceback.print_exc()
                    self.args.update({"ncnn": True, "selected_model": "rife-v2"})
                    import inference_A as inference

            """Update RIFE Core"""
            self.rife_core = inference.RifeInterpolation(self.args)
            self.rife_core.initiate_rife(args)

            if not self.args["ncnn"]:
                self.nvidia_vram_test()

            """Get Renderer"""
            if self.args.get("remove_dup_mode", 0) in [0, 3]:
                self.rife_thread = threading.Thread(target=self.rife_run_any_fps, name="[ARGS] RifeTaskThread", )
            else:  # 1, 2 => 去重一拍二或一拍三
                self.rife_thread = threading.Thread(target=self.rife_run, name="[ARGS] RifeTaskThread", )
            self.rife_thread.start()

            chunk_cnt, start_frame = self.check_chunk()  # start_frame = 0

            """Get Renderer"""
            self.render_thread = threading.Thread(target=self.render, name="[ARGS] RenderThread",
                                                  args=(chunk_cnt, start_frame,))
            self.render_thread.start()

            """Build SR"""
            if self.args["use_sr"]:
                input_resolution = self.video_info["size"][0] * self.video_info["size"][1]
                output_resolution = self.args.get("resize_width", 0) * self.args.get("resize_height", 0)
                resolution_rate = math.sqrt(output_resolution / input_resolution)
                if input_resolution and resolution_rate > 1:
                    sr_scale = Utils.get_exp_edge(resolution_rate)
                    if self.args["use_sr_algo"] == "waifu2x":
                        self.sr_module = SvfiWaifu(model=self.args["use_sr_model"], scale=sr_scale)
                    elif self.args["use_sr_algo"] == "realSR":
                        self.sr_module = SvfiRealSR(model=self.args["use_sr_model"], scale=sr_scale)
                    self.logger.info(
                        f"Load AI SR at {self.args['use_sr_algo']}, {self.args['use_sr_model']}, scale = {sr_scale}")
                else:
                    self.logger.warning("Abort to load AI SR since Resolution Rate < 1")

            previous_cnt = start_frame
            now_frame = start_frame

            pbar = tqdm.tqdm(total=self.all_frames_cnt, unit="it")
            pbar.update(n=start_frame)
            pbar.unpause()
            task_acquire_time = time.time()
            process_time = time.time()
            while True:
                task_acquire_time = time.time()
                task = self.rife_task_queue.get()
                task_acquire_time = time.time() - task_acquire_time
                if task is None:
                    self.feed_to_render([None], is_end=True)
                    break
                """
                task = {"now_frame", "img0", "img1", "n", "exp","scale", "is_end", "is_scene", "add_scene"}
                """
                process_time = time.time()
                now_frame = task["now_frame"]
                img0 = task["img0"]
                img1 = task["img1"]
                n = task["n"]
                exp = task["exp"]
                scale = task["scale"]
                is_end = task["is_end"]
                add_scene = task["add_scene"]
                frames_list = [img0]

                debug = False

                if self.args["use_sr"] and self.args.get("use_sr_mode", 0) == 0:
                    """先超后补"""
                    img0, img1 = self.sr_module.svfi_process(img0), self.sr_module.svfi_process(img1)

                if n > 0:
                    interp_list = self.rife_core.generate_n_interp(img0, img1, n=n, scale=scale, debug=debug)
                    frames_list.extend(interp_list)
                elif exp > 0:
                    interp_list = self.rife_core.generate_interp(img0, img1, exp=exp, scale=scale, debug=debug)
                    frames_list.extend(interp_list)

                if add_scene:
                    frames_list.append(img1)

                if self.args["use_sr"] and self.args.get("use_sr_mode", 0) == 1:
                    """先补后超"""
                    for i in range(len(frames_list)):
                        frames_list[i] = self.sr_module.svfi_process(frames_list[i])

                feed_list = [[now_frame, i] for i in frames_list]

                self.feed_to_render(feed_list, is_end=is_end)
                process_time = time.time() - process_time
                update_progress()
                if is_end:
                    break

            while (self.render_thread is not None and self.render_thread.is_alive()) or \
                    (self.rife_thread is not None and self.rife_thread.is_alive()):
                """等待渲染线程结束"""
                update_progress()
                time.sleep(0.1)

            pbar.update(abs(self.all_frames_cnt - now_frame))
            pbar.close()

            self.logger.info(f"Scedet Status Quo: {self.scene_detection.get_scene_status()}")

            """Check Finished Safely"""
            if self.main_error is not None:
                raise self.main_error

            """Concat the chunks"""
            if not self.args["no_concat"] and not self.args["img_output"]:
                self.concat_all()
                return

        self.logger.info(f"Program finished at {datetime.datetime.now()}")
        pass

    def extract_only(self):
        chunk_cnt, start_frame = self.check_chunk()
        videogen = self.generate_frame_reader(start_frame)

        img1 = self.crop_read_img(Utils.gen_next(videogen))
        if img1 is None:
            raise OSError(f"Input file not valid: {self.input}")

        renderer = ImgSeqIO(folder=self.output, is_read=False)
        pbar = tqdm.tqdm(total=self.all_frames_cnt, unit="frames")
        pbar.update(n=start_frame)
        img_cnt = 0
        while img1 is not None:
            renderer.writeFrame(img1)
            pbar.update(n=1)
            img_cnt += 1
            pbar.set_description(
                f"Process at Extracting Img {img_cnt}")
            img1 = self.crop_read_img(Utils.gen_next(videogen))

        renderer.close()

    def concat_all(self):
        """
        Concat all the chunks
        :return:
        """

        os.chdir(self.project_dir)
        concat_path = os.path.join(self.project_dir, "concat.ini")
        self.logger.info("Final Round Finished, Start Concating")
        concat_list = list()

        for f in os.listdir(self.project_dir):
            chunk_regex = rf"chunk-[\d+].*?\{self.output_ext}"
            if re.match(chunk_regex, f):
                concat_list.append(os.path.join(self.project_dir, f))
            else:
                self.logger.debug(f"concat escape {f}")

        concat_list.sort(key=lambda x: int(os.path.basename(x).split('-')[2]))  # sort as start-frame

        if os.path.exists(concat_path):
            os.remove(concat_path)

        with open(concat_path, "w+", encoding="UTF-8") as w:
            for f in concat_list:
                w.write(f"file '{f}'\n")

        """Check Input file ext"""
        output_ext = os.path.splitext(self.input)[-1]
        if output_ext not in [".mp4", ".mov", ".mkv"]:
            output_ext = self.output_ext
        if self.args["encoder"] == "ProRes":
            output_ext = ".mov"

        concat_filepath = f"{os.path.join(self.output, Utils.get_filename(self.input))}"
        concat_filepath += f"_{int(self.target_fps)}fps"  # 输出帧率
        if self.args["slow_motion"]:  # 慢动作
            concat_filepath += f"_slowmo_{self.args['slow_motion_fps']}"
        concat_filepath += f"_scale{self.args['scale']}"  # 全局光流尺度
        if not self.args["ncnn"]:  # 不使用NCNN
            concat_filepath += f"_{os.path.basename(self.args['selected_model'])}"  # 添加模型信息
        else:
            concat_filepath += f"_rife-ncnn"
        if self.args["remove_dup_mode"]:  # 去重模式
            concat_filepath += f"_RD{self.args['remove_dup_mode']}"
        if self.args["use_sr"]:  # 使用超分
            concat_filepath += f"_{self.args['use_sr_algo']}_{self.args['use_sr_model']}"
        concat_filepath += output_ext  # 添加后缀名

        if self.args["save_audio"] and not self.args["img_input"]:
            audio_path = self.input
            map_audio = f'-i "{audio_path}" -map 0:v:0 -map 1:a? -map 1:s? -c:a copy -c:s copy '  # TODO No shortest, Check VA Sync Only
            if self.args.get("start_point", None) is not None or self.args.get("end_point", None) is not None:
                time_fmt = "%H:%M:%S"
                start_point = datetime.datetime.strptime(self.args["start_point"], time_fmt)
                end_point = datetime.datetime.strptime(self.args["end_point"], time_fmt)
                if end_point > start_point:
                    self.logger.info(
                        f"Update Concat Audio Range: in {self.args['start_point']} -> out {self.args['end_point']}")
                    map_audio = f'-ss {self.args["start_point"]} -to {self.args["end_point"]} -i "{audio_path}" -map 0:v:0 -map 1:a? -c:a aac -ab 640k '
                else:
                    self.logger.warning(
                        f"Input Time Section change to origianl course")

        else:
            map_audio = ""

        ffmpeg_command = f'{self.ffmpeg} -hide_banner -f concat -safe 0 -i "{concat_path}" {map_audio} -c:v copy {Utils.fillQuotation(concat_filepath)} -y'
        self.logger.debug(f"Concat command: {ffmpeg_command}")
        os.system(ffmpeg_command)
        if self.args["output_only"] and os.path.exists(concat_filepath):
            if not os.path.getsize(concat_filepath):
                self.logger.error(f"Concat Error, {output_ext}, empty output")
                raise FileExistsError("Concat Error, empty output, Check Output Extension!!!")
            self.check_chunk(del_chunk=True)

    def concat_check(self, concat_list, concat_filepath):
        """
        Check if concat output is valid
        :param concat_filepath:
        :param concat_list:
        :return:
        """
        original_concat_size = 0
        for f in concat_list:
            original_concat_size += os.path.getsize(f)
        output_concat_size = os.path.getsize(concat_filepath)
        if output_concat_size < original_concat_size * 0.9:
            return False
        return True


interpworkflow = InterpWorkFlow(args)
interpworkflow.run()
sys.exit(0)
