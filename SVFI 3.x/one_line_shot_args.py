# coding: utf-8
import argparse
import datetime
import math
import os
import re
import sys
import threading
import time
import traceback
from queue import Queue

import cv2
import numpy as np
import psutil
import tqdm

from Utils.LicenseModule import EULAWriter
from Utils.StaticParameters import appDir, SupportFormat, HDR_STATE, RGB_TYPE, RT_RATIO, LUTS_TYPE, RIFE_TYPE
from Utils.utils import ArgumentManager, DefaultConfigParser, Tools, VideoInfoProcessor, \
    ImageRead, ImageWrite, TransitionDetection_ST, \
    VideoFrameInterpolationBase, Hdr10PlusProcessor, DoviProcessor, \
    SuperResolutionBase, overtime_reminder_deco, OverTimeReminderTask
from skvideo.io import FFmpegWriter, FFmpegReader, EnccWriter, SVTWriter
from steamworks import STEAMWORKS
from steamworks.exceptions import GenericSteamException

print(f"INFO - ONE LINE SHOT ARGS {ArgumentManager.ols_version} {datetime.date.today()}")

"""Validation Module Initiation"""
if ArgumentManager.is_steam:
    from Utils.LicenseModule import SteamValidation as ValidationModule

    try:
        _steamworks = STEAMWORKS(ArgumentManager.app_id)
    except:
        pass
else:
    from Utils.LicenseModule import RetailValidation as ValidationModule

"""设置环境路径"""
os.chdir(appDir)
sys.path.append(appDir)

"""Parse Args"""
global_args_parser = argparse.ArgumentParser(prog="#### SVFI CLI tool by Jeanna ####",
                                             description='Interpolation for long video/imgs footage')
global_basic_parser = global_args_parser.add_argument_group(title="Basic Settings, Necessary")
global_basic_parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                                 help="原视频/图片序列文件夹路径")
global_basic_parser.add_argument("-c", '--config', dest='config', type=str, required=True, help="配置文件路径")
global_basic_parser.add_argument("-t", '--task-id', dest='task_id', type=str, required=True, help="任务id")
global_basic_parser.add_argument('--concat-only', dest='concat_only', action='store_true', help='只执行合并已有区块操作')
global_basic_parser.add_argument('--extract-only', dest='extract_only', action='store_true', help='只执行拆帧操作')
global_basic_parser.add_argument('--render-only', dest='render_only', action='store_true', help='只执行渲染操作')

"""Clean Args"""
global_args_read = global_args_parser.parse_args()
global_config_parser = DefaultConfigParser(allow_no_value=True)  # 把SVFI GUI传来的参数格式化
global_config_parser.read(global_args_read.config, encoding='utf-8')
global_config_parser_items = dict(global_config_parser.items("General"))
global_args = Tools.clean_parsed_config(global_config_parser_items)
global_args.update(vars(global_args_read))  # update -i -o -c，将命令行参数更新到config生成的字典

"""Set Global Logger"""
logger = Tools.get_logger('TMP', '')


class TaskArgumentManager(ArgumentManager):
    """
        For OLS's current input's arguments validation
    """

    def __init__(self, _args: dict):
        super().__init__(_args)
        global logger
        self.interp_times = 0
        self.max_frame_cnt = 10 ** 10
        self.all_frames_cnt = 0
        self.frames_queue_len = 0
        self.dup_skip_limit = 0
        self.task_info = {"chunk_cnt": 0, "render": 0,
                          "read_now_frame": 0,
                          "rife_now_frame": 0,
                          "recent_scene": 0, "scene_cnt": 0,
                          "decode_process_time": 0,
                          "render_process_time": 0,
                          "rife_task_acquire_time": 0,
                          "rife_process_time": 0,
                          "rife_queue_len": 0,
                          "sr_now_frame": 0,
                          "sr_task_acquire_time": 0,
                          "sr_process_time": 0,
                          "sr_queue_len": 0, }  # 有关任务的实时信息

        self.__validate_io_path()
        self.__initiate_logger()
        self.__set_ffmpeg_path()
        self.video_info_instance = VideoInfoProcessor(input_file=self.input, logger=logger,
                                                      project_dir=self.project_dir,
                                                      interp_exp=self.rife_exp,
                                                      hdr_cube_mode=self.hdr_cube_mode)
        self.__update_io_info()
        self.__update_hdr_mode()
        self.__update_frames_cnt()
        self.__update_frame_size()
        self.__update_task_queue_size_by_memory()
        self.__update_io_ext()

        """Check Initiation Info"""
        logger.info(
            f"Check Input Source: "
            f"FPS: {self.input_fps} -> {self.target_fps}, FRAMES_CNT: {self.all_frames_cnt}, "
            f"INTERP_TIMES: {self.interp_times}, "
            f"HDR: {self.hdr_mode.name}, FRAME_SIZE: {self.frame_size}, QUEUE_LEN: {self.frames_queue_len}, "
            f"INPUT_EXT: {self.input_ext}, OUTPUT_EXT: {self.output_ext}")

        self.main_error = list()

        """Preview"""
        self.preview_imgs = list()

    def __update_io_ext(self):
        """update extension"""
        self.input_ext = os.path.splitext(self.input)[1] if os.path.isfile(self.input) else ""
        self.input_ext = self.input_ext.lower()
        if not self.output_ext.startswith('.'):
            self.output_ext = "." + self.output_ext
        if "ProRes" in self.render_encode_format and not self.is_img_output:
            self.output_ext = ".mov"
        if self.is_img_output and self.output_ext not in SupportFormat.img_outputs:
            self.output_ext = ".png"

    def __update_io_info(self):
        """
        Update io fps and interp times
        :return:
        """
        """Set input and target(output) fps"""
        self.is_img_input = self.video_info_instance.is_img_input  # update input file type
        self.input = self.video_info_instance.input_file  # update true input file
        if not self.is_img_input:  # 输入不是文件夹，使用检测到的帧率
            self.input_fps = self.video_info_instance.fps
        elif not self.input_fps:  # 输入是文件夹，使用用户的输入帧率; 用户有毒，未发现有效的输入帧率
            raise OSError("Not Find Input FPS, Input File is not valid")
        if self.render_only:
            self.target_fps = self.input_fps
            logger.info(f"Render only, target fps is changed to input fps: {self.target_fps}")
        else:
            if not self.target_fps:  # 未找到用户的输出帧率
                self.target_fps = (2 ** self.rife_exp) * self.input_fps  # default
            if self.is_img_input or not len(self.video_info_instance.audio_info):  # 图片序列输入，不保留音频（也无音频可保留
                logger.warning("Image Sequence input is found or Video does not contain audio, will not mux audio")
                self.is_save_audio = False

        """Set interpolation exp related to hdr mode"""
        self.interp_times = round(self.target_fps / self.input_fps)
        if self.hdr_mode in [HDR_STATE.HDR10_PLUS, HDR_STATE.DOLBY_VISION]:
            self.target_fps = self.interp_times * self.input_fps
            logger.info(f"DoVi or HDR10+ Content Detected, target fps is changed to {self.target_fps}")

        if self.is_img_input:
            if self.is_16bit_workflow:
                self.is_16bit_workflow = False  # 图片序列输入不使用高精度工作流
                RGB_TYPE.change_8bit(True)
                logger.warning("Image Sequence input is found, will not use 16bit workflow")

    def __update_task_queue_size_by_memory(self):
        """Guess Memory and Fix Resolution"""
        if self.use_manual_buffer:
            # 手动指定内存占用量
            free_mem = self.manual_buffer_size * 1024
        else:
            mem = psutil.virtual_memory()
            free_mem = round(mem.free / 1024 / 1024)
        self.frames_queue_len = round(free_mem / (sys.getsizeof(
            np.random.rand(3, self.frame_size[0], self.frame_size[1])) / 1024 / 1024)) // 3
        if not self.use_manual_buffer:
            self.frames_queue_len = int(min(max(self.frames_queue_len, 8), 88))  # [8, 88]
        self.dup_skip_limit = int(0.5 * self.input_fps) + 1  # 当前跳过的帧计数超过这个值，将结束当前判断循环
        logger.info(f"Free RAM: {free_mem / 1024:.1f}G, "
                    f"Update Task Queue Len:  {self.frames_queue_len}, "
                    f"Duplicate Frames Cnt Upper Limit: {self.dup_skip_limit}")

    def __update_frame_size(self):
        """规整化输出输入分辨率"""
        self.frame_size = (round(self.video_info_instance.frame_size[0]),
                           round(self.video_info_instance.frame_size[1]))
        self.first_frame_size = (round(self.video_info_instance.first_img_frame_size[0]),
                                 round(self.video_info_instance.first_img_frame_size[1]))

    def __update_frames_cnt(self):
        """Update All Frames Count"""
        self.all_frames_cnt = abs(int(self.video_info_instance.duration * self.target_fps))
        if self.all_frames_cnt > self.max_frame_cnt:
            raise OSError(f"SVFI can't afford input exceeding {self.max_frame_cnt} frames")

    def __update_hdr_mode(self):
        if self.hdr_mode == HDR_STATE.AUTO:  # Auto
            self.hdr_mode = self.video_info_instance.hdr_mode
            logger.info(f"Auto Sets HDR mode to {self.hdr_mode.name}")
            # no hdr at -1, 0 checked and None, 1 hdr, 2 hdr10, 3 DV, 4 HLG
            # hdr_check_status indicates the final process mode for (hdr) input

    def __set_ffmpeg_path(self):
        """Set FFmpeg"""
        self.ffmpeg = "ffmpeg"

    def __validate_io_path(self):
        if not len(self.input):
            raise OSError("Input Path is empty, Program will not proceed")
        if not len(self.output_dir):
            """未填写输出文件夹"""
            self.output_dir = os.path.dirname(self.input)
        if os.path.isfile(self.output_dir):
            self.output_dir = os.path.dirname(self.output_dir)

        self.project_name = f"{Tools.get_filename(self.input)}_{self.task_id}"
        self.project_dir = os.path.join(self.output_dir, self.project_name)
        os.makedirs(self.project_dir, exist_ok=True)
        sys.path.append(self.project_dir)

        """Extract Only Mode"""
        if self.extract_only and self.output_ext not in SupportFormat.img_outputs:
            self.is_img_output = True
            self.output_ext = ".png"
            logger.warning("Output extension is changed to .png")

        """Check Img IO status"""
        if self.is_img_output:
            self.output_dir = os.path.join(self.output_dir, self.project_name)
            os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.isfile(self.input):
            self.is_img_input = True

    def __initiate_logger(self):
        """Set Global Logger"""
        global logger
        logger = Tools.get_logger("CLI", self.project_dir, debug=self.debug)
        logger.info(f"Initial New Interpolation Project: PROJECT DIR: %s, INPUT FILEPATH: %s", self.project_dir,
                    self.input)

    def update_task_info(self, update_dict: dict):
        self.task_info.update(update_dict)

    def get_main_error(self):
        if not len(self.main_error):
            return None
        else:
            return self.main_error[-1]

    def save_main_error(self, e: Exception):
        self.main_error.append(e)

    def update_preview_imgs(self, imgs: list):
        self.preview_imgs = imgs

    def get_preview_imgs(self):
        return self.preview_imgs


class ValidationFlow(ValidationModule):
    def __init__(self, _args: TaskArgumentManager):
        self.ARGS = _args
        super().__init__(logger)
        self.kill = False
        pass

    def steam_update_achv(self, output_path):
        """
        Update Steam Achievement
        :return:
        """
        if not self.ARGS.is_steam or self.kill:
            """If encountered serious error in the process, end steam update"""
            return
        if self.ARGS.concat_only or self.ARGS.render_only or self.ARGS.extract_only:
            return
        """Get Stat"""
        STAT_INT_FINISHED_CNT = self.GetStat("STAT_INT_FINISHED_CNT", int)
        STAT_FLOAT_FINISHED_MINUTE = self.GetStat("STAT_FLOAT_FINISHED_MIN", float)

        """Update Stat"""
        STAT_INT_FINISHED_CNT += 1
        self.SetStat("STAT_INT_FINISHED_CNT", STAT_INT_FINISHED_CNT)
        if self.ARGS.all_frames_cnt >= 0:
            """Update Mission Process Time only in interpolation"""
            STAT_FLOAT_FINISHED_MINUTE += self.ARGS.all_frames_cnt / self.ARGS.target_fps / 60
            self.SetStat("STAT_FLOAT_FINISHED_MIN", round(STAT_FLOAT_FINISHED_MINUTE, 2))

        """Get ACHV"""
        ACHV_Task_Frozen = self.GetAchv("ACHV_Task_Frozen")
        ACHV_Task_Cruella = self.GetAchv("ACHV_Task_Cruella")
        ACHV_Task_Suzumiya = self.GetAchv("ACHV_Task_Suzumiya")
        ACHV_Task_1000M = self.GetAchv("ACHV_Task_1000M")
        ACHV_Task_10 = self.GetAchv("ACHV_Task_10")
        ACHV_Task_50 = self.GetAchv("ACHV_Task_50")

        """Update ACHV"""
        if 'Frozen' in output_path and not ACHV_Task_Frozen:
            self.SetAchv("ACHV_Task_Frozen")
        if 'Cruella' in output_path and not ACHV_Task_Cruella:
            self.SetAchv("ACHV_Task_Cruella")
        if any([i in output_path for i in ['Suzumiya', 'Haruhi', '涼宮', '涼宮ハルヒの憂鬱', '涼宮ハルヒの消失', '凉宫春日']]) \
                and not ACHV_Task_Suzumiya:
            self.SetAchv("ACHV_Task_Suzumiya")
        if STAT_INT_FINISHED_CNT > 10 and not ACHV_Task_10:
            self.SetAchv("ACHV_Task_10")
        if STAT_INT_FINISHED_CNT > 50 and not ACHV_Task_50:
            self.SetAchv("ACHV_Task_50")
        if STAT_FLOAT_FINISHED_MINUTE > 1000 and not ACHV_Task_1000M:
            self.SetAchv("ACHV_Task_1000M")
        self.Store()


class IOFlow(threading.Thread):
    def __init__(self, __args: TaskArgumentManager, __logger):
        super().__init__()
        self.ARGS = __args
        self.logger = __logger
        self.initiation_event = threading.Event()
        self.initiation_event.clear()
        self._kill = False
        self.task_done = False

    def _release_initiation(self):
        self.initiation_event.set()

    def acquire_initiation_clock(self):
        if not self.is_alive():
            self.initiation_event.set()
        self.initiation_event.wait()

    def _get_color_tag_dict(self):
        color_tag_map = {'-color_range': 'color_range',
                         '-color_primaries': 'color_primaries',
                         '-colorspace': 'color_space', '-color_trc': 'color_transfer'}
        output_dict = {}
        for ct in color_tag_map:
            ct_data = self.ARGS.video_info_instance.video_info[color_tag_map[ct]]
            if len(ct_data):
                output_dict.update({ct: ct_data})
        return output_dict

    def check_chunk(self):
        """
        Get Chunk Start
        :param: del_chunk: delete all chunks existed
        :return: chunk, start_frame
        """
        if self.ARGS.is_img_output:
            """IMG OUTPUT"""
            img_writer = ImageWrite(logger, folder=self.ARGS.output_dir, start_frame=self.ARGS.interp_start,
                                    output_ext=self.ARGS.output_ext, is_tool=True)
            last_img = img_writer.get_write_start_frame()
            if self.ARGS.interp_start not in [-1, ]:
                return int(self.ARGS.output_chunk_cnt), int(self.ARGS.interp_start)  # Manually Prioritized
            if last_img == 0:
                return 1, 0

        if self.ARGS.interp_start != -1 or self.ARGS.output_chunk_cnt != -1:
            return int(self.ARGS.output_chunk_cnt), int(self.ARGS.interp_start)

        chunk_paths, chunk_cnt, last_frame = Tools.get_existed_chunks(self.ARGS.project_dir)
        if not len(chunk_paths):
            return 1, 0
        return chunk_cnt + 1, last_frame + 1

    def get_valid_input_resolution_for_test(self):
        if all(self.ARGS.frame_size):
            w, h = self.ARGS.frame_size
        elif all(self.ARGS.first_frame_size):
            w, h = self.ARGS.first_frame_size
        else:
            w, h = (480, 270)
        return w, h

    def kill(self):
        self._kill = True

    def _task_done(self):
        self.task_done = True

    def is_task_done(self):
        """
        Only available for specific task
        :return:
        """
        return self.task_done


class ReadFlow(IOFlow):

    def __init__(self, _args: TaskArgumentManager, __logger, _output_queue: Queue):
        super().__init__(_args, __logger)
        self.name = 'Reader'
        self._output_queue = _output_queue
        self.scene_detection = TransitionDetection_ST(self.ARGS.project_dir, int(0.3 * self.ARGS.input_fps),
                                                      scdet_threshold=self.ARGS.scdet_threshold,
                                                      no_scdet=self.ARGS.is_no_scdet,
                                                      use_fixed_scdet=self.ARGS.use_scdet_fixed,
                                                      fixed_max_scdet=self.ARGS.scdet_fixed_max,
                                                      scdet_output=self.ARGS.is_scdet_output)
        self.vfi_core = VideoFrameInterpolationBase(self.ARGS, logger)
        self.decode_timer_on = False
        self.decode_time_start = time.time()

    def __crop(self, img):
        """
        Crop using self.crop parameters
        :param img:
        :return:
        """
        if img is None or not any(self.ARGS.crop_param):
            return img

        h, w, _ = img.shape
        cw, ch = self.ARGS.crop_param
        if cw > w or ch > h:
            """奇怪的黑边参数，不予以处理"""
            return img
        return img[ch:h - ch, cw:w - cw]

    def __input_check(self, dedup=False):
        """
        perform input availability check and return generator of frames
        :return: chunk_cnt, start_frame, videogen, videogen_check
        """
        _debug = False
        chunk_cnt, start_frame = self.check_chunk()  # start_frame = 0
        self.logger.info("Resuming Checkpoint...")

        """Get Frames to interpolate"""
        # TODO Optimize this since progress bar started after reading initiation is complete
        _over_time_reminder_task = OverTimeReminderTask(10, "Decode Input",
                                                        "Decoding takes too long, Please consider to terminate the program, and check input's parameters and restart. It's normal to wait for at least 10 minutes when it's 4K input at performing resume of workflow")
        self.ARGS.put_overtime_task(_over_time_reminder_task)

        videogen = self.__generate_frame_reader(start_frame).nextFrame()
        videogen_check = None
        if dedup:
            videogen_check = self.__generate_frame_reader(start_frame, frame_check=True).nextFrame()
        videogen_available_check = self.__generate_frame_reader(start_frame, frame_check=True).nextFrame()

        check_img1 = self.__crop(Tools.gen_next(videogen_available_check))
        _over_time_reminder_task.deactive()

        videogen_available_check.close()
        if check_img1 is None:
            main_error = OSError(
                f"Input file is not available: {self.ARGS.input}, is img input?: {self.ARGS.is_img_input},"
                f"Please Check Your Input Parameters: "
                f"Start Chunk, Start Frame, Start Point, Start Frame")
            self.ARGS.save_main_error(main_error)
            self._release_initiation()
            raise main_error
        return chunk_cnt, start_frame, videogen, videogen_check

    def __generate_frame_reader(self, start_frame=0, frame_check=False):
        """
        输入帧迭代器
        :return:
        """
        """If input is sequence of frames"""
        if self.ARGS.is_img_input:
            resize_param = self.ARGS.resize_param
            if self.ARGS.use_sr:
                # modify input resolution for super resolution
                resize_param = RT_RATIO.get_modified_resolution(self.ARGS.frame_size, self.ARGS.transfer_ratio)

            img_reader = ImageRead(self.logger, folder=self.ARGS.input, start_frame=self.ARGS.interp_start,
                                   exp=self.ARGS.rife_exp, resize=resize_param, )
            self.ARGS.all_frames_cnt = img_reader.get_frames_cnt()
            self.logger.info(f"Img Input Found, update frames count to {self.ARGS.all_frames_cnt}")
            return img_reader

        """If input is a video"""
        input_dict = {"-vsync": "0", }
        if self.ARGS.use_hwaccel_decode:
            input_dict.update({"-hwaccel": "auto"})

        if self.ARGS.input_start_point or self.ARGS.input_end_point:
            """任意时段任务"""
            time_fmt = "%H:%M:%S"
            start_point = datetime.datetime.strptime("00:00:00", time_fmt)
            end_point = datetime.datetime.strptime("00:00:00", time_fmt)
            if self.ARGS.input_start_point is not None:
                start_point = datetime.datetime.strptime(self.ARGS.input_start_point, time_fmt) - start_point
                input_dict.update({"-ss": self.ARGS.input_start_point})
            else:
                start_point = start_point - start_point
            if self.ARGS.input_end_point is not None:
                end_point = datetime.datetime.strptime(self.ARGS.input_end_point, time_fmt) - end_point
                input_dict.update({"-to": self.ARGS.input_end_point})
            elif self.ARGS.video_info_instance.duration:
                # no need to care about img input
                end_point = datetime.datetime.fromtimestamp(
                    self.ARGS.video_info_instance.duration) - datetime.datetime.fromtimestamp(0.0)
            else:
                end_point = end_point - end_point

            if end_point > start_point:
                start_frame = -1
                clip_duration = end_point - start_point
                clip_fps = self.ARGS.target_fps
                self.ARGS.all_frames_cnt = round(clip_duration.total_seconds() * clip_fps)
                self.logger.info(
                    f"Update Input Section: in {self.ARGS.input_start_point} -> out {self.ARGS.input_end_point}, "
                    f"all_frames_cnt -> {self.ARGS.all_frames_cnt}")
            else:
                if '-ss' in input_dict:
                    input_dict.pop('-ss')
                if '-to' in input_dict:
                    input_dict.pop('-to')
                self.logger.warning(
                    f"Invalid Input Section, changed to original section")

        output_dict = {"-map": "0:v:0", "-vframes": str(10 ** 10),
                       # "-sws_flags": "bicubic",
                       }  # use read frames cnt to avoid ffprobe, fuck

        output_dict.update(self._get_color_tag_dict())

        vf_args = f"copy"

        """Checking Start Time"""
        if start_frame not in [-1, 0]:
            # not start from the beginning
            if self.ARGS.risk_resume_mode:
                """Quick Locate"""
                input_dict.update({"-ss": f"{start_frame / self.ARGS.target_fps:.3f}"})
            else:
                vf_args += f",trim=start={start_frame / self.ARGS.target_fps:.3f}"

        if frame_check:
            """用以文件检查或一拍N除重模式的预处理"""
            output_dict.update({"-s": f"300x300"})
        else:
            """Normal Input Process"""

            """Deinterlace"""
            if self.ARGS.use_deinterlace:
                vf_args += f",yadif=parity=auto"

            """Checking Input Resolution"""
            if not self.ARGS.use_sr:
                """直接用最终输出分辨率"""
                if self.ARGS.frame_size != self.ARGS.resize_param and all(self.ARGS.resize_param):
                    output_dict.update({"-s": f"{self.ARGS.resize_param[0]}x{self.ARGS.resize_param[1]}"})
            else:
                """超分"""
                if self.ARGS.transfer_ratio not in [RT_RATIO.AUTO, RT_RATIO.WHOLE] and all(self.ARGS.frame_size):
                    w, h = RT_RATIO.get_modified_resolution(self.ARGS.frame_size, self.ARGS.transfer_ratio)
                    output_dict.update({"-s": f"{w}x{h}"})

            """Applying One Click HDR"""
            if self.ARGS.hdr_cube_mode != LUTS_TYPE.NONE:
                lut_path = LUTS_TYPE.get_lut_path(self.ARGS.hdr_cube_mode)
                if not os.path.exists(lut_path):
                    self.logger.warning("Could not find target cube, skip applying one click HDR")
                else:
                    vf_args += f",lut3d={lut_path}"

            """Checking Color Management"""
            scale_args = ""
            if '-colorspace' in output_dict:
                scale_args = f",scale=in_color_matrix={output_dict['-colorspace']}:out_color_matrix={output_dict['-colorspace']}"
            if not frame_check and not self.ARGS.is_quick_extract:
                scale_args = f",format=yuv444p10le{scale_args},format=rgb48be"
                if RGB_TYPE.DTYPE == np.uint8 and self.ARGS.hdr_mode == HDR_STATE.NONE:  # No format filter for hdr in 8bit
                    scale_args += ",format=rgb24"
                output_dict.update({"-sws_flags": "+bicubic+full_chroma_int+accurate_rnd", })
            vf_args += scale_args
        vf_args += f",minterpolate=fps={self.ARGS.target_fps:.3f}:mi_mode=dup"

        """Update video filters"""
        output_dict["-vf"] = vf_args
        self.logger.debug(f"reader: {input_dict} {output_dict}")
        return FFmpegReader(filename=self.ARGS.input, inputdict=input_dict, outputdict=output_dict)

    def __run_rest(self, run_time: float):
        rest_exp = 3600
        if self.ARGS.multi_task_rest and self.ARGS.multi_task_rest_interval and \
                time.time() - run_time > self.ARGS.multi_task_rest_interval * rest_exp:
            self.logger.info(
                f"\n\nINFO - Time to Rest for 10 minutes! Rest for every {self.ARGS.multi_task_rest_interval} hour. ")
            time.sleep(600)
            return time.time()
        return run_time

    def __update_decode_process_time(self):
        if not self.decode_timer_on:
            self.decode_timer_on = True
            self.decode_time_start = time.time()
            return
        else:
            self.decode_timer_on = False
            decode_time = time.time() - self.decode_time_start
            self.ARGS.update_task_info({'decode_process_time': decode_time})

    def remove_duplicate_frames(self, videogen_check: FFmpegReader.nextFrame, init=False) -> (list, list, dict):
        """
        获得新除重预处理帧数序列
        :param init: 第一次重复帧
        :param videogen_check:
        :return:
        """
        flow_dict = dict()
        canny_dict = dict()
        predict_dict = dict()
        resize_param = (40, 40)

        def get_img(i0):
            if i0 in check_frame_data:
                return check_frame_data[i0]
            else:
                return None

        def sobel(src):
            src = cv2.GaussianBlur(src, (3, 3), 0)
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, -1, 3, 0, ksize=5)
            grad_y = cv2.Sobel(gray, -1, 0, 3, ksize=5)
            return cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

        def calc_flow_distance(pos0: int, pos1: int, _use_flow=True):
            if not _use_flow:
                return diff_canny(pos0, pos1)
            if (pos0, pos1) in flow_dict:
                return flow_dict[(pos0, pos1)]
            if (pos1, pos0) in flow_dict:
                return flow_dict[(pos1, pos0)]

            prev_gray = cv2.cvtColor(cv2.resize(get_img(pos0), resize_param), cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(cv2.resize(get_img(pos1), resize_param), cv2.COLOR_BGR2GRAY)
            flow0 = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                                 flow=None, pyr_scale=0.5, levels=1, iterations=20,
                                                 winsize=15, poly_n=5, poly_sigma=1.1, flags=0)
            flow1 = cv2.calcOpticalFlowFarneback(curr_gray, prev_gray,
                                                 flow=None, pyr_scale=0.5, levels=1, iterations=20,
                                                 winsize=15, poly_n=5, poly_sigma=1.1, flags=0)
            flow = (flow0 - flow1) / 2
            _x = flow[:, :, 0]
            _y = flow[:, :, 1]
            dis = np.linalg.norm(_x) + np.linalg.norm(_y)
            flow_dict[(pos0, pos1)] = dis
            return dis

        def diff_canny(pos0, pos1):
            if (pos0, pos1) in canny_dict:
                return canny_dict[(pos0, pos1)]
            if (pos1, pos0) in canny_dict:
                return canny_dict[(pos1, pos0)]
            img0, img1 = get_img(pos0), get_img(pos1)
            if self.ARGS.use_dedup_sobel:
                img0, img1 = sobel(img0), sobel(img1)
            canny_diff = cv2.Canny(cv2.absdiff(img0, img1), 100, 200).mean()
            canny_dict[(pos0, pos1)] = canny_diff
            return canny_diff

        def predict_scale(pos0, pos1):
            if (pos0, pos1) in predict_dict:
                return predict_dict[(pos0, pos1)]
            if (pos1, pos0) in predict_dict:
                return predict_dict[(pos1, pos0)]

            w, h, _ = get_img(pos0).shape
            diff = cv2.Canny(cv2.absdiff(get_img(pos0), get_img(pos0)), 100, 200)
            mask = np.where(diff != 0)
            try:
                xmin = min(list(mask)[0])
            except:
                xmin = 0
            try:
                xmax = max(list(mask)[0]) + 1
            except:
                xmax = w
            try:
                ymin = min(list(mask)[1])
            except:
                ymin = 0
            try:
                ymax = max(list(mask)[1]) + 1
            except:
                ymax = h
            W = xmax - xmin
            H = ymax - ymin
            S0 = w * h
            S1 = W * H
            prediction = -2 * (S1 / S0) + 3
            predict_dict[(pos0, pos1)] = prediction
            return prediction

        use_flow = True
        check_queue_size = max(self.ARGS.frames_queue_len, 200)  # 预处理长度，非重复帧
        check_frame_list = list()  # 采样图片帧数序列,key ~ LabData
        scene_frame_list = list()  # 转场图片帧数序列,key,和check_frame_list同步
        check_frame_data = dict()  # 用于判断的采样图片数据
        if init:
            self.logger.info("Initiating Duplicated Frames Removal(Dedup) Process...This might take some time")
            pbar = tqdm.tqdm(total=check_queue_size, unit="frames")
        else:
            pbar = None
        """
            check_frame_list contains key, check_frame_data contains (key, frame_data)
        """
        check_frame_cnt = -1
        while len(check_frame_list) < check_queue_size:
            check_frame_cnt += 1
            check_frame = Tools.gen_next(videogen_check)
            if check_frame is None:
                break
            check_frame = Tools.get_u1_from_u2_img(check_frame)
            if len(check_frame_list):  # len>1
                diff_result = Tools.get_norm_img_diff(check_frame_data[check_frame_list[-1]], check_frame)
                if diff_result < 0.001:
                    continue
            if init:
                pbar.update(1)
                pbar.set_description(
                    f"Process at Extract Frame {check_frame_cnt}")
            check_frame_data[check_frame_cnt] = check_frame
            check_frame_list.append(check_frame_cnt)  # key list
        if not len(check_frame_list):
            if init:
                pbar.close()
            return [], [], {}

        if init:
            pbar.close()
            pbar = tqdm.tqdm(total=len(check_frame_list), unit="frames")

        """Scene Batch Detection"""
        for i in range(len(check_frame_list) - 1):
            if init:
                pbar.update(1)
                pbar.set_description(f"Process at Scene Detect Frame {i}")
            i1 = check_frame_data[check_frame_list[i]]
            i2 = check_frame_data[check_frame_list[i + 1]]
            result = self.scene_detection.check_scene(i1, i2)
            if result:
                scene_frame_list.append(check_frame_list[i + 1])  # at i find scene

        if init:
            pbar.close()
            self.logger.info("Start Removing First Batch of Duplicated Frames")

        max_epoch = self.ARGS.remove_dup_mode  # 一直去除到一拍N，N为max_epoch，默认去除一拍二
        opt = []  # 已经被标记，识别的帧
        for queue_size, _ in enumerate(range(1, max_epoch), start=4):
            Icount = queue_size - 1  # 输入帧数
            Current = []  # 该轮被标记的帧
            i = 1
            try:
                while i < len(check_frame_list) - Icount:
                    c = [check_frame_list[p + i] for p in range(queue_size)]  # 读取queue_size帧图像 ~ 对应check_frame_list中的帧号
                    first_frame = c[0]
                    last_frame = c[-1]
                    count = 0
                    for step in range(1, queue_size - 2):
                        pos = 1
                        while pos + step <= queue_size - 2:
                            m0 = c[pos]
                            m1 = c[pos + step]
                            d0 = calc_flow_distance(first_frame, m0, use_flow)
                            d1 = calc_flow_distance(m0, m1, use_flow)
                            d2 = calc_flow_distance(m1, last_frame, use_flow)
                            value_scale = predict_scale(m0, m1)
                            if value_scale * d1 < d0 and value_scale * d1 < d2:
                                count += 1
                            pos += 1
                    if count == (queue_size * (queue_size - 5) + 6) / 2:
                        Current.append(i)  # 加入标记序号
                        i += queue_size - 3
                    i += 1
            except:
                self.logger.error(traceback.format_exc(limit=ArgumentManager.traceback_limit))
            for x in Current:
                if x not in opt:  # 优化:该轮一拍N不可能出现在上一轮中
                    for t in range(queue_size - 3):
                        opt.append(t + x + 1)
        delgen = sorted(set(opt))  # 需要删除的帧
        for d in delgen:
            if check_frame_list[d] not in scene_frame_list:
                check_frame_list[d] = -1

        max_key = np.max(list(check_frame_data.keys()))
        if max_key not in check_frame_list:
            check_frame_list.append(max_key)
        if 0 not in check_frame_list:
            check_frame_list.insert(0, 0)
        check_frame_list = [i for i in check_frame_list if i > -1]
        return check_frame_list, scene_frame_list, check_frame_data

    def __no_dedup_run(self):
        """
        Extract Frames Without any dedup(or scene detection)
        :return:
        """

        self.logger.info("Activate Any-FPS Mode without Dedup/Scene Detection)")
        chunk_cnt, now_frame, videogen, _ = self.__input_check(dedup=False)
        img1 = self.__crop(Tools.gen_next(videogen))
        self.logger.info("Input Frames loaded")
        is_end = False
        """Start Process"""
        run_time = time.time()
        self._release_initiation()
        while True:
            if is_end:
                break

            if self._kill or self.ARGS.get_main_error() is not None:
                self.logger.debug("Reader Thread Exit")
                break

            run_time = self.__run_rest(run_time)
            self.__update_decode_process_time()
            img0 = img1
            img1 = self.__crop(Tools.gen_next(videogen))

            now_frame += 1

            # Decode Review, should be annoted
            # title = f"decode:"
            # comp_stack = cv2.resize(img0, (2880, 1620))
            # # comp_stack = img0
            # cv2.imshow(title, cv2.cvtColor(comp_stack, cv2.COLOR_BGR2RGB))
            # cv2.moveWindow(title, 0, 0)
            # cv2.resizeWindow(title, 2880, 1620)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            if img1 is None:
                is_end = True
                self.__feed_to_rife(now_frame, img0, img0, n=0, is_end=is_end)
                break
            self.__feed_to_rife(now_frame, img0, img0, n=0, is_end=is_end)  # 当前模式下非重复帧间没有空隙，仅输入img0
            self.scene_detection.update_scene_status(now_frame, "normal")

            self.ARGS.update_task_info({"read_now_frame": now_frame})
            self.__update_scene_status()
            pass

        self._output_queue.put(None)  # bad way to end
        videogen.close()

    # @profile
    def __dedup_1xn_run(self):
        """
        Go through all procedures to produce interpolation result in dedup mode
        :return:
        """

        self.logger.info("Activate Duplicate Frames Removal Mode")
        chunk_cnt, now_frame_key, videogen, videogen_check = self.__input_check(dedup=True)
        self.logger.info("Input Frames loaded")
        is_end = False

        """Start Process"""
        run_time = time.time()
        first_run = True
        now_frame_cnt = now_frame_key
        while True:
            if is_end:
                break

            if self._kill or self.ARGS.get_main_error() is not None:
                self.logger.debug("Reader Thread Exit")
                break

            run_time = self.__run_rest(run_time)
            self.__update_decode_process_time()
            check_frame_list, scene_frame_list, input_frame_data = self.remove_duplicate_frames(videogen_check,
                                                                                                init=first_run)
            input_frame_data = dict(input_frame_data)
            first_run = False
            self._release_initiation()
            if not len(check_frame_list):
                while True:
                    img1 = self.__crop(Tools.gen_next(videogen))
                    if img1 is None:
                        is_end = True
                        self.__feed_to_rife(now_frame_cnt, img1, img1, n=0,
                                            is_end=is_end)
                        break
                    self.__feed_to_rife(now_frame_cnt, img1, img1, n=0)
                break

            else:
                img0 = self.__crop(Tools.gen_next(videogen))
                img1 = img0.copy()
                last_frame_key = check_frame_list[0]
                now_a_key = last_frame_key
                for frame_cnt in range(1, len(check_frame_list)):
                    now_b_key = check_frame_list[frame_cnt]
                    img1 = img0.copy()
                    """A - Interpolate -> B"""
                    while True:
                        last_possible_scene = img1
                        if now_a_key != now_b_key:
                            img1 = self.__crop(Tools.gen_next(videogen))
                            now_a_key += 1
                        else:
                            break
                    now_frame_key = now_b_key
                    self.ARGS.update_task_info({"read_now_frame": now_frame_cnt + now_frame_key})
                    if now_frame_key in scene_frame_list:
                        self.scene_detection.update_scene_status(now_frame_cnt + now_frame_key, "scene")
                        potential_key = now_frame_key - 1
                        if potential_key > 0 and potential_key in input_frame_data:
                            before_img = last_possible_scene
                        else:
                            before_img = img0

                        # Scene Review, should be annoted
                        # title = f"try:"
                        # comp_stack = np.hstack((img0, before_img, img1))
                        # comp_stack = cv2.resize(comp_stack, (1440, 270))
                        # cv2.imshow(title, cv2.cvtColor(comp_stack, cv2.COLOR_BGR2RGB))
                        # cv2.moveWindow(title, 0, 0)
                        # cv2.resizeWindow(title, 1440, 270)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                        if frame_cnt < 1:
                            # pass
                            self.__feed_to_rife(now_frame_cnt + now_frame_key, img0, img0, n=0,
                                                is_end=is_end)
                        elif self.ARGS.is_scdet_mix:
                            self.__feed_to_rife(now_frame_cnt + now_frame_key, img0, img1,
                                                n=now_frame_key - last_frame_key - 1,
                                                add_scene=True,
                                                is_end=is_end)
                        else:
                            self.__feed_to_rife(now_frame_cnt + now_frame_key, img0, before_img,
                                                n=now_frame_key - last_frame_key - 2,
                                                add_scene=True,
                                                is_end=is_end)
                    else:
                        self.scene_detection.update_scene_status(now_frame_cnt + now_frame_key, "normal")
                        self.__feed_to_rife(now_frame_cnt + now_b_key, img0, img1, n=now_frame_key - last_frame_key - 1,
                                            is_end=is_end)
                    last_frame_key = now_frame_key
                    img0 = img1
                    self.__update_scene_status()
                self.__feed_to_rife(now_frame_cnt + now_frame_key, img1, img1, n=0, is_end=is_end)
                self.ARGS.update_task_info({"read_now_frame": now_frame_cnt + check_frame_list[-1]})
                now_frame_cnt += last_frame_key

        pass
        self._output_queue.put(None)
        videogen.close()
        videogen_check.close()

    # @profile
    def __dedup_any_fps_run(self):
        """
        Go through all procedures to produce interpolation result in any fps mode(from a fps to b fps)
        :return:
        """

        self.logger.info("Activate Any FPS Mode")
        chunk_cnt, now_frame, videogen, _ = self.__input_check(dedup=False)
        img1 = self.__crop(Tools.gen_next(videogen))
        self.logger.info("Input Frames loaded")
        is_end = False

        """Update Interp Mode Info"""
        if self.ARGS.remove_dup_mode == 1:  # 单一模式
            self.ARGS.remove_dup_threshold = self.ARGS.remove_dup_threshold if self.ARGS.remove_dup_threshold > 0.01 else 0.01
        else:  # 0， 不去除重复帧
            self.ARGS.remove_dup_threshold = 0.001

        """Start Process"""
        run_time = time.time()
        self._release_initiation()

        while True:
            if is_end:
                break

            if self._kill or self.ARGS.get_main_error() is not None:
                self.logger.debug("Reader Thread Exit")
                break

            run_time = self.__run_rest(run_time)
            self.__update_decode_process_time()
            img0 = img1
            img1 = self.__crop(Tools.gen_next(videogen))

            now_frame += 1

            if img1 is None:
                is_end = True
                self.__feed_to_rife(now_frame, img0, img0, is_end=is_end)
                break

            diff = Tools.get_norm_img_diff(img0, img1)
            skip = 0  # 用于记录跳过的帧数

            """Find Scene"""
            if self.scene_detection.check_scene(img0, img1, use_diff=diff):
                self.__feed_to_rife(now_frame, img0, img1, n=0,
                                    is_end=is_end)  # add img0 only, for there's no gap between img0 and img1
                self.scene_detection.update_scene_status(now_frame, "scene")
                continue
            else:
                if diff < self.ARGS.remove_dup_threshold:
                    before_img = img1.copy()
                    is_scene = False
                    while diff < self.ARGS.remove_dup_threshold:
                        skip += 1
                        self.scene_detection.update_scene_status(now_frame, "dup")
                        last_frame = img1.copy()
                        img1 = self.__crop(Tools.gen_next(videogen))

                        if img1 is None:
                            img1 = last_frame
                            is_end = True
                            break

                        diff = Tools.get_norm_img_diff(img0, img1)

                        is_scene = self.scene_detection.check_scene(img0, img1, use_diff=diff)  # update scene stack
                        if is_scene:
                            break
                        if skip == self.ARGS.dup_skip_limit * self.ARGS.target_fps // self.ARGS.input_fps:
                            """超过重复帧计数限额，直接跳出"""
                            break

                    # 除去重复帧后可能im0，im1依然为转场，因为转场或大幅度运动的前一帧可以为重复帧
                    if is_scene:
                        if self.ARGS.is_scdet_mix:
                            self.__feed_to_rife(now_frame, img0, img1, n=skip, add_scene=True,
                                                is_end=is_end)
                        else:
                            self.__feed_to_rife(now_frame, img0, before_img, n=skip - 1, add_scene=True,
                                                is_end=is_end)
                            """
                            0 (1 2 3) 4[scene] => 0 (1 2) 3 4[scene] 括号内为RIFE应该生成的帧
                            """
                        self.scene_detection.update_scene_status(now_frame, "scene")

                    elif skip != 0:  # skip >= 1
                        assert skip >= 1
                        """Not Scene"""
                        self.__feed_to_rife(now_frame, img0, img1, n=skip, is_end=is_end)
                        self.scene_detection.update_scene_status(now_frame, "normal")
                    now_frame += skip
                else:
                    """normal frames"""
                    self.__feed_to_rife(now_frame, img0, img1, n=0, is_end=is_end)  # 当前模式下非重复帧间没有空隙，仅输入img0
                    self.scene_detection.update_scene_status(now_frame, "normal")

                self.ARGS.update_task_info({"read_now_frame": now_frame})
                self.__update_scene_status()
            pass

        self._output_queue.put(None)  # bad way to end
        videogen.close()

    def __feed_to_rife(self, now_frame: int, img0, img1, n=0, exp=0, is_end=False, add_scene=False, ):
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

        def normalize_dtype(img):
            # if img.dtype in (np.uint16, np.dtype('>u2'), np.dtype('<u2')):
            #     img = img.astype(np.float32)
            return img

        scale = self.__get_auto_scale(img0, img1)
        img0 = normalize_dtype(img0)
        img1 = normalize_dtype(img1)
        self.__update_decode_process_time()
        self._output_queue.put(
            {"now_frame": now_frame, "img0": img0, "img1": img1, "n": n, "scale": scale,
             "is_end": is_end, "add_scene": add_scene})

    def __get_auto_scale(self, img0, img1):
        scale = self.ARGS.rife_scale
        # if self.ARGS.use_rife_auto_scale and not self.ARGS.render_only and not self.ARGS.extract_only:
        #     """使用动态光流"""
        #     if img0 is None or img1 is None:
        #         scale = 1.0
        #     else:
        #         scale = self.vfi_core.get_auto_scale(img0, img1)
        return scale

    def update_vfi_core(self, vfi_core: VideoFrameInterpolationBase):
        self.vfi_core = vfi_core

    def __update_scene_status(self):
        scene_status = self.scene_detection.get_scene_status()
        update_dict = {'recent_scene': scene_status['recent_scene'], 'scene_cnt': scene_status['scene']}
        self.ARGS.update_task_info(update_dict)

    def run(self):
        try:
            if self.ARGS.render_only and self.ARGS.is_no_dedup_render:
                self.__no_dedup_run()
            elif self.ARGS.remove_dup_mode in [0, 1]:
                self.__dedup_any_fps_run()
            else:  # 1, 2 => 去重一拍二或一拍三
                self.__dedup_1xn_run()
            self._task_done()
        except Exception as e:
            self.logger.critical("Read Thread Panicked")
            self.logger.critical(traceback.format_exc(limit=ArgumentManager.traceback_limit))
            self.ARGS.save_main_error(e)
            return


class RenderFlow(IOFlow):
    def __init__(self, _args: TaskArgumentManager, __logger, _reader_queue: Queue):
        super().__init__(_args, __logger)
        self.name = 'Render'
        self.__ffmpeg = "ffmpeg"
        self.__hdr10_metadata_processer = Hdr10PlusProcessor(self.logger, self.ARGS.project_dir, self.ARGS.render_gap,
                                                             self.ARGS.interp_times,
                                                             self.ARGS.video_info_instance.getInputHdr10PlusMetadata())
        self._input_queue = _reader_queue
        self.is_audio_failed_concat = False
        self.validation_flow = None

    def __modify_hdr_params(self):
        if self.ARGS.is_img_input or self.ARGS.hdr_mode == HDR_STATE.CUSTOM_HDR:  # img input or ordinary hdr
            return

        if self.ARGS.hdr_mode in [HDR_STATE.HDR10, HDR_STATE.HDR10_PLUS]:
            """HDR10"""
            self.ARGS.render_encoder = "CPU"
            if "H265" in self.ARGS.render_encode_format:
                self.ARGS.render_encode_format = "H265, 10bit"
            elif "H264" in self.ARGS.render_encode_format:
                self.ARGS.render_encode_format = "H264, 10bit"
            # do not change encoder preset
            # self.ARGS.render_encoder_preset = "medium"
        elif self.ARGS.hdr_mode == HDR_STATE.HLG:
            """HLG"""
            self.ARGS.render_encode_format = "H265, 10bit"
            self.ARGS.render_encoder = "CPU"
            # self.ARGS.render_encoder_preset = "medium"

    def __generate_frame_writer(self, start_frame: int, output_path: str, _assign_last_n_frames=0):
        """
        渲染帧
        :param start_frame: for IMG IO, select start_frame to generate IO instance
        :param output_path:
        :return:
        """
        hdr10plus_metadata_path = self.__hdr10_metadata_processer.get_hdr10plus_metadata_path_at_point(start_frame)
        params_libx265s = {
            "fast": "asm=avx512:ref=2:rd=2:ctu=32:min-cu-size=16:limit-refs=3:limit-modes=1:rect=0:amp=0:early-skip=1:fast-intra=1:b-intra=1:rdoq-level=0:tu-intra-depth=1:tu-inter-depth=1:limit-tu=0:max-merge=2:me=1:subme=3:merange=25:weightb=1:strong-intra-smoothing=0:open-gop=0:keyint=250:min-keyint=1:rc-lookahead=15:lookahead-slices=8:b-adapt=1:bframes=4:aq-mode=2:aq-strength=1:qg-size=16:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:deblock=-1:sao=0:info=0",
            "fast_FD+ZL": "asm=avx512:ref=2:rd=2:ctu=32:min-cu-size=16:limit-refs=3:limit-modes=1:rect=0:amp=0:early-skip=1:fast-intra=1:b-intra=0:rdoq-level=0:tu-intra-depth=1:tu-inter-depth=1:limit-tu=0:max-merge=2:me=1:subme=3:merange=25:weightp=0:strong-intra-smoothing=0:open-gop=0:keyint=50:min-keyint=1:rc-lookahead=25:lookahead-slices=8:b-adapt=0:bframes=0:aq-mode=2:aq-strength=1:qg-size=16:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:deblock=false:sao=0:info=0",
            "slow": "asm=avx512:pmode=1:ref=4:rd=4:ctu=32:min-cu-size=8:limit-refs=1:limit-modes=1:rect=0:amp=0:early-skip=0:fast-intra=0:b-intra=1:rdoq-level=2:tu-intra-depth=1:tu-inter-depth=1:limit-tu=0:max-merge=4:me=3:subme=5:merange=25:weightb=1:strong-intra-smoothing=0:psy-rd=2:psy-rdoq=1:open-gop=0:keyint=250:min-keyint=1:rc-lookahead=35:lookahead-slices=4:b-adapt=2:bframes=6:aq-mode=2:aq-strength=0.8:qg-size=8:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:deblock=-1:sao=0:info=0",
            "slow_FD+ZL": "asm=avx512:pmode=1:ref=4:rd=4:ctu=32:min-cu-size=8:limit-refs=1:limit-modes=1:rect=0:amp=0:early-skip=0:fast-intra=0:b-intra=0:rdoq-level=2:tu-intra-depth=1:tu-inter-depth=1:limit-tu=0:max-merge=4:me=3:subme=5:merange=25:weightp=0:strong-intra-smoothing=0:psy-rd=2:psy-rdoq=1:open-gop=0:keyint=50:min-keyint=1:rc-lookahead=25:lookahead-slices=4:b-adapt=0:bframes=0:aq-mode=2:aq-strength=0.8:qg-size=8:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:deblock=false:sao=0:info=0",
            "hdr10": 'asm=avx512:pmode=1:ref=4:rd=4:ctu=32:min-cu-size=8:limit-refs=1:limit-modes=1:rect=0:amp=0:early-skip=0:fast-intra=0:b-intra=1:rdoq-level=2:tu-intra-depth=1:tu-inter-depth=1:limit-tu=0:max-merge=4:me=3:subme=5:merange=25:weightb=1:strong-intra-smoothing=0:psy-rd=2:psy-rdoq=1:open-gop=0:keyint=250:min-keyint=1:rc-lookahead=35:lookahead-slices=4:b-adapt=2:bframes=6:aq-mode=2:aq-strength=0.8:qg-size=8:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:deblock=-1:sao=0:range=limited:colorprim=9:transfer=16:colormatrix=9:hdr10-opt=1:repeat-headers=1:info=0',
            "hdr10+": 'asm=avx512:pmode=1:ref=4:rd=4:ctu=32:min-cu-size=8:limit-refs=1:limit-modes=1:rect=0:amp=0:early-skip=0:fast-intra=0:b-intra=1:rdoq-level=2:tu-intra-depth=1:tu-inter-depth=1:limit-tu=0:max-merge=4:me=3:subme=5:merange=25:weightb=1:strong-intra-smoothing=0:psy-rd=2:psy-rdoq=1:open-gop=0:keyint=250:min-keyint=1:rc-lookahead=35:lookahead-slices=4:b-adapt=2:bframes=6:aq-mode=2:aq-strength=0.8:qg-size=8:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:deblock=-1:sao=0:range=limited:colorprim=9:transfer=16:colormatrix=9:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50):hdr10-opt=1:repeat-headers=1:info=0',
        }

        params_libx264s = {
            "fast": "keyint=250:min-keyint=1:bframes=3:b-adapt=1:open-gop=0:ref=2:rc-lookahead=20:chroma-qp-offset=-1:aq-mode=1:aq-strength=0.9:mbtree=0:qcomp=0.60:weightp=1:me=hex:merange=16:subme=7:psy-rd='1.0:0.0':mixed-refs=0:trellis=1:deblock='-1:-1'",
            "fast_FD+ZL": "keyint=50:min-keyint=1:bframes=0:b-adapt=0:open-gop=0:ref=2:rc-lookahead=25:chroma-qp-offset=-1:aq-mode=1:aq-strength=0.9:mbtree=0:qcomp=0.60:weightp=0:me=hex:merange=16:subme=7:psy-rd='1.0:0.0':mixed-refs=0:trellis=1:deblock=false:cabac=0:weightb=0",
            "slow": "keyint=250:min-keyint=1:bframes=6:b-adapt=2:open-gop=0:ref=8:rc-lookahead=35:chroma-qp-offset=0:aq-mode=1:aq-strength=0.9:mbtree=1:qcomp=0.75:partitions=all:direct=auto:weightp=2:me=umh:merange=24:subme=10:psy-rd='1.0:0.1':mixed-refs=1:trellis=2:deblock='-1:-1'",
            "slow_FD+ZL": "keyint=50:min-keyint=1:bframes=0:b-adapt=0:open-gop=0:ref=8:rc-lookahead=25:chroma-qp-offset=0:aq-mode=1:aq-strength=0.9:mbtree=1:qcomp=0.75:partitions=all:direct=auto:weightp=0:me=umh:merange=24:subme=10:psy-rd='1.0:0.1':mixed-refs=1:trellis=2:deblock=false:cabac=0:weightb=0",
            "hdr10": "keyint=250:min-keyint=1:bframes=6:b-adapt=2:open-gop=0:ref=8:rc-lookahead=35:chroma-qp-offset=0:aq-mode=1:aq-strength=0.9:mbtree=1:qcomp=0.75:partitions=all:direct=auto:me=umh:merange=24:subme=10:psy-rd='1.0:0.1':mixed-refs=1:trellis=2:deblock='-1:-1'",
            "hdr10+": "keyint=250:min-keyint=1:bframes=6:b-adapt=2:open-gop=0:ref=8:rc-lookahead=35:chroma-qp-offset=0:aq-mode=1:aq-strength=0.9:mbtree=1:qcomp=0.75:partitions=all:direct=auto:me=umh:merange=24:subme=10:psy-rd='1.0:0.1':mixed-refs=1:trellis=2:deblock='-1:-1':mastering-display='G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)'"
        }
        if self.ARGS.use_crf:
            for k in params_libx264s:
                params_libx264s[k] = f"crf={self.ARGS.render_crf}:" + params_libx264s[k]
            for k in params_libx265s:
                params_libx265s[k] = f"crf={self.ARGS.render_crf}:" + params_libx265s[k]
        else:  # vbr control
            for k in params_libx264s:
                params_libx264s[k] = f"bitrate={self.ARGS.render_bitrate * 1024:.0f}:" + params_libx264s[k]
            for k in params_libx265s:
                params_libx265s[k] = f"bitrate={self.ARGS.render_bitrate * 1024:.0f}:" + params_libx265s[k]

        """If output is sequence of frames"""
        if self.ARGS.is_img_output:
            resize_width, resize_height = self.__get_cropped_resize()
            img_io = ImageWrite(self.logger, folder=self.ARGS.output_dir, start_frame=start_frame,
                                exp=self.ARGS.rife_exp,
                                resize=(resize_width, resize_height), output_ext=self.ARGS.output_ext, )
            return img_io

        """HDR Check"""
        if self.ARGS.hdr_mode != HDR_STATE.CUSTOM_HDR:
            self.__modify_hdr_params()

        """Output Video"""
        input_dict = {"-vsync": "cfr"}

        output_dict = {"-r": f"{self.ARGS.target_fps:.3f}", "-preset:v": self.ARGS.render_encoder_preset,
                       "-metadata": f'title="Powered By SVFI {self.ARGS.version}"'}

        output_dict.update(self._get_color_tag_dict())

        if not self.ARGS.is_img_input:
            input_dict.update({"-r": f"{self.ARGS.target_fps:.3f}"})
        else:
            """Img Input"""
            input_dict.update({"-r": f"{self.ARGS.input_fps * self.ARGS.interp_times:.3f}"})

        """Slow motion design"""
        if self.ARGS.is_render_slow_motion:
            if self.ARGS.render_slow_motion_fps:
                input_dict.update({"-r": f"{self.ARGS.render_slow_motion_fps:.3f}"})
            else:
                input_dict.update({"-r": f"{self.ARGS.target_fps:.3f}"})
            output_dict.pop("-r")

        vf_args = "format=yuv444p10le"
        if '-colorspace' in output_dict:
            vf_args = f"scale=out_color_matrix={output_dict['-colorspace']},{vf_args}"
        output_dict.update({"-vf": vf_args})

        resize_width, resize_height = self.__get_cropped_resize()
        if resize_height and resize_width:
            output_dict.update({"-sws_flags": "bicubic+accurate_rnd+full_chroma_int",
                                "-s": f"{resize_width}x{resize_height}"})

        """Assign Render Codec"""
        """CRF / Bitrate Control"""
        if self.ARGS.render_encoder == "CPU":
            if "H264" in self.ARGS.render_encode_format:
                output_dict.update({"-c:v": "libx264", "-preset:v": self.ARGS.render_encoder_preset})
                if "8bit" in self.ARGS.render_encode_format:
                    output_dict.update({"-pix_fmt": "yuv420p", "-profile:v": "high"})
                else:
                    """10bit"""
                    output_dict.update({"-pix_fmt": "yuv420p10le", "-profile:v": "high10"})
                if self.ARGS.hdr_mode == HDR_STATE.HDR10:
                    """HDR10"""
                    output_dict.update({"-x264-params": params_libx264s["hdr10"]})
                elif self.ARGS.hdr_mode == HDR_STATE.HDR10_PLUS:
                    """HDR10"""
                    output_dict.update({"-x264-params": params_libx264s["hdr10+"]})
                elif self.ARGS.use_render_zld:
                    if 'fast' in self.ARGS.render_encoder_preset:
                        output_dict.update({"-tune:v": "fastdecode,zerolatency", "-x264-params": params_libx264s["fast_FD+ZL"]})
                    if 'slow' in self.ARGS.render_encoder_preset:
                        output_dict.update({"-tune:v": "fastdecode,zerolatency", "-x264-params": params_libx264s["slow_FD+ZL"]})
                else:
                    if 'fast' in self.ARGS.render_encoder_preset:
                        output_dict.update({"-x264-params": params_libx264s["fast"]})
                    if 'slow' in self.ARGS.render_encoder_preset:
                        output_dict.update({"-x264-params": params_libx264s["slow"]})

                if self.ARGS.use_render_encoder_default_preset:
                    output_dict.pop('-x264-params')

            elif "H265" in self.ARGS.render_encode_format:
                output_dict.update({"-c:v": "libx265", "-preset:v": self.ARGS.render_encoder_preset})

                if "8bit" in self.ARGS.render_encode_format:
                    output_dict.update({"-pix_fmt": "yuv420p", "-profile:v": "main"})
                else:
                    """10bit"""
                    output_dict.update({"-pix_fmt": "yuv420p10le", "-profile:v": "main10"})

                if self.ARGS.hdr_mode == HDR_STATE.HDR10:
                    """HDR10"""
                    output_dict.update({"-x265-params": params_libx265s["hdr10"]})
                elif self.ARGS.hdr_mode == HDR_STATE.HDR10_PLUS:
                    """HDR10+"""
                    hdr_param = params_libx265s["hdr10+"]
                    if os.path.isfile(hdr10plus_metadata_path):
                        hdr_param += f":dhdr10-info='{hdr10plus_metadata_path}'"
                    output_dict.update({"-x265-params": hdr_param})
                elif self.ARGS.use_render_zld:
                    if 'fast' in self.ARGS.render_encoder_preset:
                        output_dict.update({"-tune:v": "fastdecode","-x265-params": params_libx265s["fast_FD+ZL"]})
                    if 'slow' in self.ARGS.render_encoder_preset:
                        output_dict.update({"-tune:v": "fastdecode","-x265-params": params_libx265s["slow_FD+ZL"]})
                else:
                    if 'fast' in self.ARGS.render_encoder_preset:
                        output_dict.update({"-x265-params": params_libx265s["fast"]})
                    if 'slow' in self.ARGS.render_encoder_preset:
                        output_dict.update({"-x265-params": params_libx265s["slow"]})

                if self.ARGS.use_render_encoder_default_preset:
                    output_dict.pop('-x265-params')
            elif "AV1" in self.ARGS.render_encode_format:
                encoder_preset_map = {"slow": "4", "ultrafast": "8", "fast": "7", "medium": "6", "veryslow": "3", }
                output_dict.update({"-c:v": "libsvtav1",
                                    "-preset:v": encoder_preset_map[self.ARGS.render_encoder_preset]})
                if "8bit" in self.ARGS.render_encode_format:
                    output_dict.update({"-pix_fmt": "yuv420p"})
                else:
                    """10bit"""
                    output_dict.update({"-pix_fmt": "yuv420p10le"})
            else:
                """ProRes"""
                if "-preset:v" in output_dict:
                    output_dict.pop("-preset:v")
                output_dict.update({"-c:v": "prores_ks", "-profile:v": self.ARGS.render_encoder_preset, })
                if "422" in self.ARGS.render_encode_format:
                    output_dict.update({"-pix_fmt": "yuv422p10le"})
                else:
                    output_dict.update({"-pix_fmt": "yuv444p10le"})

        elif self.ARGS.render_encoder == "NVENC":
            output_dict.update({"-pix_fmt": "yuv420p", "-profile:v": "main", "-rc:v": "constqp"})
            if "10bit" in self.ARGS.render_encode_format:
                output_dict.update({"-pix_fmt": "yuv420p10le", "-profile:v": "main10"})
                pass
            if "H264" in self.ARGS.render_encode_format:
                output_dict.update(
                    {f"-g": f"{int(self.ARGS.target_fps * 3)}", "-c:v": "h264_nvenc", })
            elif "H265" in self.ARGS.render_encode_format:
                output_dict.update({"-c:v": "hevc_nvenc",
                                    f"-g": f"{int(self.ARGS.target_fps * 3)}", })

            if self.ARGS.render_encoder_preset != "loseless":
                hwacccel_preset = self.ARGS.render_nvenc_preset
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
                output_dict.update({"-preset:v": "10", })

        elif self.ARGS.render_encoder == "NVENCC":
            _input_dict = {  # '--avsw': '',
                'encc': "NVENCC",
                '--fps': output_dict['-r'] if '-r' in output_dict else input_dict['-r'],
                "-pix_fmt": "rgb24",
            }
            _output_dict = {
                # "--chroma-qp-offset": "-2",
                "--lookahead": "16",
                "--gop-len": "250",
                "-b": "4",
                "--ref": "8",
                "--aq": "",
                "--aq-temporal": "",
                "--bref-mode": "middle"}
            self.convert_encc_color_tag(output_dict, _output_dict)

            if '-s' in output_dict:
                _output_dict.update({'--output-res': output_dict['-s']})
            if "10bit" in self.ARGS.render_encode_format:
                _output_dict.update({"--output-depth": "10"})
            if "H264" in self.ARGS.render_encode_format:
                _output_dict.update({f"-c": f"h264",
                                     "--profile": "high10" if "10bit" in self.ARGS.render_encode_format else "high", })
            elif "H265" in self.ARGS.render_encode_format:
                _output_dict.update({"-c": "hevc",
                                     "--profile": "main10" if "10bit" in self.ARGS.render_encode_format else "main",
                                     "--tier": "main", "-b": "5"})

            if self.ARGS.hdr_mode in [HDR_STATE.HDR10, HDR_STATE.HDR10_PLUS]:
                """HDR10"""
                _output_dict.update({"-c": "hevc",
                                     "--profile": "main10",
                                     "--tier": "main", "-b": "5", })
                if self.ARGS.hdr_mode == HDR_STATE.HDR10_PLUS:
                    _output_dict.update({"--max-cll": "1000,100",
                                         "--master-display": "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)"})
                    if os.path.isfile(hdr10plus_metadata_path):
                        _output_dict.update({"--dhdr10-info": hdr10plus_metadata_path})

            else:
                if self.ARGS.render_encoder_preset != "loseless":
                    _output_dict.update({"--preset": self.ARGS.render_encoder_preset})
                else:
                    _output_dict.update({"--lossless": "", "--preset": self.ARGS.render_encoder_preset})

            input_dict = _input_dict
            output_dict = _output_dict
            pass
        elif self.ARGS.render_encoder == "QSVENCC":
            _input_dict = {  # '--avsw': '',
                'encc': "QSVENCC",
                '--fps': output_dict['-r'] if '-r' in output_dict else input_dict['-r'],
                "-pix_fmt": "rgb24",
            }
            _output_dict = {
                "--fallback-rc": "", "--la-depth": "50", "--la-quality": "slow", "--extbrc": "", "--mbbrc": "",
                "--i-adapt": "",
                "--b-adapt": "", "--gop-len": "250", "-b": "6", "--ref": "8", "--b-pyramid": "", "--weightb": "",
                "--weightp": "", "--adapt-ltr": "",
            }
            self.convert_encc_color_tag(_output_dict, output_dict)

            if '-s' in output_dict:
                _output_dict.update({'--output-res': output_dict['-s']})
            if "10bit" in self.ARGS.render_encode_format:
                _output_dict.update({"--output-depth": "10"})
            if "H264" in self.ARGS.render_encode_format:
                _output_dict.update({f"-c": f"h264",
                                     "--profile": "high", "--repartition-check": "", "--trellis": "all"})
            elif "H265" in self.ARGS.render_encode_format:
                _output_dict.update({"-c": "hevc",
                                     "--profile": "main10" if "10bit" in self.ARGS.render_encode_format else "main",
                                     "--tier": "main", "--sao": "luma", "--ctu": "64", })
            if self.ARGS.hdr_mode in [HDR_STATE.HDR10, HDR_STATE.HDR10_PLUS]:
                _output_dict.update({"-c": "hevc",
                                     "--profile": "main10" if "10bit" in self.ARGS.render_encode_format else "main",
                                     "--tier": "main", "--sao": "luma", "--ctu": "64",
                                     })
                if self.ARGS.hdr_mode == HDR_STATE.HDR10_PLUS:
                    _output_dict.update({"--max-cll": "1000,100",
                                         "--master-display": "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)"})
            _output_dict.update({"--quality": self.ARGS.render_encoder_preset})

            input_dict = _input_dict
            output_dict = _output_dict
            pass
        elif self.ARGS.render_encoder == "SVT":
            _input_dict = {  # '--avsw': '',
                "-pix_fmt": "rgb24",
                '-n': f"{self.ARGS.render_gap}",
            }
            if _assign_last_n_frames:
                _input_dict.update({'n': f"{_assign_last_n_frames}"})
            if "H265" in self.ARGS.render_encode_format:
                _output_dict = {
                    "encc": "hevc",
                    '-fps': output_dict['-r'] if '-r' in output_dict else input_dict['-r'],
                    "-bit-depth": "8",
                    "-profile": "1",
                    "-level": "6.2",
                    "-lad": "17",
                    "-brr": "1", "-sharp": "1",
                }
            elif "VP9" in self.ARGS.render_encode_format:
                _output_dict = {
                    "encc": "vp9",
                    '-fps': output_dict['-r'] if '-r' in output_dict else input_dict['-r'],
                    "-bit-depth": "8",
                    "-tune": "0",
                    "-level": "6.2",
                }
                """Source Height must be a multiple of 8"""
                shit_flag = False
                if self.ARGS.resize_param[1] > 0:
                    if self.ARGS.resize_param[1] % 8:
                        shit_flag = True
                elif self.ARGS.frame_size[1] > 0 and self.ARGS.frame_size[1] % 8:
                    shit_flag = True
                if shit_flag:
                    raise OSError("For VP9 Encode, Source Height must be a multiple of 8, please alter input's height "
                                  "to a valid one")
            else:
                """AV1"""
                _output_dict = {
                    "encc": "av1",
                    "--input-depth": "8",
                    "--profile": "0",
                    "--level": "6.2",
                }
            preset_mapper = {"slowest": "2", "slow": "4", "fast": "6", "faster": "8"}
            if "H265" in self.ARGS.render_encode_format:
                _output_dict.update({"-encMode": preset_mapper[self.ARGS.render_encoder_preset]})
            elif "VP9" in self.ARGS.render_encode_format:
                _output_dict.update({"-enc-mode": preset_mapper[self.ARGS.render_encoder_preset]})
            else:
                """AV1"""
                _output_dict.update({"--preset": preset_mapper[self.ARGS.render_encoder_preset]})

            if '-s' in output_dict:
                _output_dict.update({'-s': output_dict['-s']})
            # if "10bit" in self.ARGS.render_encoder:
            #     _output_dict.update({"-bit-depth": "10"})
            # else:
            input_dict = _input_dict
            output_dict = _output_dict
            pass

        else:
            """QSV"""
            output_dict.update({"-pix_fmt": "yuv420p"})
            if "10bit" in self.ARGS.render_encode_format:
                output_dict.update({"-pix_fmt": "yuv420p10le"})
                pass
            if "H264" in self.ARGS.render_encode_format:
                output_dict.update({"-c:v": "h264_qsv",
                                    "-i_qfactor": "0.75", "-b_qfactor": "1.1",
                                    f"-rc-lookahead": "120", })
            elif "H265" in self.ARGS.render_encode_format:
                output_dict.update({"-c:v": "hevc_qsv",
                                    f"-g": f"{int(self.ARGS.target_fps * 3)}", "-i_qfactor": "0.75",
                                    "-b_qfactor": "1.1",
                                    f"-look_ahead": "120", })

        if "ProRes" not in self.ARGS.render_encode_format and self.ARGS.render_encoder_preset != "loseless":

            if self.ARGS.render_crf and self.ARGS.use_crf:
                encoder = self.ARGS.render_encoder
                if encoder == "CPU":
                    if 'AV1' in self.ARGS.render_encode_format:
                        output_dict.update({"-qp": str(self.ARGS.render_crf)})
                    else:
                        output_dict.update({"-crf": str(self.ARGS.render_crf)})
                elif encoder == "NVENC":
                    output_dict.update({"-cq:v": str(self.ARGS.render_crf)})
                elif encoder == "QSV":
                    output_dict.update({"-q": str(self.ARGS.render_crf)})
                elif encoder == "NVENCC":
                    output_dict.update({"--vbr": "0", "--vbr-quality": str(self.ARGS.render_crf)})
                elif encoder == "QSVENCC":
                    output_dict.update({"--la-icq": str(self.ARGS.render_crf)})
                elif encoder == "SVT":
                    if "VP9" in self.ARGS.render_encode_format or "H265" in self.ARGS.render_encode_format:
                        output_dict.update({"-q": str(self.ARGS.render_crf)})
                    else:
                        """AV1"""
                        output_dict.update({"--crf": str(self.ARGS.render_crf)})
            if self.ARGS.render_bitrate and self.ARGS.use_bitrate:
                if self.ARGS.render_encoder in ["NVENCC", "QSVENCC"]:
                    output_dict.update({"--vbr": f'{int(self.ARGS.render_bitrate * 1024)}'})
                elif self.ARGS.render_encoder == "SVT":
                    if "VP9" in self.ARGS.render_encode_format or "H265" in self.ARGS.render_encode_format:
                        output_dict.update({"-tbr": f'{int(self.ARGS.render_bitrate * 1024)}'})
                    else:
                        """AV1"""
                        output_dict.update({"--tbr": f'{int(self.ARGS.render_bitrate * 1024)}'})
                else:
                    """CPU"""
                    # if 'AV1' in self.ARGS.render_encode_format:
                    #     output_dict.update({"-rc:v": "vbr"})
                    if 'NVENC' in self.ARGS.render_encoder:
                        output_dict.update({"-rc:v": f'vbr'})
                    output_dict.update({"-b:v": f'{self.ARGS.render_bitrate}M'})
                if self.ARGS.render_encoder == "QSV":
                    output_dict.update({"-maxrate": "200M"})

        if self.ARGS.use_manual_encode_thread and self.ARGS.render_encoder == "CPU":
            output_dict.update({"-threads": f"{self.ARGS.render_encode_thread}"})

        self.logger.debug(f"render system parameters: {output_dict}, {input_dict}")

        """Customize FFmpeg Render Parameters"""
        ffmpeg_customized_command = {}
        if type(self.ARGS.render_ffmpeg_customized) is str and len(self.ARGS.render_ffmpeg_customized):
            for param, arg in Tools.get_custom_cli_params(self.ARGS.render_ffmpeg_customized).items():
                ffmpeg_customized_command.update({param: arg})
        self.logger.debug(f"render detected custom parameters: {ffmpeg_customized_command}")
        output_dict.update(ffmpeg_customized_command)
        if self.ARGS.render_encoder in ["NVENCC", "QSVENCC"]:
            return EnccWriter(filename=output_path, inputdict=input_dict, outputdict=output_dict,
                              verbosity=self.ARGS.debug)
        elif self.ARGS.render_encoder in ["SVT"]:
            return SVTWriter(filename=output_path, inputdict=input_dict, outputdict=output_dict,
                             verbosity=self.ARGS.debug)
        return FFmpegWriter(filename=output_path, inputdict=input_dict, outputdict=output_dict,
                            verbosity=self.ARGS.debug)

    def __get_cropped_resize(self):
        if not all(self.ARGS.resize_param):
            return self.ARGS.resize_param
        resize_width, resize_height = self.ARGS.resize_param
        if resize_width - self.ARGS.crop_param[0] * 2 and resize_height - self.ARGS.crop_param[1] * 2:
            resize_width = resize_width - self.ARGS.crop_param[0] * 2
            resize_height = resize_height - self.ARGS.crop_param[1] * 2
        return resize_width, resize_height

    @staticmethod
    def convert_encc_color_tag(ffmpeg_param_dict: dict, encc_param_dict: dict):
        if '-color_range' in ffmpeg_param_dict:
            encc_param_dict.update({"--colorrange": ffmpeg_param_dict["-color_range"]})
        if '-colorspace' in ffmpeg_param_dict:
            encc_param_dict.update({"--colormatrix": ffmpeg_param_dict["-colorspace"]})
        if '-color_trc' in ffmpeg_param_dict:
            encc_param_dict.update({"--transfer": ffmpeg_param_dict["-color_trc"]})
        if '-color_primaries' in ffmpeg_param_dict:
            encc_param_dict.update({"--colorprim": ffmpeg_param_dict["-color_primaries"]})

    def __rename_chunk(self, chunk_from_path: str, chunk_cnt: int, start_frame: int, end_frame: int):
        """Maintain Chunk json"""
        if self.ARGS.is_img_output or self._kill:
            return
        chunk_desc_path = "chunk-{:0>3d}-{:0>8d}-{:0>8d}{}".format(chunk_cnt, start_frame, end_frame,
                                                                   self.ARGS.output_ext)
        chunk_desc_path = os.path.join(self.ARGS.project_dir, chunk_desc_path)
        if os.path.exists(chunk_desc_path):
            os.remove(chunk_desc_path)
        if os.path.exists(chunk_from_path):
            os.rename(chunk_from_path, chunk_desc_path)
        else:
            self.logger.warning(f"Renamed Chunk Not found: {chunk_from_path}")

    def __check_audio_concat(self, chunk_tmp_path: str, fail_signal=0):
        """Check Input file ext"""
        if not self.ARGS.is_save_audio or self.ARGS.is_encode_audio or self.ARGS.get_main_error() is not None:
            return
        if self.ARGS.is_img_output:
            return
        concat_filepath = f"{os.path.join(self.ARGS.output_dir, 'concat_test')}" + self.ARGS.output_ext
        map_audio = f'-i "{self.ARGS.input}" -map 0:v:0 -map 1:a:0 -map 1:s? -c:a copy -c:s copy -shortest '
        ffmpeg_command = f'{self.__ffmpeg} -hide_banner -i "{chunk_tmp_path}" {map_audio} -c:v copy ' \
                         f'{Tools.fillQuotation(concat_filepath)} -y'

        self.logger.info("Start Audio Mux Test")
        sp = Tools.popen(ffmpeg_command)
        sp.wait()
        concat_return_code = sp.returncode
        if not os.path.exists(concat_filepath) or not os.path.getsize(concat_filepath) or concat_return_code:
            self.logger.warning(f"Audio Mux Test found unavailable audio codec for output extension: "
                                f"{self.ARGS.output_ext}, audio codec is changed to AAC 640kbps")
            self.is_audio_failed_concat = True
        else:
            self.logger.info("Audio Mux Test Succeeds")
            os.remove(concat_filepath)

    def get_output_path(self):
        """
        Get Output Path for Process
        :return:
        """
        """Check Input file ext"""
        output_ext = self.ARGS.output_ext
        if "ProRes" in self.ARGS.render_encode_format:
            output_ext = ".mov"

        output_filepath = f"{os.path.join(self.ARGS.output_dir, Tools.get_filename(self.ARGS.input))}"
        if self.ARGS.render_only:
            output_filepath += ".SVFI_Render"  # 仅渲染
        output_filepath += f".{int(self.ARGS.target_fps)}fps"  # 补帧

        if self.ARGS.is_render_slow_motion:  # 慢动作
            output_filepath += f".SLM={self.ARGS.render_slow_motion_fps}fps"
        if self.ARGS.is_16bit_workflow:
            output_filepath += f".16bit"
        if self.ARGS.is_quick_extract:
            output_filepath += f".QE"
        if self.ARGS.use_hwaccel_decode:
            output_filepath += f".HW"
        if self.ARGS.use_deinterlace:
            output_filepath += f".DI"
        if self.ARGS.use_fast_denoise:
            output_filepath += f".DN"
        if self.ARGS.hdr_cube_mode != LUTS_TYPE.NONE:
            output_filepath += f".OCHDR"

        if not self.ARGS.render_only:
            """RIFE"""
            if self.ARGS.use_rife_auto_scale:
                output_filepath += f".DS"
            else:
                output_filepath += f".S={self.ARGS.rife_scale}"  # 全局光流尺度
            if self.ARGS.use_ncnn:
                output_filepath += ".NCNN"
            output_filepath += f".RIFE={os.path.basename(self.ARGS.rife_model_name)}"  # 添加模型信息
            if self.ARGS.use_rife_fp16:
                output_filepath += ".FP16"
            if self.ARGS.is_rife_reverse:
                output_filepath += ".RR"
            if self.ARGS.use_rife_forward_ensemble:
                output_filepath += ".RFE"
            if VideoFrameInterpolationBase.get_model_version(self.ARGS.rife_model_name) == RIFE_TYPE.RIFEvAnyTime:
                if self.ARGS.rife_layer_connect_mode == 0:
                    output_filepath += ".LM=cunet"
                if self.ARGS.rife_layer_connect_mode == 1:
                    output_filepath += ".LM=residual"
                else:
                    output_filepath += ".LM=direct"

            if self.ARGS.rife_tta_mode:
                output_filepath += f".TTA={self.ARGS.rife_tta_mode}-{self.ARGS.rife_tta_iter}"
            if self.ARGS.remove_dup_mode:  # 去重模式
                output_filepath += f".FD={self.ARGS.remove_dup_mode}"

        if self.ARGS.use_sr:  # 使用超分
            sr_model = os.path.splitext(self.ARGS.use_sr_model)[0]
            output_filepath += f".SR={os.path.splitext(sr_model)[0]}"

        output_filepath += f"_{self.ARGS.task_id[-6:]}"
        output_filepath += output_ext  # 添加后缀名
        return output_filepath, output_ext

    def __check_concat_fail_circumstances(self):
        if not self.ARGS.is_encode_audio:
            self.ARGS.is_encode_audio = True
            self.logger.warning("Audio will be Encoded into AAC 640kbps to avoid mux error")
            return True
        return False

    # @profile
    @overtime_reminder_deco(120, "Concat Chunks",
                            "This is normal for long footage concat which more than 30 chunks, please wait patiently until concat is done")
    def concat_all(self):
        """
        Concat all the chunks
        :return:
        """

        os.chdir(self.ARGS.project_dir)
        concat_path = os.path.join(self.ARGS.project_dir, "concat.ini")
        self.logger.info("Final Interpolation Round is Finished, Start Concating Chunks")
        concat_list = list()

        for f in os.listdir(self.ARGS.project_dir):
            if re.match("chunk-\d+-\d+-\d+", f):
                concat_list.append(os.path.join(self.ARGS.project_dir, f))
            else:
                self.logger.debug(f"Concat escape {f}")

        concat_list.sort(key=lambda x: int(os.path.basename(x).split('-')[2]))  # sort as start-frame

        if not len(concat_path):
            raise OSError(
                f"Could not find any chunks in your output project folder, the chunks could have already been concatenated or removed, "
                f"please check your output folder.")

        if os.path.exists(concat_path):
            os.remove(concat_path)

        with open(concat_path, "w+", encoding="UTF-8") as w:
            for f in concat_list:
                w.write(f"file '{f}'\n")

        concat_filepath, output_ext = self.get_output_path()

        if self.ARGS.is_save_audio and not self.ARGS.is_img_input:
            audio_path = self.ARGS.input
            map_audio = f'-i "{audio_path}" -map 0:v:0 -map 1:a? -map 1:s? -c:a copy -c:s copy '
            if self.ARGS.input_start_point or self.ARGS.input_end_point:
                map_audio = f'-i "{audio_path}" -map 0:v:0 -map 1:a? -c:a aac -ab 640k -map_chapters -1 '
                if self.ARGS.input_end_point is not None:
                    map_audio = f'-to {self.ARGS.input_end_point} {map_audio}'
                if self.ARGS.input_start_point is not None:
                    map_audio = f'-ss {self.ARGS.input_start_point} {map_audio}'
            elif self.is_audio_failed_concat or self.ARGS.is_encode_audio:
                # not specific io point, and audio concat test failed, so audio is encoded into aac compulsorily,
                # and subtitle is disabled
                map_audio = f'-i "{audio_path}" -map 0:v:0 -map 1:a? -map 1:s? -c:a aac -ab 640k '
            # Special Case Optimization
            # if self.ARGS.input_ext in ['.vob'] and self.ARGS.output_ext in ['.mkv']:
            #     map_audio += "-map_chapters -1 "
        else:
            map_audio = ""

        color_dict = self._get_color_tag_dict()
        color_info_str = ""
        for ck, cd in color_dict.items():
            color_info_str += f" {ck} {cd}"

        ffmpeg_command = f'{self.__ffmpeg} -hide_banner -f concat -safe 0 -i "{concat_path}" {map_audio} -c:v copy ' \
                         f'-metadata title="Powered By SVFI {self.ARGS.version}" ' \
                         f'{color_info_str} ' \
                         f'-y {Tools.fillQuotation(concat_filepath)}'

        self.logger.debug(f"Concat command: {ffmpeg_command}")
        concat_return_code = 0
        try:
            sp = Tools.popen(ffmpeg_command)
            sp.wait()
            concat_return_code = sp.returncode
        except Exception as e:
            if self.__check_concat_fail_circumstances():  # audio mux type modified to aac
                self.logger.warning("Retry Concat after FFmpeg failed")
                self.concat_all()
            else:
                self.logger.info("Failed To Concat Chunks, all chunks will be preserved")
                self.ARGS.save_main_error(e)
                raise e

        self.logger.info(f"{len(concat_list)} files concatenated to {os.path.basename(concat_filepath)}")
        if not os.path.exists(concat_filepath) or not os.path.getsize(concat_filepath) or concat_return_code:
            # If return code is not 0, then there must be error
            if self.__check_concat_fail_circumstances():
                self.logger.warning("Retry Concat after Output File Validity Check failed")
                self.concat_all()
            else:
                main_error = FileExistsError(
                    f"Concat Error with output extension {output_ext}, empty output detected, Please Check Your Output Extension!!!\n"
                    "WARNING - e.g. mkv input should match .mkv as output extension to avoid possible muxing issues")
                self.ARGS.save_main_error(main_error)
                raise main_error
        if self.ARGS.hdr_mode == HDR_STATE.DOLBY_VISION:
            self.__run_dovi(concat_filepath)
        if self.ARGS.is_output_only:
            self.__del_existed_chunks()

        if self.validation_flow is not None:
            self.validation_flow.steam_update_achv(concat_filepath)

    def check_concat_result(self):
        concat_filepath, output_ext = self.get_output_path()
        if os.path.exists(concat_filepath):
            self.logger.warning("Project with same Task ID is already finished, "
                                "Jump to Dolby Vision Check")
            if self.ARGS.hdr_mode == HDR_STATE.DOLBY_VISION:
                """Dolby Vision"""
                self.__run_dovi(concat_filepath)
            else:
                return True
        return False

    def __del_existed_chunks(self):
        chunk_paths, chunk_cnt, last_frame = Tools.get_existed_chunks(self.ARGS.project_dir)
        for f in chunk_paths:
            os.remove(os.path.join(self.ARGS.project_dir, f))

    def __run_dovi(self, concat_filepath: str):
        self.logger.info("Start DOVI Conversion")
        dovi_maker = DoviProcessor(concat_filepath, self.ARGS.input, self.ARGS.project_dir,
                                   self.ARGS.interp_times, self.logger)
        dovi_maker.run()

    def wait_for_input(self):
        outside_ok = True
        while self._input_queue.qsize() == 0:
            time.sleep(0.1)
            if self.ARGS.get_main_error():
                """Sth out there dead"""
                outside_ok = False
                break
        return outside_ok

    def run(self):
        """
                Render thread
                :return:
        """
        concat_test_flag = True
        chunk_cnt, start_frame = self.check_chunk()
        chunk_frame_cnt = start_frame  # number of frames of current output chunk
        chunk_tmp_path = os.path.join(self.ARGS.project_dir, f"chunk-tmp{self.ARGS.output_ext}")
        if self.ARGS.all_frames_cnt < self.ARGS.render_gap:
            self.ARGS.render_gap = self.ARGS.render_gap
        try:
            frame_writer = self.__generate_frame_writer(start_frame, chunk_tmp_path, )  # get frame renderer
            frame_written = False
            self._release_initiation()
            while True:
                if self._kill or not self.wait_for_input():
                    if frame_written:
                        frame_writer.close()
                    self.logger.debug("Render thread exit")
                    self.__rename_chunk(chunk_tmp_path, chunk_cnt, start_frame, chunk_frame_cnt)
                    break

                frame_data = self._input_queue.get()
                if frame_data is None:
                    if frame_written:
                        frame_writer.close()
                    self.__rename_chunk(chunk_tmp_path, chunk_cnt, start_frame, chunk_frame_cnt)
                    break

                now_frame = frame_data[0]
                frame = frame_data[1]
                frame = frame.astype(RGB_TYPE.DTYPE)
                if self.ARGS.use_fast_denoise and not self.ARGS.is_16bit_workflow:
                    frame = cv2.fastNlMeansDenoising(frame)
                _over_time_reminder_task = OverTimeReminderTask(15, "Encoder",
                                                                "Low Encoding speed detected (>15s per image), Please check your encode settings to avoid performance issues")
                self.ARGS.put_overtime_task(_over_time_reminder_task)

                render_process_time = time.time()

                # Write Review, should be annoted
                # title = f"render:"
                # comp_stack = cv2.resize(frame, (1440, 819))
                # # comp_stack = img0
                # cv2.imshow(title, cv2.cvtColor(comp_stack, cv2.COLOR_BGR2RGB))
                # cv2.moveWindow(title, 0, 0)
                # cv2.resizeWindow(title,1440, 819 )
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                if frame is not None:
                    frame_written = True
                    frame_writer.writeFrame(frame)
                render_process_time = time.time() - render_process_time
                self.ARGS.update_task_info({'render_process_time': render_process_time})
                _over_time_reminder_task.deactive()

                chunk_frame_cnt += 1
                self.ARGS.update_task_info({"chunk_cnt": chunk_cnt, "render": now_frame})  # update render info

                if not chunk_frame_cnt % self.ARGS.render_gap:
                    # start_frame start from 0, chunk_frame_cnt is ahead of start_frame by 1 frame,
                    # so start_frame is head of chunk_frame_cnt
                    frame_writer.close()
                    if concat_test_flag:
                        self.__check_audio_concat(chunk_tmp_path)
                        concat_test_flag = False
                    self.__rename_chunk(chunk_tmp_path, chunk_cnt, start_frame, chunk_frame_cnt - 1)
                    chunk_cnt += 1
                    start_frame = chunk_frame_cnt
                    _assign_last_n_frames = 0
                    frame_writer = self.__generate_frame_writer(start_frame, chunk_tmp_path, _assign_last_n_frames)
            if not self.ARGS.is_no_concat and not self.ARGS.is_img_output \
                    and not self._kill and self.ARGS.get_main_error() is None:
                # Do not concat when main error is detected
                self.concat_all()
        except Exception as e:
            self.logger.critical("Render Thread Panicked")
            self.logger.critical(traceback.format_exc(limit=ArgumentManager.traceback_limit))
            self.ARGS.save_main_error(e)
            return
        return

    def update_validation_flow(self, validation_flow: ValidationFlow):
        self.validation_flow = validation_flow


class SuperResolutionFlow(IOFlow):
    def __init__(self, _args: TaskArgumentManager, __logger, _reader_queue: Queue, _render_queue: Queue):
        super().__init__(_args, __logger)
        self.name = 'SuperResolution'
        self._input_queue = _reader_queue
        self._output_queue = _render_queue
        self.sr_module = SuperResolutionBase()  # 超分类
        self._vram_check_lock = threading.Event()
        self._vram_check_lock.clear()
        if not self.ARGS.use_sr:
            self._release_vram_check_lock()
            return

        sr_scale = self.ARGS.resize_exp
        sr_module_exp = self.ARGS.sr_module_exp

        if all(self.ARGS.resize_param) and all(self.ARGS.frame_size):
            resize_resolution = self.ARGS.resize_param[0] * self.ARGS.resize_param[1]
            original_resolution = self.ARGS.frame_size[0] * self.ARGS.frame_size[1]
            sr_scale = int(math.ceil(math.sqrt(resize_resolution / original_resolution)))
        elif self.ARGS.resize_exp:
            sr_scale = self.ARGS.resize_exp
        if sr_module_exp:
            sr_times = sr_scale / sr_module_exp
            ref_ratio = RT_RATIO.get_auto_transfer_ratio(sr_times)
            if self.ARGS.transfer_ratio == RT_RATIO.AUTO:
                self.ARGS.transfer_ratio = ref_ratio
            if sr_times <= 1:
                sr_scale = RT_RATIO.get_surplus_sr_scale(sr_scale, self.ARGS.transfer_ratio)

        frame_size = self.ARGS.resize_param  # it's useless as redundant input parameters for api compatibility
        # sr_scale = int(math.ceil(math.sqrt(sr_scale)))

        if any(self.ARGS.crop_param):
            self.ARGS.crop_param = RT_RATIO.get_modified_resolution(self.ARGS.crop_param, self.ARGS.transfer_ratio,
                                                                    keep_single=True)
        if sr_scale == 0:
            sr_scale = 2  # TODO wtf is here

        try:
            if self.ARGS.use_sr_algo == "waifu2x":
                import SuperResolution.SuperResolutionModule
                self.sr_module = SuperResolution.SuperResolutionModule.SvfiWaifu(model=self.ARGS.use_sr_model,
                                                                                 scale=sr_scale,
                                                                                 num_threads=self.ARGS.ncnn_thread,
                                                                                 resize=frame_size)
            # elif self.ARGS.use_sr_algo == "realSR":
            #     import SuperResolution.SuperResolutionModule
            #     self.sr_module = SuperResolution.SuperResolutionModule.SvfiRealSR(model=self.ARGS.use_sr_model,
            #                                                                       scale=sr_times,
            #                                                                       resize=frame_size)
            elif self.ARGS.use_sr_algo == "realESR":
                import SuperResolution.RealESRModule
                self.sr_module = SuperResolution.RealESRModule.SvfiRealESR(model=self.ARGS.use_sr_model,
                                                                           gpu_id=self.ARGS.use_specific_gpu,
                                                                           # TODO Assign another card here
                                                                           scale=sr_scale, tile=self.ARGS.sr_tilesize,
                                                                           half=self.ARGS.use_realesr_fp16,
                                                                           resize=frame_size)
            elif self.ARGS.use_sr_algo == "waifuCuda":
                import SuperResolution.WaifuCudaModule
                self.sr_module = SuperResolution.WaifuCudaModule.SvfiWaifuCuda(model=self.ARGS.use_sr_model,
                                                                               gpu_id=self.ARGS.use_specific_gpu,
                                                                               scale=sr_scale,
                                                                               tile=self.ARGS.sr_tilesize,
                                                                               half=self.ARGS.use_realesr_fp16,
                                                                               resize=frame_size)
            elif self.ARGS.use_sr_algo == "realCUGAN":
                import SuperResolution.RealCUGANModule
                self.sr_module = SuperResolution.RealCUGANModule.SvfiRealCUGAN(model=self.ARGS.use_sr_model,
                                                                               gpu_id=self.ARGS.use_specific_gpu,
                                                                               scale=sr_scale,
                                                                               tile_mode=self.ARGS.sr_realCUGAN_tilemode,
                                                                               half=self.ARGS.use_realesr_fp16,
                                                                               resize=frame_size)

            self.logger.info(
                f"Load Super Resolution Module at {self.ARGS.use_sr_algo}, "
                f"Model at {self.ARGS.use_sr_model}, "
                f"Transfer Mode = {self.ARGS.transfer_ratio.name}, "
                f"scale = {sr_scale}")

        except ImportError:
            self.logger.error(
                f"Import SR Module failed\n"
                f"{traceback.format_exc(limit=ArgumentManager.traceback_limit)}")
            self._release_vram_check_lock()
            self.ARGS.use_sr = False

    def _release_vram_check_lock(self):
        self._vram_check_lock.set()

    def acquire_vram_check_lock(self):
        self._vram_check_lock.wait()

    def vram_test(self):
        """
        VRAM Check for SR Module
        :return:
        """
        if not self.ARGS.use_sr:
            self._release_vram_check_lock()
            return
        try:
            resolution = self.get_valid_input_resolution_for_test()
            w, h = RT_RATIO.get_modified_resolution(resolution, self.ARGS.transfer_ratio)

            self.logger.info(f"Start Super Resolution VRAM Test: {w}x{h}")

            test_img0 = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
            self.sr_module.svfi_process(test_img0)
            self.logger.info(f"SR VRAM Test Success")
            self._release_vram_check_lock()
            del test_img0
        except Exception as e:
            self.logger.error("SR VRAM Check Failed, PLS Lower your transfer resolution or tilesize for RealESR\n" +
                              traceback.format_exc(limit=ArgumentManager.traceback_limit))
            self._release_vram_check_lock()
            raise e

    def wait_for_input(self):
        outside_ok = True
        while self._input_queue.qsize() == 0:
            time.sleep(0.1)
            if self.ARGS.get_main_error():
                """Sth out there dead"""
                outside_ok = False
                break
        return outside_ok

    def run(self):
        """
        SR thread
        :return:
        """
        self._release_initiation()
        try:
            self.vram_test()
            while True:
                task_acquire_time = time.time()
                if self._kill or not self.wait_for_input():
                    self.logger.debug("Super Resolution thread exit")
                    break
                task = self._input_queue.get()
                task_acquire_time = time.time() - task_acquire_time
                if task is None:
                    break
                if self.ARGS.use_sr:
                    """
                        task = {"now_frame", "img0", "img1", "n","scale", "is_end", "is_scene", "add_scene"}
                    """
                    now_frame = task["now_frame"]
                    img0 = task["img0"]
                    img1 = task["img1"]  # TODO in one image input scenario, this case will output two identical image.
                    _over_time_reminder_task = OverTimeReminderTask(60,
                                                                    "Super Resolution",
                                                                    "Low Super Resolution speed detected (>60s per image), Please consider lower your output settings to enhance speed")
                    self.ARGS.put_overtime_task(_over_time_reminder_task)

                    sr_process_time = time.time()
                    is_img01_equals = False
                    if img0 is not None and img1 is not None and Tools.get_norm_img_diff(img0, img1) == 0:
                        is_img01_equals = True
                    if img0 is not None:
                        img0 = self.sr_module.svfi_process(img0)
                    if img1 is not None:
                        if not is_img01_equals:
                            img1 = self.sr_module.svfi_process(img1)
                        else:
                            img1 = img0
                    sr_process_time = time.time() - sr_process_time
                    task['img0'] = img0
                    task['img1'] = img1
                    self.ARGS.update_task_info({'sr_now_frame': now_frame,
                                                'sr_task_acquire_time': task_acquire_time,
                                                'sr_process_time': sr_process_time,
                                                'sr_queue_len': self._input_queue.qsize()})
                    _over_time_reminder_task.deactive()
                self._output_queue.put(task)
        except Exception as e:
            self.logger.critical("Super Resolution Thread Panicked")
            self.logger.critical(traceback.format_exc(limit=ArgumentManager.traceback_limit))
            self._output_queue.put(None)
            self.ARGS.save_main_error(e)
            self._release_vram_check_lock()
            return
        self._task_done()
        return


class ProgressUpdateFlow(IOFlow):
    def __init__(self, _args: TaskArgumentManager, __logger, _read_flow: ReadFlow):
        super().__init__(_args, __logger)
        self.name = 'Pbar'
        self.start_frame = 0
        self.read_flow = _read_flow

    def __task_progress_complete_check(self, now_frame: int):
        if now_frame < self.ARGS.all_frames_cnt * 4 / 5:
            self.logger.warning("Detect that Task Complete Progress Rate (now frame / all frame cnt) "
                                "is lower than 80%, Please Check whether Your Disk of input file was disconnected "
                                "during the process and check whether output file exists and is in full length!")

    def __preview_vfi(self, now_frame):
        if not self.ARGS.is_preview_imgs:
            time.sleep(0.1)
            return
        preview_imgs = self.ARGS.get_preview_imgs()
        if len(preview_imgs) < 2:
            time.sleep(0.1)
            return
        screen_h, screen_w = self.ARGS.get_screen_size()
        title = f"SVFI Preview of Interpolated/Uplifted Frame"
        preview_w = screen_w // 2

        comp_stack = preview_imgs[len(preview_imgs) // 2]
        h, w, _ = comp_stack.shape
        if w > preview_w * 4:
            comp_stack = comp_stack[::4, ::4, :]
        elif w > preview_w * 2:
            comp_stack = comp_stack[::2, ::2, :]
        comp_stack = (comp_stack / RGB_TYPE.SIZE * 255.).astype(np.uint8)

        stack_h, stack_w, _ = comp_stack.shape
        preview_h = int(stack_h / stack_w * preview_w)
        comp_stack = cv2.resize(comp_stack, (preview_w, preview_h))

        cv2.putText(comp_stack,
                    f"Frame {now_frame}, {now_frame / self.ARGS.all_frames_cnt * 100:.2f}%",
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0))
        comp_stack = cv2.cvtColor(comp_stack, cv2.COLOR_BGR2RGB)
        # cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, comp_stack)
        cv2.waitKey(41)  # 1/24

    def update_start_frame(self, start_frame: int):
        self.start_frame = start_frame

    def __proceed_overtime_reminder_task(self):
        if self.ARGS.is_empty_overtime_task_queue():
            return
        _over_time_reminder_task = self.ARGS.get_overtime_task()
        # assert type(_over_time_reminder_task) is OverTimeReminderTask, "[OverTimeReminderProcess]: Not Match Type"
        if _over_time_reminder_task.is_active():
            if _over_time_reminder_task.is_overdue():
                function_name, interval, function_warning = _over_time_reminder_task.get_msgs()
                self.logger.warning(f"Function [{function_name}] exceeds {interval} seconds, {function_warning}")
            else:
                self.ARGS.put_overtime_task(_over_time_reminder_task)

    def run(self):
        """
        Start Progress Update Bar
        :return:
        """
        """(chunk_cnt, start_frame, end_frame, frame_cnt)"""
        self.read_flow.acquire_initiation_clock()
        pbar = tqdm.tqdm(total=self.ARGS.all_frames_cnt, unit="frames")
        pbar.moveto(n=self.start_frame)
        pbar.unpause()
        previous_cnt = self.start_frame
        self._release_initiation()
        while True:
            self.__proceed_overtime_reminder_task()
            task_status = self.ARGS.task_info  # render status quo
            postfix_dict = {"R": f"{task_status['render']}",
                            "C": f"{task_status['read_now_frame']}",
                            "RPT": f"{task_status['decode_process_time']:.2f}s",
                            "WPT": f"{task_status['render_process_time']:.2f}s"}
            if self.ARGS.render_only or self.ARGS.extract_only:
                now_frame = task_status['render']
                pbar_description = f"Process at Frame {now_frame}"
            else:
                now_frame = task_status['rife_now_frame']
                pbar_description = f"Process at Chunk {task_status['chunk_cnt']:0>3d}"
                postfix_dict.update({
                    "C": f"{now_frame}",
                    "S": f"{task_status['recent_scene']}",
                    "SC": f"{task_status['scene_cnt']}",
                    "TAT": f"{task_status['rife_task_acquire_time']:.2f}s",
                    "PT": f"{task_status['rife_process_time']:.2f}s",
                    "QL": f"{task_status['rife_queue_len']}"})
            if self.ARGS.use_sr:
                postfix_dict.update({'SR': f"{task_status['sr_now_frame']}",
                                     'SRTAT': f"{task_status['sr_task_acquire_time']:.2f}s",
                                     'SRPT': f"{task_status['sr_process_time']:.2f}s",
                                     "SRL": f"{task_status['sr_queue_len']}", })
            if self._kill or self.ARGS.get_main_error() is not None:
                self.__task_progress_complete_check(now_frame)
                break
            pbar.set_description(pbar_description)
            pbar.set_postfix(postfix_dict)
            pbar.update(now_frame - previous_cnt)
            previous_cnt = now_frame
            self.__preview_vfi(now_frame)
            # time.sleep(0.1)
        pbar.update(abs(self.ARGS.all_frames_cnt - previous_cnt))
        pbar.close()


class InterpWorkFlow:
    # @profile
    def __init__(self, __args: TaskArgumentManager, **kwargs):
        global logger
        self.ARGS = __args
        self.run_all_time = datetime.datetime.now()

        """EULA"""
        self.eula = EULAWriter()
        self.eula.boom()

        """Set Queues"""
        queue_len = self.ARGS.frames_queue_len
        self.read_task_queue = Queue(maxsize=queue_len)
        self.sr_task_queue = Queue(maxsize=queue_len)
        self.rife_task_queue = Queue(maxsize=queue_len)
        self.render_task_queue = Queue(maxsize=queue_len)

        self.validation_flow = ValidationFlow(self.ARGS)
        self.sr_flow = SuperResolutionFlow(self.ARGS, logger, self.read_task_queue, self.rife_task_queue)
        self.read_flow = ReadFlow(self.ARGS, logger, self.read_task_queue)
        self.render_flow = RenderFlow(self.ARGS, logger, self.render_task_queue)
        self.update_progress_flow = ProgressUpdateFlow(self.ARGS, logger, self.read_flow)

        """Set VFI Core"""
        self.vfi_core = VideoFrameInterpolationBase(self.ARGS, logger)

        """Set 'Global' Reminder"""

    def __feed_to_render(self, frames_list: list, is_end=False):
        """
        维护输出帧数组的输入（往输出渲染线程喂帧
        :param frames_list:
        :param is_end: 是否是视频结尾
        :return:
        """
        for frame_i, frame_data in enumerate(frames_list):
            if frame_data is None:
                self.render_task_queue.put(None)
                logger.debug("Put None to render_task_queue in advance")
                break
            self.render_task_queue.put(frame_data)  # 往输出队列（消费者）喂正常的帧
        if is_end:
            self.render_task_queue.put(None)
            logger.debug("Put None to render_task_queue")
        pass

    def vram_test(self):
        """
        显存测试
        :return:
        """
        if self.ARGS.use_sr:
            logger.debug("Waiting for SR Module VRAM Check Lock")
            self.sr_flow.acquire_vram_check_lock()
        try:
            if all(self.ARGS.resize_param):
                w, h = self.ARGS.resize_param
            elif all(self.ARGS.frame_size):
                w, h = self.ARGS.frame_size
            elif all(self.ARGS.first_frame_size):
                w, h = self.ARGS.first_frame_size
            else:
                w, h = (480, 270)
            if self.ARGS.use_rife_auto_scale:
                self.ARGS.rife_scale = 1
            logger.info(f"Start VFI VRAM Test: {w}x{h} with scale {self.ARGS.rife_scale}, "
                        f"Auto Scale {'on' if self.ARGS.use_rife_auto_scale else 'off'}, "
                        f"interlace inference mode: {self.ARGS.rife_interlace_inference}")

            test_img0, test_img1 = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8), \
                                   np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
            self.vfi_core.generate_n_interp(test_img0, test_img1, 1, self.ARGS.rife_scale)
            logger.info(f"Interpolation VRAM Test Success, Resume of workflow ahead")
            del test_img0, test_img1
        except Exception as e:
            logger.error("Interpolation VRAM Check Failed, PLS Lower your presets\n" + traceback.format_exc(
                limit=ArgumentManager.traceback_limit))
            raise e

    def __check_validation(self):
        if not self.validation_flow.CheckValidateStart():
            if self.validation_flow.GetValidateError() is not None:
                logger.error(f"Validation Failed: ")
                raise self.validation_flow.GetValidateError()
            return
        steam_dlc_check = self.validation_flow.CheckProDLC(0)
        if not steam_dlc_check:
            _msg = "SVFI - Professional DLC Not Purchased,"
            # if self.ARGS.extract_only or self.ARGS.render_only:
            #     raise GenericSteamException(f"{_msg} Extract/Render ToolBox Unavailable")
            if self.ARGS.input_start_point is not None or self.ARGS.input_end_point is not None:
                raise GenericSteamException(f"{_msg} Manual Input Section Unavailable")
            if self.ARGS.is_scdet_output or self.ARGS.is_scdet_mix:
                raise GenericSteamException(f"{_msg} Scdet Output/Mix Unavailable")
            if self.ARGS.use_sr:
                raise GenericSteamException(f"{_msg} Super Resolution Module Unavailable")
            if self.ARGS.use_rife_multi_cards:
                raise GenericSteamException(f"{_msg} Multi Video Cards Work flow Unavailable")
            if self.ARGS.use_deinterlace:
                raise GenericSteamException(f"{_msg} DeInterlace is Unavailable")
            if self.ARGS.use_rife_auto_scale:
                raise GenericSteamException(f"{_msg} RIFE Dynamic Scale is Unavailable")
            if self.ARGS.is_rife_reverse:
                raise GenericSteamException(f"{_msg} RIFE Reversed Flow is Unavailable")

    def __check_interp_prerequisite(self):
        if self.ARGS.render_only or self.ARGS.extract_only or self.ARGS.concat_only:
            return

        model_name_lower = self.ARGS.rife_model_name.lower()
        if 'abme' in model_name_lower:
            """model: abme_best"""
            _over_time_reminder_task = OverTimeReminderTask(15, "ABME VFI Module Load Failed",
                                                            "Import Cracked(>15s so far), Please terminate the process and check your CUDA version according to the manual")
            self.ARGS.put_overtime_task(_over_time_reminder_task)
            from ABME import inference_abme as inference
            self.vfi_core = inference.ABMEInterpolation(self.ARGS, logger)
            logger.warning("ABME VFI Module Loaded, Note that this is alpha only")
            _over_time_reminder_task.deactive()
        elif 'xvfi' in model_name_lower:
            """model: xvfi_*"""
            _over_time_reminder_task = OverTimeReminderTask(15, "XVFI Module Load Failed",
                                                            "Import Cracked(>15s so far), Please terminate the process and check your Environment according to the manual")
            self.ARGS.put_overtime_task(_over_time_reminder_task)
            from XVFI import inference_xvfi as inference
            self.vfi_core = inference.XVFInterpolation(self.ARGS, logger)
            logger.warning("XVFI VFI Module Loaded, Note that this is alpha only")
            _over_time_reminder_task.deactive()
        elif ('v7' in model_name_lower and 'multi' in model_name_lower) \
                or '4.' in model_name_lower \
                or 'anytime' in model_name_lower:
            """model: rife's official_v7_multi / official 4.0 / Master Zhe's hybrid model"""
            _over_time_reminder_task = OverTimeReminderTask(15, "RIFE Multi VFI Module Load Failed",
                                                            "Import Cracked(>15s so far), Please terminate the process and check your Environment according to the manual")
            self.ARGS.put_overtime_task(_over_time_reminder_task)
            from RIFE import inference_rife as inference
            self.vfi_core = inference.RifeMultiInterpolation(self.ARGS, logger)
            logger.warning("RIFE VFI Module Multi Version Loaded")
            _over_time_reminder_task.deactive()
            pass
        else:
            _over_time_reminder_task = OverTimeReminderTask(15, "RIFE VFI Module Load Failed",
                                                            "Import Cracked(>15s so far), Please terminate the process and check your Environment according to the manual")
            self.ARGS.put_overtime_task(_over_time_reminder_task)
            if self.ARGS.use_ncnn:
                self.ARGS.rife_model_name = os.path.basename(self.ARGS.rife_model)
                from RIFE import inference_rife_ncnn as inference
            else:
                try:
                    # raise Exception("Load Torch Failed Test")
                    from RIFE import inference_rife as inference
                except Exception:
                    logger.warning("Import Torch Failed, use NCNN-RIFE instead")
                    logger.error(traceback.format_exc(limit=ArgumentManager.traceback_limit))
                    self.ARGS.use_ncnn = True
                    self.ARGS.rife_model = "rife-v2"
                    self.ARGS.rife_model_name = "rife-v2"
                    from RIFE import inference_rife_ncnn as inference
            """Update RIFE Core"""
            self.vfi_core = inference.RifeInterpolation(self.ARGS, logger)
            _over_time_reminder_task.deactive()
            logger.info("RIFE VFI Module Loaded")
        self.vfi_core.initiate_algorithm()

        if not self.ARGS.use_ncnn:
            self.vram_test()
        self.read_flow.update_vfi_core(self.vfi_core)

    def __check_outside_error(self):
        if self.ARGS.get_main_error() is not None:
            logger.error("Error outside RIFE:")
            self.__feed_to_render([None], is_end=True)
            self.task_failed()

    def task_finish(self):
        self.__check_outside_error()
        logger.info(f"Program finished at {datetime.datetime.now()}: "
                    f"Duration: {datetime.datetime.now() - self.run_all_time}")
        logger.info("Please Note That Commercial Use of SVFI's Output is Strictly PROHIBITED, "
                    "Check EULA for more details")
        return 0

    def task_failed(self):
        self.read_flow.kill()
        self.render_flow.kill()
        self.sr_flow.kill()
        self.update_progress_flow.kill()
        logger.error(f"\n\n\nProgram Failed at {datetime.datetime.now()}: "
                     f"Duration: {datetime.datetime.now() - self.run_all_time}")
        return 1
        # if self.ARGS.get_main_error():
        #     raise self.ARGS.get_main_error()

    def wait_for_input(self):
        outside_ok = True
        while self.rife_task_queue.qsize() == 0:
            time.sleep(0.1)
            if self.ARGS.get_main_error():
                """Sth out there dead"""
                outside_ok = False
                break
        return outside_ok

    def run(self):
        """
        Main Thread of SVFI
        :return:
        """

        """Check Steam Validation"""
        self.__check_validation()

        """Go through the process"""
        if self.ARGS.concat_only:
            self.render_flow.concat_all()
            return self.task_finish()

        """Concat Already / Mission Conflict Check & Dolby Vision Sort"""
        if self.render_flow.check_concat_result():
            return self.task_finish()

        """Start Process"""
        try:
            """Get SR - Read Flow Task Thread"""
            self.sr_flow.start()

            """Load RIFE Model"""
            self.read_flow.start()
            self.update_progress_flow.start()
            self.__check_interp_prerequisite()
            self.render_flow.update_validation_flow(self.validation_flow)
            self.render_flow.start()

            PURE_SCENE_THRESHOLD = 20

            self.__check_outside_error()
            self.read_flow.acquire_initiation_clock()
            self.render_flow.acquire_initiation_clock()
            self.update_progress_flow.acquire_initiation_clock()

            while True:
                task_acquire_time = time.time()
                if not self.wait_for_input():
                    logger.debug("Main thread about to exit")
                    break
                task = self.rife_task_queue.get(timeout=3600)
                task_acquire_time = time.time() - task_acquire_time
                process_time = time.time()
                if task is None:
                    self.__feed_to_render([None], is_end=True)
                    break
                """
                task = {"now_frame", "img0", "img1", "n","scale", "is_end", "is_scene", "add_scene"}
                """
                now_frame = task["now_frame"]
                img0 = task["img0"]
                img1 = task["img1"]
                n = task["n"]
                scale = task["scale"]
                is_end = task["is_end"]
                add_scene = task["add_scene"]

                debug = False
                """Test
                1. 正常4K，解码编码
                2. 一拍N卡顿
                """

                if img1 is None:
                    self.__feed_to_render([None], is_end=True)
                    break

                frames_list = [img0]
                if self.ARGS.is_scdet_mix and add_scene:
                    mix_list = Tools.get_mixed_scenes(img0, img1, n + 1)
                    frames_list.extend(mix_list)
                else:
                    _over_time_reminder_task = OverTimeReminderTask(60, "Video Frame Interpolation",
                                                                    "Low interpolate speed detected (>60s per pair), Please consider lower your output settings to enhance speed")
                    self.ARGS.put_overtime_task(_over_time_reminder_task)
                    if n > 0:
                        if n > PURE_SCENE_THRESHOLD and Tools.check_pure_img(img0):
                            """It's Pure Img Sequence, Copy img0"""
                            for i in range(n):
                                frames_list.append(img0)
                        else:
                            interp_list = self.vfi_core.generate_n_interp(img0, img1, n=n, scale=scale, debug=debug)
                            frames_list.extend(interp_list)
                    if add_scene:  # [AA BBB CC DDD] E
                        frames_list.append(img1)
                    _over_time_reminder_task.deactive()

                feed_list = list()
                for i in frames_list:
                    feed_list.append([now_frame, i])
                if self.ARGS.use_evict_flicker or self.ARGS.use_rife_fp16:
                    img_ori = frames_list[0].copy()
                    frames_list[0] = self.vfi_core.generate_n_interp(img_ori, img_ori, n=1, scale=scale,
                                                                     debug=debug)
                    if add_scene:
                        img_ori = frames_list[-1].copy()
                        frames_list[-1] = self.vfi_core.generate_n_interp(img_ori, img_ori, n=1, scale=scale,
                                                                          debug=debug)

                process_time = time.time() - process_time
                self.__update_rife_progress(now_frame, task_acquire_time, process_time)
                self.__feed_to_render(feed_list, is_end=is_end)
                preview_imgs = [feed[1] for feed in feed_list]
                preview_imgs.append(img1)
                self.ARGS.update_preview_imgs(preview_imgs)
                if is_end:
                    break
        except Exception as e:
            logger.critical("Main Thread Panicked")
            logger.critical(traceback.format_exc(limit=ArgumentManager.traceback_limit))
            self.ARGS.save_main_error(e)
        if self.ARGS.get_main_error() is not None:
            """Shit happened after receiving None as end signal"""
            return self.task_failed()

        while self.render_flow.is_alive() or self.sr_flow.is_alive() or self.read_flow.is_alive():
            """等待渲染线程结束"""
            time.sleep(1)
        self.update_progress_flow.kill()
        return self.task_finish()

    def __update_rife_progress(self, now_frame, task_acquire_time, process_time, ):
        update_dict = {'rife_now_frame': now_frame,
                       'rife_task_acquire_time': task_acquire_time,
                       'rife_process_time': process_time,
                       'rife_queue_len': self.rife_task_queue.qsize()}
        self.ARGS.update_task_info(update_dict)
        pass


global_task_args_manager = TaskArgumentManager(global_args)

"""设置可见的gpu"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if int(global_task_args_manager.rife_cuda_cnt) != 0 and global_task_args_manager.use_rife_multi_cards:
    global_cuda_devices = [str(i) for i in range(global_task_args_manager.rife_cuda_cnt)]
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{','.join(global_cuda_devices)}"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{global_task_args_manager.use_specific_gpu}"

"""强制使用CPU"""
if global_task_args_manager.force_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = f""

global_workflow = InterpWorkFlow(global_task_args_manager)
global_workflow.run()
sys.exit(0)
