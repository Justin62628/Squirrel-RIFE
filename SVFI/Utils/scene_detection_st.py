# encoding=utf-8
import math
import warnings
from collections import deque
from queue import Queue

import cv2
import numpy as np
from sklearn import linear_model

from Utils.utils import Utils

warnings.filterwarnings("ignore")


class TransitionDetection:
    def __init__(self, scene_queue_length, scdet_threshold=50, output="", no_scdet=False,
                 use_fixed_scdet=False, fixed_max_scdet=50, **kwargs):
        """
        转场检测类
        :param scene_queue_length: 转场判定队列长度
        :param fixed_scdet:
        :param scdet_threshold: （标准输入）转场阈值，具体判定规则在__judge_mean里
        :param output: 输出（没用）
        :param no_scdet: 不进行转场识别
        :param use_fixed_scdet: 使用固定转场阈值
        :param fixed_max_scdet: 使用的最大转场阈值（配合use_fixed_scdet使用）
        :param kwargs:
        """
        self.view = False  # 控制预览
        self.utils = Utils()
        self.scdet_cnt = 0
        self.scdet_threshold = scdet_threshold
        self.scene_dir = os.path.join(os.path.dirname(output), "scene")  # 存储转场图片的文件夹路径
        self.dead_thres = 80  # 写死最高的absdiff
        self.born_thres = 1  # 写死判定为非转场的最低阈值

        self.scene_queue_len = scene_queue_length
        if kwargs.get("remove_dup_mode", 0) in [1, 2]:
            """去除重复帧一拍二或N"""
            self.scene_queue_len = 8  # 写死
        self.flow_queue = deque(maxlen=self.scene_queue_len)  # flow_cnt队列
        self.absdiff_queue = deque(maxlen=self.scene_queue_len)  # absdiff队列
        self.scene_stack = Queue(maxsize=self.scene_queue_len)  # 转场识别队列
        self.no_scdet = no_scdet
        self.use_fixed_scdet = use_fixed_scdet
        self.scedet_info = {"scene": 0, "normal": 0, "dup": 0, "recent_scene": -1}
        # 帧种类，scene为转场，normal为正常帧，dup为重复帧，即两帧之间的计数关系

        self.img1 = None
        self.img2 = None
        self.before_img = None
        if self.use_fixed_scdet:
            self.dead_thres = fixed_max_scdet

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
        return sub_array.var() ** 0.5

    def __judge_mean(self, flow_cnt, diff, flow):
        var_before = self.__check_var()
        self.flow_queue.append(flow_cnt)
        var_after = self.__check_var()
        if var_after - var_before > self.scdet_threshold and diff > self.born_thres and \
                flow_cnt > self.flow_queue[-2] and flow_cnt > np.mean(self.flow_queue):
            """Detect new scene"""
            self.see_flow(
                f"flow_cnt: {flow_cnt:.3f}, diff: {diff:.3f}, before: {var_before:.3f}, after: {var_after:.3f}, "
                f"cnt: {self.scdet_cnt + 1}", flow)
            self.flow_queue.clear()
            self.scdet_cnt += 1
            return True
        else:
            if diff > self.dead_thres:
                """不漏掉死差转场"""
                self.flow_queue.clear()
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
        if not self.view:
            return
        comp_stack = np.hstack((self.img1, self.img2))
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, cv2.cvtColor(comp_stack, cv2.COLOR_BGR2RGB))
        cv2.moveWindow(title, 0, 0)
        cv2.resizeWindow(title, 960, 270)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
                self.absdiff_queue.append(0)
            return False
        elif np.mean(self.absdiff_queue) == 0:
            """检测到00000001"""
            self.absdiff_queue.clear()
            self.scdet_cnt += 1
            self.see_result(f"absdiff: {diff:.3f}, Pure Scene Alarm, cnt: {self.scdet_cnt}")
            return True

        flow_cnt, flow = self.utils.get_norm_img_flow(img1, img2)

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


class TransitionDetection_ST:
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
        self.scdet_output = scdet_output
        self.scdet_threshold = scdet_threshold
        self.use_fixed_scdet = use_fixed_scdet
        if self.use_fixed_scdet:
            self.scdet_threshold = fixed_max_scdet
        self.scdet_cnt = 0
        self.scene_stack_len = scene_queue_length
        self.absdiff_queue = deque(maxlen=self.scene_stack_len)  # absdiff队列
        self.black_scene_queue = deque(maxlen=self.scene_stack_len)  # 黑场开场特判队列
        self.utils = Utils()
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
        return math.sqrt(sub_array.var())

    def __judge_mean(self, diff):
        var_before = self.__check_var()
        self.absdiff_queue.append(diff)
        var_after = self.__check_var()
        if var_after - var_before > self.scdet_threshold and diff > self.born_thres:
            """Detect new scene"""
            self.save_flow(
                f"diff: {diff:.3f}, before: {var_before:.3f}, after: {var_after:.3f}, cnt: {self.scdet_cnt + 1}")
            self.absdiff_queue.clear()
            self.scdet_cnt += 1
            return True
        else:
            if diff > self.dead_thres:
                self.absdiff_queue.clear()
                self.save_flow(f"diff: {diff:.3f}, False Alarm, cnt: {self.scdet_cnt + 1}")
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
            self.save_flow(title)

    def save_flow(self, title):
        # return
        if not self.scdet_output:
            return
        try:
            comp_stack = np.hstack((self.img1, self.img2))
            # comp_stack = cv2.cvtColor(comp_stack, cv2.COLOR_RGB2BGR)
            cv2.putText(comp_stack,
                        title,
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0))
            cv2.imencode('.png', comp_stack)[1].tofile(
                os.path.join(self.scene_dir, f"{self.scdet_cnt:08d}.png"))
            return
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
        :return: 是转场则返回帧
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
            if diff < self.scdet_threshold:
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
            self.save_flow(f"absdiff: {diff:.3f}, Pure Scene Alarm, cnt: {self.scdet_cnt}")
            # self.save_flow()
            return True

        self.img1 = img1
        self.img2 = img2
        # if diff == 0:
        #     """重复帧，不可能是转场，也不用添加到判断队列里"""
        #     return False

        if len(self.absdiff_queue) < self.scene_stack_len or add_diff:
            if diff not in self.absdiff_queue:
                self.absdiff_queue.append(diff)
            # if diff > dead_thres:
            #     if not add_diff:
            #         see_result(f"compare: True, diff: {diff:.3f}, Sparse Stack, cnt: {self.scdet_cnt + 1}")
            #     self.scene_stack.clear()
            #     return True
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


if __name__ == "__main__":
    import os

    detector = TransitionDetection_ST(scene_queue_length=10, scdet_threshold=16,
                                      project_dir=r"D:\60-fps-Project\input_or_ref\Test\output\test_scene",
                                      scdet_output=True)
    scene_path = r"D:\60-fps-Project\input_or_ref\Test\scenes"
    files = sorted(os.listdir(scene_path))
    for i in range(len(files) - 1):
        i1 = os.path.join(scene_path, files[i])
        i2 = os.path.join(scene_path, files[i + 1])
        r = detector.check_scene(cv2.imread(i1), cv2.imread(i2))
        if r:
            print(f"at {i}, find scene")
    pass
