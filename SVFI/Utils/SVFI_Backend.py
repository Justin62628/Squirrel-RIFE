import datetime
import json
import math
import os
import re
import traceback

import cv2
import torch
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QTextCursor, QIcon
import html
import sys
import subprocess as sp
import shlex
import time
import psutil
from Utils import RIFE_GUI
from Utils.utils import Utils, EncodePresetAssemply

MAC = True
try:
    from PyQt5.QtGui import qt_mac_set_native_menubar
except ImportError:
    MAC = False

Utils = Utils()
abspath = os.path.abspath(__file__)
dname = os.path.dirname(os.path.dirname(abspath))
ddname = os.path.dirname(abspath)
appDataPath = os.path.join(dname, "SVFI.ini")  # TODO multi task designed
appData = QSettings(appDataPath, QSettings.IniFormat)
appData.setIniCodec("UTF-8")

logger = Utils.get_logger("GUI", dname)
ols_potential = os.path.join(dname, "one_line_shot_args.exe")
appData.setValue("OneLineShotPath", ols_potential)
appData.setValue("ffmpeg", dname)
appData.setValue("model", os.path.join(ddname, "train_log"))
if not os.path.exists(ols_potential):
    appData.setValue("OneLineShotPath",
                     r"D:\60-fps-Project\Projects\RIFE_GUI\one_line_shot_args.py")
    appData.setValue("ffmpeg", "ffmpeg")
    appData.setValue("model", r"D:\60-fps-Project\Projects\RIFE_GUI\Utils\train_log")
    logger.info("Change to Debug Path")


class RIFE_Run_Other_Threads(QThread):
    run_signal = pyqtSignal(str)

    def __init__(self, command, task_id=0, data=None, parent=None):
        """

        :param command:
        :param task_id:
        :param data: 信息回传时的数据
        :param parent:
        """
        super(RIFE_Run_Other_Threads, self).__init__(parent)
        self.command = command
        self.task_id = task_id
        self.data = data

    def fire_finish_signal(self):
        emit_json = {"id": self.task_id, "status": 1, "data": self.data}
        self.run_signal.emit(json.dumps(emit_json))

    def run(self):
        logger.info(f"[CMD Thread]: Start execute {self.command}")
        os.system(self.command)
        self.fire_finish_signal()
        pass

    pass


class RIFE_Run_Thread(QThread):
    run_signal = pyqtSignal(str)

    def __init__(self, parent=None, concat_only=False):
        super(RIFE_Run_Thread, self).__init__(parent)
        self.concat_only = concat_only
        self.command = ""
        self.current_proc = None
        self.kill = False
        self.pause = False
        self.all_cnt = 0
        self.silent = False
        self.tqdm_re = re.compile("Process at .*?\]")

    def fillQuotation(self, string):
        if string[0] != '"':
            return f'"{string}"'

    def build_command(self, input_file):
        if os.path.splitext(appData.value("OneLineShotPath"))[-1] == ".exe":
            self.command = appData.value("OneLineShotPath") + " "
        else:
            self.command = f'python {appData.value("OneLineShotPath")} '
        if not len(input_file) or not os.path.exists(input_file):
            self.command = ""
            return ""
        if float(appData.value("fps", -1.0, type=float)) <= 0 or float(
                appData.value("target_fps", -1.0, type=float)) <= 0:
            return ""

        self.command += f'--input {Utils.fillQuotation(input_file)} '
        if os.path.isfile(appData.value("output")):
            logger.info("[GUI]: OutputPath with FileName detected")
            output_path = appData.value("output")
            appData.setValue("output", os.path.dirname(output_path))
        self.command += f'--output {Utils.fillQuotation(appData.value("output"))} '
        self.command += f'--config {Utils.fillQuotation(appDataPath)} '
        if self.concat_only:
            self.command += f"--concat-only "

        self.command = self.command.replace("\\", "/")
        return self.command

    def update_status(self, current_step, finished=False, notice="", sp_status="", returncode=-1):
        """
        update sub process status
        :return:
        """
        emit_json = {"cnt": self.all_cnt, "current": current_step, "finished": finished,
                     "notice": notice, "subprocess": sp_status, "returncode": returncode}
        emit_json = json.dumps(emit_json)
        self.run_signal.emit(emit_json)

    def maintain_multitask(self):
        appData.setValue("chunk", 1)
        appData.setValue("interp_start", 0)

    def run(self):
        logger.info("[GUI]: Start")

        file_list = appData.value("InputFileName", "").split(";")

        command_list = list()
        for f in file_list:
            command = self.build_command(f)
            if not len(command):
                continue
            command_list.append((f, command))

        current_step = 0
        self.all_cnt = len(command_list)

        appData.setValue("batch", False)
        if self.all_cnt > 1:
            """MultiTask"""
            appData.setValue("output_only", True)
            appData.setValue("batch", True)
            appData.setValue("start_point", "")
            appData.setValue("end_point", "")

        if not self.all_cnt:
            logger.info("[GUI]: Task List Empty, Please Check Your Settings! (input fps for example)")
            self.update_status(current_step, True, "\nTask List is Empty!\n")
            return
        interval_time = time.time()
        try:
            for f in command_list:
                logger.info(f"[GUI]: Designed Command:\n{f}")
                # if appData.value("debug", type=bool):
                #     logger.info(f"DEBUG: {f[1]}")
                #     continue
                proc_args = shlex.split(f[1])
                self.current_proc = sp.Popen(args=proc_args, stdout=sp.PIPE, stderr=sp.STDOUT, encoding='gb18030',
                                             errors='ignore',
                                             universal_newlines=True)
                flush_lines = ""
                while self.current_proc.poll() is None:
                    if self.kill:
                        self.current_proc.terminate()
                        self.update_status(current_step, False, notice=f"\n\nWARNING, 补帧已被强制结束", returncode=-1)
                        break

                    if self.pause:
                        pid = self.current_proc.pid
                        pause = psutil.Process(pid)  # 传入子进程的pid
                        pause.suspend()  # 暂停子进程
                        self.update_status(current_step, False, notice=f"\n\nWARNING, 补帧已被手动暂停", returncode=-1)
                        while True:
                            if self.kill:
                                pause.resume()
                                time.sleep(0.5)
                                self.current_proc.terminate()
                                self.update_status(current_step, False, notice=f"\n\nWARNING, 补帧已在暂停后被强制结束",
                                                   returncode=-1)
                                break
                            elif not self.pause:
                                pause.resume()
                                self.update_status(current_step, False, notice=f"\n\nWARNING, 补帧已继续",
                                                   returncode=-1)
                                break
                            time.sleep(0.2)
                    else:
                        line = self.current_proc.stdout.readline()
                        self.current_proc.stdout.flush()
                        flush_lines += line.replace("[A", "")
                        if "error" in flush_lines.lower():
                            """Imediately Upload"""
                            logger.error(f"[In ONE LINE SHOT]: f{flush_lines}")
                            self.update_status(current_step, False, sp_status=f"{flush_lines}")
                            flush_lines = ""
                        elif len(flush_lines) and time.time() - interval_time > 0.1:
                            interval_time = time.time()
                            self.update_status(current_step, False, sp_status=f"{flush_lines}")
                            flush_lines = ""
                self.update_status(current_step, False, sp_status=f"{flush_lines}")  # emit last possible infos

                current_step += 1
                self.update_status(current_step, False, f"\nINFO - {datetime.datetime.now()} {f[0]} 完成\n\n")
                self.maintain_multitask()

        except Exception:
            logger.error(traceback.format_exc())

        self.update_status(current_step, True, returncode=self.current_proc.returncode)
        logger.info("[GUI]: Tasks Finished")
        pass

    def kill_proc_exec(self):
        self.kill = True
        logger.info("Kill Process Command Fired")

    def pause_proc_exec(self):
        self.pause = not self.pause
        if self.pause:
            logger.info("Pause Process Command Fired")
        else:
            logger.info("Resume Process Command Fired")

    pass


class RIFE_GUI_BACKEND(QDialog, RIFE_GUI.Ui_RIFEDialog):
    kill_proc = pyqtSignal(int)
    notfound = pyqtSignal(int)

    def __init__(self, parent=None):
        super(RIFE_GUI_BACKEND, self).__init__()
        self.setupUi(self)
        self.thread = None
        self.Exp = int(math.log(float(appData.value("exp", "2")), 2))

        if appData.value("ffmpeg") != "ffmpeg":
            self.ffmpeg = os.path.join(appData.value("ffmpeg"), "ffmpeg.exe")
        else:
            self.ffmpeg = appData.value("ffmpeg")

        if os.path.exists(appDataPath):
            logger.info("[GUI]: Previous Settings, Found, Loaded")

        self.InputFileName.currentItemChanged.connect(self.on_InputFileName_currentItemChanged)
        self.check_gpu = False
        self.silent = False
        self.tqdm_re = re.compile(".*?Process at .*?\]")
        self.current_failed = False
        self.formatted_option_check = []
        self.pause = False

        """Initiate and Check GPU"""
        self.hasNVIDIA = True
        self.update_gpu_info()
        self.update_model_info()
        self.init_before_settings()
        self.on_EncoderSelector_currentTextChanged()  # Flush Encoder Sets

    def init_before_settings(self):
        input_list = appData.value("InputFileName", "").split(";")
        for i in input_list:
            if len(i):
                self.InputFileName.addItem(i)
        self.OutputFolder.setText(appData.value("output"))
        self.InputFPS.setText(appData.value("fps", "0"))
        self.OutputFPS.setText(appData.value("target_fps"))
        self.ExpSelecter.setCurrentText("x" + str(2 ** int(appData.value("exp", "1"))))

        self.UseCRF.setChecked(appData.value("UseCRF", True, type=bool))
        self.CRFSelector.setValue(appData.value("crf", 16, type=int))
        self.UseTargetBitrate.setChecked(appData.value("UseTargetBitrate", False, type=bool))
        self.BitrateSelector.setValue(appData.value("bitrate", 90, type=float))
        self.PresetSelector.setCurrentText(appData.value("preset", "fast[软编, 硬编]"))
        self.EncoderSelector.setCurrentText(appData.value("encoder", "H264/AVC"))
        self.ExtSelector.setCurrentText(appData.value("output_ext", "mp4"))
        self.ScedetChecker.setChecked(not appData.value("no_scdet", False, type=bool))
        self.UseFixedScdet.setChecked(appData.value("fixed_scdet", False, type=bool))
        self.RenderGapSelector.setValue(appData.value("render_gap", 1000, type=int))
        self.SaveAudioChecker.setChecked(appData.value("save_audio", True, type=bool))
        self.StartPoint.setTime(QTime.fromString(appData.value("start_point", "00:00:00"), "HH:mm:ss"))
        self.EndPoint.setTime(QTime.fromString(appData.value("end_point", "00:00:00"), "HH:mm:ss"))

        self.ScdetSelector.setValue(appData.value("scdet_threshold", 12, type=int))
        self.DupRmChecker.setChecked(appData.value("remove_dup", False, type=bool))
        self.DupFramesTSelector.setValue(appData.value("dup_threshold", 1.00, type=float))
        self.UseAnyFPS.setChecked(appData.value("any_fps", False, type=bool))
        self.on_UseAnyFPS_clicked()

        self.CropSettings.setText(appData.value("crop"))
        self.ResizeSettings.setText(appData.value("resize"))
        self.FFmpegCustomer.setText(appData.value("ffmpeg_customized", ""))

        self.QuickExtractChecker.setChecked(appData.value("quick_extract", True, type=bool))
        self.ImgOutputChecker.setChecked(appData.value("img_output", False, type=bool))

        # self.HwaccelChecker.setChecked(appData.value("hwaccel", False, type=bool))
        self.HwaccelSelector.setCurrentText(appData.value("hwaccel_mode", "None", type=str))
        self.MBufferChecker.setChecked(appData.value("manual_buffer", False, type=bool))
        self.BufferSizeSelector.setValue(appData.value("manual_buffer_size", 1, type=int))
        self.FP16Checker.setChecked(appData.value("fp16", False, type=bool))
        self.InterpScaleSelector.setCurrentText(appData.value("scale", "1.00"))
        self.ReverseChecker.setChecked(appData.value("reverse", False, type=bool))

        self.UseNCNNButton.setChecked(appData.value("ncnn", False, type=bool))

        j_settings_values = list(map(lambda x: int(x), appData.value("j_settings", "2:4:4").split(":")))
        self.ncnnReadThreadCnt.setValue(j_settings_values[0])
        self.ncnnInterpThreadCnt.setValue(j_settings_values[1])
        self.ncnnOutputThreadCnt.setValue(j_settings_values[2])
        self.slowmotion.setChecked(appData.value("slow_motion", False, type=bool))
        self.SlowmotionFPS.setText(appData.value("slow_motion_fps", "", type=str))
        appData.setValue("img_input", appData.value("img_input", False))

        desktop = QApplication.desktop()
        pos = appData.value("pos", QVariant(QPoint(960, 540)))
        size = appData.value("size", QVariant(QSize(int(desktop.width() * 0.25), int(desktop.height() * 0.4))))

        self.resize(size)
        self.move(pos)

        self.setAttribute(Qt.WA_TranslucentBackground)

    def load_current_settings(self):
        input_file_names = ""
        for i in self.load_input_files():
            if len(i):
                input_file_names += f"{i};"
        appData.setValue("InputFileName", input_file_names)
        appData.setValue("output", self.OutputFolder.text())
        appData.setValue("fps", self.InputFPS.text())
        appData.setValue("target_fps", self.OutputFPS.text())
        appData.setValue("crf", self.CRFSelector.value())
        appData.setValue("exp", int(math.log(int(self.ExpSelecter.currentText()[1:]), 2)))
        appData.setValue("bitrate", self.BitrateSelector.value())
        appData.setValue("preset", self.PresetSelector.currentText())
        appData.setValue("encoder", self.EncoderSelector.currentText())
        # appData.setValue("hwaccel", self.HwaccelChecker.isChecked())
        appData.setValue("hwaccel_mode", self.HwaccelSelector.currentText())
        appData.setValue("no_scdet", not self.ScedetChecker.isChecked())
        appData.setValue("fixed_scdet", self.UseFixedScdet.isChecked())
        appData.setValue("scdet_threshold", self.ScdetSelector.value())
        appData.setValue("any_fps", self.UseAnyFPS.isChecked())
        appData.setValue("remove_dup", self.DupRmChecker.isChecked())
        if self.UseAnyFPS.isChecked():
            appData.setValue("remove_dup", True)

        appData.setValue("dup_threshold", self.DupFramesTSelector.value())
        appData.setValue("crop", self.CropSettings.text())
        appData.setValue("resize", self.ResizeSettings.text())

        appData.setValue("save_audio", self.SaveAudioChecker.isChecked())
        appData.setValue("quick_extract", self.QuickExtractChecker.isChecked())
        appData.setValue("img_output", self.ImgOutputChecker.isChecked())
        # appData.setValue("img_input", self.ImgInputChecker.isChecked())
        appData.setValue("no_concat", False)  # always concat
        appData.setValue("output_only", True)  # always output only
        appData.setValue("fp16", self.FP16Checker.isChecked())
        appData.setValue("reverse", self.ReverseChecker.isChecked())
        appData.setValue("UseCRF", self.UseCRF.isChecked())
        appData.setValue("UseTargetBitrate", self.UseTargetBitrate.isChecked())
        appData.setValue("start_point", self.StartPoint.time().toString("HH:mm:ss"))
        appData.setValue("end_point", self.EndPoint.time().toString("HH:mm:ss"))

        appData.setValue("encoder", self.EncoderSelector.currentText())
        appData.setValue("pix_fmt", self.PixFmtSelector.currentText())
        appData.setValue("output_ext", self.ExtSelector.currentText())

        appData.setValue("chunk", self.StartChunk.text() if len(self.StartChunk.text()) else 1)
        appData.setValue("interp_start", self.StartFrame.text() if len(self.StartFrame.text()) else 0)
        appData.setValue("render_gap", int(self.RenderGapSelector.value()))

        appData.setValue("manual_buffer", self.MBufferChecker.isChecked())
        appData.setValue("manual_buffer_size", self.BufferSizeSelector.value())
        appData.setValue("ncnn", self.UseNCNNButton.isChecked())
        appData.setValue("scale", self.InterpScaleSelector.currentText())
        appData.setValue("SelectedModel", os.path.join(appData.value("model"), self.ModuleSelector.currentText()))
        appData.setValue("use_specific_gpu", self.DiscreteCardSelector.currentIndex())
        appData.setValue("pos", QVariant(self.pos()))
        appData.setValue("size", QVariant(self.size()))
        appData.setValue("ffmpeg_customized", self.FFmpegCustomer.text())
        appData.setValue("debug", self.DebugChecker.isChecked())

        j_settings = f"{self.ncnnReadThreadCnt.value()}:{self.ncnnInterpThreadCnt.value()}:{self.ncnnOutputThreadCnt.value()}"
        appData.setValue("j_settings", j_settings)
        appData.setValue("slow_motion", self.slowmotion.isChecked())
        appData.setValue("slow_motion_fps", self.SlowmotionFPS.text())
        if appData.value("slow_motion", False, type=bool):
            appData.setValue("save_audio", False)
            self.SaveAudioChecker.setChecked(False)

        logger.info("[Main]: Download all settings")
        self.OptionCheck.isReadOnly = True
        appData.sync()
        pass

    def update_rife_process(self, json_data):
        """
        Communicate with RIFE Thread
        :return:
        """

        def generate_error_log():
            status_check = "[导出设置预览]\n\n"
            for key in appData.allKeys():
                status_check += f"{key} => {appData.value(key)}\n"
            status_check += "\n\n[错误信息]\n\n"
            status_check += self.OptionCheck.toPlainText()
            error_path = os.path.join(self.OutputFolder.text(), "log", f"{datetime.datetime.now().date()}.error.log")
            error_path_dir = os.path.dirname(error_path)
            if not os.path.exists(error_path_dir):
                os.mkdir(error_path_dir)
            with open(error_path, "w", encoding="utf-8") as w:
                w.write(status_check)
            os.startfile(error_path_dir)

        def remove_last_line():
            cursor = self.OptionCheck.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.select(QTextCursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deletePreviousChar()
            self.OptionCheck.setTextCursor(cursor)

        def error_handle():
            now_text = self.OptionCheck.toPlainText()
            if self.current_failed:
                return
            if "Input File not valid" in now_text:
                self.sendWarning("Inputs Failed", "你的输入文件有问题！请检查输入文件是否能播放，路径有无特殊字符", )
                self.current_failed = True
                return
            elif "JSON" in now_text:
                self.sendWarning("Input File Failed", "文件信息读取失败，请确保软件和视频文件路径均为纯英文、无空格且无特殊字符", )
                self.current_failed = True
                return
            elif "ascii" in now_text:
                self.sendWarning("Software Path Failure", "请把软件所在文件夹移到纯英文、无中文、无空格路径下", )
                self.current_failed = True
                return
            elif "CUDA out of memory" in now_text:
                self.sendWarning("CUDA Failed", "你的显存不够啦！去清一下后台占用显存的程序，或者去'高级设置'降低视频分辨率/使用半精度模式~", )
                self.current_failed = True
                return
            elif "Concat Test Error" in now_text:
                self.sendWarning("Concat Failed", "区块合并音轨测试失败，请检查输出文件格式是否支持源文件音频", )
                self.current_failed = True
                return
            elif "error" in data.get("subprocess", "").lower():
                logger.error(f"[At the end of One Line Shot]: \n {data.get('subprocess')}")
                self.sendWarning("Something Went Wrong", f"程序运行出现错误！\n{data.get('subprocess')}\n联系开发人员解决", )
                self.current_failed = True
                return

        data = json.loads(json_data)
        self.progressBar.setMaximum(int(data["cnt"]))
        self.progressBar.setValue(int(data["current"]))
        new_text = ""

        if len(data.get("notice", "")):
            new_text += data["notice"] + "\n"

        if len(data.get("subprocess", "")):
            dup_keys_list = ["Process at", "frame="]
            if any([i in data["subprocess"] for i in dup_keys_list]):
                tmp = ""
                lines = data["subprocess"].splitlines()
                for line in lines:
                    if not any([i in line for i in dup_keys_list]):
                        tmp += line + "\n"
                if tmp.strip() == lines[-1].strip():
                    lines[-1] = ""
                data["subprocess"] = tmp + lines[-1]
                remove_last_line()
            new_text += data["subprocess"]

        for line in new_text.splitlines():
            line = html.escape(line)
            if "process at" in line.lower():
                add_line = f'<p><span style=" font-weight:600;">{line}</span></p>'
            elif "program finished" in line.lower():
                add_line = f'<p><span style=" font-weight:600; color:#55aa00;">{line}</span></p>'
            elif "info" in line.lower():
                add_line = f'<p><span style=" font-weight:600; color:#0000ff;">{line}</span></p>'
            elif any([i in line.lower() for i in
                      ["error", "invalid", "incorrect", "critical", "fail", "can't", "can not"]]):
                add_line = f'<p><span style=" font-weight:600; color:#ff0000;">{line}</span></p>'
            elif "warn" in line.lower():
                add_line = f'<p><span style=" font-weight:600; color:#ffaa00;">{line}</span></p>'
            elif "duration" in line.lower():
                add_line = f'<p><span style=" font-weight:600; color:#550000;">{line}</span></p>'
            else:
                add_line = f'<p><span>{line}</span></p>'
            self.OptionCheck.append(add_line)
        if data["finished"]:
            """Error Handle"""
            returncode = data["returncode"]
            complete_msg = f"共 {data['cnt']} 个补帧任务\n"
            if returncode == 0:
                complete_msg += '成功！'
                os.startfile(self.OutputFolder.text())
            else:
                complete_msg += f'失败, 返回码：{returncode}\n请将弹出的文件夹内error.txt发送至交流群排疑，' \
                                f'并尝试前往高级设置恢复补帧进度'
                error_handle()
                generate_error_log()

            self.sendWarning("补帧任务完成", complete_msg, 2)
            self.ProcessStart.setEnabled(True)
            self.ConcatAllButton.setEnabled(True)
            self.current_failed = False

        self.OptionCheck.moveCursor(QTextCursor.End)

    def check_args(self) -> bool:
        """
        Check are all args available
        :return:
        """
        videos = self.load_input_files()
        output_dir = self.OutputFolder.text()

        if not len(videos) or not len(output_dir):
            self.sendWarning("Empty Input", "请输入要补帧的文件和输出文件夹")
            return False

        if not os.path.exists(output_dir):
            logger.info("Not Exists OutputFolder")
            self.sendWarning("Output Folder Not Found", "输入文件或输出文件夹不存在！请确认输入")
            return False

        if os.path.isfile(output_dir):
            """Auto set Output Dir to correct form"""
            self.OutputFolder.setText(os.path.dirname(output_dir))

        for v in videos:
            if not os.path.exists(v):
                logger.info(f"Not Exists Input Source: {v}")
                self.sendWarning("Input Source Not Found", f"输入文件:\n{v}\n不存在！请确认输入!")
                return False

        video = videos[0]  # first video
        if not os.path.isfile(video):
            """Input is a folder"""
            self.SaveAudioChecker.setChecked(False)
            input_fps = self.InputFPS.text()
            appData.setValue("img_input", True)
            if not len(input_fps):
                self.sendWarning("Empty Input", "请输入图片序列文件夹帧率")
                return False
        else:
            appData.setValue("img_input", False)

        try:
            float(self.InputFPS.text())
            float(self.OutputFPS.text())
        except Exception:
            self.sendWarning("Wrong Inputs", "请确认输入和输出帧率")
            return False

        try:
            if self.slowmotion.isChecked():
                float(self.SlowmotionFPS.text())
        except Exception:
            self.sendWarning("Wrong Inputs", "请确认慢动作输入帧率")
            return False

        return True

    def sendWarning(self, title, string, msg_type=1):
        """

        :param title:
        :param string:
        :param msg_type: 1 warning 2 info 3 question
        :return:
        """
        if self.silent:
            return
        QMessageBox.setWindowIcon(self, QIcon('svfi.png'))
        if msg_type == 1:
            reply = QMessageBox.warning(self,
                                        f"{title}",
                                        f"{string}",
                                        QMessageBox.Yes)
        elif msg_type == 2:
            reply = QMessageBox.information(self,
                                            f"{title}",
                                            f"{string}",
                                            QMessageBox.Yes)
        elif msg_type == 3:
            reply = QMessageBox.information(self,
                                            f"{title}",
                                            f"{string}",
                                            QMessageBox.Yes | QMessageBox.No)
        else:
            return
        return reply

    def select_file(self, filename, folder=False, _filter=None, multi=False):
        if folder:
            directory = QFileDialog.getExistingDirectory(None, caption="选取文件夹")
            return directory
        if multi:
            files = QFileDialog.getOpenFileNames(None, caption=f"选择{filename}", filter=_filter)
            return files[0]
        directory = QFileDialog.getOpenFileName(None, caption=f"选择{filename}", filter=_filter)
        return directory[0]

    def quick_concat(self):
        input_v = self.ConcatInputV.text()
        input_a = self.ConcatInputA.text()
        output_v = self.OutputConcat.text()
        self.load_current_settings()
        if not input_v or not input_a or not output_v:
            self.sendWarning("Parameters unfilled", "请填写输入或输出视频路径！")
            return

        ffmpeg_command = f"""
            {self.ffmpeg} -i {Utils.fillQuotation(input_a)} -i {Utils.fillQuotation(input_v)} 
            -map 1:v:0 -map 0:a:0 -c:v copy -c:a copy -shortest {Utils.fillQuotation(output_v)} -y
        """.strip().strip("\n").replace("\n", "").replace("\\", "/")
        logger.info(f"[GUI] concat {ffmpeg_command}")
        os.system(ffmpeg_command)
        self.sendWarning("音视频合并操作完成！", f"请查收", msg_type=2)

    def update_gif_making(self, emit_json: str):
        self.GifButton.setEnabled(True)
        emit_json = json.loads(emit_json)
        target_fps = emit_json.get("data", {"target_fps": appData.value("target_fps", 50)})["target_fps"]
        self.sendWarning("GIF操作完成！", f'GIF帧率:{target_fps}', 2)

    def quick_gif(self):
        input_v = self.GifInput.text()
        output_v = self.GifOutput.text()
        self.load_current_settings()
        if not input_v or not output_v:
            self.sendWarning("Parameters unfilled", "请填写输入或输出视频路径！")
            return
        if not appData.value("target_fps"):
            appData.setValue("target_fps", 50)
            logger.info("Not find output GIF fps, Auto set GIF output fps to 50 as it's smooth enough")
        target_fps = appData.value("target_fps", 50, type=float)
        if target_fps > 50:
            target_fps = 50
            logger.info("Auto set GIF output fps to 50 as it's smooth enough")
        resize = self.ResizeSettings.text()
        ffmpeg_command = f'{self.ffmpeg} -hide_banner -i {Utils.fillQuotation(input_v)} -r {target_fps} ' \
                         f'-lavfi "scale={resize},split[s0][s1];' \
                         f'[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=floyd_steinberg" ' \
                         f'{Utils.fillQuotation(output_v)} -y'.strip().strip("\n").replace("\n", "").replace("\\", "/")

        logger.info(f"[GUI] create gif: {ffmpeg_command}")
        self.GifButton.setEnabled(False)
        GIF_Thread = RIFE_Run_Other_Threads(ffmpeg_command, 23333, data={"target_fps": target_fps})
        GIF_Thread.run_signal.connect(self.update_gif_making)
        GIF_Thread.start()

    def set_start_info(self, sf, scf, sc):
        """
        :return:
        """
        self.StartFrame.setText(str(sf))
        self.StartChunk.setText(str(sc))
        return

    def auto_set_fps(self, sample_file):
        if not os.path.isfile(sample_file):
            return
        currentExp = self.ExpSelecter.currentText()[1:]
        try:
            input_stream = cv2.VideoCapture(sample_file)
            input_fps = input_stream.get(cv2.CAP_PROP_FPS)
            self.InputFPS.setText(f"{input_fps:.5f}")
            self.OutputFPS.setText(f"{float(input_fps) * float(currentExp):.5f}")
        except Exception:
            logger.error(traceback.format_exc())

    def auto_set(self):
        chunk_list = list()
        output_dir = self.OutputFolder.text()
        ratio = float(self.OutputFPS.text()) / float(self.InputFPS.text())
        for f in os.listdir(output_dir):
            if re.match("chunk-[\d+].*?\.(mp4|mov)", f):
                chunk_list.append(os.path.join(output_dir, f))
        if not len(chunk_list):
            self.set_start_info(0, 1, 1)
            logger.info("AutoSet find None")
            return

        logger.info("Found Previous Chunks")
        chunk_list.sort(key=lambda x: int(os.path.basename(x).split('-')[2]))

        reply = self.sendWarning(f"恢复进度？", f"检测到上次还有未完成的补帧，要继续吗？", 3)
        if reply == QMessageBox.No:
            self.set_start_info(0, 1, 1)
            for c in chunk_list:
                os.remove(c)
            return
        last_chunk = chunk_list[-1]
        match_result = re.findall("chunk-(\d+)-(\d+)-(\d+)\.(mp4|mov)", last_chunk)[0]
        chunk = int(match_result[0])
        last_frame = int(match_result[2])
        first_interp_cnt = (last_frame + 1) * ratio + 1
        self.set_start_info(last_frame + 1, first_interp_cnt, chunk + 1)
        logger.info("AutoSet Ready")
        pass

    def get_filename(self, path):
        if not os.path.isfile(path):
            return os.path.basename(path)
        return os.path.splitext(os.path.basename(path))[0]

    def update_gpu_info(self):
        infos = {}
        for i in range(torch.cuda.device_count()):
            card = torch.cuda.get_device_properties(i)
            info = f"{card.name}, {card.total_memory / 1024 ** 3:.1f} GB"
            infos[f"{i}"] = info
        logger.info(f"NVIDIA data: {infos}")

        if not len(infos):
            self.hasNVIDIA = False
            self.sendWarning("No NVIDIA Card Found", "未找到N卡，将使用A卡或核显")
            appData.setValue("ncnn", True)
            self.UseNCNNButton.setChecked(True)
            self.UseNCNNButton.setEnabled(False)
            self.on_UseNCNNButton_clicked()
            return
        else:
            if self.UseNCNNButton.isChecked():
                appData.setValue("ncnn", True)
            else:
                appData.setValue("ncnn", False)

        self.DiscreteCardSelector.clear()
        for gpu in infos:
            self.DiscreteCardSelector.addItem(f"{gpu}: {infos[gpu]}")
        self.check_gpu = True
        return infos

    def update_model_info(self):
        model_dir = appData.value("model")
        if not os.path.exists(model_dir):
            logger.info(f"Not find Module dir at {model_dir}")
            self.sendWarning("Model Dir Not Found", "未找到补帧模型路径，请检查！")
            return
        model_list = list()
        for m in os.listdir(model_dir):
            if not os.path.isfile(os.path.join(model_dir, m)):
                model_list.append(m)
        model_list.reverse()
        self.ModuleSelector.clear()
        for mod in model_list:
            self.ModuleSelector.addItem(f"{mod}")

    def load_input_files(self):
        widgetres = []
        count = self.InputFileName.count()
        for i in range(count):
            widgetres.append(self.InputFileName.item(i).text())
        return widgetres

    def on_InputFileName_currentItemChanged(self):
        if self.InputFileName.currentItem() is None:
            return
        text = self.InputFileName.currentItem().text().strip('"')
        self.InputFileName.disconnect()
        if text == "":
            return
        """empty text"""
        input_filename = text.strip(";").split(";")[0]
        self.auto_set_fps(input_filename)

        self.InputFileName.currentItemChanged.connect(self.on_InputFileName_currentItemChanged)
        return

    @pyqtSlot(bool)
    def on_InputDirButton_clicked(self):
        try:
            self.InputFileName.disconnect()
        except TypeError:
            pass
        self.InputFileName.clear()
        input_directory = self.select_file("要补帧的图片序列文件夹", folder=True)
        self.InputFileName.addItem(input_directory)
        self.InputFileName.currentItemChanged.connect(self.on_InputFileName_currentItemChanged)
        return

    @pyqtSlot(bool)
    def on_InputButton_clicked(self):
        try:
            self.InputFileName.disconnect()
        except TypeError:
            pass

        input_files = self.select_file('要补帧的视频', multi=True)
        if not len(input_files):
            return
        self.InputFileName.clear()
        for f in input_files:
            self.InputFileName.addItem(f)

        self.OutputFolder.setText(os.path.dirname(input_files[0]))
        sample_file = input_files[0]
        self.auto_set_fps(sample_file)
        self.InputFileName.currentItemChanged.connect(self.on_InputFileName_currentItemChanged)

    @pyqtSlot(bool)
    def on_OutputButton_clicked(self):
        folder = self.select_file('要输出项目的文件夹', folder=True)
        self.OutputFolder.setText(folder)

    @pyqtSlot(bool)
    def on_AllInOne_clicked(self):
        """
        Alas
        :return:
        """
        if not self.check_args():
            return
        self.auto_set()
        self.on_EncoderSelector_currentTextChanged()  # update Encoders
        self.load_current_settings()  # update settings
        self.on_ProcessStart_clicked()
        self.tabWidget.setCurrentIndex(2)  # redirect to info page

    @pyqtSlot(bool)
    def on_AutoSet_clicked(self):
        if not len(self.load_input_files()) or not len(self.OutputFolder.text()):
            self.sendWarning("Invalid Inputs", "请检查你的输入和输出文件夹")
            return
        if self.check_args():
            self.auto_set()

    @pyqtSlot(bool)
    def on_ConcatButton_clicked(self):
        if not self.ConcatInputV.text():
            self.load_current_settings()  # update settings
            input_filename = self.select_file('请输入要进行音视频合并的视频文件')
            self.ConcatInputV.setText(input_filename)
            self.ConcatInputA.setText(input_filename)
            self.OutputConcat.setText(
                os.path.join(os.path.dirname(input_filename), f"{self.get_filename(input_filename)}_concat.mp4"))
            return
        self.quick_concat()
        # self.auto_set()
        pass

    @pyqtSlot(bool)
    def on_GifButton_clicked(self):
        if not self.GifInput.text():
            self.load_current_settings()  # update settings
            input_filename = self.select_file('请输入要制作成gif的视频文件')
            self.GifInput.setText(input_filename)
            self.GifOutput.setText(
                os.path.join(os.path.dirname(input_filename), f"{self.get_filename(input_filename)}.gif"))
            return
        self.quick_gif()
        pass

    @pyqtSlot(str)
    def on_HwaccelSelector_currentTextChanged(self):
        logger.info("Switch To HWACCEL Mode: %s" % self.HwaccelSelector.currentText())
        self.on_EncoderSelector_currentTextChanged()

    @pyqtSlot(bool)
    def on_MBufferChecker_clicked(self):
        logger.info("Switch To Manual Assign Buffer Size Mode: %s" % self.MBufferChecker.isChecked())
        self.BufferSizeSelector.setEnabled(self.MBufferChecker.isChecked())

    @pyqtSlot(bool)
    def on_UseFixedScdet_clicked(self):
        logger.info("Switch To FixedScdetThreshold Mode: %s" % self.UseFixedScdet.isChecked())
        # self.ScdetSelector.setEnabled(self.UseFixedScdet.isChecked())

    @pyqtSlot(bool)
    def on_UseNCNNButton_clicked(self):
        if self.hasNVIDIA and self.UseNCNNButton.isChecked():
            reply = self.sendWarning(f"确定使用NCNN？", f"你有N卡，确定使用A卡/核显？", 3)
            if reply == QMessageBox.Yes:
                logger.info("Switch To NCNN Mode: %s" % self.UseNCNNButton.isChecked())
            else:
                self.UseNCNNButton.setChecked(False)
        else:
            logger.info("Switch To NCNN Mode: %s" % self.UseNCNNButton.isChecked())
        if self.UseNCNNButton.isChecked():
            """Nvidia Special Functions"""
            # self.HwaccelSelector.setE(False)
            self.DupRmChecker.setChecked(False)
            self.UseAnyFPS.setChecked(False)
        bool_result = not self.UseNCNNButton.isChecked()
        self.FP16Checker.setEnabled(bool_result)
        self.ReverseChecker.setEnabled(bool_result)
        self.InterpScaleSelector.setEnabled(bool_result)
        self.ModuleSelector.setEnabled(bool_result)
        self.DiscreteCardSelector.setEnabled(bool_result)
        self.ncnnReadThreadCnt.setEnabled(not bool_result)
        self.ncnnInterpThreadCnt.setEnabled(not bool_result)
        self.ncnnOutputThreadCnt.setEnabled(not bool_result)
        self.ncnnGPUCnt.setEnabled(not bool_result)
        self.UseAnyFPS.setEnabled(bool_result)
        self.DupRmChecker.setEnabled(bool_result)
        self.DupFramesTSelector.setEnabled(bool_result)
        # self.HwaccelChecker.setEnabled(bool_result)
        self.on_UseAnyFPS_clicked()
        self.on_ExpSelecter_currentTextChanged()

    @pyqtSlot(bool)
    def on_UseAnyFPS_clicked(self):
        if not self.hasNVIDIA and self.UseAnyFPS.isChecked():
            reply = self.sendWarning(f"未检测到N卡，不能勾选此项！", 1)
            self.UseAnyFPS.setChecked(False)
            # self.HwaccelChecker.setChecked(False)
            return
        bool_result = self.UseAnyFPS.isChecked()
        self.ExpSelecter.setEnabled(not bool_result)
        self.OutputFPS.setEnabled(True)
        if not bool_result:
            self.on_ExpSelecter_currentTextChanged()

    @pyqtSlot(bool)
    def on_slowmotion_clicked(self):
        self.SlowmotionFPS.isEnabled(self.slowmotion.isChecked())

    @pyqtSlot(str)
    def on_EncoderSelector_currentTextChanged(self):
        self.PresetSelector.clear()
        self.PixFmtSelector.clear()
        currentEncoder = self.EncoderSelector.currentText()
        currentHwaccel = self.HwaccelSelector.currentText()
        presets = []
        pixfmts = []
        if currentEncoder == "H265/HEVC":
            if currentHwaccel == "None":
                # x265
                presets = EncodePresetAssemply.preset["HEVC"]["x265"]
                pixfmts = EncodePresetAssemply.pixfmt["HEVC"]["x265"]
            else:
                # hwaccel
                presets = EncodePresetAssemply.preset["HEVC"][currentHwaccel]
                pixfmts = EncodePresetAssemply.pixfmt["HEVC"][currentHwaccel]
        elif currentEncoder == "H264/AVC":
            if currentHwaccel == "None":
                # x264
                presets = EncodePresetAssemply.preset["H264"]["x264"]
                pixfmts = EncodePresetAssemply.pixfmt["H264"]["x264"]
            else:
                # hwaccel
                presets = EncodePresetAssemply.preset["H264"][currentHwaccel]
                pixfmts = EncodePresetAssemply.pixfmt["H264"][currentHwaccel]
        elif currentEncoder == "ProRes":
            presets = EncodePresetAssemply.preset["ProRes"]
            pixfmts = EncodePresetAssemply.pixfmt["ProRes"]

        for preset in presets:
            self.PresetSelector.addItem(preset)
        for pixfmt in pixfmts:
            self.PixFmtSelector.addItem(pixfmt)

    @pyqtSlot(str)
    def on_ExpSelecter_currentTextChanged(self):
        input_files = self.load_input_files()
        if not len(input_files):
            return
        input_filename = input_files[0]
        self.auto_set_fps(input_filename)

    @pyqtSlot(str)
    def on_InputFPS_textChanged(self, text):
        currentExp = self.ExpSelecter.currentText()[1:]
        try:
            self.OutputFPS.setText(f"{float(text) * float(currentExp):.5f}")
        except ValueError:
            self.sendWarning("Pls Enter Valid InputFPS", "请输入正常的视频帧率")

    @pyqtSlot(int)
    def on_tabWidget_currentChanged(self, tabIndex):
        if tabIndex in [2, 3]:
            """Step 3"""
            if tabIndex == 2:
                self.progressBar.setValue(0)
            logger.info("[Main]: Start Loading Settings")
            self.load_current_settings()

    @pyqtSlot(bool)
    def on_ProcessStart_clicked(self):
        if not self.check_args():
            return
        reply = self.sendWarning("Confirm Start Info", f"补帧将会从区块[{self.StartChunk.text()}], "
                                                       f"起始帧[{self.StartFrame.text()}]启动。\n请确保上述两者皆不为空。"
                                                       f"是否执行补帧？", 3)
        if reply == QMessageBox.No:
            return
        self.ProcessStart.setEnabled(False)
        self.progressBar.setValue(0)
        RIFE_thread = RIFE_Run_Thread()
        RIFE_thread.run_signal.connect(self.update_rife_process)
        RIFE_thread.start()
        self.thread = RIFE_thread
        update_text = f"""
            [补帧操作启动]
            显示“Program finished”则任务完成
            如果遇到任何问题，请将命令行（黑色界面）、基础设置、高级设置和输出窗口截全图并联系开发人员解决，
            群号在首页说明\n
            第一个文件的输入帧率：{self.InputFPS.text()}， 输出帧率：{self.OutputFPS.text()}， 启用慢动作：{self.slowmotion.isChecked()}， 慢动作帧率：{self.SlowmotionFPS.text()} 启用任意帧率：{self.UseAnyFPS.isChecked()}， 使用动漫优化：{self.DupRmChecker.isChecked()}
        """
        if appData.value("ncnn", type=bool):
            update_text += "使用A卡或核显：True\n"

        self.OptionCheck.setText(update_text)
        self.current_failed = False

    @pyqtSlot(bool)
    def on_ConcatAllButton_clicked(self):
        """

        :return:
        """
        self.ConcatAllButton.setEnabled(False)
        self.tabWidget.setCurrentIndex(2)
        self.progressBar.setValue(0)
        RIFE_thread = RIFE_Run_Thread(concat_only=True)
        RIFE_thread.run_signal.connect(self.update_rife_process)
        RIFE_thread.start()
        self.thread = RIFE_thread
        self.OptionCheck.setText("[仅合并操作启动，请移步命令行查看进度详情]\n显示“Program finished”则任务完成\n"
                                 "如果遇到任何问题，请将命令行（黑色界面）和软件运行界面的Step1、Step2、Step3截图并联系开发人员解决，"
                                 "群号在首页说明\n\n\n\n\n")

    @pyqtSlot(bool)
    def on_KillProcButton_clicked(self):
        """
        :return:
        """
        if self.thread is not None:
            self.thread.kill_proc_exec()

    @pyqtSlot(bool)
    def on_PauseProcess_clicked(self):
        """
        :return:
        """
        if self.thread is not None:
            self.thread.pause_proc_exec()
            if not self.pause:
                self.pause = True
                self.PauseProcess.setText("继续补帧！")
            else:
                self.pause = False
                self.PauseProcess.setText("暂停补帧！")

    @pyqtSlot(bool)
    def on_CloseButton_clicked(self):
        self.load_current_settings()
        sys.exit()


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        form = RIFE_GUI_BACKEND()
        form.show()
        app.exec_()
        form.load_current_settings()
        sys.exit()
    except Exception:
        logger.critical(traceback.format_exc())
        sys.exit()
