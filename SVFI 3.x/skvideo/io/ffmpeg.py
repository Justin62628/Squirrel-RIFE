""" Plugin that uses ffmpeg to read and write series of images to
a wide range of video formats.

"""

# Heavily inspired from Almar Klein's imageio code
# Copyright (c) 2015, imageio contributors
# distributed under the terms of the BSD License (included in release).

import subprocess as sp

import cv2
import numpy as np

from .abstract import VideoReaderAbstract, VideoWriterAbstract
from .ffprobe import ffprobe
from .. import _FFMPEG_APPLICATION
from .. import _FFMPEG_PATH
from .. import _FFMPEG_SUPPORTED_DECODERS
from .. import _FFMPEG_SUPPORTED_ENCODERS
from .. import _HAS_FFMPEG
from ..utils import *

startupinfo = sp.STARTUPINFO()
startupinfo.dwFlags = sp.CREATE_NEW_CONSOLE | sp.STARTF_USESHOWWINDOW
startupinfo.wShowWindow = sp.SW_HIDE


# uses FFmpeg to read the given file with parameters
class FFmpegReader(VideoReaderAbstract):
    """Reads frames using FFmpeg

    Using FFmpeg as a backend, this class
    provides sane initializations meant to
    handle the default case well.

    """

    INFO_AVERAGE_FRAMERATE = "@r_frame_rate"
    INFO_WIDTH = "@width"
    INFO_HEIGHT = "@height"
    INFO_PIX_FMT = "@pix_fmt"
    INFO_DURATION = "@duration"
    INFO_NB_FRAMES = "@nb_frames"
    OUTPUT_METHOD = "image2pipe"

    def __init__(self, *args, **kwargs):
        assert _HAS_FFMPEG, "Cannot find installation of real FFmpeg (which comes with ffprobe)."
        super(FFmpegReader, self).__init__(*args, **kwargs)

    def _createProcess(self, inputdict, outputdict, verbosity):
        if '-vcodec' not in outputdict:
            outputdict['-vcodec'] = "rawvideo"

        iargs = self._dict2Args(inputdict)
        oargs = self._dict2Args(outputdict)

        if verbosity > 0:
            cmd = [_FFMPEG_PATH + "/" + _FFMPEG_APPLICATION] + iargs + ['-i', self._filename] + oargs + ['-']
            try:
                print("FFmpeg Read Command:", " ".join(cmd))
            except UnicodeEncodeError:
                print("FFmpeg Read Command: NON-ASCII character exists in command, not shown")
            self._proc = sp.Popen(cmd, stdin=sp.PIPE,
                                  stdout=sp.PIPE, stderr=None, startupinfo=startupinfo)
        else:
            cmd = [_FFMPEG_PATH + "/" + _FFMPEG_APPLICATION, "-nostats", "-loglevel", "0"] + iargs + ['-i',
                                                                                                      self._filename] + oargs + [
                      '-']
        self._proc = sp.Popen(cmd, stdin=sp.PIPE,
                              stdout=sp.PIPE, stderr=sp.PIPE, startupinfo=startupinfo)
        self._cmd = " ".join(cmd)

    def _probCountFrames(self):
        # open process, grabbing number of frames using ffprobe
        probecmd = [_FFMPEG_PATH + "/ffprobe"] + ["-v", "error", "-count_frames", "-select_streams", "v:0",
                                                  "-show_entries", "stream=nb_read_frames", "-of",
                                                  "default=nokey=1:noprint_wrappers=1", self._filename]
        return np.int(check_output(probecmd).decode().split('\n')[0])

    def _probe(self):
        return ffprobe(self._filename)

    def _getSupportedDecoders(self):
        return _FFMPEG_SUPPORTED_DECODERS


startupinfo = sp.STARTUPINFO()
startupinfo.dwFlags = sp.CREATE_NEW_CONSOLE | sp.STARTF_USESHOWWINDOW
startupinfo.wShowWindow = sp.SW_HIDE


class FFmpegWriter(VideoWriterAbstract):
    """Writes frames using FFmpeg

    Using FFmpeg as a backend, this class
    provides sane initializations for the default case.
    """

    def __init__(self, *args, **kwargs):
        assert _HAS_FFMPEG, "Cannot find installation of real FFmpeg (which comes with ffprobe)."
        super(FFmpegWriter, self).__init__(*args, **kwargs)

    def _getSupportedEncoders(self):
        return _FFMPEG_SUPPORTED_ENCODERS

    def _createProcess(self, inputdict, outputdict, verbosity):
        iargs = self._dict2Args(inputdict)
        oargs = self._dict2Args(outputdict)

        cmd = [_FFMPEG_PATH + "/" + _FFMPEG_APPLICATION, "-y"] + iargs + ["-i", "-"] + oargs + [self._filename]

        self._cmd = " ".join(cmd)

        # Launch process
        if self.verbosity > 0:
            try:
                print("FFmpeg Write Command:", self._cmd)
            except UnicodeEncodeError:
                print("FFmpeg Write Command: NON-ASCII character exists in command, not shown")
            self._proc = sp.Popen(cmd, stdin=sp.PIPE,
                                  stdout=sp.PIPE, stderr=None, startupinfo=startupinfo)
        else:
            self._proc = sp.Popen(cmd, stdin=sp.PIPE,
                                  stdout=self.DEVNULL, stderr=sp.STDOUT, startupinfo=startupinfo)


class EnccWriter(VideoWriterAbstract):
    """Writes frames using NVencc

    Using NVencc as a backend, this class
    provides sane initializations for the default case.
    """

    def __init__(self, *args, **kwargs):
        assert _HAS_FFMPEG, "Cannot find installation of real NVencc (which comes with ffmpeg)."
        super(EnccWriter, self).__init__(*args, **kwargs)
        self.bit_depth = 8

    def _getSupportedEncoders(self):
        return _FFMPEG_SUPPORTED_ENCODERS

    def _createProcess(self, inputdict, outputdict, verbosity):
        if inputdict['encc'] == "NVENCC":
            _ENCC_APPLICATION = "NVEncC64"
        else:
            _ENCC_APPLICATION = "QSVEncC64"
        _ENCC_APPLICATION += ".exe"
        inputdict.pop('encc')
        n_inputdict = self._dealWithFFmpegArgs(inputdict)
        n_outputdict = self._dealWithFFmpegArgs(outputdict)
        if '-s' in inputdict:
            n_inputdict.update({'--input-res': inputdict['-s']})
            n_inputdict.pop('-s')
        if '--output-depth' in outputdict:
            if outputdict['--output-depth'] == '10':
                self.bit_depth = 10
        """
        !!!Attention!!!
        "--input-csp", "yv12" Yes
        "--input-csp yv12" No
        """

        iargs = self._dict2Args(inputdict)
        oargs = self._dict2Args(outputdict)
        n_iargs = self._dict2Args(n_inputdict)
        n_oargs = self._dict2Args(n_outputdict)

        cmd = [_FFMPEG_PATH + "/" + _ENCC_APPLICATION] + ["--raw", "--input-csp", "yv12"] + n_iargs + ["-i",
                                                                                                       "-", ] + n_oargs + [
                  "-o"] + [
                  self._filename]
        # cmd = ['D:\\60-fps-Project\\ffmpeg/QSVEncC64.exe', '--raw', '--input-csp', 'yv12', '--fps', '24.0',
        #        '--input-res', '1920x1080', '-i', '-', '--la-depth', '50', '--la-quality', 'slow', '--extbrc', '--mbbrc',
        #        '--i-adapt', '--b-adapt', '--gop-len', '250', '-b', '6', '--ref', '8', '--b-pyramid', '--weightb',
        #        '--weightp', '--adapt-ltr', '--colorrange', 'tv', '--colormatrix', 'bt709', '--transfer', 'bt709',
        #        '--colorprim', 'bt709', '-c', 'hevc', '--profile', 'main', '--tier', 'main', '--sao', 'luma', '--ctu',
        #        '64', '--la-icq', '16', '-o',
        #        'D:\\60-fps-Project\\input_or_ref\\Test\\输出output\\【3】批处理1_SVFI_Render_847013.mp4']
        # H264 8bit QSVEncC64
        # cmd = ['D:\\60-fps-Project\\ffmpeg/QSVEncC64.exe', '--raw', '--input-csp', 'yv12', '--fps', '24.0', '--input-res', '1920x1080', '-i', '-',
        #         '--la-depth', '50', '--la-quality', 'slow', '--extbrc', '--mbbrc', '--i-adapt', '--b-adapt', '--gop-len', '250',
        #         '-b', '6', '--ref', '8', '--b-pyramid', '--weightb', '--weightp', '--adapt-ltr',
        #         '--colorrange', 'tv', '--colormatrix', 'bt709', '--transfer', 'bt709', '--colorprim', 'bt709',
        #         '-c', 'h264', '--profile', 'high', '--repartition-check', '--trellis', 'all', 
        #         '--la-icq', '16',
        #        '-o','D:\\60-fps-Project\\input_or_ref\\Test\\输出output\\【3】批处理1_SVFI_Render_847013.mp4']
        self._cmd = " ".join(cmd)

        # Launch process
        if self.verbosity > 0:
            try:
                print("EnCc Write Command:", self._cmd)
            except UnicodeEncodeError:
                print("EnCc Write Command: NON-ASCII character exists in command, not shown")
            self._proc = sp.Popen(cmd, stdin=sp.PIPE,
                                  stdout=sp.PIPE, stderr=None, startupinfo=startupinfo)
        else:
            self._proc = sp.Popen(cmd, stdin=sp.PIPE,
                                  stdout=self.DEVNULL, stderr=sp.STDOUT, startupinfo=startupinfo, shell=True)

    def writeFrame(self, im: np.ndarray):
        """Sends ndarray frames to NVENCC

        """
        vid = vshape(im)
        T, M, N, C = vid.shape
        if not self.warmStarted:
            self._warmStart(M, N, C, im.dtype)

        assert self._proc is not None  # Check status

        try:
            im = im.astype(np.uint8)
            vid = cv2.cvtColor(im, cv2.COLOR_BGR2YUV_YV12)
            # if self.bit_depth == 10:
            #     vid <<= 2
            self._proc.stdin.write(vid.tostring())
        except IOError as e:
            # Show the command and stderr from pipe
            msg = '{0:}\n\nENCODE COMMAND:\n{1:}\n\nENCODE STDERR ' \
                  'OUTPUT:{2:}\n'.format(e, self._cmd, sp.STDOUT)
            raise IOError(msg)

    def _dealWithFFmpegArgs(self, args: dict):
        input_args = args.copy()
        pop_list = ['-f', '-pix_fmt']
        for p in pop_list:
            if p in input_args:
                input_args.pop(p)
        return input_args
        pass

class SVTWriter(EnccWriter):
    """Writes frames using SVT

    Using SVT as a backend, this class
    provides sane initializations for the default case.
    """

    def __init__(self, *args, **kwargs):
        assert _HAS_FFMPEG, "Cannot find installation of real SVT (which comes with ffmpeg)."
        super(SVTWriter, self).__init__(*args, **kwargs)

    def _createProcess(self, inputdict, outputdict, verbosity):
        if outputdict['encc'] == "hevc":
            _SVT_APPLICATION = "SvtHevcEncApp"
        elif outputdict['encc'] == "vp9":
            _SVT_APPLICATION = "SvtVp9EncApp"
        else:  # av1
            _SVT_APPLICATION = "SvtAv1EncApp"
        _SVT_APPLICATION += ".exe"
        outputdict.pop('encc')
        n_inputdict = self._dealWithFFmpegArgs(inputdict)
        n_outputdict = self._dealWithFFmpegArgs(outputdict)
        if '-s' in inputdict:
            input_resolution = inputdict['-s'].split('x')
            n_inputdict.update({'-w': input_resolution[0], '-h': input_resolution[1]})
            n_inputdict.pop('-s')
        if '-s' in outputdict:
            n_outputdict.pop('-s')
        if '-bit-depth' in outputdict:
            if outputdict['-bit-depth'] == '10':
                self.bit_depth = 10
        """
        !!!Attention!!!
        "--input-csp", "yv12" Yes
        "--input-csp yv12" No
        """

        iargs = self._dict2Args(inputdict)
        oargs = self._dict2Args(outputdict)
        n_iargs = self._dict2Args(n_inputdict)
        n_oargs = self._dict2Args(n_outputdict)

        # _cmd = [_FFMPEG_PATH + "/" + _FFMPEG_APPLICATION] + ["-i", "-"]
        # if outputdict['-bit-depth'] in ['8']:
        #     _cmd += ["-pix_fmt", "yuv420p"]
        # else:
        #     """10bit"""
        #     _cmd += ["-pix_fmt", "yuv420p10le"]
        # _cmd += ["-f", "rawvideo", "-", "|"]

        cmd = [_FFMPEG_PATH + "/" + _SVT_APPLICATION] + ["-i", "stdin"] + n_iargs + n_oargs + ["-b",
                                                                                                           self._filename]
        self._cmd = " ".join(cmd)
        # self._cmd = r"D:\60-fps-Project\ffmpeg\ffmpeg.exe -i D:\60-fps-Project\input_or_ref\Test\【4】暗场+黑边裁切+时长片段+字幕轨合并.mkv -pix_fmt yuv420p10le -f rawvideo - | D:\60-fps-Project\ffmpeg\SvtHevcEncApp.exe -i stdin -fps 25 -n 241 -w 3840 -h 2160 -brr 1 -sharp 1 -q 16 -bit-depth 10 -b D:\60-fps-Project\input_or_ref\Test\svt_output.mp4"
        # cmd = [_FFMPEG_PATH + "/" + _SVT_APPLICATION] + ["-i", "stdin"] + n_iargs + n_oargs + ["-b", self._filename]
        # self._cmd = " ".join(cmd)

        # Launch process
        if self.verbosity > 0:
            try:
                print("SVT Write Command:", self._cmd)
            except UnicodeEncodeError:
                print("SVT Write Command: NON-ASCII character exists in command, not shown")
            self._proc = sp.Popen(cmd, stdin=sp.PIPE,
                                  stdout=sp.PIPE, stderr=None, startupinfo=startupinfo)
        else:
            self._proc = sp.Popen(cmd, stdin=sp.PIPE,
                                  stdout=self.DEVNULL, stderr=sp.STDOUT, startupinfo=startupinfo)
