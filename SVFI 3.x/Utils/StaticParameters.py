import enum
import os

import numpy as np

abspath = os.path.abspath(__file__)
appDir = os.path.dirname(os.path.dirname(abspath))

INVALID_CHARACTERS = ["'", '"']


class TASKBAR_STATE(enum.Enum):
    TBPF_NOPROGRESS = 0x00000000
    TBPF_INDETERMINATE = 0x00000001
    TBPF_NORMAL = 0x00000002
    TBPF_ERROR = 0x00000004
    TBPF_PAUSED = 0x00000008


class HDR_STATE(enum.Enum):
    AUTO = -2
    NOT_CHECKED = -1
    NONE = 0
    CUSTOM_HDR = 1
    HDR10 = 2
    HDR10_PLUS = 3
    DOLBY_VISION = 4
    HLG = 5


class RT_RATIO(enum.Enum):
    """
    Resolution Transfer Ratio
    """
    AUTO = 0
    WHOLE = 1
    THREE_QUARTERS = 2
    HALF = 3
    QUARTER = 4

    @staticmethod
    def get_auto_transfer_ratio(sr_times: float):
        if sr_times >= 1:
            return RT_RATIO.WHOLE
        elif 0.75 <= sr_times < 1:
            return RT_RATIO.THREE_QUARTERS
        elif 0.5 <= sr_times < 0.75:
            return RT_RATIO.HALF
        else:
            return RT_RATIO.QUARTER

    @staticmethod
    def get_surplus_sr_scale(scale: float, ratio):
        if ratio == RT_RATIO.WHOLE:
            return scale
        elif ratio == RT_RATIO.THREE_QUARTERS:
            return scale * (4 / 3)
        elif ratio == RT_RATIO.HALF:
            return scale * 2
        elif ratio == RT_RATIO.QUARTER:
            return scale * 4
        else:
            return scale

    @staticmethod
    def get_modified_resolution(params: tuple, ratio, keep_single=False):
        w, h = params
        if ratio == RT_RATIO.WHOLE:
            w, h = int(w), int(h)
        elif ratio == RT_RATIO.THREE_QUARTERS:
            w, h = int(w / 4 * 3), int(h / 4 * 3)
        elif ratio == RT_RATIO.HALF:
            w, h = int(w / 2), int(h / 2)
        elif ratio == RT_RATIO.QUARTER:
            w, h = int(w / 4), int(h / 4)
        else:
            w, h = int(w), int(h)
        if not keep_single:
            if w % 2:
                w += 1
            if h % 2:
                h += 1
        return w, h


class SR_TILESIZE_STATE(enum.Enum):
    NONE = 0
    CUSTOM = 1
    VRAM_2G = 2
    VRAM_4G = 3
    VRAM_6G = 4
    VRAM_8G = 5
    VRAM_12G = 6

    @staticmethod
    def get_tilesize(state):
        if state == SR_TILESIZE_STATE.NONE:
            return 0
        if state == SR_TILESIZE_STATE.VRAM_2G:
            return 100
        if state == SR_TILESIZE_STATE.VRAM_4G:
            return 200
        if state == SR_TILESIZE_STATE.VRAM_6G:
            return 1000
        if state == SR_TILESIZE_STATE.VRAM_8G:
            return 1200
        if state == SR_TILESIZE_STATE.VRAM_12G:
            return 2000
        return 100


class SupportFormat:
    img_inputs = ['.png', '.tif', '.tiff', '.jpg', '.jpeg']
    img_outputs = ['.png', '.tiff', '.jpg']
    vid_outputs = ['.mp4', '.mkv', '.mov']


class EncodePresetAssemply:
    encoder = {
        "CPU": {
            "H264,8bit": ["slow", "ultrafast", "fast", "medium", "veryslow", "placebo", ],
            "H264,10bit": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "H265,8bit": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "H265,10bit": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "AV1,8bit": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "AV1,10bit": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "ProRes,422": ["hq", "4444", "4444xq"],
            "ProRes,444": ["hq", "4444", "4444xq"],
        },
        "NVENC":
            {"H264,8bit": ["slow", "fast", "hq", "bd", "llhq", "loseless", "p7"],
             "H265,8bit": ["slow", "fast", "hq", "bd", "llhq", "loseless", "p7"],
             "H265,10bit": ["slow", "fast", "hq", "bd", "llhq", "loseless", "p7"], },
        "NVENCC":
            {"H264,8bit": ["default", "performance", "quality"],
             "H265,8bit": ["default", "performance", "quality"],
             "H265,10bit": ["default", "performance", "quality"], },
        "QSVENCC":
            {"H264,8bit": ["best", "higher", "high", "balanced", "fast", "faster", "fastest"],
             "H265,8bit": ["best", "higher", "high", "balanced", "fast", "faster", "fastest"],
             "H265,10bit": ["best", "higher", "high", "balanced", "fast", "faster", "fastest"], },
        "QSV":
            {"H264,8bit": ["slow", "fast", "medium", "veryslow", ],
             "H265,8bit": ["slow", "fast", "medium", "veryslow", ],
             "H265,10bit": ["slow", "fast", "medium", "veryslow", ], },
        "SVT":
            {"VP9,8bit": ["slowest", "slow", "fast", "faster"],
             "H265,8bit": ["slowest", "slow", "fast", "faster"],
             "AV1,8bit": ["slowest", "slow", "fast", "faster"],
             },

    }


class RGB_TYPE:
    SIZE = 65535.
    DTYPE = np.uint8 if SIZE == 255. else np.uint16

    @staticmethod
    def change_8bit(d8: bool):
        if d8:
            RGB_TYPE.SIZE = 255.
            RGB_TYPE.DTYPE = np.uint8 if RGB_TYPE.SIZE == 255. else np.uint16


class LUTS_TYPE(enum.Enum):
    NONE = 0
    PreserveSaturation = 1

    @staticmethod
    def get_lut_path(lut_type):
        if lut_type is LUTS_TYPE.NONE:
            return
        elif lut_type is LUTS_TYPE.PreserveSaturation:
            return "1x3d.cube"


class RIFE_TYPE(enum.Enum):
    """
    0000000 = RIFEv2, RIFEv3, RIFEv6, RIFEv7/RIFE 4.0, RIFEvNew from Master Zhe, XVFI, ABME

        Rv2 Rv3 Rv6 Rv4 Rv7 RvNew XVFI ABME
    DS   1   1   1   1   1    1    0   0
    TTA  1   1   1   0   0    0    0   1
    MC   0   0   0   0   0    0    0   0
    EN   1   1   1   1   1    1    0   0
    OE   0   0   0   0   0    1    0   0
    """
    RIFEv2 = 0b10000000
    RIFEv3 = 0b01000000
    RIFEv6 = 0b00100000
    RIFEv4 = 0b00010000
    RIFEv7 = 0b00001000
    RIFEvAnyTime = 0b00000100
    XVFI = 0b00000010
    ABME = 0b00000001

    DS = 0b11111100
    TTA = 0b11100001
    ENSEMBLE = 0b11111100
    MULTICARD = 0b00000000
    OUTPUTMODE = 0b00000100
