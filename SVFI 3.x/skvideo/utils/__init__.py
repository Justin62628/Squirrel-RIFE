""" Various utility functions for manipulating video data

"""
import os
import platform
import itertools
from .xmltodict import parse as xmltodictparser
import subprocess as sp
import numpy as np
from .edge import canny
from .stpyr import SpatialSteerablePyramid, rolling_window
from .mscn import compute_image_mscn_transform, gen_gauss_window
from .stats import ggd_features, aggd_features, paired_product


# dictionary based on pix_fmt keys
# first element is number of components
# second element is number of bits per pixel
bpplut = {}
bpplut["yuv420p"] = [3, 12]
bpplut["yuyv422"] = [3, 16]
bpplut["rgb24"] = [3, 24]
bpplut["bgr24"] = [3, 24]
bpplut["yuv422p"] = [3, 16]
bpplut["yuv444p"] = [3, 24]
bpplut["yuv410p"] = [3, 9]
bpplut["yuv411p"] = [3, 12]
bpplut["gray"] = [1, 8]
bpplut["monow"] = [1, 1]
bpplut["monob"] = [1, 1]
bpplut["pal8"] = [1, 8]
bpplut["yuvj420p"] = [3, 12]
bpplut["yuvj422p"] = [3, 16]
bpplut["yuvj444p"] = [3, 24]
bpplut["xvmcmc"] = [0, 0]
bpplut["xvmcidct"] = [0, 0]
bpplut["uyvy422"] = [3, 16]
bpplut["uyyvyy411"] = [3, 12]
bpplut["bgr8"] = [3, 8]
bpplut["bgr4"] = [3, 4]
bpplut["bgr4_byte"] = [3, 4]
bpplut["rgb8"] = [3, 8]
bpplut["rgb4"] = [3, 4]
bpplut["rgb4_byte"] = [3, 4]
bpplut["nv12"] = [3, 12]
bpplut["nv21"] = [3, 12]
bpplut["argb"] = [4, 32]
bpplut["rgba"] = [4, 32]
bpplut["abgr"] = [4, 32]
bpplut["bgra"] = [4, 32]
bpplut["gray16be"] = [1, 16]
bpplut["gray16le"] = [1, 16]
bpplut["yuv440p"] = [3, 16]
bpplut["yuvj440p"] = [3, 16]
bpplut["yuva420p"] = [4, 20]
bpplut["vdpau_h264"] = [0, 0]
bpplut["vdpau_mpeg1"] = [0, 0]
bpplut["vdpau_mpeg2"] = [0, 0]
bpplut["vdpau_wmv3"] = [0, 0]
bpplut["vdpau_vc1"] = [0, 0]
bpplut["rgb48be"] = [3, 48]
bpplut["rgb48le"] = [3, 48]
bpplut["rgb565be"] = [3, 16]
bpplut["rgb565le"] = [3, 16]
bpplut["rgb555be"] = [3, 15]
bpplut["rgb555le"] = [3, 15]
bpplut["bgr565be"] = [3, 16]
bpplut["bgr565le"] = [3, 16]
bpplut["bgr555be"] = [3, 15]
bpplut["bgr555le"] = [3, 15]
bpplut["vaapi_moco"] = [0, 0]
bpplut["vaapi_idct"] = [0, 0]
bpplut["vaapi_vld"] = [0, 0]
bpplut["yuv420p16le"] = [3, 24]
bpplut["yuv420p16be"] = [3, 24]
bpplut["yuv422p16le"] = [3, 32]
bpplut["yuv422p16be"] = [3, 32]
bpplut["yuv444p16le"] = [3, 48]
bpplut["yuv444p16be"] = [3, 48]
bpplut["vdpau_mpeg4"] = [0, 0]
bpplut["dxva2_vld"] = [0, 0]
bpplut["rgb444le"] = [3, 12]
bpplut["rgb444be"] = [3, 12]
bpplut["bgr444le"] = [3, 12]
bpplut["bgr444be"] = [3, 12]
bpplut["ya8"] = [2, 16]
bpplut["bgr48be"] = [3, 48]
bpplut["bgr48le"] = [3, 48]
bpplut["yuv420p9be"] = [3, 13]
bpplut["yuv420p9le"] = [3, 13]
bpplut["yuv420p10be"] = [3, 15]
bpplut["yuv420p10le"] = [3, 15]
bpplut["yuv422p10be"] = [3, 20]
bpplut["yuv422p10le"] = [3, 20]
bpplut["yuv444p9be"] = [3, 27]
bpplut["yuv444p9le"] = [3, 27]
bpplut["yuv444p10be"] = [3, 30]
bpplut["yuv444p10le"] = [3, 30]
bpplut["yuv422p9be"] = [3, 18]
bpplut["yuv422p9le"] = [3, 18]
bpplut["vda_vld"] = [0, 0]
bpplut["gbrp"] = [3, 24]
bpplut["gbrp9be"] = [3, 27]
bpplut["gbrp9le"] = [3, 27]
bpplut["gbrp10be"] = [3, 30]
bpplut["gbrp10le"] = [3, 30]
bpplut["gbrp16be"] = [3, 48]
bpplut["gbrp16le"] = [3, 48]
bpplut["yuva420p9be"] = [4, 22]
bpplut["yuva420p9le"] = [4, 22]
bpplut["yuva422p9be"] = [4, 27]
bpplut["yuva422p9le"] = [4, 27]
bpplut["yuva444p9be"] = [4, 36]
bpplut["yuva444p9le"] = [4, 36]
bpplut["yuva420p10be"] = [4, 25]
bpplut["yuva420p10le"] = [4, 25]
bpplut["yuva422p10be"] = [4, 30]
bpplut["yuva422p10le"] = [4, 30]
bpplut["yuva444p10be"] = [4, 40]
bpplut["yuva444p10le"] = [4, 40]
bpplut["yuva420p16be"] = [4, 40]
bpplut["yuva420p16le"] = [4, 40]
bpplut["yuva422p16be"] = [4, 48]
bpplut["yuva422p16le"] = [4, 48]
bpplut["yuva444p16be"] = [4, 64]
bpplut["yuva444p16le"] = [4, 64]
bpplut["vdpau"] = [0, 0]
bpplut["xyz12le"] = [3, 36]
bpplut["xyz12be"] = [3, 36]
bpplut["nv16"] = [3, 16]
bpplut["nv20le"] = [3, 20]
bpplut["nv20be"] = [3, 20]
bpplut["yvyu422"] = [3, 16]
bpplut["vda"] = [0, 0]
bpplut["ya16be"] = [2, 32]
bpplut["ya16le"] = [2, 32]
bpplut["qsv"] = [0, 0]
bpplut["mmal"] = [0, 0]
bpplut["d3d11va_vld"] = [0, 0]
bpplut["rgba64be"] = [4, 64]
bpplut["rgba64le"] = [4, 64]
bpplut["bgra64be"] = [4, 64]
bpplut["bgra64le"] = [4, 64]
bpplut["0rgb"] = [3, 24]
bpplut["rgb0"] = [3, 24]
bpplut["0bgr"] = [3, 24]
bpplut["bgr0"] = [3, 24]
bpplut["yuva444p"] = [4, 32]
bpplut["yuva422p"] = [4, 24]
bpplut["yuv420p12be"] = [3, 18]
bpplut["yuv420p12le"] = [3, 18]
bpplut["yuv420p14be"] = [3, 21]
bpplut["yuv420p14le"] = [3, 21]
bpplut["yuv422p12be"] = [3, 24]
bpplut["yuv422p12le"] = [3, 24]
bpplut["yuv422p14be"] = [3, 28]
bpplut["yuv422p14le"] = [3, 28]
bpplut["yuv444p12be"] = [3, 36]
bpplut["yuv444p12le"] = [3, 36]
bpplut["yuv444p14be"] = [3, 42]
bpplut["yuv444p14le"] = [3, 42]
bpplut["gbrp12be"] = [3, 36]
bpplut["gbrp12le"] = [3, 36]
bpplut["gbrp14be"] = [3, 42]
bpplut["gbrp14le"] = [3, 42]
bpplut["gbrap"] = [4, 32]
bpplut["gbrap16be"] = [4, 64]
bpplut["gbrap16le"] = [4, 64]
bpplut["yuvj411p"] = [3, 12]
bpplut["bayer_bggr8"] = [3, 8]
bpplut["bayer_rggb8"] = [3, 8]
bpplut["bayer_gbrg8"] = [3, 8]
bpplut["bayer_grbg8"] = [3, 8]
bpplut["bayer_bggr16le"] = [3, 16]
bpplut["bayer_bggr16be"] = [3, 16]
bpplut["bayer_rggb16le"] = [3, 16]
bpplut["bayer_rggb16be"] = [3, 16]
bpplut["bayer_gbrg16le"] = [3, 16]
bpplut["bayer_gbrg16be"] = [3, 16]
bpplut["bayer_grbg16le"] = [3, 16]
bpplut["bayer_grbg16be"] = [3, 16]
bpplut["yuv440p10le"] = [3, 20]
bpplut["yuv440p10be"] = [3, 20]
bpplut["yuv440p12le"] = [3, 24]
bpplut["yuv440p12be"] = [3, 24]
bpplut["ayuv64le"] = [4, 64]
bpplut["ayuv64be"] = [4, 64]
bpplut["videotoolbox_vld"] = [0, 0]



# python2 only
binary_type = str

def check_dict(dic, key, valueifnot):
    if key not in dic:
        dic[key] = valueifnot


# patch for python 2.6
def check_output(*popenargs, **kwargs):
    closeNULL = 0
    try:
        from subprocess import DEVNULL
        closeNULL = 0
    except ImportError:
        import os
        try:
            DEVNULL = open(os.devnull, 'wb')
        except FileNotFoundError:
            DEVNULL = open('null', 'wb')
        closeNULL = 1
    startupinfo = sp.STARTUPINFO()
    startupinfo.dwFlags = sp.CREATE_NEW_CONSOLE | sp.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = sp.SW_HIDE
    try:
        process = sp.Popen(stdout=sp.PIPE, stderr=DEVNULL, *popenargs, **kwargs, startupinfo=startupinfo)
    except FileNotFoundError:
        DEVNULL = open('null', 'wb')
        closeNULL = 1
        process = sp.Popen(stdout=sp.PIPE, stderr=DEVNULL, *popenargs, **kwargs, startupinfo=startupinfo)
    output, unused_err = process.communicate()
    retcode = process.poll()

    if closeNULL:
        DEVNULL.close()

    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        error = sp.CalledProcessError(retcode, cmd)
        error.output = output
        raise error
    return output

try:
    # on py27 make map/filter behave like an iterator
    map = itertools.imap
    filter = itertools.ifilter
except AttributeError:
    # py3+
    pass


def where( filename ):
    """ Returns all matching file paths. """
    return list(iwhere(filename))


def first( filename ):
    """ Returns first matching file path. """
    try:
        return next(iwhere(filename))
    except StopIteration:
        return None


def iwhere( filename ):
    possible_paths = _gen_possible_matches(filename)
    existing_file_paths = filter(os.path.isfile, possible_paths)
    return existing_file_paths

def iter_unique(iterable):
    yielded = set()
    for i in iterable:
        if i in yielded:
            continue
        yield i
        yielded.add(i)

def imapchain(*a, **kwa):
    """ Like map but also chains the results. """

    imap_results = map(*a, **kwa)
    return itertools.chain(*imap_results)

def _gen_possible_matches(filename):
    path_parts =     os.environ.get("PATH", "").split(os.pathsep)
    path_parts =     itertools.chain((os.curdir,), path_parts)
    possible_paths = map(lambda path_part: os.path.join(path_part, filename), path_parts)

    if platform.system() == "Windows":
        possible_paths = imapchain(lambda path: (path, path+".bat", path+".com", path+".exe"), possible_paths)

    possible_paths = map(os.path.abspath, possible_paths)

    result = iter_unique(possible_paths)

    return result

def vshape(videodata):
    """Standardizes the input data shape.

    Transforms video data into the standardized shape (T, M, N, C), where
    T is number of frames, M is height, N is width, and C is number of 
    channels.

    Parameters
    ----------
    videodata : ndarray
        Input data of shape (T, M, N, C), (T, M, N), (M, N, C), or (M, N), where
        T is number of frames, M is height, N is width, and C is number of 
        channels.

    Returns
    -------
    videodataout : ndarray
        Standardized version of videodata, shape (T, M, N, C)

    """
    if not isinstance(videodata, np.ndarray):
        videodata = np.array(videodata)

    if len(videodata.shape) == 2: 
        a, b = videodata.shape
        return videodata.reshape(1, a, b, 1) 
    elif len(videodata.shape) == 3: 
        a, b, c = videodata.shape
        # check the last dimension small
        # interpret as color channel
        if c in [1, 2, 3, 4]:
            return videodata.reshape(1, a, b, c) 
        else:
            return videodata.reshape(a, b, c, 1) 
    elif len(videodata.shape) == 4: 
        return videodata
    else:
        raise ValueError("Improper data input")

def rgb2gray(videodata):
    """Computes the grayscale video.

    Computes the grayscale video from the input video returning the 
    standardized shape (T, M, N, C), where T is number of frames, 
    M is height, N is width, and C is number of channels (here always 1).

    Parameters
    ----------
    videodata : ndarray
        Input data of shape (T, M, N, C), (T, M, N), (M, N, C), or (M, N), where
        T is number of frames, M is height, N is width, and C is number of 
        channels.

    Returns
    -------
    videodataout : ndarray
        Standardized grayscaled version of videodata, shape (T, M, N, 1)
    """
    videodata = vshape(videodata)
    T, M, N, C = videodata.shape

    if C == 1:
        return videodata
    elif C == 3: # assume RGB
        videodata = videodata[:, :, :, 0]*0.2989 + videodata[:, :, :, 1]*0.5870 + videodata[:, :, :, 2]*0.1140 
        return vshape(videodata)
    else:
        raise NotImplemented

__all__ = [
    'xmltodictparser',
    'bpplut',
    'where',
    'check_output',
    'rgb2gray',
    'vshape',
    'canny',
    'SpatialSteerablePyramid',
    'compute_image_mscn_transform',
    'gen_gauss_window',
    'paired_product',
    'ggd_features',
    'aggd_features',
    'rolling_window'
]
