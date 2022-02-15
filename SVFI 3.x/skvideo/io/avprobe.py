import subprocess as sp

from ..utils import *
from .. import _HAS_AVCONV, _LIBAV_MAJOR_VERSION
from .. import _AVCONV_PATH
from .. import _AVPROBE_APPLICATION
import json


def avprobe(filename):
    """get metadata by using avprobe

    Checks the output of avprobe on the desired video
    file. MetaData is then parsed into a dictionary.

    Parameters
    ----------
    filename : string
        Path to the video file

    Returns
    -------
    metaDict : dict
       Dictionary containing all header-based information 
       about the passed-in source video.

    """
    # check if FFMPEG exists in the path
    assert _HAS_AVCONV, "Cannot find installation of avprobe."
    assert int(_LIBAV_MAJOR_VERSION) >= 10, "Version of libav (" + str(_LIBAV_MAJOR_VERSION) +") < 9. Please update libav or use ffmpeg."

    try:
        command = [_AVCONV_PATH + "/" + _AVPROBE_APPLICATION, "-v", "error", "-show_streams", "-of", "json", filename]

        # simply get std output
        jsonstr = check_output(command)
        probedict = json.loads(jsonstr.decode())

        d = probedict["streams"]

        # check type
        streamsbytype = {}
        if type(d) is list:
            # go through streams
            for stream in d:
                streamsbytype[stream["codec_type"].lower()] = stream
        else:
            streamsbytype[d["codec_type"].lower()] = d

        return streamsbytype
    except:
        return {}
