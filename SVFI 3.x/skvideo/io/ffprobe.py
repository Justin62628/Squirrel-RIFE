import subprocess as sp

from ..utils import *
from .. import _HAS_FFMPEG
from .. import _FFMPEG_PATH
from .. import _FFPROBE_APPLICATION

def ffprobe(filename):
    """get metadata by using ffprobe

    Checks the output of ffprobe on the desired video
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
    assert _HAS_FFMPEG, "Cannot find installation of real FFmpeg (which comes with ffprobe)."

    try:
        command = [_FFMPEG_PATH + "/" + _FFPROBE_APPLICATION, "-v", "error", "-show_streams", "-print_format", "xml", filename]

        # simply get std output
        xml = check_output(command)

        d = xmltodictparser(xml)["ffprobe"]

        d = d["streams"]

        #import json
        #print json.dumps(d, indent = 4)
        #exit(0)

        # check type
        streamsbytype = {}
        if type(d["stream"]) is list:
            # go through streams
            for stream in d["stream"]:
                streamsbytype[stream["@codec_type"].lower()] = stream
        else:
            streamsbytype[d["stream"]["@codec_type"].lower()] = d["stream"]

        return streamsbytype
    except:
        return {}
