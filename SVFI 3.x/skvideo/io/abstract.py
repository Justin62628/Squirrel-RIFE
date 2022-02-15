import os
import time
import warnings

import numpy as np

from Utils.StaticParameters import RGB_TYPE
from .. import _HAS_FFMPEG
from ..utils import *


class VideoReaderAbstract(object):
    """Reads frames
    """

    INFO_AVERAGE_FRAMERATE = None  # "avg_frame_rate"
    INFO_WIDTH = None  # "width"
    INFO_HEIGHT = None  # "height"
    INFO_PIX_FMT = None  # "pix_fmt"
    INFO_DURATION = None  # "duration"
    INFO_NB_FRAMES = None  # "nb_frames"
    DEFAULT_FRAMERATE = 25.
    DEFAULT_INPUT_PIX_FMT = "yuvj444p"
    OUTPUT_METHOD = None  # "rawvideo"

    def __init__(self, filename, inputdict=None, outputdict=None, verbosity=0):
        """Initializes FFmpeg in reading mode with the given parameters

        During initialization, additional parameters about the video file
        are parsed using :func:`skvideo.io.ffprobe`. Then FFmpeg is launched
        as a subprocess. Parameters passed into inputdict are parsed and
        used to set as internal variables about the video. If the parameter,
        such as "Height" is not found in the inputdict, it is found through
        scanning the file's header information. If not in the header, ffprobe
        is used to decode the file to determine the information. In the case
        that the information is not supplied and connot be inferred from the
        input file, a ValueError exception is thrown.

        Parameters
        ----------
        filename : string
            Video file path

        inputdict : dict
            Input dictionary parameters, i.e. how to interpret the input file.

        outputdict : dict
            Output dictionary parameters, i.e. how to encode the data
            when sending back to the python process.

        Returns
        -------
        none

        """
        # check if FFMPEG exists in the path
        self._proc = None
        assert _HAS_FFMPEG, "Cannot find installation of real FFmpeg (which comes with ffprobe)."

        self._filename = filename
        self.verbosity = verbosity

        if not inputdict:
            inputdict = {}

        if not outputdict:
            outputdict = {}

        # General information
        _, self.extension = os.path.splitext(filename)
        self.size = os.path.getsize(filename)
        self.probeInfo = self._probe()

        # smartphone video data is weird
        self.rotationAngle = '0'  # specific FFMPEG

        viddict = {}
        if "video" in self.probeInfo:
            viddict = self.probeInfo["video"]

        self.inputfps = -1
        if ("-r" in inputdict):
            self.inputfps = np.int(inputdict["-r"])
        elif self.INFO_AVERAGE_FRAMERATE in viddict:
            # check for the slash
            frtxt = viddict[self.INFO_AVERAGE_FRAMERATE]
            parts = frtxt.split('/')
            if len(parts) > 1:
                if np.float(parts[1]) == 0.:
                    self.inputfps = self.DEFAULT_FRAMERATE
                else:
                    self.inputfps = np.float(parts[0]) / np.float(parts[1])
            else:
                self.inputfps = np.float(frtxt)
        else:
            self.inputfps = self.DEFAULT_FRAMERATE

        # check for transposition tag
        if ('tag' in viddict):
            tagdata = viddict['tag']
            if not isinstance(tagdata, list):
                tagdata = [tagdata]

            for tags in tagdata:
                if tags['@key'] == 'rotate':
                    self.rotationAngle = tags['@value']

        # if we don't have width or height at all, raise exception
        if ("-s" in inputdict):
            widthheight = inputdict["-s"].split('x')
            self.inputwidth = np.int(widthheight[0])
            self.inputheight = np.int(widthheight[1])
        elif ((self.INFO_WIDTH in viddict) and (self.INFO_HEIGHT in viddict)):
            self.inputwidth = np.int(viddict[self.INFO_WIDTH])
            self.inputheight = np.int(viddict[self.INFO_HEIGHT])
        else:
            raise ValueError(
                "No way to determine width or height from video. Need `-s` in `inputdict`. Consult documentation on I/O.")

        # smartphone recordings seem to store data about rotations
        # in tag format. Just swap the width and height
        if self.rotationAngle == '90' or self.rotationAngle == '270':
            self.inputwidth, self.inputheight = self.inputheight, self.inputwidth

        self.bpp = -1  # bits per pixel
        self.pix_fmt = ""
        # completely unsure of this:
        if ("-pix_fmt" in inputdict):
            self.pix_fmt = inputdict["-pix_fmt"]
        elif (self.INFO_PIX_FMT in viddict):
            # parse this bpp
            self.pix_fmt = viddict[self.INFO_PIX_FMT]
        else:
            self.pix_fmt = self.DEFAULT_INPUT_PIX_FMT
            if verbosity != 0:
                warnings.warn("No input color space detected. Assuming {}.".format(self.DEFAULT_INPUT_PIX_FMT),
                              UserWarning)

        self.inputdepth = np.int(bpplut[self.pix_fmt][0])
        self.bpp = np.int(bpplut[self.pix_fmt][1])

        israw = str.encode(self.extension) in [b".raw", b".yuv"]
        iswebcam = not os.path.isfile(filename)

        if ("-vframes" in outputdict):
            self.inputframenum = np.int(outputdict["-vframes"])
        elif ("-r" in outputdict):
            inputfps = np.int(outputdict["-r"])
            inputduration = np.float(viddict[self.INFO_DURATION])
            self.inputframenum = np.int(round(inputfps * inputduration) + 1)
        elif (self.INFO_NB_FRAMES in viddict):
            self.inputframenum = np.int(viddict[self.INFO_NB_FRAMES])
        elif israw:
            # we can compute it based on the input size and color space
            self.inputframenum = np.int(self.size / (self.inputwidth * self.inputheight * (self.bpp / 8.0)))
        elif iswebcam:
            # webcam can stream frames endlessly, lets use the special default value of 0 to indicate that
            self.inputframenum = 0
        else:
            self.inputframenum = self._probCountFrames()
            if verbosity != 0:
                warnings.warn(
                    "Cannot determine frame count. Scanning input file, this is slow when repeated many times. Need `-vframes` in inputdict. Consult documentation on I/O.",
                    UserWarning)

        if israw or iswebcam:
            inputdict['-pix_fmt'] = self.pix_fmt
        else:
            decoders = self._getSupportedDecoders()
            if decoders != NotImplemented:
                # check that the extension makes sense
                assert str.encode(
                    self.extension).lower() in decoders, "Unknown decoder extension: " + self.extension.lower()

        if '-f' not in outputdict:
            outputdict['-f'] = self.OUTPUT_METHOD

        if '-pix_fmt' not in outputdict:
            if RGB_TYPE.SIZE == 255.:
                outputdict['-pix_fmt'] = "rgb24"
            else:
                outputdict['-pix_fmt'] = "rgb48be"  # 48
        self.output_pix_fmt = outputdict['-pix_fmt']

        if '-s' in outputdict:
            widthheight = outputdict["-s"].split('x')
            self.outputwidth = np.int(widthheight[0])
            self.outputheight = np.int(widthheight[1])
        else:
            self.outputwidth = self.inputwidth
            self.outputheight = self.inputheight

        self.outputdepth = np.int(bpplut[outputdict['-pix_fmt']][0])
        self.outputbpp = np.int(bpplut[outputdict['-pix_fmt']][1])
        bitpercomponent = self.outputbpp // self.outputdepth
        if bitpercomponent == 8:
            self.dtype = np.dtype('u1')  # np.uint8
        elif bitpercomponent == 16:
            suffix = outputdict['-pix_fmt'][-2:]
            if suffix == 'le':
                self.dtype = np.dtype('<u2')
            elif suffix == 'be':
                self.dtype = np.dtype('>u2')
        else:
            raise ValueError(outputdict['-pix_fmt'] + 'is not a valid pix_fmt for numpy conversion')

        self._createProcess(inputdict, outputdict, verbosity)

    def __next__(self):
        return next(self.nextFrame())

    def __iter__(self):
        for frame in self.nextFrame():
            yield frame

    def _createProcess(self, inputdict, outputdict, verbosity):
        pass

    def _probCountFrames(self):
        return NotImplemented

    def _probe(self):
        pass

    def _getSupportedDecoders(self):
        return NotImplemented

    def _dict2Args(self, dict):
        args = []
        for key in dict.keys():
            args.append(key)
            args.append(dict[key])
        return args

    def getShape(self):
        """Returns a tuple (T, M, N, C)

        Returns the video shape in number of frames, height, width, and channels per pixel.
        """

        return self.inputframenum, self.outputheight, self.outputwidth, self.outputdepth

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.stdout.close()
            self._proc.stderr.close()
            self._terminate(0.2)

    def _terminate(self, timeout=1.0):
        """ Terminate the sub process.
        """
        # Check
        if self._proc is None:  # pragma: no cover
            return  # no process
        if self._proc.poll() is not None:
            return  # process already dead
        # Terminate process
        self._proc.terminate()
        # Wait for it to close (but do not get stuck)
        etime = time.time() + timeout
        while time.time() < etime:
            time.sleep(0.01)
            if self._proc.poll() is not None:
                break

    def _read_frame_data(self):
        # Init and check
        framesize = self.outputdepth * self.outputwidth * self.outputheight
        assert self._proc is not None

        try:
            # Read framesize bytes
            pipe_read = self._proc.stdout.read(framesize * self.dtype.itemsize)
            arr = np.frombuffer(pipe_read, dtype=self.dtype)
            if len(arr) != framesize:
                return np.array([])
            # assert len(arr) == framesize
        except Exception as err:
            self._terminate()
            err1 = str(err)
            raise RuntimeError("%s" % (err1,))
        return arr

    def _readFrame(self):
        # Read and convert to numpy array
        frame = self._read_frame_data()
        if len(frame) == 0:
            return frame

        if self.output_pix_fmt in ['rgb24', 'rgb48be']:
            self._lastread = frame.reshape((self.outputheight, self.outputwidth, self.outputdepth))
        elif self.output_pix_fmt.startswith('yuv444p') or self.output_pix_fmt.startswith(
                'yuvj444p') or self.output_pix_fmt.startswith('yuva444p'):
            self._lastread = frame.reshape((self.outputdepth, self.outputheight, self.outputwidth)).transpose((1, 2, 0))
        else:
            if self.verbosity > 0:
                warnings.warn(
                    'Unsupported reshaping from raw buffer to images frames  for format {:}. Assuming HEIGHTxWIDTHxCOLOR'.format(
                        self.output_pix_fmt), UserWarning)
            self._lastread = frame.reshape((self.outputheight, self.outputwidth, self.outputdepth))

        return self._lastread

    def nextFrame(self):
        """Yields frames using a generator

        Returns T ndarrays of size (M, N, C), where T is number of frames,
        M is height, N is width, and C is number of channels per pixel.

        """
        while True:
            frame = self._readFrame()
            if len(frame) == 0:
                break
            yield frame

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class VideoWriterAbstract(object):
    """Writes frames

    this class provides sane initializations for the default case.
    """
    NEED_RGB2GRAY_HACK = False
    DEFAULT_OUTPUT_PIX_FMT = "yuvj444p"

    def __init__(self, filename, inputdict=None, outputdict=None, verbosity=0):
        """Prepares parameters

        Does not instantiate the an FFmpeg subprocess, but simply
        prepares the required parameters.

        Parameters
        ----------
        filename : string
            Video file path for writing

        inputdict : dict
            Input dictionary parameters, i.e. how to interpret the data coming from python.

        outputdict : dict
            Output dictionary parameters, i.e. how to encode the data
            when writing to file.

        Returns
        -------
        none

        """
        self._proc = None
        self._cmd = None
        try:
            self.DEVNULL = open(os.devnull, 'wb')
        except FileNotFoundError:
            self.DEVNULL = open('null', 'wb')

        filename = os.path.abspath(filename)
        _, self.extension = os.path.splitext(filename)

        # check that the extension makes sense
        encoders = self._getSupportedEncoders()
        if encoders != NotImplemented:
            assert str.encode(
                self.extension).lower() in encoders, "Unknown encoder extension: " + self.extension.lower()

        self._filename = filename
        basepath, _ = os.path.split(filename)

        # check to see if filename is a valid file location
        assert os.access(basepath, os.W_OK), "Cannot write to directory: " + basepath

        if not inputdict:
            inputdict = {}

        if not outputdict:
            outputdict = {}

        self.inputdict = inputdict
        self.outputdict = outputdict
        self.verbosity = verbosity

        if "-f" not in self.inputdict:
            self.inputdict["-f"] = "rawvideo"
        self.warmStarted = False

    def _warmStart(self, M, N, C, dtype):
        self.warmStarted = True

        if "-pix_fmt" not in self.inputdict:
            # check the number channels to guess
            if dtype.kind == 'u' and dtype.itemsize == 2:
                suffix = 'le' if dtype.byteorder else 'be'
                if C == 1:
                    if self.NEED_RGB2GRAY_HACK:
                        self.inputdict["-pix_fmt"] = "rgb48" + suffix
                        self.rgb2grayhack = True
                        C = 3
                    else:
                        self.inputdict["-pix_fmt"] = "gray16" + suffix
                elif C == 2:
                    self.inputdict["-pix_fmt"] = "ya16" + suffix
                elif C == 3:
                    self.inputdict["-pix_fmt"] = "rgb48" + suffix
                elif C == 4:
                    self.inputdict["-pix_fmt"] = "rgba64" + suffix
                else:
                    raise NotImplemented
            else:
                if C == 1:
                    if self.NEED_RGB2GRAY_HACK:
                        self.inputdict["-pix_fmt"] = "rgb24"
                        self.rgb2grayhack = True
                        C = 3
                    else:
                        self.inputdict["-pix_fmt"] = "gray"
                elif C == 2:
                    self.inputdict["-pix_fmt"] = "ya8"
                elif C == 3:
                    self.inputdict["-pix_fmt"] = "rgb24"
                elif C == 4:
                    self.inputdict["-pix_fmt"] = "rgba"
                else:
                    raise NotImplemented

        self.bpp = bpplut[self.inputdict["-pix_fmt"]][1]
        self.inputNumChannels = bpplut[self.inputdict["-pix_fmt"]][0]
        bitpercomponent = self.bpp // self.inputNumChannels
        if bitpercomponent == 8:
            self.dtype = np.dtype('u1')  # np.uint8
        elif bitpercomponent == 16:
            suffix = self.inputdict['-pix_fmt'][-2:]
            if suffix == 'le':
                self.dtype = np.dtype('<u2')
            elif suffix == 'be':
                self.dtype = np.dtype('>u2')
        else:
            raise ValueError(self.inputdict['-pix_fmt'] + 'is not a valid pix_fmt for numpy conversion')

        assert self.inputNumChannels == C, "Failed to pass the correct number of channels %d for the pixel format %s." % (
            self.inputNumChannels, self.inputdict["-pix_fmt"])

        if ("-s" in self.inputdict):
            widthheight = self.inputdict["-s"].split('x')
            self.inputwidth = np.int(widthheight[0])
            self.inputheight = np.int(widthheight[1])
        else:
            self.inputdict["-s"] = str(N) + "x" + str(M)
            self.inputwidth = N
            self.inputheight = M

        # prepare output parameters, if raw
        if self.extension == ".yuv":
            if "-pix_fmt" not in self.outputdict:
                self.outputdict["-pix_fmt"] = self.DEFAULT_OUTPUT_PIX_FMT
                if self.verbosity > 0:
                    warnings.warn("No output color space provided. Assuming {}.".format(self.DEFAULT_OUTPUT_PIX_FMT),
                                  UserWarning)

        self._createProcess(self.inputdict, self.outputdict, self.verbosity)

    def _createProcess(self, inputdict, outputdict, verbosity):
        pass

    def _prepareData(self, data):
        return data  # general case : do nothing

    def close(self):
        """Closes the video and terminates FFmpeg process

        """
        if self._proc is None:  # pragma: no cover
            return  # no process
        if self._proc.poll() is not None:
            return  # process already dead
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait()
        self.DEVNULL.close()

    def writeFrame(self, im):
        """Sends ndarray frames to FFmpeg

        """
        vid = vshape(im)
        T, M, N, C = vid.shape
        if not self.warmStarted:
            self._warmStart(M, N, C, im.dtype)

        vid = vid.clip(0, (1 << (self.dtype.itemsize << 3)) - 1).astype(self.dtype)
        vid = self._prepareData(vid)
        T, M, N, C = vid.shape  # in case of hack ine prepareData to change the image shape (gray2RGB in libAV for exemple)

        # check if we need to do some bit-plane swapping
        # for the raw data format
        if self.inputdict["-pix_fmt"].startswith('yuv444p') or self.inputdict["-pix_fmt"].startswith('yuvj444p') or \
                self.inputdict["-pix_fmt"].startswith('yuva444p'):
            vid = vid.transpose((0, 3, 1, 2))

        # Check size of image
        if M != self.inputheight or N != self.inputwidth:
            raise ValueError('All images in a movie should have same size')
        if C != self.inputNumChannels:
            raise ValueError('All images in a movie should have same '
                             'number of channels')

        assert self._proc is not None  # Check status

        # Write
        try:
            self._proc.stdin.write(vid.tostring())
        except IOError as e:
            # Show the command and stderr from pipe
            msg = '{0:}\n\nFFMPEG COMMAND:\n{1:}\n\nFFMPEG STDERR ' \
                  'OUTPUT:\n'.format(e, self._cmd)
            raise IOError(msg)

    def _getSupportedEncoders(self):
        return NotImplemented

    def _dict2Args(self, dict):
        args = []
        for key in dict.keys():
            args.append(key)
            if len(dict[key]):
                args.append(dict[key])
        return args

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
