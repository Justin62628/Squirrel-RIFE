using System;
using System.Diagnostics;
using System.IO;

namespace RIFE_APP
{
    class Media
    {
        private const string V = @"pkgs\avtools\ffmpeg\ffprobe.exe";
        private const string V1 = @"pkgs\avtools\ffmpeg\ffmpeg.exe";
        public string ffmpegPath = AppDomain.CurrentDomain.SetupInformation.ApplicationBase + V1;
        public string ffprobePath = AppDomain.CurrentDomain.SetupInformation.ApplicationBase + V;
        public ProjectFFMPEG.FFMPEG ff;

        //提取帧
        public void Frame_Exc(string input, string outputdir,bool fast_exc, bool dup = false, string scale = "", string framerate = "", string format = "png", bool hwaccel = false, DataReceivedEventHandler d = null, EventHandler e = null)
        {
            string cmd = "";
            if (hwaccel)
            {
                cmd += "-hwaccel auto ";
            }
            if (fast_exc)
            {
                cmd += $"-y -i \"{input}\" ";
                if (dup)
                { 
                    cmd += "-vf mpdecimate ";
                }
            }
            else
            {
                cmd += $"-y -i \"{input}\" -vf ";
                if (dup)
                {
                    cmd += "mpdecimate,";
                }
                cmd += "zscale=matrix=709:matrixin=709:chromal=input:cin=input,format=yuv444p10le,format=rgb48be,format=rgb24 ";
            }
            if (dup)
            {
                cmd += "-vsync 0 ";
            }
            if (scale != "")
            {
                cmd += "-s " + scale + " ";
            }
            if (framerate != "")
            {
                cmd += $"-r {framerate} ";
            }
            if (outputdir.Substring(outputdir.Length) == @"\")
            {
                outputdir.Substring(0, outputdir.Length);
            }
            cmd += $"-f image2  \"{outputdir}\\%09d.{format}\"";
            ff = new ProjectFFMPEG.FFMPEG();
            ff.SetPath(ffmpegPath);
            ff.Run(cmd, dataReceived: d,Exited:e);
        }

        //合成视频
        public void CreateVideo(string input, string output, string FrameRate1,string FrameRate2, string vcodec = "libx264", string pixfmt = "yuv420p", int crf_video = 16, string preset_video = "", string AudioFile = "", bool recode = true, string acodec = "aac", string abitrate = "320", DataReceivedEventHandler d = null, EventHandler e = null)
        {
            string cmd = $"-r {FrameRate1} -y -i \"{input}\" ";
            if (AudioFile != "")
            {
                if (File.Exists(AudioFile))
                {
                    cmd += $"-i \"{AudioFile}\"";
                    if (!recode)
                    {
                        cmd += " -c:a copy ";
                    }
                    else
                    {
                        cmd += $" -c:a {acodec} -b:a {abitrate}k ";
                    }
                }
            }
            cmd += $"-c:v {vcodec} -pix_fmt {pixfmt} -crf {crf_video} -r {FrameRate2} ";
            if (preset_video != "")
            {
                cmd += $"-preset {preset_video} ";
            }
            cmd += $"\"{output}\"";
            ff = new ProjectFFMPEG.FFMPEG();
            ff.SetPath(ffmpegPath);
            ff.Run(cmd, dataReceived: d, Exited: e);
        }

        //提取音频
        public void Audio_Exc(string input, string acodec,string output, DataReceivedEventHandler d = null,EventHandler e = null)
        {
            ff = new ProjectFFMPEG.FFMPEG();
            ff.SetPath(ffmpegPath);
            ff.Run($"-y -i \"{input}\" -vn -c:a {acodec} \"{output}\"", dataReceived: d, Exited: e);
        }

        public void Kill()
        {
            if (ff != null)
            {
                ff.Kill();
            }
        }

    }
}
