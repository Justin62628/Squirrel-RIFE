using System;
using System.Diagnostics;
using System.IO;

namespace RIFE_APP
{
    class Media
    {
        public Process P;
        public string ffmpegPath = @"pkgs\avtools\ffmpeg\ffmpeg.exe";
        public string ffprobePath = @"pkgs\avtools\ffmpeg\ffprobe.exe";

        //读取视频原始帧率
        public string ReadFPS(string videoPath)
        {
            P = new Process();
            P.StartInfo.FileName = ffprobePath;
            P.StartInfo.Arguments = $"-v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate \"{videoPath}\"";
            P.StartInfo.RedirectStandardInput = true;
            P.StartInfo.RedirectStandardOutput = true;
            P.StartInfo.RedirectStandardError = true;
            P.StartInfo.UseShellExecute = false;
            P.StartInfo.CreateNoWindow = true;
            P.EnableRaisingEvents = true;
            P.Start();
            return P.StandardOutput.ReadToEnd();
        }

        //读取视频总像素数
        public double GetPixelCount(string videoPath)
        {
            try
            {
                double width = double.Parse(ReadWidth(videoPath));
                double height = double.Parse(ReadHeight(videoPath));
                return width * height;
            }
            catch { return 0; }
        }

        //读取视频宽度
        private string ReadWidth(string videoPath)
        {
            var P = new Process();
            P.StartInfo.FileName = ffprobePath;
            P.StartInfo.Arguments = $"-v error -select_streams v:0 -of default=noprint_wrappers=1:nokey=1 -show_entries stream=coded_width \"{videoPath}\"";
            P.StartInfo.RedirectStandardInput = true;
            P.StartInfo.RedirectStandardOutput = true;
            P.StartInfo.RedirectStandardError = true;
            P.StartInfo.UseShellExecute = false;
            P.StartInfo.CreateNoWindow = true;
            P.EnableRaisingEvents = true;
            P.Start();
            return P.StandardOutput.ReadToEnd().Trim().ToLower();
        }

        //读取视频宽度
        private string ReadHeight(string videoPath)
        {
            var P = new Process();
            P.StartInfo.FileName = ffprobePath;
            P.StartInfo.Arguments = $"-v error -select_streams v:0 -of default=noprint_wrappers=1:nokey=1 -show_entries stream=coded_height \"{videoPath}\"";
            P.StartInfo.RedirectStandardInput = true;
            P.StartInfo.RedirectStandardOutput = true;
            P.StartInfo.RedirectStandardError = true;
            P.StartInfo.UseShellExecute = false;
            P.StartInfo.CreateNoWindow = true;
            P.EnableRaisingEvents = true;
            P.Start();
            return P.StandardOutput.ReadToEnd().Trim().ToLower();
        }

    }
}
