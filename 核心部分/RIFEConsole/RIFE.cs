using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace RIFE_APP
{
    class RIFE
    {
        private readonly string python = AppDomain.CurrentDomain.SetupInformation.ApplicationBase + @"pkgs\py\python.exe";
        private readonly string ffmpeg = AppDomain.CurrentDomain.SetupInformation.ApplicationBase + @"pkgs\avtools\ffmpeg\ffmpeg.exe";
        private readonly string rife_imgs = AppDomain.CurrentDomain.SetupInformation.ApplicationBase + @"pkgs\rife\inference_img_seq_by_guile.py";
        private readonly string rife_json_low = AppDomain.CurrentDomain.SetupInformation.ApplicationBase + @"pkgs\rife\json_interpolation_low_memory_by_guile.py";
        public int count = 0;
        Process P;

        /// <summary>
        /// Json文件记录插帧
        /// </summary>
        /// <param name="vector">方向</param>
        /// <param name="json">JSON文件记录</param>
        /// <param name="tempdir">临时目录</param>
        /// <param name="gpu_id">GPU ID</param>
        /// <param name="uhd">UHD模式</param>
        /// <param name="batchsize">batchsize</param>
        /// <param name="d">数据接收事件</param>
        /// <param name="e">退出事件</param>
        public void JsonInterpolate(int vector, string json, string tempdir, int gpu_id = 0, bool uhd = true,int batchsize = 50, DataReceivedEventHandler d = null, EventHandler e = null)
        {
            if (uhd)
            {
                Run(rife_json_low, $"--gpu_id {gpu_id} --UHD --vector {vector} --json=\"{json}\" --batch_size {batchsize} --output=\"{tempdir}\"", d, e);
            }
            else
            {
                Run(rife_json_low, $"--gpu_id {gpu_id} --vector {vector} --json=\"{json}\" --batch_size {batchsize} --output=\"{tempdir}\"", d, e);
            }

        }

        /// <summary>
        /// inference_img_seq
        /// </summary>
        /// <param name="direct">直接导出为mp4</param>
        /// <param name="vector">先输入I0，再输入I1 vector=1</param>
        /// <param name="img">图片所在的文件路径</param>
        /// <param name="output">文件保存路径</param>
        /// <param name="gpu_id">gpu_id</param>
        /// <param name="uhd">UHD模式</param>
        /// <param name="exp">导出帧率应为 2的exp次方</param>
        /// <param name="scene">识别场景切换阈值</param>
        /// <param name="audio">音频文件</param>
        /// <param name="read_fps">读取帧率</param>
        /// <param name="out_fps">导出帧率</param>
        /// <param name="crf">CRF数值</param>
        /// <param name="batchsize">batchsize</param>
        /// <param name="d"></param>
        /// <param name="e"></param>
        public void Interpolate(bool direct,int vector, string img,string output, int gpu_id, bool uhd, int exp = 1, double scene = 50,string audio = "",double read_fps = 60.00,double out_fps = 60.00,int crf = 16, int batchsize = 50, DataReceivedEventHandler d = null, EventHandler e = null)
        {
            if (direct)
            {
                if (uhd)
                {
                    Run(rife_imgs, $"--direct --ffmpeg \"{ffmpeg}\" --gpu_id {gpu_id} --UHD --vector {vector} --output \"{output}\" --batch_size {batchsize} --read_fps {read_fps} --img \"{img}\" --exp {exp} --audio \"{audio}\" --scene {scene} --out_fps {out_fps} --crf {crf}", d, e);
                }
                else
                {
                    Run(rife_imgs, $"--direct --ffmpeg \"{ffmpeg}\" --gpu_id {gpu_id} --vector {vector} --output \"{output}\" --batch_size {batchsize} --read_fps {read_fps} --img \"{img}\" --exp {exp} --audio \"{audio}\" --scene {scene} --out_fps {out_fps} --crf {crf}", d, e);
                }
            }
            else
            {
                if (uhd)
                {
                    Run(rife_imgs, $"--gpu_id {gpu_id} --UHD --vector {vector} --img \"{img}\" --exp {exp} --output \"{output}\" --batch_size {batchsize} --scene {scene}", d, e);
                }
                else
                {
                    Run(rife_imgs, $"--gpu_id {gpu_id} --vector {vector} --img \"{img}\" --exp {exp} --output \"{output}\" --batch_size {batchsize} --scene {scene}", d, e);
                }
            }
        }

        private void Run(string file,string cmd, DataReceivedEventHandler d = null, EventHandler e = null)
        {
            P = new Process();
            P.StartInfo.FileName = python;
            P.StartInfo.Arguments = $"\"{file}\" {cmd}";
            P.StartInfo.RedirectStandardInput = true;
            P.StartInfo.RedirectStandardOutput = true;
            P.StartInfo.RedirectStandardError = true;
            P.StartInfo.UseShellExecute = false;
            P.OutputDataReceived += d;
            P.ErrorDataReceived += d;
            P.Exited += e;
            P.StartInfo.CreateNoWindow = true;
            P.EnableRaisingEvents = true;
            P.Start();
            P.BeginOutputReadLine();
            P.BeginErrorReadLine();
        }

        public void Kill()
        {
            if (P != null)
            {
                while (!P.HasExited)
                {
                    try { P.Kill(); } catch { }
                }
            }
        }
    }
}
