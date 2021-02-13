using RIFE_APP;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;

namespace RIFEConsole
{
    class Program
    {
        static readonly Process p = new Process();
        static TaskClass tc;
        public delegate bool ControlCtrlDelegate(int CtrlType);
        [DllImport("kernel32.dll")]
        private static extern bool SetConsoleCtrlHandler(ControlCtrlDelegate HandlerRoutine, bool Add);
        private static ControlCtrlDelegate cancelHandler = new ControlCtrlDelegate(HandlerRoutine);

        public static bool HandlerRoutine(int CtrlType)
        {
            switch (CtrlType)
            {
                case 0:
                    Kill(); //Ctrl+C关闭  
                    break;
                case 2:
                    Kill();//按控制台关闭按钮关闭  
                    break;
            }
            Console.ReadLine();
            return false;
        }

        static void Main(string[] args)
        {
            Console.Title = "Squirrel Video Frame Interpolation 1.5";
            WriteLine("RIFE算法作者:",msgtype.pass);
            WriteLine("Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, Shuchang Zhou", msgtype.warning);
            WriteLine("SVFI作者:", msgtype.pass);
            WriteLine("YiWeiHuang-stack,ABlyh-LEO,GuileCyclone,NULL204,hzwer", msgtype.warning);
            WriteLine("Github地址:",msgtype.pass);
            WriteLine("https://github.com/YiWeiHuang-stack/Squirrel-Video-Frame-Interpolation", msgtype.warning);
            SetConsoleCtrlHandler(cancelHandler, true);
            //开启内存回收(防止内存溢出)
            p.StartInfo.FileName = AppDomain.CurrentDomain.SetupInformation.ApplicationBase + "pkgs\\RAMcollector.exe";
            p.StartInfo.RedirectStandardInput = true;
            p.StartInfo.UseShellExecute = false;
            p.StartInfo.CreateNoWindow = true;
            p.Start();
            p.StandardInput.WriteLine("2048"); // 最大运行内存占用量(已弃用)
            p.StandardInput.WriteLine("60"); //回收频率(秒)
            while (true)
            {
                List<string> configs = new List<string>();
                while (true)
                {
                    //读取配置文件
                    Console.Write("config:"); 
                    string sf = Console.ReadLine();
                    if (sf.ToLower() == "y") { break; }
                    if (sf.IndexOf("\"") != -1)
                    {
                        sf = sf.Replace("\"", "");
                    }
                    if (Directory.Exists(sf))
                    {
                        foreach (string s in Directory.EnumerateFiles(sf, "*.config", SearchOption.TopDirectoryOnly))
                        {
                            configs.Add(s);
                        }
                    }
                    if (File.Exists(sf))
                    {
                        configs.Add(sf);
                    }
                }
                int n = 0;
                foreach (string cf in configs)
                {
                    n += 1;
                    string perfix = $"[{n}/{configs.Count}]";//消息前缀
                    Dictionary<string, string> dic = new Dictionary<string, string>();
                    //将配置文件的信息存入dic
                    foreach (string s in File.ReadLines(cf))
                    {
                        string[] s1 = s.Split(':');
                        string s2 = "";
                        int i = 0;
                        foreach (string s3 in s1)
                        {
                            i += 1;
                            if (i != 1)
                            {
                                s2 += $"{s3}:";
                            }
                        }
                        s2 = s2.Substring(0, s2.Length - 1);
                        dic.Add(s1[0], s2);
                    }
                    //检查dic键的数量
                    if (dic.Count != 20)
                    {
                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.WriteLine("损坏的config文件...");
                        Console.ResetColor();
                    }
                    else
                    {
                        //读取dic值
                        TaskClass.Input_type it = TaskClass.Input_type.video;
                        if (dic["Input_Type"] == "imgs")
                        {
                            it = TaskClass.Input_type.imgs;
                        }
                        if (dic["Input_Type"] == "resume")
                        {
                            it = TaskClass.Input_type.resume;
                        }
                        string in_path = dic["Input_Path"];
                        string out_path = dic["Output_Path"] + in_path.Substring(in_path.LastIndexOf(@"\") + 1);
                        if (in_path.IndexOf(".") != -1)
                        {
                            string s1 = in_path.Substring(0, in_path.LastIndexOf("."));
                            out_path = dic["Output_Path"] + "\\" + s1.Substring(s1.LastIndexOf(@"\") + 1);
                        }
                        if (Directory.Exists(out_path))
                        {
                            int i = 1;
                            string ex_path = $"{out_path} (1)";
                            while (Directory.Exists(ex_path))
                            {
                                i += 1;
                                ex_path = $"{out_path} ({i})";
                            }
                            Directory.CreateDirectory($"{ex_path}");
                            out_path = ex_path;
                        }
                        else
                        {
                            Directory.CreateDirectory(out_path);
                        }
                        TaskClass.InterpolationMode im = TaskClass.InterpolationMode.normal;
                        if (dic["mode"] == "dup_absdiff")
                        {
                            im = TaskClass.InterpolationMode.dup_absdiff;
                        }
                        if (dic["mode"] == "dup_ffmpeg")
                        {
                            im = TaskClass.InterpolationMode.dup_ffmepg;
                        }
                        if (dic["mode"] == "pro")
                        {
                            im = TaskClass.InterpolationMode.pro;
                        }
                        int gid = int.Parse(dic["gpu_id"]);
                        TaskClass.Times t = TaskClass.Times.two;
                        if (dic["times"] == "4")
                        {
                            t = TaskClass.Times.four;
                        }
                        if (dic["times"] == "8")
                        {
                            t = TaskClass.Times.Eight;
                        }
                        if (dic["times"] == "16")
                        {
                            t = TaskClass.Times.Sixteen;
                        }
                        double interpolatedFps = double.Parse(dic["InterpolatedFps"]);
                        bool fast_exc = bool.Parse(dic["FastExc"]);
                        bool uhd = bool.Parse(dic["OpenUHD"]);
                        TaskClass.Type_output to = TaskClass.Type_output.mp4;
                        if (dic["type_out"] == "pngs")
                        {
                            to = TaskClass.Type_output.pngs;
                        }
                        if (dic["type_out"] == "gif")
                        {
                            to = TaskClass.Type_output.gif;
                        }
                        int vector = int.Parse(dic["Vector"]);
                        double dup = double.Parse(dic["Value_dup"]);
                        double scene = double.Parse(dic["Value_scene"]);
                        int stf = int.Parse(dic["Value_static_frame"]);
                        int threads = int.Parse(dic["Threads_detect"]);
                        bool oa = bool.Parse(dic["Output_Audio"]);
                        int crf = int.Parse(dic["Output_crf_value"]);
                        double out_fps = double.Parse(dic["Output_fps"]);
                        bool out_final = bool.Parse(dic["Output_only_final"]);
                        int batch_size = int.Parse(dic["batch_size"]);

                        //新建TaskClass类
                        tc = new TaskClass(perfix, it, in_path, out_path, im, gid, t,interpolatedFps,fast_exc,uhd, vector, to, dup, scene, stf, threads,oa,crf,out_fps,out_final,batch_size);
                        //tc.Exc_Frames();
                        //tc.Exc_Audio();
                        //tc.Dup_Scene_Detect();
                        //tc.Interpolate();
                        //tc.CreateVideo();
                        tc.Start(); //一键完成所有步骤
                    }
                    Console.WriteLine("进行下一个任务");
                }
                Console.WriteLine("已完成所有任务");
                Console.WriteLine("进入下一个循环");
            }
        }

        static void Kill()
        {
            try { p.Kill(); } catch { }
            try { tc.Kill(); } catch { }
        }

        enum msgtype
        {
            info,
            warning,
            pass,
            error
        }

        private static object sync = new object();
        private static void WriteLine(string msg, msgtype ms)
        {
            lock (sync)
            {
                if (ms == msgtype.warning)
                {
                    Console.ForegroundColor = ConsoleColor.Yellow;
                }
                if (ms == msgtype.error)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                }
                if (ms == msgtype.pass)
                {
                    Console.ForegroundColor = ConsoleColor.Green;
                }
                Console.WriteLine(msg);
                Console.ResetColor();
            }
        }

    }
}
