using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace RIFE_APP
{
    class TaskClass
    {

        private Media m { get; set; } 

        private RIFE r { get; set; }
                                                               
        public string prefix { private get; set; } = "1/1"; //前缀

        public enum Input_type
        {
            video,
            imgs,
            resume
        }

        public Input_type input_Type { private set; get; } = Input_type.video; //输入文件类型

        public string Input_Path { private set; get; } = ""; //输入路径

        public string Output_Path { private set; get; } = ""; //输出路径

        public bool Output_Audio { private set; get; } = false; //是否输出音频

        public double InterpolatedFps { private set; get; } = 60.00; //补帧后帧率

        public enum InterpolationMode
        {
            normal,
            dup_ffmepg,
            dup_absdiff,
            pro
        }

        public InterpolationMode mode { private set; get; } = InterpolationMode.normal; //补帧模式

        public int gpu_id { private set; get; } = 0; //gpu_id(下一版本将改为string)(支持多GPU)

        public enum Times
        {
            two,
            four,
            Eight,
            Sixteen
        }

        public Times times { private set; get; } = Times.two; //X倍帧率

        public bool Fast_Exc { private set; get; } = false; //是否进行快去拆帧

        public int CRFValue { private set; get; } = 16; //CRF数值

        public double Output_fps { private set; get; } = 60.00; //导出帧率

        public bool OpenUHD { private set; get; } = true; //开去UHD模式

        public int vector { private set; get; } = 1; //先输入I0后输入I1，vector=1

        public enum Type_output
        {
            mp4,
            gif,
            pngs
        }

        public Type_output type_out { private set; get; } = Type_output.mp4; //导出类型

        public double Value_dup { private set; get; } = 1; //识别重复帧阈值

        public double Value_scene { private set; get; } = 50;//识别镜头切换阈值

        public int Value_static_frame { private set; get; } = 30; //当多余X帧时识别为刻意静止

        public int Threads_detect { private set; get; } = 16; //识别重复帧与镜头切换程序使用的线程

        public int batchsize { private set; get; } = 50;

        public bool Output_only_final { private set; get; } = true; //只保留最终导出文件


        /// <summary>
        /// 新建任务
        /// </summary>
        /// <param name="pre">前缀</param>
        /// <param name="it">输入类型</param>
        /// <param name="in_path">输入路径</param>
        /// <param name="out_path">输出路径</param>
        /// <param name="im">补帧模式</param>
        /// <param name="gid">gpu_id</param>
        /// <param name="t">补帧倍数</param>
        /// <param name="IntpoFps">补帧后帧率</param>
        /// <param name="uhd">开始UHD模式</param>
        /// <param name="vec">方向(取一为正)取其他为负</param>
        /// <param name="to">导出类型</param>
        /// <param name="dup">重复帧阈值</param>
        /// <param name="scene">镜头切换阈值</param>
        /// <param name="stf">重复帧大于该值时识别为刻意静止</param>
        /// <param name="threads_det">识别线程</param>
        /// <param name="out_final">只保留最终导出文件</param>
        /// <param name="bs">batchsize</param>
        public TaskClass(string pre, Input_type it, string in_path, string out_path, InterpolationMode im, int gid, Times t, double interpofps,bool fast_exc,bool uhd, int vec, Type_output to, double dup, double scene, int stf, int threads_det,bool oa,int crf,double output_fps,bool out_final,int bs)
        {
            //为以上变量赋值
            prefix = pre;
            input_Type = it;
            Input_Path = in_path;
            Output_Path = out_path;
            mode = im;
            gpu_id = gid;
            times = t;
            InterpolatedFps = interpofps;
            Fast_Exc = fast_exc;
            OpenUHD = uhd;
            vector = vec;
            type_out = to;
            Value_dup = dup;
            Value_scene = scene;
            Value_static_frame = stf;
            Threads_detect = threads_det;
            Output_Audio = oa;
            CRFValue = crf;
            Output_fps = output_fps;
            Output_only_final = out_final;
            batchsize = bs;

            Origin_frames = $"{out_path}\\origin_frames";
            Interpolated_Frames = Output_Path + "\\interpolated_frames";
            Temp_Dir = Output_Path + "\\temp";
            Audio_File = Output_Path + "\\audio.flac";
            Video_File = Output_Path + "\\output.mp4";
            Temp_Dir1 = Output_Path + "\\temp1";
        }

        private string Origin_frames { set; get; } = ""; //原始帧存放的位置

        public bool finish_frames_exc { private set; get; } = false; //是否完成了视频帧提取

        private string Audio_File { set; get; } = ""; //音频文件路径

        private string Video_File { set; get; } = ""; //视频文件路径

        private bool finish_audio_exc { set; get; } = false; //是否完成了音频提取

        private bool end_audio_exc { set; get; } = false; //音频提取程序是否结束

        public void Exc_Audio()
        {
            Media m1 = new Media();
            //提取音频，flac格式
            m1.Audio_Exc(Input_Path, "flac", Audio_File, (le, se) => {
                if (se.Data != null)
                {
                    WriteLine(se.Data, msgtype.info);
                }
            }, (le, se) => {
                if (File.Exists(Audio_File))
                {
                    FileInfo fi = new FileInfo(Audio_File);
                    if (fi.Length > 1024)
                    {
                        finish_audio_exc = true;
                        WriteLine(prefix + "音频提取完成", msgtype.pass);
                    }
                }
                end_audio_exc = true;
            });
        }

        public void Exc_Frames()
        {
            if (!Directory.Exists(Origin_frames))
            {
                Directory.CreateDirectory(Origin_frames);
            }
            m = new Media();
            if (input_Type == Input_type.video)
            {
                if (mode == InterpolationMode.dup_ffmepg)
                {
                    m.Frame_Exc(Input_Path,Origin_frames,Fast_Exc,true,"","","png",false, (d, e) => {
                        if (e.Data != null)
                        {
                            //prograss = m.ff.CompiledTime.TotalSeconds / m.ff.Duration.TotalSeconds * 100;
                            WriteLine(prefix + e.Data, msgtype.info);
                        }
                    }, (w, e) => {
                        finish_frames_exc = true;
                        WriteLine(prefix + "视频帧拆解完成", msgtype.pass);
                    });
                }
                else
                {
                    m.Frame_Exc(Input_Path, Origin_frames, Fast_Exc,false, "", "", "png", false, (d, e) => {
                        if (e.Data != null)
                        {
                            //prograss = m.ff.CompiledTime.TotalSeconds / m.ff.Duration.TotalSeconds * 100;
                            WriteLine(prefix + e.Data, msgtype.info);
                        }
                    }, (w, e) => { finish_frames_exc = true; WriteLine(prefix + "视频帧拆解完成", msgtype.pass); });
                }
            }
            if (input_Type == Input_type.imgs)
            {
                List<string> copy = new List<string>();
                try
                {
                    foreach (string s in Directory.EnumerateFiles(Input_Path, "*.png", SearchOption.TopDirectoryOnly))
                    {
                        copy.Add(s);
                    }
                }
                catch { WriteLine(prefix + "输入的目录不存在或无法访问", msgtype.error); }
                if (copy.Count == 0)
                {
                    WriteLine(prefix + "输入的目录内无PNG文件", msgtype.error);
                }
                else
                {
                    WriteLine(prefix + "正在复制PNG文件", msgtype.warning);
                    copy.Sort();
                    try
                    {
                        string str = "000000000";
                        for (int f = 0; f != copy.Count; f++)
                        {
                            File.Copy(copy[f], $"{Origin_frames}\\{str.Substring(0, str.Length - f.ToString().Length)}\\{f}.png");
                            WriteLine(prefix + copy[f], msgtype.info);
                        }
                        finish_frames_exc = true;
                        WriteLine(prefix + "视频帧复制完成", msgtype.pass);
                    }
                    catch (IOException e)
                    {
                        WriteLine(prefix + "复制PNG文件时失败", msgtype.error);
                        WriteLine(prefix + e.Message, msgtype.error);
                    }
                }
            }
        }

        public bool finish_dup_scene { private set; get; } = false; //是否完成重复帧与镜头切换识别

        public void Dup_Scene_Detect()
        {
            WriteLine(prefix + "识别重复帧和镜头切换,创建文件列表", msgtype.warning);
            List<string> l = new List<string>();
            foreach (string s in Directory.EnumerateFiles(Origin_frames, "*.png", SearchOption.AllDirectories))
            {
                l.Add(s);
            }
            l.Sort();
            double total = l.Count; //计算文件总数(delgen需要传入)
            Process p = new Process();
            p.StartInfo.FileName = AppDomain.CurrentDomain.SetupInformation.ApplicationBase + @"pkgs\avtools\delgen.exe";
            p.StartInfo.RedirectStandardInput = true;
            p.StartInfo.RedirectStandardOutput = true;
            p.StartInfo.RedirectStandardError = true;
            p.StartInfo.UseShellExecute = false;
            p.OutputDataReceived += (o, sd) => {
                if (sd.Data != null)
                {
                    WriteLine(sd.Data, msgtype.info);
                }
            };
            p.ErrorDataReceived += (o, sd) => {
                if (sd.Data != null)
                {
                    WriteLine(sd.Data, msgtype.info);
                }
            };
            p.StartInfo.CreateNoWindow = true;
            p.Start();
            //delgen需要传入5个值
            p.StandardInput.WriteLine(Origin_frames); //原始帧存放路径
            p.StandardInput.WriteLine(Value_dup); //重复帧阈值
            p.StandardInput.WriteLine(Value_scene); //场景切换阈值
            p.StandardInput.WriteLine(total); //总文件数
            p.StandardInput.WriteLine(Threads_detect); //线程数
            p.BeginOutputReadLine();
            p.BeginErrorReadLine();
            while (!p.HasExited)
            {
                Task.Delay(1).Wait();
            }
            finish_dup_scene = true;
        }

        private string Temp_Dir { set; get; } = ""; //临时目录

        private string Temp_Dir1 { set; get; } = ""; //临时目录

        private string Json_File { set; get; } = ""; //Json文件路径

        private string Need_File { set; get; } = ""; //记载需要帧数的路径

        private string Interpolated_Frames { set; get; } = ""; //已导出后图片序列的路径

        public bool finish_interpolate { private set; get; } = false; //是否完成了补帧

        public void Interpolate()
        {
            if (!Directory.Exists(Interpolated_Frames) && !Output_only_final)
            {
                Directory.CreateDirectory(Interpolated_Frames);
            }
            r = new RIFE();
            if (mode == InterpolationMode.pro)
            {
                Json_File = Output_Path + "\\tasks.json";
                Need_File = Output_Path + "\\need.txt";

                WriteLine(prefix + "创建补帧队列", msgtype.warning);
                List<string> Files = new List<string>();
                foreach (string s in Directory.EnumerateFiles(Origin_frames, "*", SearchOption.TopDirectoryOnly))
                {
                    Files.Add(s);
                }
                Files.Sort();
                List<int[]> IndexList = new List<int[]>();
                for (var i = 0; i < Files.Count - 1; i++)
                {
                    var s0 = int.Parse(Files[i].Substring(Files[i].LastIndexOf(@"\") + 1, 9));
                    var s1 = int.Parse(Files[i + 1].Substring(Files[i].LastIndexOf(@"\") + 1, 9));
                    var exp = 1;
                    string bs = "000000000";
                    if ((s1 - s0) > 1)
                    {
                        if (s1 - s0 - 1 == 1)
                        {
                            exp = 1;
                        }
                        else
                        {
                            //计算exp
                            do
                            {
                                exp += 1;
                                //MessageBox.Show($"num:{Math.Pow(2, exp) - 2} exp:{exp} need:{need}");
                            } while (Math.Pow(2, exp) - 2 < (s1 - s0 - 1)); 
                        }
                        if (s1 - s0 - 1 > Value_static_frame)
                        {
                            //当重复帧多余Value_static_frame时识别为刻意静止，采用帧复制
                            WriteLine(prefix + $"skip {Files[i]}...", msgtype.warning);
                            for (int po = s0 + 1; po != s1; po++)
                            {
                                try { File.Copy(Files[i], $"{Files[i].Substring(0, Files[i].LastIndexOf(@"\"))}\\{bs.Substring(0, bs.Length - po.ToString().Length)}{po}.png"); } catch { }
                            }
                        }
                        else
                        {
                            IndexList.Add(new int[] { i, i + 1, exp, (s1 - s0 - 1) });
                        }
                    }
                }
                WriteLine(prefix + "创建JSON文件记录", msgtype.warning);
                //创建JSON记录文件
                string sn = "";
                JsonMaker j = new JsonMaker();
                List<JsonItem> l = new List<JsonItem>();
                int n = 0;
                //记录同时将图片对复制到临时目录
                foreach (int[] i in IndexList)
                {
                    n += 1;
                    Directory.CreateDirectory($"{Temp_Dir}\\{n}");
                    l.Add(new JsonItem($"{Temp_Dir}\\{n}".Replace("\\", "/"), i[2], i[3].ToString()));
                    string s0 = Files[i[0]];
                    string s1 = Files[i[1]];
                    File.Copy(s0, $"{Temp_Dir}\\{n}\\000000001.png", true);
                    File.Copy(s1, $"{Temp_Dir}\\{n}\\000000002.png", true);
                    WriteLine(prefix + $"{n}/{IndexList.Count}", msgtype.info);
                    sn += $"{Temp_Dir}\\{n}?{i[3]}\r\n";
                }
                j.Make(Json_File, l);
                File.WriteAllText(Need_File, sn);
                if (!Directory.Exists(Temp_Dir1))
                {
                    Directory.CreateDirectory(Temp_Dir1);
                }
                WriteLine(prefix + "第一轮补帧", msgtype.warning);
                r.JsonInterpolate(vector, Json_File, Temp_Dir1, gpu_id, OpenUHD, batchsize,(s, se) =>
                {
                    if (se.Data != null)
                    {
                        WriteLine(prefix + se.Data, msgtype.info);
                    }

                }, (c, xe) =>
                {
                    WriteLine(prefix + "移动文件", msgtype.warning);
                    Dictionary<string, double> npairs = new Dictionary<string, double>();
                    n = 0;
                    foreach (string s in File.ReadLines(Need_File))
                    {
                        string[] p = s.Split('?');
                        npairs.Add(p[0], double.Parse(p[1]));
                        n += 1;
                    }
                    int iw = 0;
                    List<string> files = new List<string>();
                    while (iw != n)
                    {
                        iw += 1;
                        foreach (string s in Directory.EnumerateFiles($"{Temp_Dir}\\{iw}", "*.png", SearchOption.TopDirectoryOnly))
                        {
                            files.Add(s);
                        }
                    }
                    double max = 0;
                    double start = 1;
                    foreach (string s in Directory.EnumerateFiles(Origin_frames, "*", SearchOption.AllDirectories))
                    {
                        double d = double.Parse(s.Substring(s.LastIndexOf("\\") + 1, 9));
                        if (d > max)
                        {
                            max = d;
                        }
                    }
                    int get = 0;
                    for (double i = start; i < max; i++)
                    {
                        string file = Origin_frames + "\\000000000".Substring(0, 10 - i.ToString().Length) + i.ToString() + ".png";
                        if (!File.Exists(file))
                        {
                            File.Move(files[get], file);
                            Console.WriteLine(prefix + $"{i}/{max} {files[get]}", msgtype.info);
                            get += 1;
                        }
                    }
                    WriteLine(prefix + "第二轮补帧", msgtype.warning);
                    var exp = 0;
                    if (times == Times.two) { exp = 1; }
                    if (times == Times.four) { exp = 2; }
                    if (times == Times.Eight) { exp = 3; }
                    if (times == Times.Sixteen) { exp = 4; }
                    if (Output_Audio && finish_audio_exc)
                    {
                        if (Output_only_final)
                        {
                            r.Interpolate(Output_only_final, vector, Origin_frames, Video_File, gpu_id, OpenUHD, exp, Value_scene, Audio_File, InterpolatedFps, Output_fps, CRFValue, batchsize, (sl, se) => {
                                if (se.Data != null)
                                {
                                    WriteLine(prefix + se.Data, msgtype.info);
                                }
                            }, (sl, se) => {
                                WriteLine(prefix + "补帧完成", msgtype.pass);
                                finish_interpolate = true;
                            });
                        }
                        else
                        {
                            r.Interpolate(Output_only_final, vector, Origin_frames, Interpolated_Frames, gpu_id, OpenUHD, exp, Value_scene, Audio_File, InterpolatedFps, Output_fps, CRFValue, batchsize, (sl, se) => {
                                if (se.Data != null)
                                {
                                    WriteLine(prefix + se.Data, msgtype.info);
                                }
                            }, (sl, se) => {
                                WriteLine(prefix + "补帧完成", msgtype.pass);
                                finish_interpolate = true;
                            });
                        }
                    }
                    else
                    {
                        if (Output_only_final)
                        {
                            r.Interpolate(Output_only_final, vector, Origin_frames, Video_File, gpu_id, OpenUHD, exp, Value_scene, "", InterpolatedFps, Output_fps, CRFValue, batchsize, (sl, se) => {
                                if (se.Data != null)
                                {
                                    WriteLine(prefix + se.Data, msgtype.info);
                                }
                            }, (sl, se) => {
                                WriteLine(prefix + "补帧完成", msgtype.pass);
                                finish_interpolate = true;
                            });
                        }
                        else
                        {
                            r.Interpolate(Output_only_final, vector, Origin_frames, Interpolated_Frames, gpu_id, OpenUHD, exp, Value_scene, "", InterpolatedFps, Output_fps, CRFValue, batchsize, (sl, se) => {
                                if (se.Data != null)
                                {
                                    WriteLine(prefix + se.Data, msgtype.info);
                                }
                            }, (sl, se) => {
                                WriteLine(prefix + "补帧完成", msgtype.pass);
                                finish_interpolate = true;
                            });
                        }
                    }
                });
            }
            else
            {
                var exp = 0;
                if (times == Times.two) { exp = 1; }
                if (times == Times.four) { exp = 2; }
                if (times == Times.Eight) { exp = 3; }
                if (times == Times.Sixteen) { exp = 4; }
                if (Output_Audio && finish_audio_exc)
                {
                    if (Output_only_final)
                    {
                        r.Interpolate(Output_only_final, vector, Origin_frames, Video_File, gpu_id, OpenUHD, exp, Value_scene, Audio_File, InterpolatedFps, Output_fps, CRFValue, batchsize, (sl, se) => {
                            if (se.Data != null)
                            {
                                WriteLine(prefix + se.Data, msgtype.info);
                            }
                        }, (sl, se) => {
                            WriteLine(prefix + "补帧完成", msgtype.pass);
                            finish_interpolate = true;
                        });
                    }
                    else
                    {
                        r.Interpolate(Output_only_final, vector, Origin_frames, Interpolated_Frames, gpu_id, OpenUHD, exp, Value_scene, Audio_File, InterpolatedFps, Output_fps, CRFValue, batchsize, (sl, se) => {
                            if (se.Data != null)
                            {
                                WriteLine(prefix + se.Data, msgtype.info);
                            }
                        }, (sl, se) => {
                            WriteLine(prefix + "补帧完成", msgtype.pass);
                            finish_interpolate = true;
                        });
                    }
                }
                else
                {
                    if (Output_only_final)
                    {
                        r.Interpolate(Output_only_final, vector, Origin_frames, Video_File, gpu_id, OpenUHD, exp, Value_scene, "", InterpolatedFps, Output_fps, CRFValue, batchsize, (sl, se) => {
                            if (se.Data != null)
                            {
                                WriteLine(prefix + se.Data, msgtype.info);
                            }
                        }, (sl, se) => {
                            WriteLine(prefix + "补帧完成", msgtype.pass);
                            finish_interpolate = true;
                        });
                    }
                    else
                    {
                        r.Interpolate(Output_only_final, vector, Origin_frames, Interpolated_Frames, gpu_id, OpenUHD, exp, Value_scene, "", InterpolatedFps, Output_fps, CRFValue, batchsize, (sl, se) => {
                            if (se.Data != null)
                            {
                                WriteLine(prefix + se.Data, msgtype.info);
                            }
                        }, (sl, se) => {
                            WriteLine(prefix + "补帧完成", msgtype.pass);
                            finish_interpolate = true;
                        });
                    }
                }
            }
        }

        private bool finish_cv { get; set; } = false; //是否完成了视频合成

        public void CreateVideo()
        {
            WriteLine(prefix + "正在创建最终导出文件", msgtype.warning);
            Media m = new Media();
            if (Output_Audio && finish_audio_exc)
            {
                m.CreateVideo($"{Interpolated_Frames}\\%09d.png",Video_File,InterpolatedFps.ToString(),Output_fps.ToString(),"h264","yuv420p",CRFValue,"",Audio_File,true,"aac","320",(sl, se) => {
                    if (se.Data != null)
                    {
                        WriteLine(prefix + se.Data, msgtype.info);
                        //prograss = m.ff.CompiledTime.TotalSeconds / m.ff.Duration.TotalSeconds * 100;
                    }
                }, (sl, se) => {
                    finish_cv = true;
                });
            }
            else
            {
                m.CreateVideo($"{Interpolated_Frames}\\%09d.png", Video_File, InterpolatedFps.ToString(), Output_fps.ToString(), "h264", "yuv420p", CRFValue, "", "", true, "aac", "320", (sl, se) => {
                    if (se.Data != null)
                    {
                        WriteLine(prefix + se.Data, msgtype.info);
                        //prograss = m.ff.CompiledTime.TotalSeconds / m.ff.Duration.TotalSeconds * 100;
                    }
                }, (sl, se) => {
                    finish_cv = true;
                });

            }

        }

        public void Start()
        {
            Exc_Audio();
            //等待音频提取程序结束
            while (!end_audio_exc)
            {
                Task.Delay(1).Wait();
            }
            Exc_Frames();
            //等待视频帧提取完成
            while (!finish_frames_exc)
            {
                Task.Delay(1).Wait();
            }
            //判断是否需要识别重复帧和场景切换
            if (mode == InterpolationMode.dup_absdiff || mode == InterpolationMode.pro)
            {
                Dup_Scene_Detect();
                //等待识别完成
                while (!finish_dup_scene)
                {
                    Task.Delay(1).Wait();
                }
            }
            Interpolate();
            //等待补帧完成
            while (!finish_interpolate)
            {
                Task.Delay(1).Wait();
            }
            if (type_out == Type_output.mp4)
            {
                if (!Output_only_final)
                {
                    CreateVideo();
                    //等待合成
                    while (!finish_cv)
                    {
                        Task.Delay(1).Wait();
                    }
                }
            }
            if (type_out == Type_output.gif)
            {
                throw new NotImplementedException();
            }
            if (Output_only_final)
            {
                if (type_out == Type_output.mp4)
                {
                    Delete(Origin_frames, true);
                    Delete(Interpolated_Frames, true);
                    Delete(Temp_Dir, true);
                    Delete(Temp_Dir1, true);
                    Delete($"{Output_Path}\\need.txt");
                    Delete($"{Output_Path}\\task.json");
                }
                if (type_out == Type_output.pngs)
                {
                    Delete(Origin_frames, true);
                    //Delete(Interpolated_Frames, true);
                    Delete(Temp_Dir, true);
                    Delete(Temp_Dir1, true);
                    Delete(Audio_File);
                    Delete($"{Output_Path}\\need.txt");
                    Delete($"{Output_Path}\\task.json");
                }
                if (type_out == Type_output.gif)
                {
                    throw new NotImplementedException();
                }
            }
            WriteLine(prefix + "已完成所有操作", msgtype.pass);
        }

        private void Delete(string f, bool dir = false)
        {
            if (dir)
            {
                try { Directory.Delete(f, true); } catch { }
            }
            else
            {
                try { File.Delete(f); } catch { }
            }
        }

        private object o = new object();

        public void Kill()
        {
            m.Kill();
            r.Kill();
        }

        enum msgtype
        {
            info,
            warning,
            pass,
            error
        }
        private void WriteLine(string msg, msgtype ms)
        {
            lock (o)
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

