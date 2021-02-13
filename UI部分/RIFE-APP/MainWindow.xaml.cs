using Microsoft.Win32;
using Microsoft.WindowsAPICodePack.Dialogs;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Threading;
using RIFE_APP.Pages;

namespace RIFE_APP
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {

        public enum controls
        {
            rbtn_v,
            btn_input,
            btn_output,
            com_mode,
            com_devices,
            txt_input_fps,
            com_times,
            txt_interpolated_fps,
            chk_fast_exc,
            chk_uhd,
            chk_vector,
            com_type_out,
            txt_dup_value,
            txt_scene_value,
            txt_static_frames,
            txt_detect_threads,
            chk_add_audio,
            txt_crf_value,
            txt_output_fps,
            chk_only_final,
        }

        /// <summary>
        /// 获取控件信息
        /// </summary>
        /// <param name="c"></param>
        /// <returns></returns>
        public object GetGUInfo(controls c)
        {
            object o = new object();
            Dispatcher.Invoke(() =>
            {
                switch (c)
                {
                    case controls.rbtn_v:
                        o = rbtn_v.IsChecked;
                        break;
                    case controls.btn_input:
                        o = btn_input.Content;
                        break;
                    case controls.btn_output:
                        o = btn_output.Content;
                        break;
                    case controls.com_mode:
                        o = p1.com_mode.SelectedIndex;
                        ; break;
                    case controls.com_devices:
                        o = p1.com_devices.Text;
                        ; break;
                    case controls.txt_input_fps:
                        o = p1.txt_input_fps.Text;
                        ; break;
                    case controls.com_times:
                        o = p1.com_times.SelectedIndex;
                        ; break;
                    case controls.txt_interpolated_fps:
                        o = p1.txt_interpolated_fps.Text;
                        break;
                    case controls.chk_fast_exc:
                        o = p1.chk_fast_exc.IsChecked;
                        break;
                    case controls.chk_uhd:
                        o = p1.chk_uhd.IsChecked;
                        break;
                    case controls.chk_vector:
                        o = p1.chk_vector.IsChecked;
                        break;
                    case controls.com_type_out:
                        o = p1.com_out_type.SelectedIndex;
                        break;
                    case controls.txt_dup_value:
                        o = p1.txt_dup_value.Text;
                        break;
                    case controls.txt_scene_value:
                        o = p1.txt_scene_value.Text;
                        break;
                    case controls.txt_static_frames:
                        o = p1.txt_static_frames.Text;
                        break;
                    case controls.txt_detect_threads:
                        o = p1.txt_detect_threads.Text;
                        break;
                    case controls.chk_add_audio:
                        o = p1.chk_add_audio.IsChecked;
                        break;
                    case controls.txt_crf_value:
                        o = p1.txt_crf_value.Text;
                        break;
                    case controls.txt_output_fps:
                        o = p1.txt_output_fps.Text;
                        break;
                    case controls.chk_only_final:
                        o = p1.chk_only_final.IsChecked;
                        break;
                }
            });

            return o;
        }

        /// <summary>
        /// 设置控件信息
        /// </summary>
        /// <param name="c"></param>
        /// <param name="o"></param>
        public void SetGUInfo(controls c, object o)
        {
            Dispatcher.Invoke(() =>
            {
                try
                {
                    switch (c)
                    {
                        case controls.btn_input:
                            btn_input.Content = (string)o;
                            break;
                        case controls.btn_output:
                            btn_output.Content = (string)o;
                            break;
                        case controls.txt_input_fps:
                            p1.txt_input_fps.Text = (string)o;
                            break;
                        case controls.txt_interpolated_fps:
                            p1.txt_interpolated_fps.Text = (string)o;
                            break;
                        case controls.chk_uhd:
                            p1.chk_uhd.IsChecked = (bool)o;
                            break;
                    }
                }
                catch { }
            });
        }

        #region var
        RIFE rife;
        Media m = new Media();
        string video = "";
        double fps = 0;
        #endregion

        Page1 p1 = new Page1();
        Page3 p3 = new Page3();

        public MainWindow()
        {
            InitializeComponent();
            LoadDeviceList();//读取设备列表
            Init(); 
            frame.Content = p1;
        }

        private void Init()
        {
            p1.btn_save_svfi.Click += btn_save_svfi_Click;
            p1.com_mode.SelectionChanged += com_mode_SelectionChanged;
            p1.com_times.SelectionChanged += com_times_SelectionChanged;
        }

        /// <summary>
        /// 加载设备列表
        /// </summary>
        private void LoadDeviceList()
        {
            rife = new RIFE();
            int n = 0;
            List<string> gl = rife.GetAvailableGPU();
            bool cb = false;
            foreach (string s in gl)
            {
                if (s != null)
                {
                    if (s.IndexOf("Traceback") != -1)
                    {
                        cb = true;
                    }
                }

            }
            if (cb)
            {
                gl.Clear();
            }
            else
            {
                foreach (string s in gl)
                {
                    if (s != null)
                    {
                        p1.com_devices.Items.Add($"{n} {s}");
                        n += 1;
                    }
                }
            }

            /*foreach (string s in gl)
            {
                if (s != null)
                {
                    com_devices.Items.Add($"{n} {s}");
                    n += 1;
                }
            }*/
            List<string> cl = rife.GetAvailableCPU();
            //设置设备选择列表的默认项
            p1.com_devices.Items.Add($"-1 {cl[0]}");
            if (gl.Count == 0)
            {
                p1.com_devices.Text = $"-1 {cl[0]}";
            }
            else
            {
                p1.com_devices.Text = $"0 {gl[0]}";
            }
        }


        double pixel = 0; //H*W,用于判断是否开启UHD
        private void btn_input_Click(object sender, RoutedEventArgs e)
        {
            if ((bool)GetGUInfo(controls.rbtn_v) == true)
            {
                OpenFileDialog fileDialog = new OpenFileDialog
                {
                    Multiselect = false,//禁止多选
                    Title = "请选择要补帧的文件",
                    Filter = "所有文件|*.*"
                };
                fileDialog.ShowDialog();
                if (IsChinese(fileDialog.FileName))
                {
                    MessageBox.Show("请保证文件名和文件路径为纯英文", "信息");
                }
                else
                {
                    if (File.Exists(fileDialog.FileName))
                    {
                        btn_output.IsEnabled = true;
                        video = fileDialog.FileName;
                        btn_input.Content = fileDialog.FileName;
                        try
                        {
                            //获取视频原始帧率
                            m = new Media();
                            string[] sf = m.ReadFPS(video).Split('/');
                            fps = double.Parse(sf[0]) / double.Parse(sf[1]);
                            SetGUInfo(controls.txt_input_fps, fps.ToString()); 
                            int t = (int)GetGUInfo(controls.com_times);
                            if (t == 0)
                            {
                                SetGUInfo(controls.txt_interpolated_fps, (fps * 2).ToString());
                            }
                            if (t == 1)
                            {
                                SetGUInfo(controls.txt_interpolated_fps, (fps * 4).ToString());
                            }
                            if (t == 2)
                            {
                                SetGUInfo(controls.txt_interpolated_fps, (fps * 8).ToString());
                            }
                            if (t == 3)
                            {
                                SetGUInfo(controls.txt_interpolated_fps, (fps * 16).ToString());
                            }
                            pixel = m.GetPixelCount(video);
                            if (pixel >= 2073600)
                            {
                                SetGUInfo(controls.chk_uhd, true);
                            }
                            else
                            {
                                SetGUInfo(controls.chk_uhd, false);
                            }

                        }
                        catch { }
                    }
                }
            }
            else
            {
                var c = new CommonOpenFileDialog
                {
                    AllowNonFileSystemItems = true,
                    IsFolderPicker = true,
                    DefaultDirectory = video.Substring(video.LastIndexOf(@"\") - 1),
                    Title = "选择存放图片序列的文件夹"
                };
                if (c.ShowDialog() == CommonFileDialogResult.Ok)
                {
                    if (IsChinese(c.FileName))
                    {
                        MessageBox.Show("请保证文件名路径为纯英文", "信息");
                    }
                    else
                    {
                        if (Directory.Exists(c.FileName))
                        {
                            btn_output.IsEnabled = true;
                            SetGUInfo(controls.btn_input,c.FileName);
                            SetGUInfo(controls.txt_input_fps,25);
                        }
                    }
                }
            }
            
        }

        private void btn_output_Click(object sender, RoutedEventArgs e)
        {
            var c = new CommonOpenFileDialog
            {
                AllowNonFileSystemItems = true,
                IsFolderPicker = true,
                DefaultDirectory = video.Substring(video.LastIndexOf(@"\") - 1),
                Title = "选择保存文件夹"
            };
            if (c.ShowDialog() == CommonFileDialogResult.Ok)
            {
                if (IsChinese(c.FileName))
                {
                    MessageBox.Show("请保证文件名和文件路径为纯英文", "消息");
                }
                else
                {
                    if (Directory.Exists(c.FileName))
                    {
                        btn_output.Content = c.FileName;
                    }
                }
            }
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            if (MessageBox.Show("确定要退出么?", "信息", MessageBoxButton.YesNo) == MessageBoxResult.Yes)
            {

            }
            else { e.Cancel = true; }
        }

        /// <summary>
        /// 判断路径中是否含有中文字符
        /// </summary>
        /// <param name="CString"></param>
        /// <returns></returns>
        private bool IsChinese(string CString)
        {
            Regex RegCHZN = new Regex("[\u4e00-\u9fa5]");
            Match m = RegCHZN.Match(CString);
            return m.Success;
        }

        private void btn_save_svfi_Click(object sender, RoutedEventArgs e)
        {
            if (checkGUI())
            {
                //从GUI中读取信息并生成配置文件
                string con = "";
                if ((bool)GetGUInfo(controls.rbtn_v) == true)
                {
                    con += $"Input_Type:video\r\n";
                }
                else
                {
                    con += $"Input_Type:pngs\r\n";
                }
                con += "Input_Path:" + (string)GetGUInfo(controls.btn_input) + "\r\n";
                con += "Output_Path:" + (string)GetGUInfo(controls.btn_output) + "\r\n";
                int m = (int)GetGUInfo(controls.com_mode);
                if (m == 0)
                {
                    con += "mode:normal\r\n";
                }
                if (m == 1)
                {
                    con += "mode:dup_ffmpeg\r\n";
                }
                if (m == 2)
                {
                    con += "mode:dup_fuzz\r\n";
                }
                if (m == 3)
                {
                    con += "mode:pro\r\n";
                }
                string[] ss = ((string)GetGUInfo(controls.com_devices)).Split(' ');
                con += $"gpu_id:{ss[0]}\r\n";
                int t = (int)GetGUInfo(controls.com_times);
                if (t == 0)
                {
                    con += "times:2\r\n";
                }
                if (t == 1)
                {
                    con += "times:4\r\n";
                }
                if (t == 2)
                {
                    con += "times:8\r\n";
                }
                if (t == 3)
                {
                    con += "times:16\r\n";
                }
                con += "InterpolatedFps:" + (string)GetGUInfo(controls.txt_interpolated_fps) + "\r\n";
                con += "FastExc:" + (bool)GetGUInfo(controls.chk_fast_exc) + "\r\n";
                con += "OpenUHD:" + (bool)GetGUInfo(controls.chk_uhd) + "\r\n";
                int ty = (int)GetGUInfo(controls.com_type_out);
                if (ty == 0)
                {
                    con += "type_out:mp4\r\n";
                }
                if (ty == 1)
                {
                    con += "type_out:pngs\r\n";
                }
                if ((bool)GetGUInfo(controls.chk_vector))
                {
                    con += "Vector:-1\r\n";
                }
                else { con += "Vector:1\r\n"; }
                con += $"Value_dup:{(string)GetGUInfo(controls.txt_dup_value)}\r\n";
                con += $"Value_scene:{(string)GetGUInfo(controls.txt_scene_value)}\r\n";
                con += $"Value_static_frame:{(string)GetGUInfo(controls.txt_static_frames)}\r\n";
                con += $"Threads_detect:{(string)GetGUInfo(controls.txt_detect_threads)}\r\n";
                con += "Output_Audio:" + (bool)GetGUInfo(controls.chk_add_audio) + "\r\n";
                con += $"Output_crf_value:{(string)GetGUInfo(controls.txt_crf_value)}\r\n";
                con += $"Output_fps:{(string)GetGUInfo(controls.txt_output_fps)}\r\n";
                con += "Output_only_final:" + (bool)GetGUInfo(controls.chk_only_final) + "\r\n";
                int batch_size = 0;
                try {
                    Dispatcher.Invoke(()=> {
                        // 6 * batchsize * width * height * 3 / 1024 / 1024 = ???MB
                        double ram = double.Parse(p3.txt_max_ram_size.Text);
                        double x = ram / 6 / 3 / pixel * 1024 * 1024;
                        if (x < 1)
                        {
                            x = 1;
                        }
                        batch_size = (int)x;
                    });
                } catch { batch_size = 1; }
                con += "batch_size:" + batch_size + "\r\n";
                SaveFileDialog sf = new SaveFileDialog()
                {
                    Title = "请选择保存配置文件的路径",
                    Filter = "config文件|*.config"
                };
                if (sf.ShowDialog() == true)
                {
                    File.WriteAllText(sf.FileName, con);
                    MessageBox.Show("配置文件导出成功", "信息");
                }
            }
            else
            {

            }

        }

        /// <summary>
        /// 检查用户是否将信息填写完整
        /// </summary>
        /// <returns></returns>
        bool checkGUI()
        {
            if ((string)GetGUInfo(controls.btn_output) == "选择保存导出内容的文件夹")
            {
                MessageBox.Show("请先选择保存导出内容的文件夹","");
                return false;
            }
            return true;
        }

        private void com_times_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            int t = (int)GetGUInfo(controls.com_times);
            if (t == 0)
            {
                SetGUInfo(controls.txt_interpolated_fps, (fps * 2).ToString());
            }
            if (t == 1)
            {
                SetGUInfo(controls.txt_interpolated_fps, (fps * 4).ToString());
            }
            if (t == 2)
            {
                SetGUInfo(controls.txt_interpolated_fps, (fps * 8).ToString());
            }
            if (t == 3)
            {
                SetGUInfo(controls.txt_interpolated_fps, (fps * 16).ToString());
            }
        }

        private void com_mode_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            int si = (int)GetGUInfo(controls.com_mode);
            if (si == 0)
            {
                p1.txt_dup_value.IsEnabled = false;
                p1.txt_scene_value.IsEnabled = true;
                p1.txt_static_frames.IsEnabled = false;
                p1.txt_detect_threads.IsEnabled = false;
            }
            if (si == 1)
            {
                p1.txt_dup_value.IsEnabled = false;
                p1.txt_scene_value.IsEnabled = true;
                p1.txt_static_frames.IsEnabled = false;
                p1.txt_detect_threads.IsEnabled = false;
            }
            if (si == 2)
            {
                p1.txt_dup_value.IsEnabled = true;
                p1.txt_scene_value.IsEnabled = true;
                p1.txt_static_frames.IsEnabled = false;
                p1.txt_detect_threads.IsEnabled = true;
            }
            if (si == 3)
            {
                p1.txt_dup_value.IsEnabled = true;
                p1.txt_scene_value.IsEnabled = true;
                p1.txt_static_frames.IsEnabled = true;
                p1.txt_detect_threads.IsEnabled = true;
            }
        }

        private void btn_about_Click(object sender, RoutedEventArgs e)
        {
            //显示作者窗口
            Aut a = new Aut();
            a.Show();
        }

        private void btn_normal_setting_Click(object sender, RoutedEventArgs e)
        {
            btn_normal_setting.IsEnabled = false;
            btn_permance_setting.IsEnabled = true;
            frame.Content = p1;
        }

        private void btn_permance_setting_Click(object sender, RoutedEventArgs e)
        {
            btn_normal_setting.IsEnabled = true;
            btn_permance_setting.IsEnabled = false;
            frame.Content = p3;
        }

    }
}
