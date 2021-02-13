using pmgr;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace ProjectFFMPEG
{
	//先前丢失的代码，使用反编译恢复
	internal class FFMPEG
	{
		private Process P { get; set; }
		public string Data { get; private set; } = "";
		public List<string> ContextData { get; private set; } = new List<string>();
		public List<string> StreamInfo { get; private set; } = new List<string>();
		public List<string> StreamEncoder { get; private set; } = new List<string>();
		public List<string> StreamMapping { get; private set; } = new List<string>();
		public bool HasVideo { get; private set; } = false;
		public bool HasAudio { get; private set; } = false;
		public string Configuration { get; private set; } = "";
		public TimeSpan Duration { get; private set; } = default;
		public string StartTime { get; private set; } = "";
		public double PrimaryFPS { get; private set; } = 0.0;
		public string PrimaryFileSize { get; private set; } = "";
		public string PrimaryBitrate { get; private set; } = "";
		public Size PrimarySize { get; private set; } = new Size(0.0, 0.0);
		public Size TargetSize { get; private set; } = new Size(0.0, 0.0);
		public string Speed { get; private set; } = "";
		public TimeSpan CompiledTime { get; private set; } = default(TimeSpan);
		public TimeSpan ForecastTime { get; private set; } = default(TimeSpan);
		public string Bitrate { get; private set; } = "";
		public int FPS { get; private set; } = 0;
		public string CompiledSize { get; private set; } = "";
		public int Frame { get; private set; } = 0;
		public int Quality { get; private set; } = 0;
		public bool SearchPath(out string[] path, string dir = null)
		{
			path = new string[3];
			bool flag = dir == null;
			if (flag)
			{
				DriveInfo[] drives = DriveInfo.GetDrives();
				foreach (DriveInfo driveInfo in drives)
				{
					foreach (string text in Directory.EnumerateFiles(driveInfo.Name, "exe", SearchOption.AllDirectories))
					{
						path[0] = text;
					}
					foreach (string text2 in Directory.EnumerateFiles(driveInfo.Name, "ffplay.exe", SearchOption.AllDirectories))
					{
						path[1] = text2;
					}
					foreach (string text3 in Directory.EnumerateFiles(driveInfo.Name, "ffprobe.exe", SearchOption.AllDirectories))
					{
						path[2] = text3;
					}
				}
			}
			else
			{
				foreach (string text4 in Directory.EnumerateFiles(dir, "exe", SearchOption.AllDirectories))
				{
					path[0] = text4;
				}
				foreach (string text5 in Directory.EnumerateFiles(dir, "ffplay.exe", SearchOption.AllDirectories))
				{
					path[1] = text5;
				}
				foreach (string text6 in Directory.EnumerateFiles(dir, "ffprobe.exe", SearchOption.AllDirectories))
				{
					path[2] = text6;
				}
			}
			return path[0] != "";
		}

		public void SetPath(string FFmpegPath, string FFplayPath = null, string FFprobePath = null)
		{
			bool flag = !File.Exists(FFmpegPath);
			if (flag)
			{
				throw new FileNotFoundException("该路径不存在所对应的文件不存在");
			}
			ffmpegPath = FFmpegPath;
			bool flag2 = FFplayPath != null;
			if (flag2)
			{
				bool flag3 = !File.Exists(FFplayPath);
				if (flag3)
				{
					throw new FileNotFoundException("该路径不存在所对应的文件不存在");
				}
				ffplayPath = FFplayPath;
			}
			bool flag4 = FFprobePath != null;
			if (flag4)
			{
				bool flag5 = !File.Exists(FFprobePath);
				if (flag5)
				{
					throw new FileNotFoundException("该路径不存在所对应的文件不存在");
				}
				ffprobePath = FFprobePath;
			}
		}

		public void AutoSetPath()
		{
			string[] array;
			SearchPath(out array, null);
			ffmpegPath = array[0].ToString();
			string text = array[1];
			ffplayPath = ((text != null) ? text.ToString() : null);
			string text2 = array[2];
			ffprobePath = ((text2 != null) ? text2.ToString() : null);
		}

		public void Run(string cmd, fftools ff = fftools.ffmpeg, string WorkingDirectory = null, EventHandler Exited = null, DataReceivedEventHandler dataReceived = null, bool WaitForExit = false)
		{
			P = new Process();
			switch (ff)
			{
				case fftools.ffmpeg:
					P.StartInfo.FileName = ffmpegPath;
					break;
				case fftools.ffplay:
					P.StartInfo.FileName = ffplayPath;
					break;
				case fftools.ffprobe:
					P.StartInfo.FileName = ffprobePath;
					break;
			}
			P.StartInfo.Arguments = cmd;
			P.StartInfo.RedirectStandardInput = true;
			P.StartInfo.RedirectStandardOutput = true;
			P.StartInfo.RedirectStandardError = true;
			P.StartInfo.UseShellExecute = false;
			P.StartInfo.CreateNoWindow = true;
			P.EnableRaisingEvents = true;
			P.StartInfo.WorkingDirectory = ((WorkingDirectory != null) ? WorkingDirectory.ToString() : null);
			P.Start();
			P.ErrorDataReceived += P_ErrorDataReceived;
			P.ErrorDataReceived += dataReceived;
			P.Exited += P_Exited;
			P.Exited += Exited;
			bool flag = !BeginErrorReadLine;
			if (flag)
			{
				P.BeginErrorReadLine();
				BeginErrorReadLine = true;
			}
			if (WaitForExit)
			{
				while (!P.HasExited)
				{
					Task.Delay(1).Wait();
				}
			}
		}

		public void Play(string path, PlayOptions po = PlayOptions.localPlayer, string PlayerPath = "", string WorkingDirectory = "")
		{
			P = new Process();
			bool flag = po == PlayOptions.localPlayer;
			if (flag)
			{
				try
				{
					P.StartInfo.FileName = path;
					bool flag2 = WorkingDirectory != "";
					if (flag2)
					{
						P.StartInfo.WorkingDirectory = WorkingDirectory;
					}
					P.Start();
				}
				catch
				{
					throw new Exception("文件无法读取或无打开方式");
				}
			}
			else
			{
				bool flag3 = po == PlayOptions.ffplay;
				if (flag3)
				{
					bool flag4 = !File.Exists(ffplayPath);
					if (flag4)
					{
						throw new FileNotFoundException("ffplay路径不存在或未设置");
					}
					P.StartInfo.FileName = ffplayPath;
					string text = "";
					bool flag5 = path.Substring(0, 1) != "\"";
					if (flag5)
					{
						text = "\"" + path;
					}
					bool flag6 = path.Substring(path.Length - 1, 1) != "\"";
					if (flag6)
					{
						text += "\"";
					}
					P.StartInfo.Arguments = "-i " + text;
					bool flag7 = WorkingDirectory != "";
					if (flag7)
					{
						P.StartInfo.WorkingDirectory = WorkingDirectory;
					}
					P.Start();
				}
				else
				{
					bool flag8 = po == PlayOptions.Custom;
					if (flag8)
					{
						bool flag9 = !File.Exists(PlayerPath);
						if (flag9)
						{
							throw new FileNotFoundException("播放器路径无效");
						}
						P.StartInfo.FileName = PlayerPath;
						P.StartInfo.Arguments = (path ?? "");
						bool flag10 = WorkingDirectory != "";
						if (flag10)
						{
							P.StartInfo.WorkingDirectory = WorkingDirectory;
						}
						P.Start();
					}
				}
			}
		}

		protected void P_Exited(object sender, EventArgs e)
		{
			/*Bitrate = "";
			CompiledSize = "";
			CompiledTime = TimeSpan.Zero;
			Configuration = "";
			ContextData.Clear();
			Data = "";
			Duration = TimeSpan.Zero;
			ForecastTime = TimeSpan.Zero;
			FPS = 0;
			Frame = 0;
			HasAudio = false;
			HasVideo = false;
			PrimaryBitrate = "";
			PrimaryFileSize = "";
			PrimaryFPS = 0.0;
			PrimarySize = new Size();
			Quality = 0;
			Speed = "";
			StartTime = "";
			StreamEncoder.Clear();
			StreamInfo.Clear();
			StreamMapping.Clear();
			TargetSize = new Size();
			GC.Collect();*/
		}

		protected void P_ErrorDataReceived(object sender, DataReceivedEventArgs e)
		{
			bool flag = e.Data != null;
			if (flag)
			{
				Data = e.Data;
				ContextData.Add(e.Data);
				bool flag2 = e.Data.IndexOf("configuration") != -1;
				if (flag2)
				{
					try
					{
						Configuration = e.Data.Substring(e.Data.IndexOf("configuration") + 14);
					}
					catch
					{
					}
				}
				bool flag3 = e.Data.IndexOf("Duration") != -1;
				if (flag3)
				{
					try
					{
						Duration = TimeSpan.Parse(e.Data.Substring(e.Data.IndexOf("Duration") + 10, 11));
					}
					catch
					{
					}
				}
				bool flag4 = e.Data.IndexOf("start") != -1;
				if (flag4)
				{
					try
					{
						StartTime = e.Data.Substring(e.Data.IndexOf("start") + 7, 8);
					}
					catch
					{
					}
				}
				bool flag5 = e.Data.IndexOf("Duration") != -1 && e.Data.IndexOf("start") != -1;
				if (flag5)
				{
					try
					{
						PrimaryBitrate = e.Data.Substring(e.Data.IndexOf("bitrate") + 8, e.Data.IndexOf("kb/s") - (e.Data.IndexOf("bitrate") + 8 - 4)).Trim().Replace(" ", "");
					}
					catch
					{
					}
					try
					{
						PrimaryFileSize = ((double)(int.Parse(PrimaryBitrate.Substring(0, PrimaryBitrate.Length - 4)) / 1024) * Duration.TotalSeconds / 8.0).ToString() + "M";
					}
					catch
					{
					}
				}
				bool flag6 = e.Data.IndexOf("Stream") != -1 && e.Data.IndexOf("mapping") == -1 && e.Data.IndexOf("->") == -1;
				if (flag6)
				{
					try
					{
						StreamInfo.Add(e.Data.Substring(e.Data.IndexOf("Stream") + 6));
					}
					catch
					{
					}
					try
					{
						PrimaryFPS = double.Parse(e.Data.Substring(e.Data.IndexOf("fps") - 6, 5));
					}
					catch
					{
					}
					try
					{
						bool flag7 = e.Data.IndexOf("Video") != -1;
						if (flag7)
						{
							HasVideo = true;
						}
					}
					catch
					{
					}
					try
					{
						bool flag8 = e.Data.IndexOf("Audio") != -1;
						if (flag8)
						{
							HasAudio = true;
						}
					}
					catch
					{
					}
				}
				bool flag9 = e.Data.IndexOf("encoder") != -1;
				if (flag9)
				{
					try
					{
						StreamEncoder.Add(e.Data.Substring(e.Data.IndexOf("encoder") + 18));
					}
					catch
					{
					}
				}
				bool flag10 = e.Data.IndexOf("->") != -1;
				if (flag10)
				{
					try
					{
						StreamMapping.Add(e.Data);
					}
					catch
					{
					}
				}
				bool flag11 = e.Data.IndexOf("frame") != -1 && e.Data.IndexOf("fps") != -1;
				if (flag11)
				{
					try
					{
						Frame = Convert.ToInt32(e.Data.Substring(e.Data.IndexOf("frame") + 6, e.Data.IndexOf("fps") - (e.Data.IndexOf("frame") + 6)).Trim());
					}
					catch
					{
					}
				}
				bool flag12 = e.Data.IndexOf("fps") != -1 && e.Data.IndexOf("q") != -1;
				if (flag12)
				{
					try
					{
						FPS = Convert.ToInt32(e.Data.Substring(e.Data.IndexOf("fps") + 4, e.Data.IndexOf("q") - (e.Data.IndexOf("fps") + 4)).Trim());
					}
					catch
					{
					}
				}
				bool flag13 = e.Data.IndexOf("q") != -1 && e.Data.IndexOf("size") != -1;
				if (flag13)
				{
					try
					{
						Quality = Convert.ToInt32(e.Data.Substring(e.Data.IndexOf("q") + 2, e.Data.IndexOf("size") - (e.Data.IndexOf("q") + 5)).Trim());
					}
					catch
					{
					}
				}
				bool flag14 = e.Data.IndexOf("size") != -1 && e.Data.IndexOf("time") != -1;
				if (flag14)
				{
					try
					{
						CompiledSize = e.Data.Substring(e.Data.IndexOf("size") + 5, e.Data.IndexOf("time") - (e.Data.IndexOf("size") + 5)).Trim();
					}
					catch
					{
					}
				}
				bool flag15 = e.Data.IndexOf("time") != -1 && e.Data.IndexOf("bitrate") != -1;
				if (flag15)
				{
					try
					{
						CompiledTime = TimeSpan.Parse(e.Data.Substring(e.Data.IndexOf("time") + 5, e.Data.IndexOf("bitrate") - (e.Data.IndexOf("time") + 5)).Trim());
					}
					catch
					{
					}
				}
				bool flag16 = e.Data.IndexOf("bitrate") != -1 && e.Data.IndexOf("speed") != -1;
				if (flag16)
				{
					try
					{
						Bitrate = e.Data.Substring(e.Data.IndexOf("bitrate") + 8, Data.IndexOf("speed") - (e.Data.IndexOf("bitrate") + 8)).Trim();
					}
					catch
					{
					}
				}
				bool flag17 = e.Data.IndexOf("speed") != -1;
				if (flag17)
				{
					try
					{
						Speed = e.Data.Substring(e.Data.IndexOf("speed") + 6).Trim();
					}
					catch
					{
					}
				}
				bool flag18 = Speed != "" && Speed != "0.00x" && Duration != TimeSpan.Zero && CompiledTime != TimeSpan.Zero;
				if (flag18)
				{
					try
					{
						ForecastTime = TimeSpan.FromSeconds((Duration.TotalSeconds - CompiledTime.TotalSeconds) / double.Parse(Speed.Replace("x", "")));
					}
					catch
					{
					}
				}
				else
				{
					ForecastTime = TimeSpan.Zero;
				}
			}
		}

		protected double ConvertToSeconds(string str)
		{
			double num = 0.0;
			string[] array = str.Split(new char[]
			{
				':'
			});
			try
			{
				num = (double)(int.Parse(array[0]) * 3600);
			}
			catch
			{
			}
			try
			{
				num += (double)(int.Parse(array[1]) * 60);
			}
			catch
			{
			}
			try
			{
				num += double.Parse(array[2]);
			}
			catch
			{
			}
			return num;
		}

		protected string ConvertToDateTime(double Seconds)
		{
			int num = 0;
			int num2 = 0;
			int num3 = 0;
			int num4 = 0;
			try
			{
				num = Convert.ToInt32(Seconds / 86400.0);
			}
			catch
			{
			}
			try
			{
				num2 = Convert.ToInt32(Seconds % 86400.0 / 3600.0);
			}
			catch
			{
			}
			try
			{
				num3 = Convert.ToInt32(Seconds % 86400.0 % 3600.0 / 60.0);
			}
			catch
			{
			}
			try
			{
				num4 = Convert.ToInt32(Seconds % 86400.0 % 3600.0 % 60.0);
			}
			catch
			{
			}
			bool flag = num != 0 && num2 != 0 && num3 != 0 && num4 != 0;
			string result;
			if (flag)
			{
				result = string.Format("{0}天{1}小时{2}分钟{3}秒", new object[]
				{
					num,
					num2,
					num3,
					Seconds
				});
			}
			else
			{
				bool flag2 = (num == 0 && num2 != 0 && num3 != 0 && num4 != 0) || (num == 0 && num2 == 0 && num3 != 0 && num4 != 0) || (num == 0 && num2 == 0 && num3 == 0 && num4 != 0);
				if (flag2)
				{
					result = string.Format("{0}小时{1}分钟{2}秒", num2, num3, Seconds);
				}
				else
				{
					result = "";
				}
			}
			return result;
		}

		public void Suspend()
		{
			if(P != null)
            {
				if(!P.HasExited)
                {
					ProcessMgr.SuspendProcess(P.Id);
				}
				
			}
			
		}

		public void Resume()
		{
			if (P != null)
			{
				if (!P.HasExited)
				{
					ProcessMgr.ResumeProcess(P.Id);
				}
			}
			
		}

		public async void Exit()
		{
			await Task.Run(delegate ()
			{
				while (!P.HasExited)
				{
					P.StandardInput.WriteLine("q");
				}
			});
		}

		public void Kill()
		{
			if (P != null)
			{
				bool flag = !P.HasExited;
				if (flag)
				{
					P.Kill();
				}
				object sender = new object();
				EventArgs e = new EventArgs();
				P_Exited(sender, e);
			}
		}

		public void Wait()
		{
			P.WaitForExit();
		}

		public void Dispose()
		{
			P.Dispose();
		}

		public void Log(LogOptions lop = LogOptions.ContextData)
		{
			object sync = Sync;
			lock (sync)
			{
				switch (lop)
				{
					case LogOptions.Bitrate:
						WriteLine("Bitrate:" + Bitrate, ConsoleColor.White);
						break;
					case LogOptions.CompiledSize:
						WriteLine("CompiledSize:" + CompiledSize, ConsoleColor.White);
						break;
					case LogOptions.CompiledTime:
						WriteLine("CompiledTime:" + CompiledTime, ConsoleColor.White);
						break;
					case LogOptions.Configuration:
						WriteLine("Configuration:" + Configuration, ConsoleColor.White);
						break;
					case LogOptions.ContextData:
						WriteLine("ContextData:\r\n", ConsoleColor.White);
						foreach (string str in ContextData)
						{
							WriteLine("  " + str + "\r\n", ConsoleColor.White);
						}
						break;
					case LogOptions.Data:
						WriteLine("Data:" + Data, ConsoleColor.White);
						break;
					case LogOptions.Duration:
						WriteLine("Duration:" + Duration, ConsoleColor.White);
						break;
					case LogOptions.ffplayPath:
						WriteLine("ffplayPath:" + ffplayPath, ConsoleColor.White);
						break;
					case LogOptions.ffprobePath:
						WriteLine("ffprobePath:" + ffprobePath, ConsoleColor.White);
						break;
					case LogOptions.ForecastTime:
						WriteLine("ForecastTime:" + ForecastTime, ConsoleColor.White);
						break;
					case LogOptions.FPS:
						WriteLine("FPS:" + FPS, ConsoleColor.White);
						break;
					case LogOptions.Frame:
						WriteLine("Frame:" + Frame, ConsoleColor.White);
						break;
					case LogOptions.ffmpegPath:
						WriteLine("ffmpegPath:" + ffmpegPath, ConsoleColor.White);
						break;
					case LogOptions.HasAudio:
						WriteLine("HasAudio:" + HasAudio.ToString(), ConsoleColor.White);
						break;
					case LogOptions.HasVideo:
						WriteLine("HasVideo:" + HasVideo.ToString(), ConsoleColor.White);
						break;
					case LogOptions.PrimaryBitrate:
						WriteLine("PrimaryBitrate:" + PrimaryBitrate, ConsoleColor.White);
						break;
					case LogOptions.PrimaryFileSize:
						WriteLine("PrimaryFileSize:" + PrimaryFileSize, ConsoleColor.White);
						break;
					case LogOptions.PrimaryFPS:
						WriteLine("PrimaryFPS:" + PrimaryFPS, ConsoleColor.White);
						break;
					case LogOptions.PrimarySize:
						WriteLine(string.Concat(new object[]
						{
						"PrimarySize:",
						PrimarySize.Width,
						"x",
						PrimarySize.Height
						}), ConsoleColor.White);
						break;
					case LogOptions.Quality:
						WriteLine("Quality:" + Quality, ConsoleColor.White);
						break;
					case LogOptions.Speed:
						WriteLine("Speed:" + Speed, ConsoleColor.White);
						break;
					case LogOptions.StartTime:
						WriteLine("StartTime:" + StartTime, ConsoleColor.White);
						break;
					case LogOptions.StreamEncoder:
						WriteLine("StreamEncoder:", ConsoleColor.White);
						foreach (string str2 in StreamEncoder)
						{
							WriteLine("   " + str2 + "\r\n", ConsoleColor.White);
						}
						break;
					case LogOptions.StreamInfo:
						WriteLine("StreamInfo:", ConsoleColor.White);
						foreach (string str3 in StreamInfo)
						{
							WriteLine("   " + str3 + "\r\n", ConsoleColor.White);
						}
						break;
					case LogOptions.StreamMapping:
						WriteLine("StreamMapping:", ConsoleColor.White);
						foreach (string str4 in StreamMapping)
						{
							WriteLine("   " + str4 + "\r\n", ConsoleColor.White);
						}
						break;
					case LogOptions.TargetSize:
						WriteLine(string.Concat(new object[]
						{
						"TargetSize:",
						TargetSize.Width,
						"x",
						TargetSize.Height
						}), ConsoleColor.White);
						break;
				}
			}
		}

		protected void WriteLine(string msg, ConsoleColor cc = ConsoleColor.White)
		{
			object sync = Sync;
			lock (sync)
			{
				Console.ForegroundColor = cc;
				Console.WriteLine(msg);
				Console.ResetColor();
			}
		}

		protected void Write(string msg, ConsoleColor cc = ConsoleColor.White)
		{
			object sync = Sync;
			lock (sync)
			{
				Console.ForegroundColor = cc;
				Console.Write(msg);
				Console.ResetColor();
			}
		}

		public void Concat(List<string> Files, string Output, string WorkingDirectory = null, EventHandler Exited = null, DataReceivedEventHandler dataReceived = null)
		{
			foreach (string str in Files)
			{
				bool flag = !File.Exists(((WorkingDirectory != null) ? WorkingDirectory.ToString() : null) + str);
				if (flag)
				{
					throw new FileNotFoundException("文件\"" + str + "\"不存在");
				}
			}
			bool flag2 = WorkingDirectory != null;
			if (flag2)
			{
				bool flag3 = WorkingDirectory.Substring(WorkingDirectory.Length - 1) != "\\";
				if (flag3)
				{
					WorkingDirectory += "\\";
				}
			}
			List<string> list = new List<string>();
			foreach (string text in Files)
			{
				Run(string.Concat(new string[]
				{
					"-y -i \"",
					text,
					"\" -c copy -bsf:v h264_mp4toannexb -f mpegts \"",
					text,
					".ts\""
				}), fftools.ffmpeg, WorkingDirectory, Exited, dataReceived, false);
				list.Add(text + ".ts");
				do
				{
					Task.Delay(1).Wait();
				}
				while (!File.Exists(((WorkingDirectory != null) ? WorkingDirectory.ToString() : null) + text + ".ts"));
				FileInfo fileInfo = new FileInfo(((WorkingDirectory != null) ? WorkingDirectory.ToString() : null) + text + ".ts");
				do
				{
					Task.Delay(1).Wait();
				}
				while (fileInfo.Length < 1024L);
			}
			string text2 = "";
			foreach (string str2 in list)
			{
				text2 = text2 + str2 + "|";
			}
			string text3 = Guid.NewGuid().ToString();
			Run(string.Concat(new string[]
			{
				"-y -i concat:\"",
				text2,
				"\" -c copy -bsf:a aac_adtstoasc -movflags +faststart \"",
				text3,
				".mp4\""
			}), fftools.ffmpeg, WorkingDirectory, Exited, dataReceived, true);
			bool flag4 = WorkingDirectory == null;
			if (flag4)
			{
				while (!File.Exists(Output.Substring(0, Output.LastIndexOf("\\") + 1) + text3 + ".mp4"))
				{
					Task.Delay(1).Wait();
				}
				FileInfo fileInfo2 = new FileInfo(Output.Substring(0, Output.LastIndexOf("\\") + 1) + text3 + ".mp4");
				while (fileInfo2.Length < 1024L)
				{
					Task.Delay(1).Wait();
				}
			}
			else
			{
				while (!File.Exists(WorkingDirectory + text3 + ".mp4"))
				{
					Task.Delay(1).Wait();
				}
				FileInfo fileInfo3 = new FileInfo(WorkingDirectory + text3 + ".mp4");
				while (fileInfo3.Length < 1024L)
				{
					Task.Delay(1).Wait();
				}
			}
			bool flag5 = Output.Substring(Output.LastIndexOf(".")) != ".mp4";
			if (flag5)
			{
				Run("-y -i " + text3 + ".mp4 -c copy " + Output, fftools.ffmpeg, WorkingDirectory, Exited, dataReceived, true);
			}
			else
			{
				File.Move(((WorkingDirectory != null) ? WorkingDirectory.ToString() : null) + text3 + ".mp4", ((WorkingDirectory != null) ? WorkingDirectory.ToString() : null) + Output);
			}
			while (!File.Exists(((WorkingDirectory != null) ? WorkingDirectory.ToString() : null) + Output))
			{
				Task.Delay(1).Wait();
			}
			FileInfo fileInfo4 = new FileInfo(((WorkingDirectory != null) ? WorkingDirectory.ToString() : null) + Output);
			while (fileInfo4.Length < 1024L)
			{
				Task.Delay(1).Wait();
			}
			bool flag6 = File.Exists(((WorkingDirectory != null) ? WorkingDirectory.ToString() : null) + Output);
			if (!flag6)
			{
				foreach (string str3 in list)
				{
					File.Delete(((WorkingDirectory != null) ? WorkingDirectory.ToString() : null) + str3);
				}
				File.Delete(((WorkingDirectory != null) ? WorkingDirectory.ToString() : null) + text3 + ".mp4");
				throw new Exception("文件合并失败或找不到合并后的文件");
			}
			FileInfo fileInfo5 = new FileInfo(((WorkingDirectory != null) ? WorkingDirectory.ToString() : null) + Output);
			long num = 0L;
			foreach (string str4 in list)
			{
				FileInfo fileInfo6 = new FileInfo(((WorkingDirectory != null) ? WorkingDirectory.ToString() : null) + str4);
				num += fileInfo6.Length;
			}
			bool flag7 = (double)fileInfo5.Length < (double)num * 0.5;
			if (flag7)
			{
				throw new Exception("合并文件失败");
			}
			foreach (string str5 in Files)
			{
				bool flag8 = File.Exists(((WorkingDirectory != null) ? WorkingDirectory.ToString() : null) + str5 + ".ts");
				if (flag8)
				{
					File.Delete(((WorkingDirectory != null) ? WorkingDirectory.ToString() : null) + str5);
				}
			}
			foreach (string str6 in list)
			{
				File.Delete(((WorkingDirectory != null) ? WorkingDirectory.ToString() : null) + str6);
			}
			File.Delete(((WorkingDirectory != null) ? WorkingDirectory.ToString() : null) + text3 + ".mp4");
		}

		public async void Spilt()
		{
			await Task.Delay(1);
		}

		private string ffmpegPath = "";

		private string ffplayPath = "";

		private string ffprobePath = "";

		private readonly object Sync = new object();

		private bool BeginErrorReadLine = false;

		public enum fftools
		{
			ffmpeg,
			ffplay,
			ffprobe
		}

		public enum PlayOptions
		{
			ffplay,
			localPlayer,
			Custom
		}

		public enum LogOptions
		{
			Bitrate,
			CompiledSize,
			CompiledTime,
			Configuration,
			ContextData,
			Data,
			Duration,
			ffplayPath,
			ffprobePath,
			ForecastTime,
			FPS,
			Frame,
			ffmpegPath,
			HasAudio,
			HasVideo,
			PrimaryBitrate,
			PrimaryFileSize,
			PrimaryFPS,
			PrimarySize,
			Quality,
			Speed,
			StartTime,
			StreamEncoder,
			StreamInfo,
			StreamMapping,
			TargetSize
		}
	}
}
