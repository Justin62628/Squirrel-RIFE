using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace RIFE_APP
{
    class RIFE
    {
        private readonly string python = AppDomain.CurrentDomain.SetupInformation.ApplicationBase + @"pkgs\py\python.exe";
        private readonly string getgpu = AppDomain.CurrentDomain.SetupInformation.ApplicationBase + @"pkgs\info\getgpu.py";
        private readonly string getcpu = AppDomain.CurrentDomain.SetupInformation.ApplicationBase + @"pkgs\info\getcpu.py";
        public int count = 0;
        Process P;

        public List<string> GetAvailableGPU()
        {
            List<string> gpus = new List<string>();
            P = new Process();
            P.StartInfo.FileName = python;
            P.StartInfo.Arguments = $"\"{getgpu}\"";
            P.StartInfo.RedirectStandardInput = true;
            P.StartInfo.RedirectStandardOutput = true;
            P.StartInfo.RedirectStandardError = true;
            P.StartInfo.UseShellExecute = false;
            P.StartInfo.CreateNoWindow = true;
            P.ErrorDataReceived += (s, e) => { gpus.Add(e.Data); };
            P.OutputDataReceived += (s, e) => { gpus.Add(e.Data); };
            P.Start();
            P.BeginOutputReadLine();
            P.BeginErrorReadLine();
            P.WaitForExit();
            return gpus;
        }

        public List<string> GetAvailableCPU()
        {
            List<string> cpus = new List<string>();
            P = new Process();
            P.StartInfo.FileName = python;
            P.StartInfo.Arguments = $"\"{getcpu}\"";
            P.StartInfo.RedirectStandardInput = true;
            P.StartInfo.RedirectStandardOutput = true;
            P.StartInfo.RedirectStandardError = true;
            P.StartInfo.UseShellExecute = false;
            P.StartInfo.CreateNoWindow = true;
            P.ErrorDataReceived += (s, e) => { cpus.Add(e.Data); };
            P.OutputDataReceived += (s, e) => { cpus.Add(e.Data); };
            P.Start();
            P.BeginOutputReadLine();
            P.BeginErrorReadLine();
            P.WaitForExit();
            return cpus;
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
