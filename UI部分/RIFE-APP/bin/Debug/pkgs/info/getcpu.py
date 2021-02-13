import wmi
cpuinfo = wmi.WMI()
for cpu in cpuinfo.Win32_Processor():
    print(cpu.Name)