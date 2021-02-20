@echo off&chcp 65001 >nul
cd /d %~dp0
fltmc>nul||mshta vbscript:CreateObject("Shell.Application").ShellExecute("%~dpnx0","%*",,"runas",1)(window.close)&&exit
cls
for /f "tokens=2* " %%i in ('reg query "HKLM\SYSTEM\ControlSet001\Services\nvlddmkm\Global\NVTweak" /v "InstalledDisplayServers"') do set InstalledDisplayServers=%%j
if exist %InstalledDisplayServers%\nvEncMFTH264.dll ( echo enable ) else ( echo disable )