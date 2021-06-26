@echo off
echo SVFI内测简易发布系统
set /p ols_v=请输入ols版本号：
set /p gui_v=请输入gui版本号：

start nuitka --standalone --mingw64 --show-memory --show-progress --nofollow-imports --plugin-enable=qt-plugins --windows-icon-from-ico=D:\60-fps-Project\Projects\RIFE_GUI\svfi.ico --windows-product-name="SVFI CLI" --windows-product-version=%ols_v% --windows-file-description="SVFI Interpolation CLI" --windows-company-name="Jeanna-SVFI"  --output-dir=release .\one_line_shot_args.py

start /wait nuitka --standalone --mingw64 --show-memory --show-progress --nofollow-imports --include-qt-plugins=sensible,styles --plugin-enable=qt-plugins  --include-package=QCandyUi,PyQt5 --windows-icon-from-ico=D:\60-fps-Project\Projects\RIFE_GUI\svfi.ico --windows-product-name="SVFI" --windows-product-version=%gui_v% --windows-file-description="Squirrel Video Frame Interpolation" --windows-company-name="SVFI" --follow-import-to=Utils --output-dir=release --windows-disable-console .\RIFE_GUI_Start.py

ren "D:\60-fps-Project\Projects\RIFE_GUI\release\RIFE_GUI_Start.dist\RIFE_GUI_Start.exe" SVFI.%gui_v%.alpha.exe

cd D:\60-fps-Project\Projects\RIFE_GUI\release\RIFE_GUI_Start.dist

xcopy /y D:\60-fps-Project\Projects\RIFE_GUI\release\RIFE_GUI_Start.dist\SVFI.%gui_v%.alpha.exe  D:\60-fps-Project\Projects\RIFE_GUI\release\SVFI.Env >nul
xcopy /y D:\60-fps-Project\Projects\RIFE_GUI\release\one_line_shot_args.dist\one_line_shot_args.exe D:\60-fps-Project\Projects\RIFE_GUI\release\SVFI.Env >nul

cd D:\60-fps-Project\Projects\RIFE_GUI\release\SVFI.Env
del 启动SVFI.bat
echo cd /d %%~dp0/Package >> 启动SVFI.bat
echo start SVFI.%gui_v%.alpha.exe >> 启动SVFI.bat
start D:\60-fps-Project\Projects\RIFE_GUI\release\SVFI.Env

