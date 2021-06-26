# encoding=utf-8
import os

root = r"D:\60-fps-Project\Projects\RIFE GUI"
ico_path = os.path.join(root, "svfi.ico")
gui_version = input("SVFI GUI Version: ")
cli_version = input("SVFI CLI Version: ")
tag_version = input("SVFI Tag Version: ")
os.system(
    f'nuitka --standalone --mingw64 --show-memory --show-progress --nofollow-imports --include-qt-plugins=sensible,styles --plugin-enable=qt-plugins  --include-package=QCandyUi,PyQt5 --windows-icon-from-ico="{ico_path}" --windows-product-name="SVFI" --windows-product-version={gui_version} --windows-file-description="Squirrel Video Frame Interpolation" --windows-company-name="SVFI" --follow-import-to=Utils --output-dir=release --windows-disable-console .\RIFE_GUI_Start.py')
# debug
# os.system(f'nuitka --standalone --mingw64 --show-memory --show-progress --nofollow-imports --include-qt-plugins=sensible,styles --plugin-enable=qt-plugins  --include-package=QCandyUi,PyQt5 --windows-icon-from-ico="{ico_path}" --windows-product-name="SVFI" --windows-product-version={gui_version} --windows-file-description="Squirrel Video Frame Interpolation" --windows-company-name="SVFI" --follow-import-to=Utils --output-dir=release .\RIFE_GUI_Start.py')
os.system(
    f'nuitka --standalone --mingw64 --show-memory --show-progress --nofollow-imports --plugin-enable=qt-plugins --windows-icon-from-ico="{ico_path}" --windows-product-name="SVFI CLI" --windows-product-version={cli_version} --windows-file-description="SVFI Interpolation CLI" --windows-company-name="Jeanna-SVFI"  --follow-import-to=Utils --output-dir=release .\one_line_shot_args.py')
pack_dir = r"D:\60-fps-Project\Projects\RIFE GUI\release\release_pack"
if not os.path.exists(pack_dir):
    os.mkdir(pack_dir)
os.replace(r".\release\one_line_shot_args.dist\one_line_shot_args.exe",
           os.path.join(pack_dir, "one_line_shot_args.exe"))
os.replace(r".\release\RIFE_GUI_Start.dist\RIFE_GUI_Start.exe", os.path.join(pack_dir, f"SVFI.{tag_version}.exe"))
os.chdir(pack_dir)
os.system(f"""
echo cd /d %%~dp0/Package > 启动SVFI.bat
echo start SVFI.{tag_version}.exe >> 启动SVFI.bat
"""
          )
"""
一定要记得改版本！
nuitka --standalone --mingw64 --show-memory --show-progress --nofollow-imports --include-qt-plugins=sensible,styles --plugin-enable=qt-plugins  --include-package=QCandyUi,PyQt5 --windows-icon-from-ico="{ico_path}" --windows-product-name="SVFI" --windows-product-version=3.2.3 --windows-file-description="Squirrel Video Frame Interpolation" --windows-company-name="SVFI" --follow-import-to=Utils --output-dir=release --windows-disable-console .\RIFE_GUI_Start.py

打包one_line_shot.exe
nuitka --standalone --mingw64 --show-memory --show-progress --nofollow-imports --plugin-enable=qt-plugins --windows-icon-from-ico="D:\60-fps-Project\Projects\RIFE GUI\svfi.ico" --windows-product-name="SVFI CLI" --windows-product-version=6.6.0 --windows-file-description="SVFI Interpolation CLI" --windows-company-name="Jeanna-SVFI"  --output-dir=release .\one_line_shot_args.py
"""
