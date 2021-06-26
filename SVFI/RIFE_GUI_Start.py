import sys
import traceback

import QCandyUi
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

"""High Resolution Support"""
if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

# if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
#     QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

try:
    from Utils import RIFE_GUI_Backend
except ImportError as e:
    traceback.print_exc()
    print("Not Find RIFE GUI Backend, please contact developers for support")
    input("Press Any Key to Quit")
    exit()
SVFI_version = "3.2 Professional"
# SVFI_version = "3.2 Platinum"
app = QApplication(sys.argv)
app_backend_module = RIFE_GUI_Backend
app_backend = app_backend_module.RIFE_GUI_BACKEND(free=False, version=SVFI_version)
try:
    form = QCandyUi.CandyWindow.createWindow(app_backend, theme="blueDeep", ico_path="svfi.png",
                                             title=f"Squirrel Video Frame Interpolation {SVFI_version}")
    form.show()
    app.exec_()
    """Save Settings"""
    app_backend.load_current_settings()
except Exception:
    app_backend_module.logger.critical(traceback.format_exc())
