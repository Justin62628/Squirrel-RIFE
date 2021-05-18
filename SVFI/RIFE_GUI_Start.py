import sys
import QCandyUi
import traceback
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

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

app = QApplication(sys.argv)
app_backend_module = RIFE_GUI_Backend
app_backend = app_backend_module.RIFE_GUI_BACKEND()
try:
    form = QCandyUi.CandyWindow.createWindow(app_backend, theme="blueDeep", ico_path="svfi.png",
                                             title="Squirrel Video Frame Interpolation 3.1.1 alpha")
    form.show()
    app.exec_()
    """Save Settings"""
    app_backend.load_current_settings()
except Exception:
    app_backend_module.logger.critical(traceback.format_exc())
