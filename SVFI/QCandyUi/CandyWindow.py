import json
import os

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import qApp

from . import WindowWithTitleBar
from . import simple_qss

"""
# 示例1和2最终效果一样, 不过推荐用1的方法
- 使用示例1:
    app = QApplication(sys.argv)
    w = CandyWindow.createWindow(LogViewer(), 'Log Viewer', 'myicon.ico', 'blueGreen')
    w.show()
    sys.exit(app.exec_())


- 使用示例2:
eg:
@colorful('blueGreen')
class LogViewer(QWidget):
    ... ...
"""
RESOURCE_DIR = 'candyUi'


def colorful(theme):
    """
    彩色主题装饰, 可以装饰所有的QWidget类使其直接拥有彩色主题 (带Titlebar)
    :param theme: 主题名, 与theme.json里面的主题名对应
    :return:
    """

    def new_func(aClass):
        def on_call(*args, **kargs):
            src_widget = aClass(*args, **kargs)
            dst_widget = createWindow(src_widget, theme)
            return dst_widget

        return on_call

    return new_func


def createWindow(mainWidget, theme=None, title='CandySweet', ico_path=''):
    """
    快速创建彩色窗 (带TitleBar)
    :param mainWidget:
    :param theme:
    :param title:
    :param ico_path:
    :return:
    """
    setTheme(theme)
    coolWindow = WindowWithTitleBar.WindowWithTitleBar(mainWidget)
    coolWindow.setWindowTitle(title)
    coolWindow.setWindowIcon(QIcon(ico_path))
    # coolWindow.setWindowRadius(3.14/2)
    return coolWindow


def setTheme(theme):
    """
    根据theme.json设置主题的qss (只改样式不加Titlebar)
    :param theme:
    :return:
    """
    THEME_FILE = RESOURCE_DIR + '/theme.json'
    if os.path.isfile(THEME_FILE):
        path = THEME_FILE
    else:
        path = (os.path.split(__file__)[0] + '\\' + THEME_FILE).replace('\\', '/')
    tDict = json.load(open(path))
    # theme.json的theme的优先级比setTheme中的theme的优先级高
    configTheme = tDict.get('theme')
    if configTheme is None or configTheme == '' or tDict.get(configTheme) is None:
        colorDict = tDict.get(theme)
    else:
        colorDict = tDict.get(configTheme)
    if colorDict is None:
        qss = simple_qss.getDefaultQss()
    else:
        qss = simple_qss.getQss(colorDict['fontLight'], colorDict['fontDark'], colorDict['normal'], colorDict['light'],
                                colorDict['deep'], colorDict['disLight'], colorDict['disDark'], theme)
    qApp.setStyleSheet(qss)
