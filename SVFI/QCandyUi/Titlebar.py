# import win32.win32gui
import win32gui
import win32con
import PyQt5.Qt, PyQt5.QtCore, PyQt5.QtWidgets
from PyQt5.Qt import QSizePolicy
from PyQt5.QtCore import QEvent, QSize, pyqtSlot
from PyQt5.QtWidgets import QLabel, QPushButton, QHBoxLayout, QApplication, QWidget

from .resourse_cfg import *


class Titlebar(QWidget):
    # 默认基础样式参数
    TITLE_TEXT_COLOR = "white"
    BGD_COLOR = "#28AAAA"
    TITLEBAR_HEIGHT = 30
    ICON_SIZE = QSize(20, 20)
    MIN_BUTT_SIZE = QSize(27, 22)
    MAX_BUTT_SIZE = QSize(27, 22)
    CLOSE_BUTT_SIZE = QSize(27, 22)
    TITLE_LABEL_NAME = "Titlebar_titleLabel"
    BACKGROUND_LABEL_NAME = "Titlebar_backgroundLabel"
    MIN_BUTT_NAME = "Titlebar_minimizeButton"
    MAX_BUTT_NAME = "Titlebar_maximizeButton"
    CLOSE_BUTT_NAME = "Titlebar_closeButton"
    THEME_IMG_DIR = 'default'

    def __init__(self, parent):
        super(Titlebar, self).__init__(parent)
        self.parentwidget = parent
        self.setFixedHeight(Titlebar.TITLEBAR_HEIGHT)
        self.m_pBackgroundLabel = QLabel(self)
        self.m_pIconLabel = QLabel(self)
        self.m_pTitleLabel = QLabel(self)
        self.m_pMinimizeButton = QPushButton(self)
        self.m_pMaximizeButton = QPushButton(self)
        self.m_pCloseButton = QPushButton(self)
        self.m_pIconLabel.setFixedSize(Titlebar.ICON_SIZE)
        self.m_pIconLabel.setScaledContents(True)
        self.m_pTitleLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.m_pBackgroundLabel.setObjectName(Titlebar.BACKGROUND_LABEL_NAME)
        # 三大金刚按钮大小
        self.m_pMinimizeButton.setFixedSize(Titlebar.MIN_BUTT_SIZE)
        self.m_pMaximizeButton.setFixedSize(Titlebar.MAX_BUTT_SIZE)
        self.m_pCloseButton.setFixedSize(Titlebar.CLOSE_BUTT_SIZE)
        # 统一设置ObjName
        self.m_pTitleLabel.setObjectName(Titlebar.TITLE_LABEL_NAME)
        self.m_pBackgroundLabel.resize(self.parentwidget.width(), Titlebar.TITLEBAR_HEIGHT)
        self.m_pMinimizeButton.setObjectName(Titlebar.MIN_BUTT_NAME)
        self.m_pMaximizeButton.setObjectName(Titlebar.MAX_BUTT_NAME)
        self.m_pCloseButton.setObjectName(Titlebar.CLOSE_BUTT_NAME)
        # 三大金刚按钮图片设置
        self.setButtonImages()
        # 布局
        pLayout = QHBoxLayout(self)
        pLayout.addWidget(self.m_pIconLabel)
        pLayout.addSpacing(10)
        pLayout.addWidget(self.m_pTitleLabel)
        pLayout.addWidget(self.m_pMinimizeButton)
        pLayout.addWidget(self.m_pMaximizeButton)
        pLayout.addWidget(self.m_pCloseButton)
        pLayout.setSpacing(1)
        pLayout.setContentsMargins(5, 0, 5, 0)
        self.setLayout(pLayout)
        # 信号连接
        self.m_pMinimizeButton.clicked.connect(self.__slot_onclicked)
        self.m_pMaximizeButton.clicked.connect(self.__slot_onclicked)
        self.m_pCloseButton.clicked.connect(self.__slot_onclicked)
        # 设置默认样式(bar的字颜色和背景颜色)
        self.setTitleBarStyle(Titlebar.BGD_COLOR, Titlebar.TITLE_TEXT_COLOR)

    def setMaximumEnable(self, isenable):
        self.m_pMaximizeButton.setEnabled(isenable)

    def setTitleBarStyle(self, backgroundColor, textColor):
        # 标题字体颜色
        # self.m_pTitleLabel.setStyleSheet("font-size:13px;margin-bottom:0px;color:%s;border-top-radius:10px;" % (textColor))
        # self.m_pTitleLabel.setStyleSheet("font-size:13px;margin-bottom:0px;color:%s;border-top-radius:10px;" % (textColor))
        # 标题栏背景颜色
        self.m_pBackgroundLabel.setStyleSheet("border-top-left-radius:10px;border-top-right-radius:5px;")

    def setButtonImages(self):
        # 三大金刚按钮Ui设置
        self.m_pCloseButton.setStyleSheet(
            self.__getButtonImgQss(IMAGE_ROOT + Titlebar.THEME_IMG_DIR + "/", IMG_CLOSE_NORM, IMG_CLOSE_HOVER, IMG_CLOSE_PRESS, IMG_CLOSE_PRESS))
        self.m_pMinimizeButton.setStyleSheet(
            self.__getButtonImgQss(IMAGE_ROOT + Titlebar.THEME_IMG_DIR + "/", IMG_MIN_NORM, IMG_MIN_HOVER, IMG_MIN_PRESS, IMG_MIN_PRESS))
        self.m_pMaximizeButton.setStyleSheet(
            self.__getButtonImgQss(IMAGE_ROOT + Titlebar.THEME_IMG_DIR + "/", IMG_MAX_NORM, IMG_MAX_HOVER, IMG_MAX_PRESS, IMG_MAX_PRESS))

    def __getButtonImgQss(self, root, norm, hover, press, disable):
        qss = str()
        qss += "QPushButton{background:transparent; background-image:url(%s); border:none}" % (
            root + norm)
        qss += "QPushButton:hover{background:transparent; background-image:url(%s)}" % (
            root + hover)
        qss += "QPushButton:pressed{background:transparent; background-image:url(%s)}" % (
            root + press)
        qss += "QPushButton:disabled{background:transparent; background-image:url(%s)}" % (
            root + disable)
        return qss



    def mouseDoubleClickEvent(self, e):
        if self.m_pMaximizeButton.isEnabled():
            self.m_pMaximizeButton.clicked.emit()  # 双击全屏

    def mousePressEvent(self, e):
        """
        使窗口能被拖动
        :param e:
        :return:
        """
        win32gui.ReleaseCapture()
        pWindow = self.window()
        if pWindow.isWindow():
            win32gui.SendMessage(pWindow.winId(), win32con.WM_SYSCOMMAND, win32con.SC_MOVE + win32con.HTCAPTION, 0)
        e.ignore()

    def eventFilter(self, object, e):
        if e.type() == QEvent.WindowTitleChange:
            if object != None:
                self.m_pTitleLabel.setText(object.windowTitle())
                return True
        if e.type() == QEvent.WindowIconChange:
            if object != None:
                icon = object.windowIcon()
                self.m_pIconLabel.setPixmap(icon.pixmap(self.m_pIconLabel.size()))
                return True
        if e.type() == QEvent.Resize:
            self.__updateMaxmize()
            self.__setTitleBarSize(self.parentwidget.width())
            return True
        # 注意!这里self要加上!!!!!!!!!
        return QWidget.eventFilter(self, object, e)

    @pyqtSlot()
    def __slot_onclicked(self):
        pButton = self.sender()
        pWindow = self.window()
        if pWindow.isWindow():
            if pButton.objectName() == Titlebar.MIN_BUTT_NAME:
                pWindow.showMinimized()
            elif pButton.objectName() == Titlebar.MAX_BUTT_NAME:
                if pWindow.isMaximized():
                    pWindow.showNormal()
                    self.m_pMaximizeButton.setStyleSheet(self.__getButtonImgQss(IMAGE_ROOT + Titlebar.THEME_IMG_DIR + "/", IMG_MAX_NORM, IMG_MAX_HOVER, IMG_MAX_PRESS, IMG_MAX_PRESS))
                else:
                    pWindow.showMaximized()
                    self.m_pMaximizeButton.setStyleSheet(self.__getButtonImgQss(IMAGE_ROOT + Titlebar.THEME_IMG_DIR + "/", IMG_RESIZE_NORM, IMG_RESIZE_HOVER, IMG_RESIZE_PRESS, IMG_RESIZE_PRESS))
            elif pButton.objectName() == Titlebar.CLOSE_BUTT_NAME:
                pWindow.close()
                os._exit(0)

    def __updateMaxmize(self):
        pWindow = self.window()
        if pWindow.isWindow() == True:
            bMaximize = pWindow.isMaximized()
            if bMaximize == 0:
                self.m_pMaximizeButton.setProperty("maximizeProperty", "restore")
            else:
                self.m_pMaximizeButton.setProperty("maximizeProperty", "maximize")
                self.m_pMaximizeButton.setStyle(QApplication.style())

    def __setTitleBarSize(self, width):
        self.m_pBackgroundLabel.resize(width, Titlebar.TITLEBAR_HEIGHT)
