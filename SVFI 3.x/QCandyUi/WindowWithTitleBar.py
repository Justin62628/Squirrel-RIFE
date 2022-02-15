"""

    本类使用的例子在下面, WindowWithTitleBar那里

"""

from PyQt5.QtGui import *
# from PyQt5.QtWidgets import QVBoxLayout, QWidget, QFrame, QLabel, QGraphicsOpacityEffect
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from .Titlebar import Titlebar

# Stretch State
NO_SELECT = 0,  # 鼠标未进入下方矩形区域
LEFT_TOP_RECT = 1  # 鼠标在左上角区域
TOP_BORDER = 2  # 鼠标在上边框区域
RIGHT_TOP_RECT = 3  # 鼠标在右上角区域
RIGHT_BORDER = 4  # 鼠标在右边框区域
RIGHT_BOTTOM_RECT = 5  # 鼠标在右下角区域
BOTTOM_BORDER = 6  # 鼠标在下边框区域
LEFT_BOTTOM_RECT = 7  # 鼠标在左下角区域
LEFT_BORDER = 8  # 鼠标在左边框区域
# 感知拉伸区域
STRETCH_RECT_WIDTH = 20
STRETCH_RECT_HEIGHT = 10

class WindowWithTitleBar(QFrame):
    def __init__(self, mainwidget, parent=0):
        super(WindowWithTitleBar, self).__init__()
        self.setObjectName('WindowWithTitleBar')
        self.m_titlebar = Titlebar(self)
        self.initWidgetsAndPack(mainwidget, self.m_titlebar)
        self.initStretch()
        self.initTipLabel(mainwidget)
        # 设置 窗口无边框和背景透明 *必须
        # self.setAttribute(Qt.WA_TranslucentBackground)
        # self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.border_width = 20

    def initTipLabel(self, parent):
        """
        消息提示标签
        :param parent:
        :return:
        """
        self.tipLabel = QLabel(parent)
        self.tipLabel.setFixedSize(200, 40)
        self.tipLabel.setAlignment(Qt.AlignCenter)
        self.tipLabel.setWordWrap(True)
        self.tipLabel.setGeometry(self.width() / 2 - self.tipLabel.width() / 2, self.tipLabel.y(),
                                  self.tipLabel.width(), self.tipLabel.height())
        self.tipLabel.hide()

    def showTip(self, text, color='#20c3ff'):
        if self.tipLabel is not None:
            self.tipLabel.show()
            self.tipLabel.setStyleSheet(
                "QLabel{background: %s;border:5px; color: #FFFFFF; border-radius: 5px}" % color)
            self.tipLabel.setText(text)
            eff = QGraphicsOpacityEffect(self)
            self.tipLabel.setGraphicsEffect(eff)
            self.animate = QPropertyAnimation(eff, "opacity")
            self.animate.setDuration(2000)
            self.animate.setStartValue(0.8)
            self.animate.setEndValue(0)
            self.animate.setEasingCurve(QEasingCurve.InCubic)
            self.animate.finished.connect(lambda: self.tipLabel.hide())
            self.animate.start()

    def initWidgetsAndPack(self, mainwidget, titlebar):
        """
        将主体Widget和titleBar拼装起来
        :param mainwidget:
        :param titlebar:
        :return:
        """
        mainwidget.setStyleSheet("QMainWindow{border-bottom-left-radius:5px;border-bottom-right-radius:5px}")
        self.mainwidget = mainwidget
        self.resize(mainwidget.width(), mainwidget.height() + Titlebar.TITLEBAR_HEIGHT)
        self.setWindowFlags(Qt.FramelessWindowHint | self.windowFlags())
        self.installEventFilter(titlebar)
        # 布局: titlbar在上主窗体在下
        pLayout = QVBoxLayout(self)
        pLayout.addWidget(titlebar)
        pLayout.addWidget(mainwidget)
        pLayout.setSpacing(0)  # 排列的几个widget为0间隔
        pLayout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(pLayout)


    def initStretch(self):
        """
        初始化拉伸功能
        :return:
        """
        self.setSupportStretch(True)
        self.m_isWindowMax = False
        self.m_stretchRectState = NO_SELECT
        self.m_isMousePressed = False
        self.setMinimumSize(self.mainwidget.minimumWidth(), self.mainwidget.minimumHeight() + Titlebar.TITLEBAR_HEIGHT)

    def getTitbar(self):
        return self.m_titlebar

    def setMinimumSize(self, width, height):
        """
        设置拉伸的最小Size
        :param width:
        :param height:
        :return:
        """
        self.m_windowMinWidth = width
        self.m_windowMinHeight = height
        super(WindowWithTitleBar, self).setMinimumSize(width, height)

    def setWindowRadius(self, n_px):
        """
        圆边
        :param n_px: 弧度
        :return:
        """
        objBitmap = QBitmap(self.size())
        painter = QPainter(objBitmap)
        painter.setBrush(QColor(0, 0, 0))
        painter.drawRoundedRect(self.rect(), n_px, n_px)
        self.setMask(objBitmap)

    def setBackgroundBorderColor(self, bgdcolor, bordercolor):
        self.setStyleSheet("WindowWithTitleBar{background:%s;border:6px solid %s;border-radius:5px;}" % (bgdcolor, bordercolor))

    def closeEvent(self, *args, **kwargs):
        self.mainwidget.close()

    def showEvent(self, event):
        self.calculateCurrentStrechRect()
        return super().showEvent(event)

    def calculateCurrentStrechRect(self):
        # 四个角Rect
        self.m_leftTopRect = QRect(0, 0, STRETCH_RECT_WIDTH, STRETCH_RECT_HEIGHT)
        self.m_leftBottomRect = QRect(0, self.height() - STRETCH_RECT_HEIGHT, STRETCH_RECT_WIDTH, STRETCH_RECT_WIDTH)
        self.m_rightTopRect = QRect(self.width() - STRETCH_RECT_WIDTH, 0, STRETCH_RECT_WIDTH, STRETCH_RECT_HEIGHT)
        self.m_rightBottomRect = QRect(self.width() - STRETCH_RECT_WIDTH, self.height() - STRETCH_RECT_HEIGHT,
                                       STRETCH_RECT_WIDTH, STRETCH_RECT_HEIGHT)
        # 四条边Rect
        self.m_topBorderRect = QRect(STRETCH_RECT_WIDTH, 0, self.width() - STRETCH_RECT_WIDTH * 2, STRETCH_RECT_HEIGHT)
        self.m_rightBorderRect = QRect(self.width() - STRETCH_RECT_WIDTH, STRETCH_RECT_HEIGHT, STRETCH_RECT_WIDTH,
                                       self.height() - STRETCH_RECT_HEIGHT * 2)
        self.m_bottomBorderRect = QRect(STRETCH_RECT_WIDTH, self.height() - STRETCH_RECT_HEIGHT,
                                        self.width() - STRETCH_RECT_WIDTH * 2, STRETCH_RECT_HEIGHT)
        self.m_leftBorderRect = QRect(0, STRETCH_RECT_HEIGHT, STRETCH_RECT_WIDTH,
                                      self.height() - STRETCH_RECT_HEIGHT * 2)

    def getCurrentStretchState(self, cursorPos):
        """
        根据鼠标的位置获取StretchState
        :param cursorPos:
        :return:
        """
        if self.m_leftTopRect.contains(cursorPos):
            stretchState = LEFT_TOP_RECT
        elif self.m_rightTopRect.contains(cursorPos):
            stretchState = RIGHT_TOP_RECT
        elif self.m_rightBottomRect.contains(cursorPos):
            stretchState = RIGHT_BOTTOM_RECT
        elif self.m_leftBottomRect.contains(cursorPos):
            stretchState = LEFT_BOTTOM_RECT
        elif self.m_topBorderRect.contains(cursorPos):
            stretchState = TOP_BORDER
        elif self.m_rightBorderRect.contains(cursorPos):
            stretchState = RIGHT_BORDER
        elif self.m_bottomBorderRect.contains(cursorPos):
            stretchState = BOTTOM_BORDER
        elif self.m_leftBorderRect.contains(cursorPos):
            stretchState = LEFT_BORDER
        else:
            stretchState = NO_SELECT
        return stretchState

    def updateMouseStyle(self, stretchState):
        """
        根据stretchState刷新鼠标的样式
        :param stretchState:
        :return:
        """
        if stretchState == NO_SELECT:
            self.setCursor(Qt.ArrowCursor)
        elif stretchState == LEFT_TOP_RECT:
            self.setCursor(Qt.SizeFDiagCursor)
        elif stretchState == RIGHT_BOTTOM_RECT:
            self.setCursor(Qt.SizeFDiagCursor)
        elif stretchState == TOP_BORDER:
            self.setCursor(Qt.SizeVerCursor)
        elif stretchState == BOTTOM_BORDER:
            self.setCursor(Qt.SizeVerCursor)
        elif stretchState == RIGHT_TOP_RECT:
            self.setCursor(Qt.SizeBDiagCursor)
        elif stretchState == LEFT_BOTTOM_RECT:
            self.setCursor(Qt.SizeBDiagCursor)
        elif stretchState == LEFT_BORDER:
            self.setCursor(Qt.SizeHorCursor)
        elif stretchState == RIGHT_BORDER:
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def mouseMoveEvent(self, event):
        """
        重写mouseMoveEvent事件，用于获取当前鼠标的位置，将位置传递给getCurrentStretchState方法，
        得到当前鼠标的状态，然后调用updateMouseStyle对鼠标的样式进行更新
        :param event:
        :return:
        """
        # 如果窗口最大化是不能拉伸的
        # 也不用更新鼠标样式
        if (self.m_isWindowMax):
            return super().mouseMoveEvent(event)
        # 如果当前鼠标未按下，则根据当前鼠标的位置更新鼠标的状态及样式
        if not self.m_isMousePressed:
            cursorPos = event.pos()
            # 根据当前鼠标的位置显示不同的样式
            self.m_stretchRectState = self.getCurrentStretchState(cursorPos)
            self.updateMouseStyle(self.m_stretchRectState)
        # 如果当前鼠标左键已经按下，则记录下第二个点的位置，并更新窗口的大小
        else:
            self.m_endPoint = self.mapToGlobal(event.pos())
            self.updateWindowSize()
        return super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        # 当前鼠标进入了以上指定的8个区域，并且是左键按下时才开始进行窗口拉伸
        self.m_isMousePressed = False
        if (self.m_stretchRectState != NO_SELECT and event.button() == Qt.LeftButton):
            self.m_isMousePressed = True
            # 记录下当前鼠标位置，为后面计算拉伸位置
            self.m_startPoint = self.mapToGlobal(event.pos())
            # 保存下拉伸前的窗口位置及大小
            self.m_windowRectBeforeStretch = QRect(self.geometry().x(), self.geometry().y(), self.geometry().width(),
                                                   self.geometry().height())
        # else:
        #     self.windowEffect.moveWindow(self.winId())
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """
        鼠标松开后意味之窗口拉伸结束，置标志位，并且重新计算用于拉伸的8个区域Rect
        :param event:
        :return:
        """
        self.m_isMousePressed = False
        self.calculateCurrentStrechRect()
        return super().mouseReleaseEvent(event)

    def updateWindowSize(self):
        """
        拉伸窗口过程中，根据记录的坐标更新窗口大小
        :return:
        """
        windowRect = QRect(self.m_windowRectBeforeStretch.x(), self.m_windowRectBeforeStretch.y(),
                           self.m_windowRectBeforeStretch.width(), self.m_windowRectBeforeStretch.height())
        delValue_X = self.m_startPoint.x() - self.m_endPoint.x()
        delValue_Y = self.m_startPoint.y() - self.m_endPoint.y()
        if self.m_stretchRectState == LEFT_BORDER:
            topLeftPoint = windowRect.topLeft()
            topLeftPoint.setX(topLeftPoint.x() - delValue_X)
            windowRect.setTopLeft(topLeftPoint)
        elif self.m_stretchRectState == RIGHT_BORDER:
            bottomRightPoint = windowRect.bottomRight()
            bottomRightPoint.setX(bottomRightPoint.x() - delValue_X)
            windowRect.setBottomRight(bottomRightPoint)
        elif self.m_stretchRectState == TOP_BORDER:
            topLeftPoint = windowRect.topLeft()
            topLeftPoint.setY(topLeftPoint.y() - delValue_Y)
            windowRect.setTopLeft(topLeftPoint)
        elif self.m_stretchRectState == BOTTOM_BORDER:
            bottomRightPoint = windowRect.bottomRight()
            bottomRightPoint.setY(bottomRightPoint.y() - delValue_Y)
            windowRect.setBottomRight(bottomRightPoint)
        elif self.m_stretchRectState == LEFT_TOP_RECT:
            topLeftPoint = windowRect.topLeft()
            topLeftPoint.setX(topLeftPoint.x() - delValue_X)
            topLeftPoint.setY(topLeftPoint.y() - delValue_Y)
            windowRect.setTopLeft(topLeftPoint)
        elif self.m_stretchRectState == RIGHT_TOP_RECT:
            topRightPoint = windowRect.topRight()
            topRightPoint.setX(topRightPoint.x() - delValue_X)
            topRightPoint.setY(topRightPoint.y() - delValue_Y)
            windowRect.setTopRight(topRightPoint)
        elif self.m_stretchRectState == RIGHT_BOTTOM_RECT:
            bottomRightPoint = windowRect.bottomRight()
            bottomRightPoint.setX(bottomRightPoint.x() - delValue_X)
            bottomRightPoint.setY(bottomRightPoint.y() - delValue_Y)
            windowRect.setBottomRight(bottomRightPoint)
        elif self.m_stretchRectState == LEFT_BOTTOM_RECT:
            bottomLeftPoint = windowRect.bottomLeft()
            bottomLeftPoint.setX(bottomLeftPoint.x() - delValue_X)
            bottomLeftPoint.setY(bottomLeftPoint.y() - delValue_Y)
            windowRect.setBottomLeft(bottomLeftPoint)
        # 避免宽或高为零窗口显示有误，这里给窗口设置最小拉伸高度、宽度
        if windowRect.width() < self.m_windowMinWidth:
            windowRect.setLeft(self.geometry().left())
            windowRect.setWidth(self.m_windowMinWidth)
        if windowRect.height() < self.m_windowMinHeight:
            windowRect.setTop(self.geometry().top())
            windowRect.setHeight(self.m_windowMinHeight)
        self.setGeometry(windowRect)

    def setSupportStretch(self, isSupportStretch):
        """
        设置当前窗口是否支持拉伸
        :param isSupportStretch:
        :return:
        """
        # 因为需要在鼠标未按下的情况下通过mouseMoveEvent事件捕捉鼠标位置，所以需要设置setMouseTracking为True（如果窗口支持拉伸）
        self.m_isSupportStretch = isSupportStretch
        self.setMouseTracking(isSupportStretch)
        # 这里对子控件也进行了设置，是因为如果不对子控件设置，当鼠标移动到子控件上时，不会发送mouseMoveEvent事件，也就获取不到当前鼠标位置，无法判断鼠标状态及显示样式了。
        widgetList = self.findChildren(QWidget)
        for widget in widgetList:
            widget.setMouseTracking(isSupportStretch)
        if (self.m_titlebar is not None):
            # titleBar同理,也需要对自己及子控件进行调用setMouseTracking进行设置，见上方注释
            # self.titleBar.setSupportStretch(isSupportStretch)
            pass

    def getSupportStretch(self):
        """
        返回当前窗口是否支持拉伸
        :return:
        """
        return self.m_isSupportStretch

    def setMaxEnable(self, isEnable):
        """
        最大化开闭
        :param isEnable
        """
        self.m_titlebar.setMaximumEnable(isEnable)

    def paintEvent(self, event):
        return
        path = QPainterPath()
        path.setFillRule(Qt.WindingFill)

        pat = QPainter(self)
        pat.setRenderHint(pat.Antialiasing)
        pat.fillPath(path, QBrush(Qt.white))

        color = QColor(192, 192, 192, 50)

        for i in range(10):
            i_path = QPainterPath()
            i_path.setFillRule(Qt.WindingFill)
            ref = QRectF(10-i, 10-i, self.width()-(10-i)*2, self.height()-(10-i)*2)
            # i_path.addRect(ref)
            i_path.addRoundedRect(ref, self.border_width, self.border_width)
            color.setAlpha(150 - i**0.5*50)
            pat.setPen(color)
            pat.drawPath(i_path)

        # 圆角
        pat2 = QPainter(self)
        pat2.setRenderHint(pat2.Antialiasing)  # 抗锯齿
        pat2.setBrush(Qt.white)
        pat2.setPen(Qt.transparent)

        rect = self.rect()
        rect.setLeft(9)
        rect.setTop(9)
        rect.setWidth(rect.width()-9)
        rect.setHeight(rect.height()-9)
        pat2.drawRoundedRect(rect, 4, 4)
