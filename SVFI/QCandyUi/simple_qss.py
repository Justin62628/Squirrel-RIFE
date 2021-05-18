from . import qss_getter as Qss
from .Titlebar import Titlebar
from .resourse_cfg import *


def getDefaultQss():
    """
    默认主题
    :return:
    """
    # fontLight, fontDark, normal, light, deep, disLight, disDark
    return getQss(Qss.WHITE, Qss.DEEPBLUEGREEN, Qss.BLUEGREEN, Qss.LIGHTGREEN, Qss.DARKBLUEGREEN, Qss.LIGHTGRAY, Qss.GRAY, "default")


def getQss(fontLight, fontDark, normal, light, deep, disLight, disDark, themeImgDir):
    """
    通用组件的Qss + CandyBar的Qss
    :param fontLight:
    :param fontDark:
    :param normal:
    :param light:
    :param deep:
    :param disLight:
    :param disDark:
    :param themeImgDir:
    :return:
    """
    themeImgDir = themeImgDir if os.path.isdir(IMAGE_ROOT + themeImgDir) else 'default'
    qss = str()
    qss += __getWidgetsQss(fontLight, fontDark, normal, light, deep, disLight, disDark, themeImgDir)
    qss += __getCandyQss(fontLight, deep, fontLight, themeImgDir)
    return qss


def __getWidgetsQss(fontLight, fontDark, normal, light, deep, disLight, disDark, themeImgDir):
    """
    通用组件(Widgets)的Qss
    :param fontLight:
    :param fontDark:
    :param normal:
    :param light:
    :param deep:
    :param disLight:
    :param disDark:
    :param themeImgDir:
    :return:
    """
    qss = str()
    qss += Qss.getFontQss("微软雅黑", fontDark)
    qss += Qss.getPushButtonQss(normal, fontLight, light, normal, disLight, fontLight, disDark, disLight)
    qss += Qss.getPlaineditQss(disLight, normal)
    qss += Qss.getTextBrowerQss(disLight, normal)
    qss += Qss.getLineeditQss(disLight, normal)
    qss += Qss.getComboxQss(fontLight, disLight, normal, IMAGE_ROOT + themeImgDir + "/" + "down_arrow.png")
    img_norm = IMAGE_ROOT + themeImgDir + "/" + "radio_normal.png"
    img_down = IMAGE_ROOT + themeImgDir + "/" + "radio_down.png"
    img_hover = IMAGE_ROOT + themeImgDir + "/" + "radio_hoverUncheck.png"
    img_downhover = IMAGE_ROOT + themeImgDir + "/" + "radio_hoverCheck.png"
    qss += Qss.getRadioButtonQss(img_norm, img_down, img_hover, img_downhover)
    img_norm = IMAGE_ROOT + themeImgDir + "/" + "checkbox_normal.png"
    img_down = IMAGE_ROOT + themeImgDir + "/" + "checkbox_down.png"
    img_hover = IMAGE_ROOT + themeImgDir + "/" + "checkbox_hoverUncheck.png"
    img_downhover = IMAGE_ROOT + themeImgDir + "/" + "checkbox_hoverCheck.png"
    qss += Qss.getCheckBoxQss(img_norm, img_down, img_hover, img_downhover)
    qss += Qss.getTabWidgetQss(normal, fontLight, normal)
    qss += Qss.getSliderQss(normal, fontLight, normal)
    qss += Qss.getScrollbarQss(normal, IMAGE_ROOT + themeImgDir + "/" + "down_arrow.png")
    return qss


def __getCandyQss(barTextColor, barColor, winBgdColor, themeImgDir):
    """
    TitleBar+CandyWindow的Qss
    :param barTextColor: 文字颜色
    :param barColor: bar主体颜色
    :param winBgdColor: 主体窗口背景颜色
    :param themeImgDir: 主题名(作用主要是为了找按钮图片)
    :return: qss
    """
    Titlebar.THEME_IMG_DIR = themeImgDir
    qss = str()
    qss += "Titlebar QLabel#%s{font-size:13px;margin-bottom:0px;color:%s;}" % (Titlebar.TITLE_LABEL_NAME, barTextColor)
    qss += "Titlebar QLabel#%s{background:%s;}" % (Titlebar.BACKGROUND_LABEL_NAME, barColor)
    # 三大金刚键的图片设置 (最大化恢复正常大小的图片设置只能在Title的onclick中设置)
    qss += "Titlebar QPushButton#%s{background:transparent; background-image:url(%s); border:none}" % \
           (Titlebar.MIN_BUTT_NAME, IMAGE_ROOT + themeImgDir + "/" + IMG_MIN_NORM)
    qss += "Titlebar QPushButton#%s:hover{background:transparent; background-image:url(%s)}" % \
           (Titlebar.MIN_BUTT_NAME, IMAGE_ROOT + themeImgDir + "/" + IMG_MIN_HOVER)
    qss += "Titlebar QPushButton#%s:pressed{background:transparent; background-image:url(%s)}" % \
           (Titlebar.MIN_BUTT_NAME, IMAGE_ROOT + themeImgDir + "/" + IMG_MIN_PRESS)
    qss += "Titlebar QPushButton#%s:disabled{background:transparent; background-image:url(%s)}" % \
           (Titlebar.MIN_BUTT_NAME, IMAGE_ROOT + themeImgDir + "/" + IMG_MIN_PRESS)
    qss += "Titlebar QPushButton#%s{background:transparent; background-image:url(%s); border:none}" % \
           (Titlebar.MAX_BUTT_NAME, IMAGE_ROOT + themeImgDir + "/" + IMG_MAX_NORM)
    qss += "Titlebar QPushButton#%s:hover{background:transparent; background-image:url(%s)}" % \
           (Titlebar.MAX_BUTT_NAME, IMAGE_ROOT + themeImgDir + "/" + IMG_MAX_HOVER)
    qss += "Titlebar QPushButton#%s:pressed{background:transparent; background-image:url(%s)}" % \
           (Titlebar.MAX_BUTT_NAME, IMAGE_ROOT + themeImgDir + "/" + IMG_MAX_PRESS)
    qss += "Titlebar QPushButton#%s:disabled{background:transparent; background-image:url(%s)}" % \
           (Titlebar.MAX_BUTT_NAME, IMAGE_ROOT + themeImgDir + "/" + IMG_MAX_PRESS)
    qss += "Titlebar QPushButton#%s{background:transparent; background-image:url(%s); border:none}" % \
           (Titlebar.CLOSE_BUTT_NAME, IMAGE_ROOT + themeImgDir + "/" + IMG_CLOSE_NORM)
    qss += "Titlebar QPushButton#%s:hover{background:transparent; background-image:url(%s)}" % \
           (Titlebar.CLOSE_BUTT_NAME, IMAGE_ROOT + themeImgDir + "/" + IMG_CLOSE_HOVER)
    qss += "Titlebar QPushButton#%s:pressed{background:transparent; background-image:url(%s)}" % \
           (Titlebar.CLOSE_BUTT_NAME, IMAGE_ROOT + themeImgDir + "/" + IMG_CLOSE_PRESS)
    qss += "Titlebar QPushButton#%s:disabled{background:transparent; background-image:url(%s)}" % \
           (Titlebar.CLOSE_BUTT_NAME, IMAGE_ROOT + themeImgDir + "/" + IMG_CLOSE_PRESS)
    # CandyWindow窗口内底色+外围描边
    qss += "WindowWithTitleBar{background:%s;border:3px solid %s}" % (winBgdColor, barColor)
    return qss
