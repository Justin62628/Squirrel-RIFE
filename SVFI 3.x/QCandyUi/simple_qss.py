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
    # qss += __getWidgetsQss(fontLight, fontDark, normal, light, deep, disLight, disDark, themeImgDir)
    # qss += __getCandyQss(fontLight, deep, fontLight, themeImgDir)
    qss += __getAllQss(fontLight, fontDark, normal, light, deep, disLight, disDark, themeImgDir)
    return qss

def __getAllQss(fontLight, fontDark, normal, light, deep, disLight, disDark, themeImgDir):
    """
    全部的Qss
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

    img_dir = os.path.join(IMAGE_ROOT, themeImgDir)

    """Global"""
    qss = f"""
    QWidget,
    QWidget:window,
    QStackedWidget,
    QPushButton,
    QRadioButton,
    QCheckBox,
    QGroupBox,
    QStatusBar,
    QToolButton,
    QComboBox,
    QDialog,
    QTabBar,
    QScrollBar,
    QScrollArea {{
        font-family: 'Segoe UI';
        background: {deep};
        /*font color*/
    }}
    
    QObject {{
        font-family: 微软雅黑;
        color: {normal};
        font-size: 14px
    }}
    """



    """QMenu"""
    qss += f"""
    QMenuBar {{
        spacing: 5px;
        /* spacing between menu bar items */
    }}
    
    QMenuBar::item {{
        background: transparent;
    }}
    
    QMenuBar::item:selected {{
        background: {light};
    }}
    
    QMenuBar::item:pressed {{
        border: 2px solid {light};
        background: {light};
    }}
    
    QMenu {{
        background: {disDark};
        padding: 8px 1px;
        /*设置菜单项文字上下和左右的内边距，效果就是菜单中的条目左右上下有了间隔*/
        margin: 2px 2px;
        /*设置菜单项的外边距*/
        border: 2px {disDark};
        border-radius: 4px;
    }}
    
    QMenu::item {{
        font-family: 'Microsoft Yahei';
        padding: 8px;
    }}
    
    QMenu::item:selected {{
        background: {light};
        border-radius: 4px;
    }}
    """

    """QPushButton"""
    qss += f"""
    QPushButton {{
        padding: 3px;
        border-radius: 5px;
        color: {normal};
        background: {light};
        border: 2px solid {light};
        font: bold;
    }}
    
    QPushButton:hover {{
        border: {light};
        color: {light};
        background: {fontLight};
    }}
    
    QPushButton:pressed {{
        color: {normal};
        background: #CCCCCC;
    }}
    
    QPushButton:disabled {{
        border: #CCCCCC;
        color: #CCCCCC;
        background: {fontDark};
    }}
    """
    custom_button_qss_list = ["RefreshStartInfo", "OutputSettingsButton", "InputDirButton", "OutputButton",
                              "InputButton", "ClearInputButton", "RemoveTemplateButton", "AddTemplateButton"]
    for q in custom_button_qss_list:
        qss += f"""
        QPushButton#{q} {{
            background: {disDark};
            border: 2px solid {disDark};
            font: light;
        }}
        
        QPushButton#{q}:hover {{
            background: {fontLight};
            border: {light};
            font: light;
        }}
        
        """

    """QLabel"""
    qss += f"""
        QLabel {{
            background: transparent
        }}
        """
    custom_label_qss_list = ["EncodeSettingsLabel", "RenderSettingsLabel", "AMDSettingsLabel", "AdvanceSettingsLabel",
                              "OutputLabel", "InputLabel"]
    for q in custom_label_qss_list:
        qss += f"""
            QLabel#{q} {{
                font: bold;
                font-size: 22px;
                font-weight: normal;
            }}
            """

    """QProgressBar"""
    qss += f"""
    QProgressBar {{
        text-align: center;
        border: 2px solid {light};
        border-radius: 5px;
        selection-color: {disDark};
        selection-background-color: {fontLight};
    }}
    """

    """QPlainTextEdit"""
    qss += f"""
    QPlainTextEdit {{
        border-style: none;
        padding: 2px;
        border-radius: 10px;
        border: 2px solid #CCCCCC;
        font-family: 微软雅黑;
        selection-color: {disDark};
        selection-background-color: {light}
    }}
    
    QPlainTextEdit:focus {{
        border: 2px solid {light};
    }}
    """

    """QGroupBox"""
    qss += f"""
    QGroupBox {{
        background: {disLight};
    
        /* 边框 */
        border: 0px solid gray;
    
        /* 倒角 */
        border-radius: 5px;
    
        /* 就像墙上挂着的两个相框,margin指的是相框与相框的距离
                       padding指的是每个相框里照片与相框边框的距离
                    margin-top: 20px;
                    padding-top: 10px;*/
        font-size: 18px;
    
    }}
    
    /* 标题设置 */
    QGroupBox::title {{
        /* 位置 */
        subcontrol-origin: margin;
        subcontrol-position: top left;
        /* 内边框,上下和左右 */
        padding: 10px 15px;
    
    }}
    """

    """QToolBox"""
    qss += f"""
    QToolBox::tab {{
        background: {disLight};
        border-radius: 5px;
        color: darkgray;
    }}
    
    QToolBox::tab:selected {{
        /* italicize selected tabs */
        font: bold;
        color: {normal};
    }}
    """

    """QTextBrowser"""
    qss += f"""
    QTextBrowser {{
        border-style: none;
        padding: 2px;
        border-radius: 5px;
        border: 2px solid #CCCCCC;
        selection-color: {disDark};
        selection-background-color: {light}
    }}
    
    QTextBrowser:focus {{
        border: 2px solid {light};
    }}
    """

    """QTextEdit"""
    qss += f"""
    QTextEdit {{
        padding: 2px;
        border-radius: 5px;
        border: 2px solid {disDark};
    }}
    
    QTextEdit:focus {{
        border: 2px solid {light};
    }}
    
    QTextEdit:hover {{
        border: 2px solid {light};
    }}
    """

    """QList"""
    qss += f"""
    QListWidget,
    QScrollArea {{
        border-style: none;
        padding: 2px;
        border-radius: 5px;
        border: 2px solid {disDark};
        selection-color: {disDark};
        selection-background-color: {light};
        background: transparent
    }}
    
    QListWidget::item:selected {{
        border-left: 3px solid {light};
    }}
    
    """

    """QLineEdit"""
    qss += f"""
    QLineEdit {{
        border-style: none;
        padding: 2px;
        border-radius: 5px;
        border: 2px solid {disDark};
        selection-color: {disDark};
        selection-background-color: {light};
    }}
    
    QLineEdit:focus {{
        border: 2px solid {light};
    }}
    
    QLineEdit:disabled {{
        color: {disDark};
    }}
    """

    """QComboBox"""
    qss += f"""
    QComboBox {{
        background: {deep};
        padding: 2px;
        border-radius: 5px;
        border: 2px solid {disDark};
    }}
    
    QComboBox:focus {{
        border: 2px solid {light};
    }}
    
    QComboBox:on {{
        border: 2px solid {light};
    }}
    
    QComboBox:disabled {{
        color: {disDark};
    }}
    
    QComboBox::drop-down {{
        border-style: solid;
        border-radius: 5px;
    }}
    
    QComboBox QAbstractItemView {{
        border: 2px solid {light};
        background: transparent;
        selection-background-color: {light};
    }}
    
    QComboBox::down-arrow {{
        image: url({os.path.join(img_dir, "down_arrow.png")});
        border: 20px;
    }}
    """

    """QSpinBox"""
    qss += f"""
    QSpinBox {{
        background: transparent;
        padding: 2px;
        border-radius: 5px;
        border: 2px solid {disDark};
    }}
    
    QSpinBox::up-arrow {{
        image: url({os.path.join(img_dir, "up_arrow.png")});
        border: 20px;
        background: transparent;
    }}
    
    QSpinBox::down-arrow {{
        image: url({os.path.join(img_dir, "down_arrow.png")});
        border: 20px;
        background: transparent;
    }}
    
    QSpinBox::hover {{
        border: 2px solid {light};
    }}
    
    QSpinBox::disabled {{
        color: {disDark};
        border-color: {disDark}
    }}
    
    """

    """QDoubleSpinBox"""
    qss += f"""
    QDoubleSpinBox {{
        background: transparent;
        padding: 2px;
        border-radius: 5px;
        border: 2px solid {disDark};
    }}
    
    QDoubleSpinBox::up-arrow {{
        image: url({os.path.join(img_dir, "up_arrow.png")});
        border: 20px;
        background: transparent;
    }}
    
    QDoubleSpinBox::down-arrow {{
        image: url({os.path.join(img_dir, "down_arrow.png")});
        border: 20px;
        background: transparent;
    }}
    
    QDoubleSpinBox::hover {{
        border: 2px solid {light};
    }}
    
    QDoubleSpinBox::disabled {{
        color: {disDark};
        border-color: {disDark}
    }}
    
    """

    """QTimeEdit"""
    qss += f"""
    QTimeEdit {{
        background: transparent;
        padding: 2px;
        border-radius: 5px;
        border: 2px solid {disDark};
    }}
    
    QTimeEdit::up-arrow {{
        image: url({os.path.join(img_dir, "up_arrow.png")});
        border: 20px;
        background: transparent;
    }}
    
    QTimeEdit::down-arrow {{
        image: url({os.path.join(img_dir, "down_arrow.png")});
        border: 20px;
        background: transparent;
    }}
    
    QTimeEdit::hover {{
        border: 2px solid {light};
    }}
    
    QTimeEdit::disabled {{
        color: {disDark};
        border-color: {disDark}
    }}
    """

    """QButtonIcon"""
    qss += f"""
    
    QPushButton#AllInOne {{
        icon: url({os.path.join(img_dir, "icons8-ok-50.png")})
    }}
    
    QPushButton#StartExtractButton {{
        icon: url({os.path.join(img_dir, "icons8-image-file-50.png")})
    }}
    
    QPushButton#StartRenderButton {{
        icon: url({os.path.join(img_dir, "icons8-file-50.png")})
    }}
    
    QPushButton#ClearInputButton {{
        icon: url({os.path.join(img_dir, "icons8-trash-50.png")})
    }}
    
    QPushButton#ShowAdvance {{
        icon: url({os.path.join(img_dir, "icons8-settings-50.png")})
    }}
    
    QPushButton#AutoSet {{
        icon: url({os.path.join(img_dir, "icons8-idea-50.png")})
    }}
    
    QRadioButton {{
        background: transparent;
    }}
    
    QRadioButton::indicator:unchecked {{
        image: url({os.path.join(img_dir, "radio_normal.png")});
    }}
    
    QRadioButton::indicator:checked {{
        image: url({os.path.join(img_dir, "radio_down.png")});
    }}
    
    QRadioButton::indicator:checked:hover {{
        image: url({os.path.join(img_dir, "radio_hoverCheck.png")});
    }}
    
    QRadioButton::disabled {{
        color: {disDark}
    }}
    
    QCheckBox {{
        background: transparent;
    }}
    
    QCheckBox::indicator:unchecked {{
        image: url({os.path.join(img_dir, "checkbox_normal.png")});
    }}
    
    QCheckBox::indicator:checked {{
        image: url({os.path.join(img_dir, "checkbox_down.png")});
    }}
    
    QCheckBox::indicator:unchecked:hover {{
        image: url({os.path.join(img_dir, "checkbox_hoverUncheck.png")});
    }}
    
    QCheckBox::indicator:checked:hover {{
        image: url({os.path.join(img_dir, "checkbox_hoverCheck.png")});
    }}
    
    QCheckBox::disabled,
    QCheckBox::indicator::disabled {{
        color: {disDark}
    }}
    """

    """QTab"""
    qss += f"""
    QTabWidget {{
        color: {normal};
        background: {light};
        font-size: 40px;
        width: 60px;
    }}
    
    QTabBar::tab {{
        color: {light};
        background: {normal};
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        padding-left: 5px;
        padding-right: 5px;
        padding-top: 2px;
        width: 60px;
    }}
    
    QTabBar::tab:hover {{
        color: {light};
        background: {normal};
        border: 2px solid {normal};
    }}
    
    QTabBar::tab:selected {{
        color: {light};
        margin-left: -3px;
        margin-right: -3px;
        border-bottom: 1px;
        border-bottom-color: {light};
    }}
    
    QTabBar::tab:!selected {{
        color: {normal};
        background: {disDark};
        border: 2px solid {disDark};
        margin-top: 2px;
        /* make non-selected tabs look smaller */
    }}
    
    QTabWidget::pane {{
        /* The tab widget frame */
        padding: 10px;
        border: 2px solid {disDark};
        border-radius: 6px;
        border-top-left-radius: 0px;
        border-top-right-radius: 0px;
    }}
    
    QTabBar::tab:only-one {{
        margin: 0;
        /* if there is only one tab, we don't want overlapping margins */
    }}
    """

    """QSlider"""
    qss += f"""
    QScrollBar::handle {{
        background: {disDark};
        border-radius: 5px;
        min-height: 10px
    }}
    
    QScrollBar::handle:pressed {{
        background: {fontDark}
    }}
    
    QScrollBar:horizontal {{
        height: 10px;
        background: {deep};
    }}
    
    QScrollBar:vertical {{
        width: 10px;
        background: {deep};
    }}
    
    QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal, 
    QScrollBar::down-arrow:vertical, QScrollBar::up-arrow:vertical,
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal,
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
    QScrollBar::sub-line:horizontal, QScrollBar::add-line:horizontal,
    QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical
    {{
        border: none;
        color: none;
        background: none;
    }}
    
    """

    """Titlebar"""
    qss += f"""
    Titlebar QLabel#Titlebar_titleLabel {{
        font-size: 13px;
        margin-bottom: 0px;
        color: {normal};
    }}
    
    Titlebar QLabel#Titlebar_backgroundLabel {{
        background: {deep};
    }}
    
    Titlebar QPushButton#Titlebar_minimizeButton {{
        background: transparent;
        background-image: url({os.path.join(img_dir, "minsize.png")});
        border: none
    }}
    
    Titlebar QPushButton#Titlebar_minimizeButton:hover {{
        background: transparent;
        background-image: url({os.path.join(img_dir, "minsizehover.png")})
    }}
    
    Titlebar QPushButton#Titlebar_minimizeButton:pressed {{
        background: transparent;
        background-image: url({os.path.join(img_dir, "minsizepress.png")})
    }}
    
    Titlebar QPushButton#Titlebar_minimizeButton:disabled {{
        background: transparent;
        background-image: url({os.path.join(img_dir, "minsizepress.png")})
    }}
    
    Titlebar QPushButton#Titlebar_maximizeButton {{
        background: transparent;
        background-image: url({os.path.join(img_dir, "maxsize.png")});
        border: none
    }}
    
    Titlebar QPushButton#Titlebar_maximizeButton:hover {{
        background: transparent;
        background-image: url({os.path.join(img_dir, "maxsizehover.png")})
    }}
    
    Titlebar QPushButton#Titlebar_maximizeButton:pressed {{
        background: transparent;
        background-image: url({os.path.join(img_dir, "maxsizepress.png")})
    }}
    
    Titlebar QPushButton#Titlebar_maximizeButton:disabled {{
        background: transparent;
        background-image: url({os.path.join(img_dir, "maxsizepress.png")})
    }}
    
    Titlebar QPushButton#Titlebar_closeButton {{
        background: transparent;
        background-image: url({os.path.join(img_dir, "close.png")});
        border: none
    }}
    
    Titlebar QPushButton#Titlebar_closeButton:hover {{
        background: transparent;
        background-image: url({os.path.join(img_dir, "closehover.png")})
    }}
    
    Titlebar QPushButton#Titlebar_closeButton:pressed {{
        background: transparent;
        background-image: url({os.path.join(img_dir, "closepress.png")})
    }}
    
    Titlebar QPushButton#Titlebar_closeButton:disabled {{
        background: transparent;
        background-image: url({os.path.join(img_dir, "closepress.png")})
    }}
    
    WindowWithTitleBar {{
        background: {normal};
        border: 3px solid {deep};
        border-radius: 2px
    }}
    
    """
    return qss.replace("\\", "/")


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
    qss += "WindowWithTitleBar{background:%s;border:3px solid %s; border-radius: 2px}" % (winBgdColor, barColor)
    return qss
