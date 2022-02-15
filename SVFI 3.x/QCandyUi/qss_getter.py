# const color string
WHITE = "#313C4F"
BLACK = "#000000"
RED = "#FF0000"
GREEN = "#00FF00"
BLUE = "#0000FF"
PURPLE = "#B23AEE"
WATCHET = "#1C86EE"
LIGHTGREEN = "#ECFEFE"
BLUEGREEN = "#33CCCC"
DEEPBLUEGREEN = "#015F5F"
DARKBLUEGREEN = "#28AAAA"
GRAY = "#999999"

DEEPBLUE = "#131820"
HARDBLUE = "#1D2839"
LIGHTBLUE = "2DA0F7"
BLUEGRAY = "#313C4F"
LIGHTGRAY = "#313C4F"
DEEPBROWN = "#85743D"
LIGHTBROWN = "F2C94C"

import os

ArrowURL = ""


def getFontQss(fontname, fontcolor):
    return """QWidget, QWidget:window, QStackedWidget, QPushButton, QRadioButton, QCheckBox, 
                QGroupBox, QStatusBar, QToolButton, QComboBox, QDialog, QTabBar, QScrollBar, QScrollArea {
                  font-family: 'Segoe UI';
                  background: #131820;
              }
              /*
              QWidget {
                border-radius: 5px;
              }  
              QMenuBar, QMenu{border-radius:0px;}
              QMainWindow{border-bottom-left-radius:5px;border-bottom-right-radius:5px}*/
              QObject{font-family:%s;color:%s;font-size:14px}
              
              QLabel{background:transparent}
              QMenuBar {
                    spacing: 5px; /* spacing between menu bar items */
                }
                
                QMenuBar::item {
                    background: transparent;
                }
                
                QMenuBar::item:selected { 
                    background: #2DA0F7;
                }
                
                QMenuBar::item:pressed {
                    border: 2px solid #2DA0F7;
                    background: #2DA0F7;
                }
                
                QMenu{
                    background: transparent;
                    padding:8px 1px; /*设置菜单项文字上下和左右的内边距，效果就是菜单中的条目左右上下有了间隔*/
                    margin:2px 2px;/*设置菜单项的外边距*/
                    border-radius:4px;
                }
                
                QMenu::item{
                    font-family: 'Microsoft Yahei';
                    background: transparent;
                    padding:8px;
                }
                
                QMenu::item:selected {
                    background: #2DA0F7;
                    border-radius: 4px;
                }
                
              """ % (fontname, fontcolor)


def getPushButtonQss(normalColor, normalTextColor, hoverColor, hoverTextColor, pressedColor, pressedTextColor,
                     disableColor, disableTextColor):
    str1 = "QPushButton{padding:3px;border-radius:5px;color:%s;background:%s;border:2px solid %s;}" % (
        normalTextColor, normalColor, normalColor)
    str2 = "QPushButton:hover{color:%s;background:%s;}" % (hoverTextColor, hoverColor)
    str3 = "QPushButton:pressed{color:%s;background:%s;}" % (pressedTextColor, pressedColor)
    str4 = "QPushButton:disabled{color:%s;background:%s;}" % (disableTextColor, disableColor)
    str5 = "QProgressBar{text-align: center; border:2px solid %s; border-radius:5px; selection-color:%s;selection-background-color:%s;}" % (
        normalColor, WHITE, hoverColor)
    str_ = """
        QPushButton#replace{background:%s;border:2px solid %s;}
    """ % (BLUEGRAY, BLUEGRAY)
    replace_list = ["InputButton", "OutputButton", "InputDirButton", "OutputSettingsButton", "RefreshStartInfo"]
    str6 = "\n".join(list(map(lambda x: str_.replace('replace', x), replace_list)))

    str_ = """
            QLabel#replace{font:bold;font-size:22px;font-weight:normal;}
        """
    replace_list = ["InputLabel", "OutputLabel", "AdvanceSettingsLabel"]
    str6 += "\n".join(list(map(lambda x: str_.replace('replace', x), replace_list)))

    str_ = """
                QLabel#replace{font:bold;font-size:16px;font-weight:bold;}
            """
    replace_list = ["AMDSettingsLabel", "RenderSettingsLabel", "EncodeSettingsLabel"]
    str6 += "\n".join(list(map(lambda x: str_.replace('replace', x), replace_list)))

    return str1 + str2 + str3 + str4 + str5 + str6


def getLineeditQss(normalColor, focusColor):
    str1 = "QLineEdit{border-style:none;padding:2px;border-radius:5px;border:2px solid %s;selection-color:%s;selection-background-color:%s;}" % (
        BLUEGRAY, WHITE, focusColor)
    str2 = "QLineEdit:focus{border:2px solid %s;}" % (focusColor)
    str3 = "QLineEdit:disabled{color:%s;}" % (LIGHTGRAY)
    return str1 + str2 + str3


def getPlaineditQss(normalColor, focusColor):
    str1 = "QPlainTextEdit{border-style:none;padding:2px;border-radius:10px;border:2px solid %s;font-family:宋体;selection-color:%s;selection-background-color:%s}" % (
        normalColor, WHITE, focusColor)
    str2 = "QPlainTextEdit:focus{border:2px solid %s;}" % (focusColor)

    str3 = """
            QGroupBox {
                /* 背景渐变色
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                  stop: 0 #E0E0E0, stop: 1 #FFFFFF);*/
                background: %s;
            
                /* 边框 */
                border: 0px solid gray;
            
                /* 倒角 */
                border-radius: 5px;
            
                /* 就像墙上挂着的两个相框,margin指的是相框与相框的距离
                   padding指的是每个相框里照片与相框边框的距离 
                margin-top: 20px;
                padding-top: 10px;*/
                font-size: 18px;
                
            }
            
            /* 标题设置 */
            QGroupBox::title {
                /* 位置 */
                subcontrol-origin: margin;
                subcontrol-position: top left;
                /* 内边框,上下和左右 */
                padding: 10px 15px;
            
            }
            
            QToolBox::tab {
                background: %s;
                border-radius: 5px;
                color: darkgray;
            }
            
            QToolBox::tab:selected { /* italicize selected tabs */
                font: bold;
                color: white;
            }
            
    """ % (HARDBLUE, HARDBLUE)
    return str1 + str2 + str3


def getTextBrowerQss(normalColor, focusColor):
    str1 = "QTextBrowser{border-style:none;padding:2px;border-radius:5px;border:2px solid %s;selection-color:%s;selection-background-color:%s}" % (
        normalColor, WHITE, focusColor)
    str2 = "QTextBrowser:focus{border:2px solid %s;}" % (focusColor)
    str3 = "QTextEdit{padding:2px;border-radius:5px;border:2px solid %s;selection-color:%s;selection-background-color:%s}" % (
        normalColor, WHITE, focusColor)
    str4 = "QTextEdit:focus{border:2px solid %s;} QTextEdit:hover{border:2px solid %s;}" % (focusColor, focusColor)
    str5 = "QListWidget,QScrollArea{border-style:none;padding:2px;border-radius:5px;border:2px solid %s;selection-color:%s;selection-background-color:%s; background:transparent}" % (
        BLUEGRAY, WHITE, focusColor)



    return str1 + str2 + str3 + str4 + str5


def getComboxQss(backgroundColor, normalColor, focusColor, arrowimageurl):
    # focusColor = BLUEGRAY
    backgroundColor = DEEPBLUE
    up_arrow_url = arrowimageurl.replace("down", "up")
    right_arrow_url = arrowimageurl.replace("down", "right")
    save_ico_url = arrowimageurl.replace("down_arrow", "select")
    load_ico_url = arrowimageurl.replace("down_arrow", "refresh")
    run_ico_url = arrowimageurl.replace("down_arrow", "dynamic-filling")
    smile_ico_url = arrowimageurl.replace("down_arrow", "smile")
    str1 = "QComboBox{background:%s;padding:2px;border-radius:5px;border:2px solid %s;}" % (
        backgroundColor, BLUEGRAY)
    # str1 = "QComboBox{background:transparent;padding:2px;border-radius:5px;border:2px solid %s;}" % (
    #     normalColor)
    str2 = "QComboBox:focus{border:2px solid %s;}" % (focusColor)
    str3 = "QComboBox:on{border:2px solid %s;}" % (focusColor)
    str4 = "QComboBox:disabled{color:%s;}" % (LIGHTGRAY)
    str5 = "QComboBox::drop-down{border-style:solid;border-radius:5px;}"
    str6 = "QComboBox QAbstractItemView{border:2px solid %s;background:transparent;selection-background-color:%s;}" % (
        focusColor, focusColor)
    str7 = "QComboBox::down-arrow{image:url(%s); border:20px;}" % (arrowimageurl)
    str8 = """
            QSpinBox {
                background:transparent;padding:2px;border-radius:5px;border:2px solid %s;
            }
            
            QSpinBox::up-arrow {
                image: url(%s);border:20px;
                background:transparent;
            }
            QSpinBox::down-arrow {
                image: url(%s);border:20px;
                background:transparent;
            }

            QSpinBox::hover {
                border:2px solid %s;
            }
            
            QSpinBox::disabled{color:%s; border-color: %s}

    """ % (BLUEGRAY, up_arrow_url, arrowimageurl, focusColor, LIGHTGRAY, LIGHTGRAY)
    str9 = str8.replace("QSpinBox", "QDoubleSpinBox")
    str10 = str8.replace("QSpinBox", "QTimeEdit")
    str11 = "QPushButton#AdvanceSettingsLabel{icon: url(%s)}" % right_arrow_url
    str11 += "QPushButton#SaveCurrentSettings{icon: url(%s)}" % save_ico_url
    str11 += "QPushButton#LoadCurrentSettings{icon: url(%s)}" % load_ico_url
    str11 += "QPushButton#AllInOne{icon: url(%s)}" % run_ico_url
    str11 += "QPushButton#AutoSet{icon: url(%s)}" % smile_ico_url
    return str1 + str2 + str3 + str4 + str5 + str6 + str7 + str8 + str9 + str10 + str11


def getProgressBarQss(normalColor, chunkColor):
    barHeight = str(8)
    barRadius = str(8)
    str1 = "QProgressBar{font:9pt;height:%spx;background:%s;border-radius:%spx;text-align:center;border:1px solid %s;}" % (
        barHeight, normalColor, barRadius, normalColor)
    str2 = "QProgressBar:chunk{border-radius:%spx;background-color:%s;margin:2px}" % (barRadius, chunkColor)
    return str1 + str2


def getSliderQss(normalColor, grooveColor, handleColor):
    sliderHeight = str(8)
    sliderRadius = str(4)
    handleWidth = str(13)
    handleRadius = str(6)
    handleOffset = str(3)
    str1 = "QSlider::groove:horizontal,QSlider::add-page:horizontal{height:%spx;border-radius:%spx;background:%s;}" % (
        sliderHeight, sliderRadius, normalColor)
    str2 = "QSlider::sub-page:horizontal{height:%spx;border-radius:%spx;background:%s;}" % (
        sliderHeight, sliderRadius, grooveColor)
    str3 = "QSlider::handle:horizontal{width:%spx;margin-top:-%spx;margin-bottom:-%spx;border-radius:%spx;" \
           "background:qradialgradient(spread:pad,cx:0.5,cy:0.5,radius:0.5,fx:0.5,fy:0.5,stop:0.6 #FFFFFF,stop:0.8 %s);}" % (
               handleWidth, handleOffset, handleOffset, handleRadius, handleColor)
    return str1 + str2 + str3


def getRadioButtonQss(normimageurl, downimageurl, normimageurlhover, downimageurlhover):
    # str1 = "QRadioButton::indicator{width:15px;height:15px;}"
    str1 = "QRadioButton{background:transparent;}"
    str2 = "QRadioButton::indicator:unchecked{image: url(%s);}" % (normimageurl)
    str3 = "QRadioButton::indicator:checked{image: url(%s);}" % (downimageurl)
    str4 = "QRadioButton::indicator:checked:hover{image: url(%s);}" % (downimageurlhover)
    str5 = "QRadioButton::disabled{color: %s}" % (LIGHTGRAY)
    return str1 + str2 + str3 + str4 + str5


def getCheckBoxQss(normimageurl, checkimageurl, normimageurlhover, checkimageurlhover):
    # str1 = "QCheckBox::indicator{width:15px;height:15px;}"
    str1 = "QCheckBox{background:transparent;}"
    str2 = "QCheckBox::indicator:unchecked{image: url(%s);}" % (normimageurl)
    str3 = "QCheckBox::indicator:checked{image: url(%s);}" % (checkimageurl)
    str4 = "QCheckBox::indicator:unchecked:hover{image: url(%s);}" % (normimageurlhover)
    str5 = "QCheckBox::indicator:checked:hover{image: url(%s);}" % (checkimageurlhover)
    str6 = "QCheckBox::disabled, QCheckBox::indicator::disabled{color: %s}" % (LIGHTGRAY)

    return str1 + str2 + str3 + str4 + str5 + str6


def getTabWidgetQss(normalTabColor, normalTabTextColor, tabBorderColor):
    str1 = "QTabWidget{color:%s; background:%s;font-size: 40px;width: 60px;}" % (normalTabTextColor, normalTabColor)
    str2 = ""
    str3 = "QTabBar::tab{color:%s; background:%s;border-top-radius:2px solid %s;" \
           "padding-left: 5px; padding-right: 5px;padding-top: 2px;width:60px;}" % (
               normalTabTextColor, BLUEGRAY, BLUEGRAY)
    str4 = "QTabBar::tab:hover{color:%s; background:%s;border:4px;}" % (normalTabColor, normalTabTextColor)
    str5 = """
              QTabBar::tab:selected{color:%s; background:%s;border-top-left-radius: 6px; border-top-right-radius: 6px;border:2px solid #313C4F; border-bottom: 1px;border-bottom-color: #313C4F;} 
              QTabBar::tab:!selected {
              margin-top: 3px; /* make non-selected tabs look smaller */}
              
              QTabWidget::pane { /* The tab widget frame */
                    padding: 10px;
                    border: 2px solid #313C4F;
                    border-radius: 6px;
                    border-top-left-radius: 0px;border-top-right-radius: 0px;
                }
                
                
                /* make use of negative margins for overlapping tabs */
                QTabBar::tab:selected {
                    /* expand/overlap to the left and right by 4px */
                    margin-left: -3px;
                    margin-right: -3px;
                }
                /*
                QTabBar::tab:first:selected {
                    margin-left: 0;
                }
                
                QTabBar::tab:last:selected {
                    margin-right: 0; 
                }*/
                
                QTabBar::tab:only-one {
                    margin: 0; /* if there is only one tab, we don't want overlapping margins */
                }
              
          """ % (normalTabColor, normalTabTextColor,)
    return str1 + str3 + str4 + str5


def getScrollbarQss(handlebarcolor, arrowimageurl):

    left_arrow_url = arrowimageurl.replace("down", "left")
    right_arrow_url = arrowimageurl.replace("down", "right")
    up_arrow_url = arrowimageurl.replace("down", "up")
    # str1 = "QScrollBar{{background:transparent;width:10px;padding-top:11px;padding-bottom:11px}}"
    # str1 = "QScrollBar{{background:transparent;padding-top:11px;padding-bottom:11px}}"
    str2 = f"QScrollBar::handle{{background:%s;border-radius:5px;min-height:10px}}" % (LIGHTGRAY)
    str3 = f"QScrollBar::handle:pressed{{background:%s}}" % (GRAY)
    str4 = f"""
            QScrollBar:horizontal {{
                height:10px;
                background:transparent;
            }}            
             QScrollBar:vertical {{
                width:10px;
                background:transparent;
            }}

            /*
            QScrollBar::add-line:horizontal {{
                border: 2px solid grey;
                background: {DEEPBLUE};
                height: 5px;
                subcontrol-position: right;
                subcontrol-origin: margin;
            }}
            
            QScrollBar::sub-line:horizontal {{
                border: 2px solid grey;
                background: {DEEPBLUE};
                height: 5px;
                subcontrol-position: left;
                subcontrol-origin: margin;
            }}

            QScrollBar::add-line:vertical {{
                border: 2px solid grey;
                background: {DEEPBLUE};
                height: 20px;
                subcontrol-position: right;
                subcontrol-origin: margin;
            }}
            
            QScrollBar::sub-line:vertical {{
                border: 2px solid grey;
                background: {DEEPBLUE};
                height: 20px;
                subcontrol-position: left;
                subcontrol-origin: margin;
            }}*/

               """
    return str2 + str3 + str4
