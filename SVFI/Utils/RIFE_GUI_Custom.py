# -*- coding: utf-8 -*-
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QTextCursor, QIcon

class MyLineWidget(QtWidgets.QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasText():  # 是否文本文件格式
            url = e.mimeData().urls()[0]
            self.setText(url.toLocalFile())
        else:
            e.ignore()


class MyListWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dropEvent(self, e):
        if e.mimeData().hasText():  # 是否文本文件格式
            for url in e.mimeData().urls():
                items = self.get_items()
                item = url.toLocalFile()
                if item not in items:
                    self.addItem(item)
        else:
            e.ignore()

    def dragEnterEvent(self, e):
        self.dropEvent(e)

    def get_items(self):
        widgetres = []
        # 获取listwidget中条目数
        count = self.count()
        # 遍历listwidget中的内容
        for i in range(count):
            widgetres.append(self.item(i).text())
        return widgetres


class MyTextWidget(QtWidgets.QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dropEvent(self, event):
        try:
            if event.mimeData().hasUrls:
                url = event.mimeData().urls()[0]
                # url_list = list()
                # url_list.append(self.toPlainText().strip(";"))
                # for url in event.mimeData().urls():
                #     url_list.append(f"{url.toLocalFile()}")
                # text = ""
                # for url in url_list:
                #     text += f"{url};"
                # text = text.strip(";")
                self.setText(f"{url.toLocalFile()}")
            else:
                event.ignore()
        except Exception as e:
            print(e)



class MyComboBox(QComboBox):
    def wheelEvent(self, e):
        if e.type() == QEvent.Wheel:
            e.ignore()


class MySpinBox(QSpinBox):
    def wheelEvent(self, e):
        if e.type() == QEvent.Wheel:
            e.ignore()

class MyDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, e):
        if e.type() == QEvent.Wheel:
            e.ignore()