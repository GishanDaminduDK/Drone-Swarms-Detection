# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main.ui'
##
## Created by: Qt User Interface Compiler version 6.7.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QMainWindow, QMenu, QMenuBar,
    QPushButton, QSizePolicy, QStatusBar, QWidget)
import logo_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1027, 737)
        self.actionClose = QAction(MainWindow)
        self.actionClose.setObjectName(u"actionClose")
        icon = QIcon()
        icon.addFile(u":/close/icons8-close-48.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionClose.setIcon(icon)
        self.actionRestart = QAction(MainWindow)
        self.actionRestart.setObjectName(u"actionRestart")
        icon1 = QIcon()
        icon1.addFile(u":/reload/icons8-reset-48.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionRestart.setIcon(icon1)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.gridLayout_4 = QGridLayout(self.frame)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.frame_3 = QFrame(self.frame)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setFrameShape(QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.gridLayout_5 = QGridLayout(self.frame_3)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.scf_display = QLabel(self.frame_3)
        self.scf_display.setObjectName(u"scf_display")
        self.scf_display.setMinimumSize(QSize(512, 350))
        self.scf_display.setMaximumSize(QSize(512, 512))
        self.scf_display.setFrameShape(QFrame.StyledPanel)
        self.scf_display.setScaledContents(True)
        self.scf_display.setAlignment(Qt.AlignCenter)

        self.gridLayout_5.addWidget(self.scf_display, 0, 0, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label = QLabel(self.frame_3)
        self.label.setObjectName(u"label")
        self.label.setMaximumSize(QSize(150, 50))
        font = QFont()
        font.setFamilies([u"Lucida Sans"])
        font.setPointSize(14)
        font.setBold(False)
        self.label.setFont(font)
        self.label.setFrameShape(QFrame.StyledPanel)
        self.label.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.label)

        self.output_label = QLabel(self.frame_3)
        self.output_label.setObjectName(u"output_label")
        self.output_label.setMaximumSize(QSize(225, 50))
        font1 = QFont()
        font1.setFamilies([u"Lucida Sans"])
        font1.setPointSize(14)
        self.output_label.setFont(font1)
        self.output_label.setFrameShape(QFrame.StyledPanel)
        self.output_label.setFrameShadow(QFrame.Plain)
        self.output_label.setLineWidth(30)
        self.output_label.setMidLineWidth(2)
        self.output_label.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.output_label)


        self.gridLayout_5.addLayout(self.horizontalLayout_2, 0, 1, 1, 1)


        self.gridLayout_4.addWidget(self.frame_3, 2, 0, 1, 1)

        self.frame_1 = QFrame(self.frame)
        self.frame_1.setObjectName(u"frame_1")
        self.frame_1.setMaximumSize(QSize(16777215, 160))
        self.frame_1.setFrameShape(QFrame.StyledPanel)
        self.frame_1.setFrameShadow(QFrame.Raised)
        self.gridLayout_2 = QGridLayout(self.frame_1)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.drone_icon = QLabel(self.frame_1)
        self.drone_icon.setObjectName(u"drone_icon")
        self.drone_icon.setMinimumSize(QSize(200, 50))
        self.drone_icon.setMaximumSize(QSize(200, 50))
        self.drone_icon.setPixmap(QPixmap(u":/DeepSi/DeepSi Logo.png"))
        self.drone_icon.setScaledContents(True)
        self.drone_icon.setAlignment(Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.drone_icon)

        self.entc_logo = QLabel(self.frame_1)
        self.entc_logo.setObjectName(u"entc_logo")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.entc_logo.sizePolicy().hasHeightForWidth())
        self.entc_logo.setSizePolicy(sizePolicy)
        self.entc_logo.setMinimumSize(QSize(400, 70))
        self.entc_logo.setMaximumSize(QSize(700, 100))
        font2 = QFont()
        font2.setFamilies([u"Lucida Sans"])
        font2.setPointSize(18)
        font2.setBold(False)
        font2.setItalic(False)
        font2.setUnderline(False)
        self.entc_logo.setFont(font2)
        self.entc_logo.setAutoFillBackground(False)
        self.entc_logo.setScaledContents(False)
        self.entc_logo.setAlignment(Qt.AlignCenter)
        self.entc_logo.setWordWrap(True)

        self.horizontalLayout.addWidget(self.entc_logo)

        self.main_title = QLabel(self.frame_1)
        self.main_title.setObjectName(u"main_title")
        self.main_title.setMinimumSize(QSize(200, 80))
        self.main_title.setMaximumSize(QSize(200, 80))
        self.main_title.setPixmap(QPixmap(u":/entc/ENTC logo.png"))
        self.main_title.setScaledContents(True)

        self.horizontalLayout.addWidget(self.main_title)


        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 0, 1, 1)


        self.gridLayout_4.addWidget(self.frame_1, 0, 0, 1, 1)

        self.frame_2 = QFrame(self.frame)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setMaximumSize(QSize(16777215, 70))
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.gridLayout_3 = QGridLayout(self.frame_2)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.run_PB = QPushButton(self.frame_2)
        self.run_PB.setObjectName(u"run_PB")
        self.run_PB.setMinimumSize(QSize(0, 40))
        self.run_PB.setMaximumSize(QSize(175, 50))
        font3 = QFont()
        font3.setFamilies([u"Lucida Sans"])
        font3.setPointSize(10)
        font3.setBold(True)
        font3.setItalic(False)
        font3.setUnderline(False)
        font3.setStrikeOut(False)
        self.run_PB.setFont(font3)
        self.run_PB.setCursor(QCursor(Qt.PointingHandCursor))
        self.run_PB.setMouseTracking(True)
        self.run_PB.setLayoutDirection(Qt.LeftToRight)
        icon2 = QIcon()
        icon2.addFile(u":/start/icons8-play-64.png", QSize(), QIcon.Normal, QIcon.Off)
        self.run_PB.setIcon(icon2)
        self.run_PB.setIconSize(QSize(30, 30))

        self.gridLayout_3.addWidget(self.run_PB, 0, 0, 1, 1)


        self.gridLayout_4.addWidget(self.frame_2, 1, 0, 1, 1)


        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1027, 26))
        self.menuMenu = QMenu(self.menubar)
        self.menuMenu.setObjectName(u"menuMenu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuMenu.menuAction())
        self.menuMenu.addAction(self.actionClose)
        self.menuMenu.addAction(self.actionRestart)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionClose.setText(QCoreApplication.translate("MainWindow", u"Close", None))
        self.actionRestart.setText(QCoreApplication.translate("MainWindow", u"Restart", None))
        self.scf_display.setText("")
        self.label.setText(QCoreApplication.translate("MainWindow", u"Output:", None))
        self.output_label.setText("")
        self.drone_icon.setText("")
        self.entc_logo.setText(QCoreApplication.translate("MainWindow", u"Doppler-Radar Drone Detection System", None))
        self.main_title.setText("")
        self.run_PB.setText(QCoreApplication.translate("MainWindow", u"   RUN", None))
        self.menuMenu.setTitle(QCoreApplication.translate("MainWindow", u"Menu", None))
    # retranslateUi

