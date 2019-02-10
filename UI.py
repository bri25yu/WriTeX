from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import * # QFileDialog, QTextEdit, QLabel, QPushButton
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt

class Ui_MainWindow(object):

    def __init__(self):
        self.file_array = []
        self.textEdit = QTextEdit()

        # variables for placement of images
        self.x = 325
        self.y = 20

        # variable for iteration 
        self.imageNum = 0
        self.file_added = False

    def setup_buttons(self):
        """
        Setup buttons for user. 
        """
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(40, 62, 240, 50))
        self.pushButton.setObjectName("pushButton")

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(40, 124, 240, 50))
        self.pushButton_3.setObjectName("pushButton_3")

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1080)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.backgroundBox = QtWidgets.QLabel(self.centralwidget)
        self.backgroundBox.setGeometry(QtCore.QRect(-300,0,1800, 1000))
        self.backgroundBox.setText("")
        self.backgroundBox.setObjectName("backgroundBox")
        self.setBackground() 

        self.setup_buttons()

        # set the image boxes
        self.imageBox_array = []
        for i in range(99):
            self.imageBox_array.append(QtWidgets.QLabel(self.centralwidget))
            self.imageBox_array[i].setGeometry(QtCore.QRect(200, 20, 450, 450))
            self.imageBox_array[i].setText("")
            self.imageBox_array[i].setObjectName("imageBox")
           
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1920, 1080))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # actions executed when buttons are clicked
        def button1_pushed():
            """ Function wrapper class for multiple actions within one 
                "button clicked" call. """
            self.file_added = True
            print("file added")
            self.setImage()
        self.pushButton.clicked.connect(button1_pushed)
        self.pushButton_3.clicked.connect(self.saveImage)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Submit Image"))
        self.pushButton_3.setText(_translate("MainWindow", "Download LaTeX PDF"))

    def setImage(self):
        '''
        takes chosen image from computer
        displays it on screen
        adjacent number of images, max: 4 images

        should implement conversion of image to LaTeX
        '''

        self.file_array.append("")
        self.file_array[self.imageNum], _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "",
                                           "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if self.file_array[self.imageNum]: 
            self.pixmap = QtGui.QPixmap(self.file_array[self.imageNum])
            self.pixmap = self.pixmap.scaled(self.imageBox_array[self.imageNum].width()/2,
                                             self.imageBox_array[self.imageNum].height()/2, 
                                             QtCore.Qt.KeepAspectRatio)
            self.imageBox_array[self.imageNum].setPixmap(self.pixmap)
            self.imageBox_array[self.imageNum].setGeometry(QtCore.QRect(self.x, self.y, self.pixmap.size().width(), self.pixmap.size().height()))
            self.imageNum += 1
 
            if self.x >= 1500:
                self.x =  325
                self.y += self.pixmap.size().width() + 28
            self.x += self.pixmap.size().width() + 10
   
    def setBackground(self):
        '''
        sets the background
        '''
        pixmap = QtGui.QPixmap('images/WXBackground.png')
        pixmap = pixmap.scaled(self.backgroundBox.width(), self.backgroundBox.height(), QtCore.Qt.KeepAspectRatio)
        self.backgroundBox.setPixmap(pixmap)
        self.backgroundBox.setAlignment(QtCore.Qt.AlignCenter)
   
    def saveImage(self):
        '''
        Waits for user input for filename, then appends the image filetype ".png"
            in order to preserve alpha channels fo rhigher quality pictures. 
        e.g. user inputs "my_image", file should save in "my_image.png"
        '''
        for file in self.file_array:
            self.pixmap = QtGui.QPixmap(file)
            fname, _ = QFileDialog.getSaveFileName(None, "Save Image")
            self.pixmap.save(fname + '.png', quality = 100)
    
def main(controller):
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    dark_palette = QPalette()

    dark_palette.setColor(QPalette.Window, QColor(0, 139, 139))
    dark_palette.setColor(QPalette.WindowText, QColor(0, 139, 139))
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)

    app.setPalette(dark_palette)
    app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")

    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    # controller.set_ui(ui)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main(None)