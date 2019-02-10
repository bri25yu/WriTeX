
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QTextEdit, QLabel

class Ui_MainWindow(object):
    
    def __init__(self):
        self.fileName = ""
        self.textEdit = QTextEdit()

    def setupUi(self, MainWindow):


        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 900)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.backgroundBox = QtWidgets.QLabel(self.centralwidget)
        self.backgroundBox.setGeometry(QtCore.QRect(0,0,1000,900))
        self.backgroundBox.setText("")
        self.backgroundBox.setObjectName("backgroundBox")
        self.setBackground() 

        #set the buttons
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(20, 20, 140, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 60, 140, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(20, 100, 140, 31))
        self.pushButton_3.setObjectName("pushButton_3")


        self.imageBox = QtWidgets.QLabel(self.centralwidget)
        self.imageBox.setGeometry(QtCore.QRect(220, 20, 700, 800))
        self.imageBox.setText("")
        self.imageBox.setObjectName("imageBox")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 480, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        # actions executed when buttons are clicked
        self.pushButton.clicked.connect(self.setImage)
        self.pushButton_3.clicked.connect(self.saveImage)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Submit Image"))
        self.pushButton_2.setText(_translate("MainWindow", "Process "))
        self.pushButton_3.setText(_translate("MainWindow", "Download Latex"))

    def setImage(self):
        print("imageBox size: " + str(self.imageBox.size()))
        self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if self.fileName: 
            self.pixmap = QtGui.QPixmap(self.fileName)
            print("pixmap size before resize to imagebox size" + str(self.pixmap.size()))
            self.pixmap = self.pixmap.scaled(self.imageBox.width(), self.imageBox.height(), QtCore.Qt.KeepAspectRatio)
            print("pixmap size after resize to imagebox size" + str(self.pixmap.size()))
            self.imageBox.setPixmap(self.pixmap)
            self.imageBox.setAlignment(QtCore.Qt.AlignCenter)
  
    def setBackground(self):
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
        fname, _ = QFileDialog.getSaveFileName(None, "Save Image")
        self.pixmap.save(fname + '.png', quality = 100)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

