
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QTextEdit, QLabel

class Ui_MainWindow(object):
    
    def __init__(self):
        
        self.file_array = []
        self.textEdit = QTextEdit()


        # variables for placement of images
        self.x = 200
        self.y = 20

        # variable for iteration 
        self.imageNum = 0

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

        # set the buttons
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(20, 20, 140, 31))
        self.pushButton.setObjectName("pushButton")

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(20, 60, 140, 31))
        self.pushButton_3.setObjectName("pushButton_3")

        # set the image boxes
        self.imageBox_array = []
        for i in range(99):
            self.imageBox_array.append(QtWidgets.QLabel(self.centralwidget))
            self.imageBox_array[i].setGeometry(QtCore.QRect(200, 20, 750, 650))
            self.imageBox_array[i].setText("")
            self.imageBox_array[i].setObjectName("imageBox")
           

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
        self.pushButton_3.setText(_translate("MainWindow", "Download Latex"))


    def setImage(self):
        '''
        takes chosen image from computer
        displays it on screen
        adjacent number of images, max: 4 images

        should implement conversion of image to LaTex
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
 
            if self.x >= 746:
                self.x =  200 - self.pixmap.size().width() - 10
                self.y += self.pixmap.size().width() * 2  - 28
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


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())