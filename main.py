from recognition import Network
import recognition, loadLatex, UI
from PyQt5.QtWidgets import QMainWindow
import sys, threading

def main():
    """Create and initialize LaTeX image to code mappingss open "notepad" """

    TRAIN_DIR, SAVE_DIR, TEST_DIR = 'C:\\Users\\bri25\\Documents\\Python\\data\\', 'data/', None
#     recognition.main(TRAIN_DIR, SAVE_DIR, TEST_DIR, train=False)
    controller = Control(TRAIN_DIR, SAVE_DIR, TEST_DIR)
    controller.run()
        # use CNN to process image pixels
        # user writes something
        # 2-3 second delay for the user to finish what they're writing
        # convert user writing into latex code
        # give user to the view pdf
        #user has option to resize things

class Control:
    def __init__(self, TRAIN_DIR, SAVE_DIR, TEST_DIR):
        self.TRAIN_DIR = TRAIN_DIR
        self.SAVE_DIR = SAVE_DIR
        self.TEST_DIR = TEST_DIR

    def set_ui(self, ui):
        self.ui = ui

    def set_CNN(self, CNN):
        self.CNN = CNN

    def run(self):
        t1 = threading.Thread(target=UI.main, args=(self, ))
        # t2 = threading.Thread(target=recognition.main, 
        #                       args=(self.TRAIN_DIR, self.SAVE_DIR, self.TEST_DIR,), 
        #                       kwargs={'implementing':True, 'controller':self})
        t2 = threading.Thread(target = lambda: print("thread2 start"))
        t1.start()
        t2.start()

        while True:
            if self.ui.file_added:
                print("added")
                image = self.ui.pixmap
                self.CNN.implement() # TODO: finish CNN function


if __name__ == '__main__':
    main()