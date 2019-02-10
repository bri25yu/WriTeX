from recognition import Network
import recognition, loadLatex, UI

def main():

    # Create and initialize LaTeX image to code mappingss
    # Open "notepad"
    
#     TRAIN_DIR, SAVE_DIR, TEST_DIR = 'C:\\Users\\bri25\\Documents\\Python\\data\\', 'data/', None
#     recognition.main(TRAIN_DIR, SAVE_DIR, TEST_DIR, train=False)

    app = UI.main()
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()

        # user writes something
        # 2-3 second delay for the user to finish what they're writing
        # convert user writing into latex code
        # give user to the view pdf
        # pass
        #user has option to resize things

    sys.exit(app.exec_())
    
    
if __name__ == '__main__':
    main()