from recognition import Network
import recognition, loadLatex, UI

def main():

    # Create and initialize LaTeX image to code mappingss
    # Open "notepad"
    
    TRAIN_DIR, TEST_DIR = 'C:\\Users\\bri25\\Documents\\Python\\data\\', None
    recognition.main(TRAIN_DIR, TEST_DIR)
    
    while False: 
        # while the user is running this program

        # user writes something
        # 2-3 second delay for the user to finish what they're writing
        # convert user writing into latex code
        # give user to the view pdf

        #user has option to resize things
        pass

        
if __name__ == '__main__':
    main()