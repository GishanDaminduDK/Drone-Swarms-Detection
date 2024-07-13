import os.path
from PySide6 import QtWidgets
from PySide6 import QtGui
from PySide6.QtGui import QPixmap, QIcon
import random
from numpy.random import randint 
from time import sleep
import os
import torch
from ui_functions import get_scf, scf_save, get_output, get_data
from ui import main
from model_types import efficientnet_1

device = "cuda" if torch.cuda.is_available() else "cpu"
script_dir = os.path.dirname(os.path.abspath(__file__)).replace('\\','/')
ls = ['data/f1', 'data/f2', 'data/f3', 'data/f4']
f = random.choice(ls)

class MyQtApp(main.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super(MyQtApp, self).__init__()
        self.setupUi(self)
        self.showMaximized()
        self.setWindowTitle("Doppler-Radar Drone Detection System")
        self.setWindowIcon(QIcon("final code/ui/Resource Files/icons8-radar-48.ico"))  # Replace with your icon path

        self.run_PB.clicked.connect(self.run)
        self.actionClose.triggered.connect(self.close_app)
        self.actionRestart.triggered.connect(self.restart_app)
        self.setStyleSheet("")
        # icon = QIcon('final code/ui/Resource Files/icons8-processing-32.png')
        # self.run_PB.setIcon(icon)

    
    def run(self):
        
        # clear output handles
        self.scf_display.setPixmap(QPixmap())
        self.output_label.setText("")
        
        self.scf_display.setText("Loading SCF")
        self.run_PB.setText('Running')
        QtWidgets.QApplication.processEvents() 
        sleep(2)

        print("Running...") #debug



        data_get_dir = os.path.join(script_dir, f)
        # get_data(data_get_dir, 'COM4', 500000, 4)
        
        # relative_path = f'data/{f}'
        data_dir = os.path.join(script_dir, f)
        data = get_scf(data_dir, self)
        scf_save(data, data_dir)
        # after SCF is generated
        relative_path = f'{f}/example.png'
        scf_path = os.path.join(script_dir, relative_path)
        pixmap = QPixmap(scf_path)
        self.scf_display.setPixmap(pixmap)
        self.scf_display.setScaledContents(True)
        self.run_PB.setText('Run')
        
        # after ML result
        relative_path = r'saved_models_type\efficientnet_clipped\from_epoch_40\efficientnet_unfreezed_40_10_epoch_82.pt'
        state_dict_path = os.path.join(script_dir, relative_path)

        ml_output = str(get_output(data, state_dict_path, device=device))
        # ml_output = str(randint(2))
        if ml_output == '1':
            ml_output = "Drone detected"
        elif ml_output == '0':
            ml_output = "No Drone detected"
        self.output_label.setText(ml_output)


    def close_app(self):
        print("Closing application")
        self.close()

    def restart_app(self):
        print("Restarting application...")
        # Restart the application
        QtWidgets.QApplication.quit()
        
        # Get the path of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.basename(__file__)
        full_path = os.path.join(current_dir, filename).replace("\\", "/")
        
        os.system('C:/Python310/python.exe "{}"'.format(full_path))
 

if __name__ == '__main__':
    app = QtWidgets.QApplication()
    app.setWindowIcon(QIcon("final code/ui/Resource Files/icons8-radar-48.ico"))
    qt_app = MyQtApp()
    qt_app.show()
    app.exec()