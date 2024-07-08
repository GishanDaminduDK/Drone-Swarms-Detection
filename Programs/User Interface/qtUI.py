from PySide6 import QtWidgets
from PySide6 import QtGui
from PySide6.QtGui import QPixmap
import os

from ui import main

class MyQtApp(main.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super(MyQtApp, self).__init__()
        self.setupUi(self)
        self.showMaximized()
        self.setWindowTitle("Doppler-Radar Drone Detection System")
        self.run_PB.clicked.connect(self.run)
        self.actionClose.triggered.connect(self.close_app)
        self.actionRestart.triggered.connect(self.restart_app)


    def run(self):
        # include code to begin running the whole program
        print("Run push button was clicked")
        
        # after SCF is generated
        path = 'scf.png'
        pixmap = QPixmap(path)
        self.scf_display.setPixmap(pixmap)
        self.scf_display.setScaledContents(True)
        
        # after ML result
        ml_output = "Drone Detected!!!"
        self.output_label.setText(ml_output)


    def close_app(self):
        print("Closing application")
        self.close()

    def restart_app(self):
        print("Restarting application")
        # Restart the application
        QtWidgets.QApplication.quit()
        
        # Get the path of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.basename(__file__)
        full_path = os.path.join(current_dir, filename).replace("\\", "/")
        
        os.system('C:/Python310/python.exe "{}"'.format(full_path))
 

if __name__ == '__main__':
    app = QtWidgets.QApplication()
    qt_app = MyQtApp()
    qt_app.show()
    app.exec()
