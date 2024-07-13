import threading 
import csv
import pandas as pd
import serial 
import time
import threading
import queue
from queue import Queue

class SerialDataReader:
    """
    This class is used to get the data from the serial port and store them in a csv file
    """

    def __init__(self, port, baudrate, file_name):
        self.ser = serial.Serial(port, baudrate)
        self.columns = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4']
        self.df = pd.DataFrame(columns=self.columns)
        self.data_queue = Queue()
        self.read_thread = None
        self.process_thread = None

    def read_serial(self, duration):
        """
        Reading serial data from the port. 
        """
        end_time = time.time() + duration
        flag = True

        while time.time() < end_time:
            serial_data = self.ser.readline().decode('ISO-8859-1')  #.split('\x00')[0].strip().split(',')
            if flag:
                flag = False
                continue
            # serial_data[-1] = serial_data[-1].split("\x00")[0]
            # if (len(serial_data) != 4) or ('' in serial_data) or ('\x00' in serial_data[-1]):
            # print(repr(serial_data))
            # values = list(map(float, serial_data))
            self.data_queue.put(serial_data)

    
    def process_queue(self):
        
        while True:
            try:
                serial_data = self.data_queue.get(timeout = 5)
                serial_data = serial_data.split('\x00')[0].strip().split(',')
                values = list(map(float, serial_data))
                self.df.loc[len(self.df)] = values
            except queue.Empty:
                if not self.read_thread.is_alive():
                    break
    
    def write_data(self, duration, file_name):
        self.read_thread = threading.Thread(target=self.read_serial, args=(duration,))
        self.process_thread = threading.Thread(target=self.process_queue)

        self.read_thread.start()
        self.process_thread.start()

        self.read_thread.join()
        self.process_thread.join()

        self.ser.close()

        self.df.to_csv(file_name, index=False)