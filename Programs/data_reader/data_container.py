import numpy as np
import serial

class dataContainer:
    def __init__(self, port, baudrate, sample_size = 1024):
        self.ser = serial.Serial(port, baudrate)
        self.sample_size = sample_size
        self.index = 0
        self.channel1 = [0.]*self.sample_size 
        self.channel2 = [0.]*self.sample_size 
        self.channel3 = [0.]*self.sample_size 
        self.channel4 = [0.]*self.sample_size 

    def read_serial(self):
        """
        Reading serial data from the port. 
        """
        try:
            flag = True
            while True:
                serial_data = self.ser.readline().decode('ISO-8859-1').strip().split(',')

                if flag:
                    flag = False
                    continue
                serial_data[-1] = serial_data[-1][:-4]
                values = list(map(float, serial_data))

                if self.index < (self.sample_size - 1):
                    self.channel1[self.index] = values[0]
                    self.channel2[self.index] = values[1]
                    self.channel3[self.index] = values[2]
                    self.channel4[self.index] = values[3]
                    self.index += 1

                else:
                    self.channel1.pop(0)
                    self.channel2.pop(0)
                    self.channel3.pop(0)
                    self.channel4.pop(0)

                    self.channel1.append(values[0])
                    self.channel2.append(values[1])
                    self.channel3.append(values[2])
                    self.channel4.append(values[3])

        except KeyboardInterrupt:
            self.ser.close()
            print("Serial port closed")

    def get_channel_values(self):
        return [self.channel1, self.channel2, self.channel3, self.channel4]


    