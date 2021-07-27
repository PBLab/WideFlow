import serial
import time


class SerialControler(serial.Serial):
    def __init__(self, port="/dev/ttyACM0", baudrate=9800, timeout=0.0, write_timeout=1.0):
        super().__init__(port, baudrate, timeout=timeout)

    def sendTTL(self):
        self.write(b'H')

    def readSerial(self):
        return self.read()