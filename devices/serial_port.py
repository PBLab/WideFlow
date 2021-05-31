import serial
import time


class SerialControler(serial.Serial):
    def __init__(self, port="/dev/ttyACM0", baudrate=9800, timeout=1.0):
        super().__init__(port, baudrate, timeout=timeout)

    def sendTTL(self):
        self.write(b'H')
        # time.sleep(0.1)
        # self.write(b'L')

    def blink(self):
        for i in range(10):
            self.write(b'H')
            time.sleep(0.2)
            self.write(b'L')
            time.sleep(0.2)