import serial
import time


encoding = 'utf-8'
startMarker = 60
endMarker = 62


class SerialControler(serial.Serial):
    def __init__(self, port="/dev/ttyACM0", baudrate=9600, timeout=None, write_timeout=None):
        super().__init__(port, baudrate, timeout=timeout, write_timeout=write_timeout)
        # self.waitForArduino()
        time.sleep(1)
        self.flushInput()
        self.flushOutput()

    def sendFeedback(self):
        self.sendToArduino("H")

    def getReadout(self):
        self.sendToArduino("R")
        ck = self.recvFromArduino()
        return ck[0]

    def openValve(self):
        self.sendToArduino("V")

    def closeValve(self):
        self.sendToArduino("v")

    def sendToArduino(self, strings):
        strings = "<" + ''.join(strings) + ">"
        self.write(strings.encode('utf-8'))

    def recvFromArduino(self):
        ck = ""
        x = "z"  # any value that is not an endMarker or startMarker
        byteCount = -1  # to allow for the fact that the last increment will be one too many

        # wait for the start character
        while ord(x) != startMarker:
            if self.in_waiting:
                x = self.read(1)
                if len(x) == 0:  # avoid calling ord(x) on an empty string
                    x = "z"

        # save data until the end marker is found
        while ord(x) != endMarker:
            if ord(x) != startMarker:
                print("serial loop")
                ck = ck + x.decode("utf-8", errors='replace')  # change for Python3
                byteCount += 1
            if self.in_waiting:
                x = self.read(1)
                if len(x) == 0:  # avoid calling ord(x) on an empty string
                    x = "z"

        return ck

    def waitForArduino(self):
        '''
        wait until the Arduino sends 'Arduino Ready' - allows time for Arduino reset
        it also ensures that any bytes left over from a previous message are discarded
        '''

        msg = ""
        while msg.find("Arduino is ready") == -1:

            while self.inWaiting() == 0:
                pass

            msg = self.recvFromArduino()
            if msg == "Arduino is ready":
                print(msg + "\n")
                break


