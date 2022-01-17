class MockSerialControler():
    def __init__(self, serial_readout):
        self.serial_readout = serial_readout

    def sendFeedback(self):
        pass

    def getReadout(self):
        return self.serial_readout[0]
        self.serial_readout.pop(0)

    def close(self):
        pass



