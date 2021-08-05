import numpy as np
import time


class Camera:
    def __init__(self, vid=None):
        self.sensor_size = (1024, 1024)
        self.exp_time = 10
        self.binning = (3,3)
        self.roi = (0, 1024, 0, 1024)
        self.clear_mode = 0
        self.vid = vid
        self.frame_idx = 0

    def open(self):
        pass

    def close(self):
        pass

    def get_frame(self):
        self.frame_idx += 1
        if self.vid is None:
            return np.random.random((self.roi[1]-self.roi[0], self.roi[3]-self.roi[2]))*255
        else:
            time.sleep(0.02)
            return self.vid[self.frame_idx, :, :]


    def get_live_frame(self):
        if self.vid is None:
            return np.random.random((self.roi[1]-self.roi[0], self.roi[3]-self.roi[2])) * 255
        else:
            self.frame_idx += 1
            return self.vid[self.frame_idx, :, :]

    def start_live(self):
        pass

    def stop_live(self):
        pass