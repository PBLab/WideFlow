import numpy as np
import time


class Camera:
    def __init__(self, vid=None, exp_time=10, channels=2):
        if vid is None:
            self.sensor_size = (100, 100)
        else:
            self.sensor_size = vid.shape[1:]
        self.exp_time = 10
        self.binning = (3, 3)
        self.roi = (0, self.sensor_size[0], 0, self.sensor_size[1])
        self.clear_mode = 0
        self.vid = vid
        self.frame_idx = -1
        self.shape = (self.roi[3]-self.roi[2], self.roi[1]-self.roi[0])
        self.channels = channels

    def start_up(self):
        pass

    def open(self):
        pass

    def close(self):
        pass

    def get_frame(self):
        self.frame_idx += 1
        if self.vid is None:
            return np.random.random(self.sensor_size)*255
        else:
            time.sleep(0.02)
            return self.vid[self.frame_idx, self.roi[2]:self.roi[3], self.roi[0]:self.roi[1]]

    def get_live_frame(self):
        self.frame_idx += 1
        if self.vid is None:
            return np.random.random(self.sensor_size) * 255
        else:
            time.sleep(0.02)
            return self.vid[self.frame_idx, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

    def start_live(self):
        pass

    def stop_live(self):
        pass

    def set_param(self, a, b):
        pass