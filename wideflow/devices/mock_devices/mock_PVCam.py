import os

from utils.load_tiff import load_tiff
from analysis.utils.sort_video_path_list import sort_video_path_list


class MockPVCamera:
    def __init__(self, camera_configs, vid_path, crop_sensor=False):
        self.camera_configs = camera_configs
        self.vid_path = vid_path
        self.videos_files_list = []
        if os.path.isfile(self.vid_path):
            self.videos_files_list.append(self.vid_path)
        elif os.path.isdir(self.vid_path):
            for file in os.listdir(self.vid_path):
                if file.endswith('.tif'):
                    self.videos_files_list.append(self.vid_path + file)
            self.videos_files_list = sort_video_path_list(self.videos_files_list)
        self.num_of_vids = len(self.videos_files_list)

        self.exp_time = self.camera_configs["core_attr"]["exp_time"]
        self.binning = self.camera_configs["core_attr"]["binning"]
        self.roi = self.camera_configs["core_attr"]["roi"]
        self.clear_mode = self.camera_configs["core_attr"]["clear_mode"]
        self.crop_sensor = crop_sensor  # return raw loaded frame or crop frame according to roi

        self.num_of_frames = None
        self.video = None
        self.load_video_data()
        self.frame_idx = -1  # for first call to get_frame method
        self.total_cap_frames = 0
        if self.crop_sensor:
            self.shape = (self.roi[3]-self.roi[2], self.roi[1]-self.roi[0])
        else:

            self.shape = (self.video.shape[2], self.video.shape[1])

    def start_up(self):
        pass

    def open(self):
        pass

    def close(self):
        pass

    def get_frame(self):
        self.frame_idx += 1
        self.total_cap_frames += 1
        if self.frame_idx == self.num_of_frames:
            self.load_video_data()
        if self.crop_sensor:
            return self.video[self.frame_idx, self.roi[2]:self.roi[3], self.roi[0]:self.roi[1]]
        else:
            return self.video[self.frame_idx]

    def poll_frame(self):
        return self.get_frame()

    def start_live(self):
        pass

    def stop_live(self):
        pass

    def set_param(self, a, b):
        pass

    def load_video_data(self):
        self.video = load_tiff(self.videos_files_list[0])
        self.videos_files_list.pop(0)
        self.num_of_frames = self.video.shape[0]
        self.frame_idx = 0
