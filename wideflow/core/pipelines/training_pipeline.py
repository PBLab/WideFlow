from wideflow.core.abstract_pipeline import AbstractPipeLine
from wideflow.core.processes import *

from wideflow.Imaging.utils.create_matching_points import MatchingPointSelector

import numpy as np
from skimage.transform import AffineTransform, warp_coords
import cupy as cp
import random


class TrainingPipe(AbstractPipeLine):
    def __init__(self, camera, min_frame_count, max_frame_count, mask, map, match_p_src, match_p_dst, capacity):
        """

        :param min_frame_count: int - minimal number of pipeline circles between rewards
        :param max_frame_count: int - maximal number of pipeline circles between rewards
        """
        self.camera = camera
        self.min_frame_count = min_frame_count
        self.max_frame_count = max_frame_count
        self.mask = mask
        self.map = map
        self.match_p_src = match_p_src
        self.match_p_dst = match_p_dst
        self.capacity = capacity

        self.new_shape = self.map.shape
        self.mapping_coordinates = self.find_mapping_coordinates(match_p_src, match_p_dst)

        self.input_shape = (self.camera.shape[1], self.camera.shape[0])

        self.frame = np.ndarray(self.input_shape)
        self.input = cp.ndarray(self.input_shape)
        self.warped_input = cp.ndarray((self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.warped_buffer = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.dff_buffer = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)

        map_coord = MapCoordinates(self.input, self.warped_input, self.mapping_coordinates, self.new_shape)
        masking = Mask(self.warped_input, self.mask, self.warped_buffer, ptr=0)
        dff = DFF(self.dff_buffer, self.warped_buffer, ptr=0)
        self.processes_list = [map_coord, masking, dff]

        self.cue = 0
        self.cue_delay = 0

        self.ptr = self.capacity - 1
        self.counter = 0

    def fill_buffers(self):
        pass

    def clear_buffers(self):
        self.input = None
        self.warped_input = None
        self.warped_buffer = None
        self.dff_buffer = None

    def get_input(self):
        self.frame = self.camera.get_live_frame()
        self.input[:] = cp.asanyarray(self.frame)

    def process(self):
        if self.ptr == self.capacity - 1:
            self.ptr = 0
        else:
            self.ptr += 1

        self.get_input()
        for process in self.processes_list:
            process.process()

    def evaluate(self):

        if self.counter == 0:
            self.cue = 0
            self.cue_delay = random.choice(range(self.min_frame_count, self.max_frame_count, 1))

        self.counter += 1
        if self.counter == self.cue_delay:
            self.counter = 0
            self.cue = 1

        return self.cue

    def find_mapping_coordinates(self, match_p_src, match_p_dst):
        tform = AffineTransform()
        tform.estimate(np.roll(match_p_src, 1, axis=1), np.roll(match_p_dst, 1, axis=1))

        warp_coor = warp_coords(tform.inverse, (self.new_shape[0], self.new_shape[1]))
        src_cols = np.reshape(warp_coor[0], (self.new_shape[0] * self.new_shape[1], 1))
        src_rows = np.reshape(warp_coor[1], (self.new_shape[0] * self.new_shape[1], 1))
        mapping_coordinates = cp.asanyarray([src_cols, src_rows])

        return mapping_coordinates