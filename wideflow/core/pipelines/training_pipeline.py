from wideflow.core.abstract_pipeline import AbstractPipeLine
from wideflow.core.processes import *

import numpy as np
import cupy as cp
import random

import h5py


class TrainingPipe(AbstractPipeLine):
    def __init__(self, camera, mapping_coordinates, min_frame_count, max_frame_count, new_shape, mask_path):
        """

        :param min_frame_count: int - minimal number of pipeline circles between rewards
        :param max_frame_count: int - maximal number of pipeline circles between rewards
        """
        self.camera = camera
        self.mapping_coordinates = mapping_coordinates
        self.min_frame_count = min_frame_count
        self.max_frame_count = max_frame_count
        self.new_shape = new_shape
        self.mask_path = mask_path
        self.mask = self.load_datasets()

        self.capacity = 1
        self.input_shape = (self.camera.shape[1], self.camera.shape[0])
        self.frame = np.ndarray(self.new_shape)
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

    def load_datasets(self):
        with h5py.File(self.mask_path, 'r') as f:
            mask = np.transpose(f["mask"][()])
        mask = cp.asanyarray(mask, dtype=cp.float32)

        return mask

