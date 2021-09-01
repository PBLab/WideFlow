from core.abstract_pipeline import AbstractPipeLine
from core.processes import *
from core.metrics import *
import cupy as cp
import numpy as np

from utils.load_tiff import load_tiff
import h5py


class DFF1CH(AbstractPipeLine):
    def __init__(self, camera, mapping_coordinates, new_shape, capacity, pattern_path, mask_path):
        self.camera = camera
        self.new_shape = new_shape
        self.capacity = capacity
        self.mapping_coordinates = mapping_coordinates
        self.pattern_path = pattern_path
        self.mask_path = mask_path
        self.mask, self.pattern = self.load_pattern()

        self.input_shape = self.camera.shape

        # allocate memory
        self.frame = np.ndarray(self.input_shape)
        self.input = cp.ndarray(self.input_shape)
        self.warped_input = cp.ndarray((self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.warped_buffer = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.dff_buffer = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)

        # set processes
        map_coord = MapCoordinates(self.input, self.warped_input, self.mapping_coordinates, self.new_shape)
        masking = Mask(self.warped_input, self.mask, self.warped_buffer, ptr=0)
        dff = DFF(self.dff_buffer, self.warped_buffer, ptr=0)
        self.processes_list = [map_coord, masking, dff]

        # set metric
        self.metric = ROIMean(self.threshold_buffer, self.pattern, ptr=0)

        self.ptr = 0

    def fill_buffers(self):
        for _ in range(self.capacity):
            self.get_input()
            self.processes_list[0].process()
            self.processes_list[1].process()

        for process in self.processes_list:
            process.initialize_buffers()

        self.metric.initialize_buffers()

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
        self.metric.evaluate()
        return self.metric.result

    def load_pattern(self):
        with h5py.File(self.mask_path, 'r') as f:
            mask = np.transpose(f["mask"][()])
        mask = cp.asanyarray(mask, dtype=cp.float32)

        pattern = load_tiff(self.pattern_path) / 65535  # convert back from uint16 to original range
        pattern = cp.asanyarray(pattern, dtype=cp.float32)
        pattern = cp.multiply(pattern, mask)

        return mask, pattern
