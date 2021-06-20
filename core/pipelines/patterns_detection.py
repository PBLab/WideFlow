from core.abstract_pipeline import AbstractPipeLine
from core.processes import *
from core.metrics.correlation import Correlation
import cupy as cp
import numpy as np


class PatternsDetection(AbstractPipeLine):
    def __init__(self, camera, new_shape, capacity, mapping_coordinates, pattern, mask):
        self.camera = camera
        self.new_shape = new_shape
        self.capacity = capacity
        self.mapping_coordinates = mapping_coordinates
        self.pattern = pattern
        self.mask = mask
        self.input_shape = self.camera.shape

        # allocate memory
        self.frame = np.ndarray(self.input_shape)
        self.input = cp.ndarray(self.input_shape)
        self.warped_input = cp.ndarray((self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.warped_buffer = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.dff_buffer = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.threshold_buffer = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)

        # set processes
        map_coord = MapCoordinates(self.input, self.warped_input, self.mapping_coordinates, self.new_shape)
        masking = Mask(self.warped_input, self.mask, self.warped_buffer, ptr=0)
        dff = DFF(self.dff_buffer, self.warped_buffer, ptr=0)
        std_threshold = StdThrehold(self.dff_buffer, self.threshold_buffer, ptr=0)
        self.processes_list = [map_coord, masking, dff, std_threshold]

        # set metric
        self.metric = Correlation(self.threshold_buffer, self.pattern, ptr=0)

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
        # self.c = self.c + 1
        # if self.c % 2:
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