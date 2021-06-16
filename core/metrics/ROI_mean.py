from core.abstract_metric import AbstractMetric
import cupy as cp
import numpy as np


class ROIMean(AbstractMetric):
    def __init__(self, x, roi_pixels_list, ptr):
        self.x = x
        self.roi_pixels_list = roi_pixels_list
        self.ptr = ptr
        self.shape = x.shape
        self.capacity = self.shape[0]
        self.pixels_inds = np.unravel_index(self.roi_pixels_list, self.shape)

    def initialize_buffers(self):
        self.ptr = self.capacity - 1

    def evaluate(self):
        return cp.mean(self.x[self.ptr, self.pixels_inds[0], self.pixels_inds[1]])

