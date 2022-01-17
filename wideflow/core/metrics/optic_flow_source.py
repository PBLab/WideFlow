from core.abstract_metric import AbstractMetric
import cupy as cp
import numpy as np
from skimage.filters import threshold_yen


class OpticFlowSource(AbstractMetric):
    def __init__(self, x, roi_pixels_list, ptr):
        self.x = x
        self.roi_pixels_list = roi_pixels_list
        self.mask = self.mask == 1  # convert to boolean type
        self.ptr = ptr

        self.shape = x.shape
        self.capacity = self.shape[0]

        self.pixels_inds = np.unravel_index(self.roi_pixels_list, (self.shape[2], self.shape[1]))

        self.result = 0

    def initialize_buffers(self):
        self.ptr = self.capacity - 1

    def evaluate(self):
        if self.ptr == self.capacity - 1:
            self.ptr = 0
        else:
            self.ptr += 1



