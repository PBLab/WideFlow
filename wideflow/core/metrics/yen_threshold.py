from core.abstract_metric import AbstractMetric
import cupy as cp
import numpy as np
from skimage.filters import threshold_yen


class YenThreshold(AbstractMetric):
    def __init__(self, x, roi_pixels_list, mask, ptr, yen_bin=256):
        self.x = x
        self.roi_pixels_list = roi_pixels_list
        self.mask = cp.asnumpy(mask)
        self.mask = self.mask == 1  # convert to boolean type
        self.ptr = ptr
        self.yen_bin = yen_bin

        self.shape = x.shape
        self.capacity = self.shape[0]

        self.pixels_inds = np.unravel_index(self.roi_pixels_list, (self.shape[2], self.shape[1]))
        self.n_metric_pixels = len(roi_pixels_list)
        self.n_pixels = np.sum(mask)

        self.scale_factor = (self.n_pixels - self.n_metric_pixels) / self.n_pixels

        self.result = 0

    def initialize_buffers(self):
        self.ptr = self.capacity - 1

    def evaluate(self):
        if self.ptr == self.capacity - 1:
            self.ptr = 0
        else:
            self.ptr += 1

        im_th = cp.asnumpy(self.x[self.ptr])
        th = np.max((0, threshold_yen(im_th[self.mask == 1], self.yen_bin)))
        im_th = im_th > th

        metric_active_pixels = np.sum(im_th[self.pixels_inds[1], self.pixels_inds[0]])
        bg_active_pixels = np.sum(im_th) + 1
        self.result = (metric_active_pixels / bg_active_pixels) * self.scale_factor
