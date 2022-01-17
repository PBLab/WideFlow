from wideflow.core.abstract_metric import AbstractMetric
import cupy as cp
import numpy as np


class ROIDiff(AbstractMetric):
    def __init__(self, x, rois_dict, eval_rois_names, delta, ptr=0):
        self.x = x
        self.rois_dict = rois_dict
        self.eval_rois_names = eval_rois_names
        self.delta = delta
        self.ptr = ptr
        self.delta_ptr = 0

        self.shape = x.shape
        self.capacity = self.shape[0]
        self.ptr_list = np.arange(self.capacity)

        self.metric_pixels = [np.ndarray((1, )), np.ndarray((1, ))]
        for roi_key in self.eval_rois_names:
            pix_ind = np.unravel_index(rois_dict[roi_key]['PixelIdxList'], (self.shape[2], self.shape[1]))
            self.metric_pixels[0] = np.concatenate((self.metric_pixels[0], pix_ind[0]))
            self.metric_pixels[1] = np.concatenate((self.metric_pixels[1], pix_ind[1]))
        self.metric_pixels[0] = self.metric_pixels[0][1:]
        self.metric_pixels[1] = self.metric_pixels[1][1:]

        self.metric_rois_mean = np.zeros((self.capacity, ), dtype=np.float32())
        self.result = 0

    def initialize_buffers(self):
        self.ptr = self.capacity - 1 - self.delta
        for i in range(self.delta):
            self.evaluate()

    def evaluate(self):
        if self.ptr == self.capacity - 1:
            self.ptr = 0
        else:
            self.ptr += 1
        self.delta_ptr = self.ptr_list[self.ptr - self.delta]

        self.metric_rois_mean[self.ptr] = cp.asnumpy(cp.mean(self.x[self.ptr, self.metric_pixels[1], self.metric_pixels[0]]))
        self.result = self.metric_rois_mean[self.ptr] - self.metric_rois_mean[self.delta_ptr]
