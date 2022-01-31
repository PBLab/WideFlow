from core.abstract_metric import AbstractMetric
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

        self.n_rois = len(self.rois_dict)
        self.shape = x.shape
        self.capacity = self.shape[0]
        self.ptr_list = np.arange(self.capacity)

        for i, (roi_key, roi_dict) in enumerate(self.rois_dict.items()):
            self.rois_dict[roi_key]['unravel_index'] = np.unravel_index(roi_dict['PixelIdxList'], (self.shape[2], self.shape[1]))

        self.metric_list = [True if key in self.eval_rois_names else False for key in self.rois_dict]
        self.non_metric_list = [False if key in self.eval_rois_names else True for key in self.rois_dict]

        self.diff = np.zeros((self.n_rois, ), dtype=cp.float32())
        self.rois_mean = cp.zeros((self.n_rois, self.capacity), dtype=cp.float32())
        self.metric_rois_mean = np.zeros((self.capacity, ), dtype=np.float32())

        self.eps = 1e-16
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

        for i, (roi_key, roi_dict) in enumerate(self.rois_dict.items()):
            self.rois_mean[i, self.ptr] = cp.mean(self.x[self.ptr, roi_dict['unravel_index'][1], roi_dict['unravel_index'][0]])

        self.diff[:] = cp.asnumpy(self.rois_mean[:, self.ptr] - self.rois_mean[:, self.delta_ptr])
        std = np.std(self.diff)
        self.result = (np.mean(self.diff[self.metric_list]) - np.mean(self.diff)) / (std + self.eps)
