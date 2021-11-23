from wideflow.core.abstract_metric import AbstractMetric
import cupy as cp
import numpy as np


class ROIContrast(AbstractMetric):
    def __init__(self, x, rois_dict, eval_rois_names, mask, ptr):
        self.x = x
        self.rois_dict = rois_dict
        self.eval_rois_names = eval_rois_names
        self.mask = mask == 1  # convert to boolean type
        self.ptr = ptr

        self.shape = x.shape
        self.capacity = self.shape[0]

        n_rois = len(self.rois_dict)
        self.rois_mean = np.zeros((n_rois, ), dtype=np.float32())
        self.eval_rois_ind = []
        for i, (roi_key, roi_dict) in enumerate(self.rois_dict.items()):
            self.rois_dict[roi_key]['unravel_index'] = np.unravel_index(roi_dict['PixelIdxList'], (self.shape[2], self.shape[1]))
            if roi_key in self.eval_rois_names:
                self.eval_rois_ind.append(i)

        self.eps = 1e-16
        self.result = 0

    def initialize_buffers(self):
        self.ptr = self.capacity - 1

    def evaluate(self):
        if self.ptr == self.capacity - 1:
            self.ptr = 0
        else:
            self.ptr += 1

        for i, (roi_key, roi_dict) in enumerate(self.rois_dict.items()):
            self.rois_mean[i] = cp.mean(self.x[self.ptr, roi_dict['unravel_index'][1], roi_dict['unravel_index'][0]])

        np_rois_mean = cp.asnumpy(self.rois_mean)
        all_rois_mean = np.mean(np_rois_mean)
        std = np.std(np_rois_mean)
        eval_rois_mean = np.mean(np_rois_mean[self.eval_rois_ind])

        self.result = (eval_rois_mean - all_rois_mean) / (std + self.eps)
