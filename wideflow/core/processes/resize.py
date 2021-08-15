from wideflow.core.abstract_process import AbstractProcess
import cupyx.scipy.ndimage as csn
import cupy as cp
import numpy as np


class Resize(AbstractProcess):
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.new_shape = dst.shape
        self.old_shape = src.shape

        trans_mat = cp.eye(3)
        trans_mat[0][0] = self.old_shape[0] / self.new_shape[0]
        trans_mat[1][1] = self.old_shape[1] / self.new_shape[1]
        self.trans_mat = trans_mat

    def initialize_buffers(self):
        pass

    def process(self):
        csn.affine_transform(self.src, self.trans_mat, output_shape=self.new_shape, output=self.dst,
                             order=1, mode='opencv')
