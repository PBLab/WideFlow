from core.abstract_process import AbstractProcess
from cupyx.scipy.ndimage import affine_transform
import cupy as cp


class AffineTrans(AbstractProcess):
    def __init__(self, src, dst, affine_mat, new_shape):
        self.src = src
        self.dst = dst
        self.affine_mat = cp.asanyarray(affine_mat)
        self.new_shape = new_shape

    def initialize_buffers(self):
        pass

    def process(self):
        affine_transform(self.src, self.affine_mat, output_shape=self.new_shape, output=self.dst)
