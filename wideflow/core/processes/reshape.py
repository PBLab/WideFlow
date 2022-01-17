from core.abstract_process import AbstractProcess
import cupy as cp


class Reshape(AbstractProcess):
    def __init__(self, src, new_shape, dst, ptr):
        self.src = src
        self.dst = dst
        self.shape = new_shape
        self.capacity = dst.shape[0]
        self.ptr = ptr

    def initialize_buffers(self):
        pass

    def process(self):
        if self.ptr == self.capacity - 1:
            self.ptr = 0
        else:
            self.ptr += 1

        self.dst[self.ptr, :, :] = cp.reshape(self.src, self.new_shape)
