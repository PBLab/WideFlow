from core.abstract_process import AbstractProcess
import cupy as cp


class Mask(AbstractProcess):
    def __init__(self, x, mask, masked_buffer, ptr):
        self.x = x
        self.mask = mask
        self.masked_buffer = masked_buffer
        self.shape = masked_buffer.shape
        self.capacity = self.shape[0]
        self.ptr = ptr

    def initialize_buffers(self):
        pass

    def process(self):
        if self.ptr == self.capacity - 1:
            self.ptr = 0
        else:
            self.ptr += 1

        self.masked_buffer[self.ptr, :, :] = cp.multiply(self.x, self.mask)