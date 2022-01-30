from core.abstract_process import AbstractProcess
import cupy as cp


class Mask(AbstractProcess):
    def __init__(self, x, mask, masked_buffer, crop_roi, ptr):
        self.x = x
        self.mask = mask
        self.masked_buffer = masked_buffer
        self.shape = masked_buffer.shape
        self.capacity = self.shape[0]
        self.crop_roi = crop_roi
        self.ptr = ptr

    def initialize_buffers(self):
        self.ptr = self.capacity - 1

    def process(self):
        if self.ptr == self.capacity - 1:
            self.ptr = 0
        else:
            self.ptr += 1

        self.masked_buffer[self.ptr, :, :] = \
            cp.multiply(self.x, self.mask)[self.ptr, self.crop_roi[0]: self.crop_roi[1], self.crop_roi[2]:self.crop_roi[3]]
