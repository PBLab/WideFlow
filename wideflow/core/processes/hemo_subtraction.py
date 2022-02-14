from core.abstract_process import AbstractProcess
import cupy as cp


class HemoSubtraction(AbstractProcess):
    def __init__(self, dff, hemo_dff, ptr=0):
        self.dff = dff
        self.hemo_dff = hemo_dff

        self.shape = dff.shape
        self.capacity = dff.shape[0]
        self.dtype = self.dff.dtype
        self.ptr = ptr

        self.sub_mean = cp.ndarray((self.shape[-2:]), dtype=self.dtype)
        self.old_sample = cp.ndarray(self.shape[-2:], dtype=self.dtype)
        self.new_sample = cp.ndarray(self.shape[-2:], dtype=self.dtype)

    def initialize_buffers(self):
        self.dff[:] = self.dff - self.hemo_dff
        self.old_sample[:] = self.dff[self.ptr, :, :]
        self.sub_mean[:] = cp.mean(self.dff, axis=0)
        self.dff[:] = self.dff - self.sub_mean

        self.ptr = self.capacity - 1

    def process(self):
        if self.ptr == self.capacity - 1:
            self.ptr = 0
        else:
            self.ptr += 1

        self.dff[self.ptr, :, :] = self.dff[self.ptr, :, :] - self.hemo_dff[self.ptr, :, :]
        self.new_sample[:] = self.dff[self.ptr, :, :]
        self.update_mean()
        self.old_sample[:] = self.dff[self.ptr, :, :]
        self.dff[self.ptr, :, :] = self.dff[self.ptr, :, :] - self.sub_mean

    def update_mean(self):
        self.sub_mean[:] = self.sub_mean - \
                         (self.old_sample - self.new_sample) / self.capacity
