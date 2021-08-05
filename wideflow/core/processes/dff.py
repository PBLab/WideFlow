from wideflow.core.abstract_process import AbstractProcess
import cupy as cp
import cupyx.scipy.ndimage as csn
import numpy as np


class DFF(AbstractProcess):
    def __init__(self, dff, signal, weights_shape=[5, 1, 1], ptr=0):
        self.dff = dff
        self.signal = signal
        self.weights_shape = weights_shape
        self.weights = cp.ones(self.weights_shape) * (1/np.prod(self.weights_shape))
        self.shape = dff.shape
        self.capacity = dff.shape[0]
        self.ptr = ptr
        self.eps = 1e-16

        self.baseline = cp.ndarray(self.shape[-2:])

    def initialize_buffers(self):
        self.calc_baseline()
        self.dff[:] = cp.divide(self.signal - cp.stack((self.baseline,) * self.capacity),
                                cp.stack((self.baseline,) * self.capacity) + self.eps)
        self.ptr = self.capacity - 1

    def process(self):
        if self.ptr == self.capacity - 1:
            self.ptr = 0
        else:
            self.ptr += 1

        self.calc_baseline()
        self.dff[self.ptr, :, :] = cp.divide(self.signal[self.ptr, :, :] - self.baseline,
                                             self.baseline + self.eps)

    def calc_baseline(self):
        self.baseline = cp.min(csn.convolve(cp.roll(self.signal, -self.ptr, 0), self.weights), 0)

