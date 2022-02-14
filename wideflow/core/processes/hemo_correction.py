from core.abstract_process import AbstractProcess
import cupy as cp
import numpy as np


class HemoCorrect(AbstractProcess):
    def __init__(self, hemo_dff, regression_coeff, ptr=0):
        self.hemo_dff = hemo_dff

        self.shape = hemo_dff.shape
        self.capacity = hemo_dff.shape[0]
        self.dtype = self.hemo_dff.dtype

        if regression_coeff is None:
            self.regression_coeff = [
                cp.ones((self.shape[-2:]), dtype=self.dtype),
                cp.zeros((self.shape[-2:]), dtype=self.dtype)
            ]
        else:
            self.regression_coeff = [None, None]
            self.regression_coeff[0] = cp.asanyarray(regression_coeff[0])
            self.regression_coeff[1] = cp.asanyarray(regression_coeff[1])
        self.ptr = ptr

    def initialize_buffers(self):
        self.hemo_dff[:] = cp.multiply(self.regression_coeff[0], self.hemo_dff) + \
                                        self.regression_coeff[1]
        self.ptr = self.capacity - 1

    def process(self):
        if self.ptr == self.capacity - 1:
            self.ptr = 0
        else:
            self.ptr += 1

        self.hemo_dff[self.ptr, :, :] = cp.multiply(self.regression_coeff[0], self.hemo_dff[self.ptr, :, :]) + \
                                        self.regression_coeff[1]
