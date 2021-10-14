from core.abstract_process import AbstractProcess
import cupy as cp
import numpy as np


class HemoCorrect(AbstractProcess):
    def __init__(self, hemo_dff, ptr=0):
        self.hemo_dff = hemo_dff

        self.shape = hemo_dff.shape
        self.capacity = hemo_dff.shape[0]
        self.dtype = self.hemo_dff.dtype

        self.regression_coeff = [
            cp.ndarray((self.shape[-2:]), dtype=self.dtype),
            cp.ndarray((self.shape[-2:]), dtype=self.dtype)
        ]

        self.ptr = self.capacity - 1

    def initialize_buffers(self, data, hemo_data):
        """

        :param data: 2D numpy array (pixels, time)
        :param hemo_data: 2D numpy array (pixels, time)
        :return: m, b regression coefficients
        """
        n_samples = np.size(data, 0)
        for i in range(self.shape[1]):
            for j in range(self.shape[2]):
                [theta, _, _, _] = np.linalg.lstsq(
                    np.stack((hemo_data[:, i, j], np.ones((n_samples,))), axis=1),
                    data[:, i, j],
                    rcond=None)
                self.regression_coeff[0][i, j] = theta[0]
                self.regression_coeff[1][i, j] = theta[1]

        self.ptr = self.capacity - 1

    def process(self):
        if self.ptr == self.capacity - 1:
            self.ptr = 0
        else:
            self.ptr += 1

        self.hemo_dff[self.ptr, :, :] = cp.multiply(self.regression_coeff[0], self.hemo_dff[self.ptr, :, :]) + \
                                        self.regression_coeff[1]
