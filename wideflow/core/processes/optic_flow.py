from core.abstract_process import AbstractProcess
import cupy as cp
import cv2
import numpy as np


class OptciFlow(AbstractProcess):
    def __init__(self, x, flow, ptr=0):
        self.x = x
        self.flow = flow
        self.ptr = ptr
        self.prev_ptr = self.ptr - 1

        self.shape = x.shape
        self.capacity = x.shape[0]

        self.nvof = cv2.cuda_DenseOpticalFlow(self.shape[2], self.shape[1], 5, False, False, False, 0)  # TODO: receive last params as args to the process
        self.u = np.ndarray(self.shape[-2:])
        self.v = np.ndarray(self.shape[-2:])

    def initialize_buffers(self):
        self.ptr = self.capacity - 1

    def process(self):
        self.prev_ptr = self.ptr
        if self.ptr == self.capacity - 1:
            self.ptr = 0
        else:
            self.ptr += 1

        self.nvof.calc(self.x[self.prev_ptr], self.x[self.ptr], self.flow)
        self.u = self.flow[:, :, 0]
        self.v = -self.flow[:, :, 1]