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

    def initialize_buffers(self):
        self.ptr = self.capacity - 1

    def process(self):
        self.prev_ptr = self.ptr
        if self.ptr == self.capacity - 1:
            self.ptr = 0
        else:
            self.ptr += 1

        cv2.cuda_DenseOpticalFlow(self.x[self.prev_ptr], self.x[self.ptr], self.flow)
