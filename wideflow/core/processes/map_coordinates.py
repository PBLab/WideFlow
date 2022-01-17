from core.abstract_process import AbstractProcess
import cupyx.scipy.ndimage as csn
import cupy as cp
import numpy as np


class MapCoordinates(AbstractProcess):
    def __init__(self, src, dst, coordinates, new_shape):
        self.src = src
        self.dst = dst
        self.new_shape = new_shape

        if int(cp.__version__[0]) < 9:
            coordinates = np.squeeze(coordinates)
            self.dst_flat = cp.ndarray((self.new_shape[0] * self.new_shape[1], ))
        else:
            self.dst_flat = cp.ndarray((self.new_shape[0] * self.new_shape[1], 1))
        self.coordinates = coordinates

    def initialize_buffers(self):
        pass

    def process(self):
        csn.map_coordinates(self.src, self.coordinates, self.dst_flat, order=1)
        self.dst[:] = cp.reshape(self.dst_flat, self.new_shape)