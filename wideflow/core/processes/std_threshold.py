from wideflow.core.abstract_process import AbstractProcess
import cupy as cp


class StdThrehold(AbstractProcess):
    def __init__(self, x3d, x3d_th, steps=1, ptr=0):
        self.x3d = x3d
        self.x3d_th = x3d_th
        self.steps = steps
        self.ptr = ptr

        self.shape = x3d.shape
        self.capacity = self.shape[0]
        self.size = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtype = self.x3d_th.dtype

        self.x3d_mean_map = cp.ndarray(x3d_th.shape[-2:], dtype=self.dtype)
        self.x3d_old_mean_map = cp.ndarray(x3d_th.shape[-2:], dtype=self.dtype)
        self.x3d_std_map = cp.ndarray(x3d_th.shape[-2:], dtype=self.dtype)
        self.old_sample = cp.ndarray(x3d_th.shape[-2:], dtype=self.dtype)
        self.new_sample = cp.ndarray(x3d_th.shape[-2:], dtype=self.dtype)

    def initialize_buffers(self):

        self.x3d_mean_map[:] = cp.mean(self.x3d, axis=0)
        self.x3d_std_map[:] = cp.std(self.x3d, axis=0)
        self.old_sample[:] = self.x3d[self.ptr, :, :]

        self.x3d_th[:] = self.x3d
        self.x3d_th[self.x3d_th < self.x3d_mean_map + self.steps * self.x3d_std_map] = 0

        self.ptr = self.capacity - 1

    def process(self):
        if self.ptr == self.capacity - 1:
            self.ptr = 0
        else:
            self.ptr += 1

        self.new_sample[:] = self.x3d[self.ptr, :, :]

        self.update_mean_map()
        self.update_std_map()

        self.new_sample[self.new_sample < self.x3d_mean_map + self.steps * self.x3d_std_map] = 0
        self.x3d_th[self.ptr, :, :] = self.new_sample

        self.old_sample[:] = self.x3d[(self.ptr + 1) % self.capacity, :, :]

    def update_mean_map(self):
        self.x3d_old_mean_map[:] = self.x3d_mean_map
        self.x3d_mean_map[:] = self.x3d_mean_map - \
                               (self.old_sample - self.new_sample) / self.capacity

    def update_std_map(self):
        self.x3d_std_map[:] = \
            cp.sqrt(
                cp.divide(
                    cp.square(self.x3d[self.ptr, :, :] - self.x3d_mean_map) -
                    cp.square(self.old_sample - self.x3d_old_mean_map) +
                    cp.multiply(self.capacity - 1, cp.square(self.x3d_mean_map) - cp.square(self.x3d_old_mean_map)) -
                    2 * cp.multiply(self.capacity * self.x3d_mean_map - self.x3d[self.ptr, :, :],
                                    self.x3d_mean_map - self.x3d_old_mean_map)
                    , self.capacity
                )
                + cp.square(self.x3d_std_map)
            )

        self.x3d_std_map[:] = \
            cp.sqrt(
                cp.divide(
                    2 * self.capacity * cp.multiply(self.x3d_mean_map - self.x3d_old_mean_map,
                                                    self.old_sample - self.x3d_old_mean_map) +
                    self.capacity * (self.capacity - 1) * cp.square(self.x3d_mean_map - self.x3d_old_mean_map)
                    , self.capacity
                )
                + cp.square(self.x3d_std_map)
            )

