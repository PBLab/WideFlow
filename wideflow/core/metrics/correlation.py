from wideflow.core.abstract_metric import AbstractMetric
import cupy as cp


class Correlation(AbstractMetric):
    def __init__(self, x, y, ptr):
        self.x = x
        self.y = y
        self.ptr = ptr

        self.shape = x.shape
        self.capacity = self.shape[0]
        self.size = self.shape[0] * self.shape[1] * self.shape[2]
        self.size2d = self.shape[1] * self.shape[2]
        self.dtype = self.x.dtype

        self.x_mean = cp.ndarray((1,), dtype=self.dtype)
        self.y_mean = cp.ndarray((1,), dtype=self.dtype)
        self.x_std = cp.ndarray((1,), dtype=self.dtype)
        self.y_std = cp.ndarray((1,), dtype=self.dtype)
        self.x_old_mean = cp.ndarray((1,), dtype=self.dtype)

        self.old_sample = cp.ndarray(self.shape[-2:], dtype=self.dtype)
        self.new_sample = cp.ndarray(self.shape[-2:], dtype=self.dtype)

        self.result = cp.float32()

    def initialize_buffers(self):
        self.x_mean[:] = cp.mean(self.x)
        self.y_mean[:] = cp.mean(self.y)
        self.x_std[:] = cp.std(self.x)
        self.y_std[:] = cp.std(self.y)

        self.old_sample[:] = self.x[self.ptr, :, :]

        self.ptr = self.capacity - 1

    def evaluate(self):
        if self.ptr == self.capacity - 1:
            self.ptr = 0
        else:
            self.ptr += 1
        self.new_sample[:] = self.x[self.ptr, :, :]

        self.update_mean()
        self.update_std()

        self.result = cp.divide(
            cp.mean(cp.multiply(cp.roll(self.x, -self.ptr, 0) - self.x_mean, self.y - self.y_mean)),
            cp.multiply(self.x_std, self.y_std))

        self.old_sample[:] = self.x[(self.ptr + 1) % self.capacity, :, :]

    def update_mean(self):
        self.x_old_mean[:] = self.x_mean
        self.x_mean[:] = self.x_mean - \
                         cp.sum(self.old_sample - self.new_sample) / self.size

    def update_std(self):
        self.x_std[:] = \
            cp.sqrt(
                cp.divide(
                    cp.sum(
                        cp.square(self.new_sample - self.x_mean) -
                        cp.square(self.old_sample - self.x_old_mean) +
                        2 * (self.x_mean - self.x_old_mean) * self.new_sample
                    )
                    - (self.size - self.size2d) * self.x_old_mean ** 2
                    - (self.size + self.size2d) * self.x_mean ** 2
                    + 2 * self.size * self.x_old_mean * self.x_mean
                    , self.size
                )
                + cp.square(self.x_std)
            )

