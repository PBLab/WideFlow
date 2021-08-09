from wideflow.Imaging.visualization.abstract_visualization import AbstractVis
import matplotlib.pyplot as plt
from multiprocessing import shared_memory
import numpy as np


class LiveVideo(AbstractVis):
    def __init__(self, query, image_shape, frame_rate=50, vmin=-0.1, vmax=0.2):
        self.query = query
        self.image_shape = image_shape
        self.frame_rate = frame_rate
        self.vmax = vmax
        self.vmin = vmin

        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Live Video')

    def __call__(self, shared_mem_name):
        existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
        image = np.ndarray(shape=self.image_shape, dtype=np.float32, buffer=existing_shm.buf)
        plt.imshow(image)
        plt.colorbar()
        while True:
            if self.query.empty():
                continue

            q = self.query.get()
            if q == "draw":
                self.ax.clear()
                self.ax.imshow(image, vmin=self.vmin, vmax=self.vmax)
                self.fig.canvas.draw()
                plt.pause(1/self.frame_rate)
            elif q == "terminate":
                print("live video terminating")
                self.terminate()
                break
            else:
                raise KeyError(f'LiveVideo query "{q}" is invalid')

    def terminate(self):
        plt.close(self.fig)

