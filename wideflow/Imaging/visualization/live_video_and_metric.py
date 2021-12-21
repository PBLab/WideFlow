from Imaging.visualization.abstract_visualization import AbstractVis

import matplotlib.pyplot as plt
from multiprocessing import shared_memory
import numpy as np


class LiveVideo(AbstractVis):
    def __init__(self, query, image_shape, frame_rate=50, vmin=-0.05, vmax=0.1):
        self.query = query
        self.image_shape = image_shape
        self.frame_rate = frame_rate
        self.vmax = vmax
        self.vmin = vmin

        f, (self.vid_ax, self.metric_ax) = plt.subplots(1,2)
        self.metric_ax.set_xlim([-1, 1])
        self.metric_ax.set_ylim([-3, 3])
        self.vid_ax.set_title('Live Video')

    def __call__(self, vid_shared_mem_name, metric_shared_mem_name, threshold_shared_mem_name):
        vid_existing_shm = shared_memory.SharedMemory(name=vid_shared_mem_name)
        image = np.ndarray(shape=self.image_shape, dtype=np.float32, buffer=vid_existing_shm.buf)

        metric_existing_shm = shared_memory.SharedMemory(name=metric_shared_mem_name)
        metric = np.ndarray(shape=self.image_shape, dtype=np.float32, buffer=metric_existing_shm.buf)

        threshold_existing_shm = shared_memory.SharedMemory(name=threshold_shared_mem_name)
        threshold = np.ndarray(shape=self.image_shape, dtype=np.float32, buffer=threshold_existing_shm.buf)

        self.vid_ax.imshow(image, vmin=self.vmin, vmax=self.vmax)
        plt.colorbar()

        metric_bar = self.metric_ax.bar(0, 0)
        threshold_line = self.metric_ax.hlines(0, -1, 1)
        while True:
            if self.query.empty():
                continue

            q = self.query.get()
            if q == "draw":
                self.vid_ax.clear()
                self.vid_ax.imshow(image, vmin=self.vmin, vmax=self.vmax)

                metric_bar.remove()
                threshold_line.remove()
                metric_bar = self.metric_ax.bar(0, metric)
                threshold_line = self.metric_ax.hlines(threshold, -1, 1)

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


