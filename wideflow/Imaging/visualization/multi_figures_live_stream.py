from Imaging.visualization.abstract_visualization import AbstractVis

import matplotlib.pyplot as plt
from multiprocessing import shared_memory
import numpy as np


class MultiFigsLiveVideo(AbstractVis):
    def __init__(self, queries, images_props):
        self.queries = queries
        self.images_props = images_props

        if len(queries) != len(images_props):
            raise ValueError("length of queries and images_props arguments must be the same")
        self.n_figs = len(queries)

        self.figures, self.axes = [], []
        for i in range(self.n_figs):
            self.figures[i], self.axes[i] = plt.subplots()

        print("Open Live Stream")

    def __call__(self, shm_names):
        images = []
        for i, name in enumerate(shm_names):
            props = self.images_props[i]
            existing_shm = shared_memory.SharedMemory(name=name)
            images[i] = np.ndarray(shape=props["image_shape"], dtype=props["image_dtype"], buffer=existing_shm.buf)
            self.axes[i].imshow(images[i], vmin=self.vmin, vmax=self.vmax)

            plt.colorbar()

        while True:
            for query, image, fig, ax in zip(self.queries, images, self.figures, self.axes):
                if self.query.empty():
                    continue

                q = query.get()
                if q == "draw":
                    ax.clear()
                    ax.imshow(image, vmin=self.vmin, vmax=self.vmax)
                    fig.canvas.draw()
                    plt.pause(1/self.frame_rate)

                elif q == "terminate":
                    print("live video terminating")
                    self.terminate()
                    break

                else:
                    raise KeyError(f'LiveVideo query "{q}" is invalid')

    def terminate(self):
        for fig in self.figures:
            plt.close(fig)


