import matplotlib.pyplot as plt
from multiprocessing import shared_memory
import numpy as np


class LiveVideo:
    def __init__(self, query, frame_rate=50):
        self.frame_rate = frame_rate
        self.query = query
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Live Video')

    def __call__(self, shared_mem_name):
        existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
        image = np.ndarray(shape=(297, 337), dtype=np.float32, buffer=existing_shm.buf)
        while True:
            if self.query.empty():
                continue

            q = self.query.get()
            print(f"-------------query: {q}-----------")
            if q == "draw":
                print(f"image max: {np.max(image)}")
                self.ax.clear()
                self.ax.imshow(image)
                self.fig.canvas.draw()
                plt.pause(1/self.frame_rate)
            elif q == "terminate":
                self.terminate()
                break
            else:
                raise KeyError(f'LiveVideo query "{q}" is invalid')

    def terminate(self):
        plt.close(self.fig)

