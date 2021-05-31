import matplotlib.pyplot as plt
from multiprocessing import Process
import numpy as np
import time


class LiveVideo:
    def __init__(self, fig=None, ax=None, nrows=337, ncols=297):
        self.fig = fig or plt.figure()
        self.ax = ax or plt.gca()
        self.im = self.ax.imshow(np.zeros((nrows, ncols), dtype=np.uint8), animated=True)
        self.ax.set_title('Live Video')

    def update_frame(self, frame):
        self.ax.clear()
        self.ax.imshow(frame)
        plt.pause(0.01)

    @staticmethod
    def _update_frame(ax, frame):
        ax.clear()
        ax.imshow(frame)
        plt.pause(0.1)


# class LiveVideo:
#     def __init__(self):
#         self.fig, self.ax = plt.subplots()
#         self.ax.set_title('Live Video')
#
#     def call_back(self):
#         while self.pipe.poll():
#             command = self.pipe.recv()
#             if command is None:
#                 return False
#             else:
#                 self.ax.clear()
#                 self.ax.imshow(command)
#         self.fig.canvas.draw()
#         return True
#
#     def __call__(self, pipe):
#
#         print('Starting Live Video...')
#
#         self.pipe = pipe
#
#         timer = self.fig.canvas.new_timer(interval=10)
#         timer.add_callback(self.call_back)
#         timer.start()
#
#         print('...done Live Video')
#         plt.show()
