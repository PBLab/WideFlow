import matplotlib.pyplot as plt
import numpy as np


class LiveVideo:
    def __init__(self, fig=None, ax=None, nrows=337, ncols=297):
        self.fig = fig or plt.figure()
        self.ax = ax or plt.gca()
        self.im = self.ax.imshow(np.zeros((nrows, ncols), dtype=np.uint8), animated=True)
        self.ax.set_title('Live Video')

    def update_frame(self, frame):
        # self.ax.imshow(frame)
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        self.ax.clear()
        self.ax.imshow(frame)
        plt.pause(0.01)




# plt.ion()
# trace_plot = TracePlot(3, 1, ['a','b','c'], 16)
# for i in range(100):
#     data = np.random.random((3,))
#     trace_plot.update_plot(data)