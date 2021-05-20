import matplotlib.pyplot as plt
import numpy as np
import random


class TracePlot:
    def __init__(self, n_traces, expected_amp, labels, n_frames, fig=None, ax=None):
        self.fig = fig or plt.figure()
        self.ax = ax or plt.gca()
        self.n_traces = n_traces
        self.expected_amp = expected_amp
        self.labels = labels
        self.n_frames = n_frames

        self.trace_space = np.linspace(start=0, stop=self.n_traces*self.expected_amp/3, num=self.n_traces)
        self.traces_data = np.zeros((n_traces, n_frames))
        self.traces_lines = [None] * self.n_traces

        self.x_data = np.linspace(start=0, stop=self.n_frames-1, num=self.n_frames, dtype=np.int)
        self.HSV_tuples = [(x * 1.0 / self.n_traces, 0.5, 0.5) for x in range(self.n_traces)]
        random.shuffle(self.HSV_tuples)
        self.set_plot_axes_labels()

    def set_plot_axes_labels(self):
        self.ax.set_xlabel("time")
        self.ax.set_xticks(self.x_data)
        self.ax.set_xticklabels(np.flip(self.x_data))

        self.ax.set_ylabel("ROIs")
        self.ax.set_yticks(self.trace_space)
        self.ax.set_yticklabels(self.labels)

        self.ax.set_title('ROIs traces')

        for i, trace in enumerate(self.traces_data):
            self.traces_data[i] = self.traces_data[i] + self.trace_space[i]
            self.traces_lines[i] = self.ax.plot(self.x_data, trace, color=self.HSV_tuples[i])

    def update_plot(self, samples):
        self.traces_data = np.roll(self.traces_data, -1, axis=1)
        self.traces_data[:, -1] = samples + self.trace_space
        for i, line in enumerate(self.traces_lines):
            line[0].set_ydata(self.traces_data[i])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# plt.ion()
# trace_plot = TracePlot(3, 1, ['a','b','c'], 16)
# for i in range(100):
#     data = np.random.random((3,))
#     trace_plot.update_plot(data)