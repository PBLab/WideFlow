import matplotlib.pyplot as plt
import numpy as np


class TracePlot:
    def __init__(self, ax, n_traces, expected_amp, labels, n_frames):
        self.ax = ax
        self.n_traces = n_traces
        self.expected_amp = expected_amp
        self.legend = labels

    def set_plot_axes_labels(self):
        self.ax.set_xlabel("time")
        self.ax.set_xticks(np.linspace(start=0, stop=self.n_frames-1, num=self.n_frames))
        self.ax.set_xticklabels(np.flip(np.linspace(start=0, stop=self.n_frames-1, num=self.n_frames)))

        self.ax.set_ylabel("ROIs")
        self.ax.set_yticks(np.linspace(start=0, stop=self.n_traces*self.expected_amp, num=self.n_traces))
        self.ax.set_yticklabels(self.labels)

        self.ax.set_title('ROIs traces')

    def update_plot(self, samples):
        pass



