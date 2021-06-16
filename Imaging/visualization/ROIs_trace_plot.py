from .multiple_trace_plot import TracePlot
import matplotlib.pyplot as plt
from multiprocessing import shared_memory
import numpy as np
from utils.load_matlab_vector_field import load_extended_rois_list


class ROIsTracePlot(TracePlot):
    def __init__(self, queue, rois_dict, n_traces, expected_amp, n_frames, update_rate=50):
        self.rois_dict = rois_dict
        labels = list(self.rois_dict.keys())
        super().__init__(queue, n_traces, expected_amp, labels, n_frames, update_rate)

    def extract_rois_data(self, image):
        shape = image.shape
        rois_data = [0] * len(self.roi_dict)
        i = 0
        for key, val in self.roi_dict.items():
            pixels_inds = np.unravel_index(val['PixelIdxList'], shape)
            rois_data[i] = np.mean(image[pixels_inds])
            i += 1

        return rois_data

    def __call__(self, shared_mem_name):
        existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
        samples = np.ndarray(shape=(self.n_traces,), buffer=existing_shm.buf)

        while True:
            if self.queue.empty():
                continue

            q = self.queue.get()
            print(q)
            if q == "draw":
                roi_trace = np.reshape(self.extract_rois_data(samples), (self.n_traces, 1))
                self.traces_data = np.roll(self.traces_data, -1, axis=1)
                self.traces_data[:, -1] = roi_trace + self.trace_space

                for i, line in enumerate(self.traces_lines):
                    line[0].set_ydata(self.traces_data[i])
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.pause(1 / self.update_rate)

            elif q == "terminate":
                self.terminate()
                break
            else:
                raise KeyError(f'TracePlot query "{q}" is invalid')