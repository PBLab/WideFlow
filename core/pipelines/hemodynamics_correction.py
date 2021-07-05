from core.abstract_pipeline import AbstractPipeLine
from core.processes import *
from core.metrics import ROIMean
import cupy as cp
import numpy as np

from utils.load_matlab_vector_field import load_extended_rois_list
import h5py

from pyvcam.constants import PARAM_LAST_MUXED_SIGNAL


class HemoDynamicsDFF(AbstractPipeLine):
    def __init__(self, camera, mapping_coordinates, new_shape, capacity, rois_dict_path, mask_path, roi_name, regression_n_samples):
        self.camera = camera
        self.new_shape = new_shape
        self.capacity = capacity + capacity % 2  # make sure capacity is an odd number
        self.mapping_coordinates = mapping_coordinates
        self.rois_dict_path = rois_dict_path
        self.mask_path = mask_path
        self.mask, self.rois_dict = self.load_datasets()
        self.roi_name = roi_name
        self.regression_n_samples = int(np.floor(regression_n_samples / (capacity * 2)) * (capacity * 2))

        self.input_shape = self.camera.shape

        # allocate memory
        self.frame = np.ndarray(self.input_shape)
        self.input = cp.ndarray(self.input_shape)
        self.warped_input = cp.ndarray((self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.warped_buffer = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.warped_buffer_ch2 = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.dff_buffer = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.dff_buffer_ch2 = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)

        self.regression_buffer = np.ndarray((self.regression_n_samples, self.new_shape[0], self.new_shape[1], 2), dtype=np.float32)

        map_coord = MapCoordinates(self.input, self.warped_input, self.mapping_coordinates, self.new_shape)
        # set processes for channel 1
        masking = Mask(self.warped_input, self.mask, self.warped_buffer, ptr=0)
        dff = DFF(self.dff_buffer, self.warped_buffer, ptr=0)
        hemo_subtract = HemoSubtraction(self.dff_buffer, self.dff_buffer_ch2, ptr=0)
        self.processes_list = [map_coord, masking, dff, hemo_subtract]

        # set processes for channel 2
        masking_ch2 = Mask(self.warped_input, self.mask, self.warped_buffer_ch2, ptr=0)
        dff_ch2 = DFF(self.dff_buffer_ch2, self.warped_buffer_ch2, ptr=0)
        Hemo_correct = HemoCorrect(self.dff_buffer_ch2, ptr=0)
        self.processes_list_ch2 = [map_coord, masking_ch2, dff_ch2, Hemo_correct]

        # set metric
        roi_pixels_list = self.rois_dict[self.roi_name]["PixelIdxList"]
        self.metric = ROIMean(self.dff_buffer, roi_pixels_list, ptr=0)

        self.camera.set_param(PARAM_LAST_MUXED_SIGNAL, 2)  # setting camera active output wires to 2 - strobbing of two LEDs
        self.ptr = self.capacity - 1

    def fill_buffers(self):

        # initialize buffers
        for i in range(self.capacity*2):
            self.get_input()
            if not i % 2:
                self.processes_list[0].process()
                self.processes_list[1].process()
            else:
                self.processes_list_ch2[0].process()
                self.processes_list_ch2[1].process()

        for process in self.processes_list[:3]:
            process.initialize_buffers()

        for process in self.processes_list_ch2[:3]:
            process.initialize_buffers()

        # collect data to calculate regression coefficient for the hemodynamic correction
        ch1i, ch2i = 0, 0
        for i in range(self.regression_n_samples*2):
            if self.ptr == self.capacity - 1:
                self.ptr = 0
            else:
                self.ptr += 1

            self.get_input()
            if not i % 2:
                print(self.camera.frame_idx)
                for process in self.processes_list[:3]:
                    process.initialize_buffers()
                self.regression_buffer[ch1i, :, :, 0] = cp.asnumpy(self.dff_buffer[self.ptr, :, :])
                ch1i += 1

            else:
                for process in self.processes_list_ch2[:3]:
                    process.initialize_buffers()
                self.regression_buffer[ch2i, :, :, 1] = cp.asnumpy(self.dff_buffer_ch2[self.ptr, :, :])
                ch2i += 1

        self.processes_list_ch2[3].initialize_buffers(
            self.regression_buffer[:, :, :, 0],
            self.regression_buffer[:, :, :, 1]
        )
        self.processes_list[3].initialize_buffers()

        self.metric.initialize_buffers()
        self.ptr = self.capacity - 1

    def get_input(self):
        self.frame = self.camera.get_live_frame()
        self.input[:] = cp.asanyarray(self.frame)

    def process(self):
        if self.ptr == self.capacity - 1:
            self.ptr = 0
        else:
            self.ptr += 1

        self.get_input()
        if not self.ptr % 2:  # first channel processing
            print(self.camera.frame_idx)
            for process in self.processes_list:
                process.process()

        else:  # second channel processing
            for process in self.processes_list_ch2:
                process.process()

    def evaluate(self):
        self.metric.evaluate()
        return self.metric.result

    def load_datasets(self):
        with h5py.File(self.mask_path, 'r') as f:
            mask = np.transpose(f["mask"][()])
        mask = cp.asanyarray(mask, dtype=cp.float32)

        rois_dict = load_extended_rois_list(self.rois_dict_path)

        return mask, rois_dict
