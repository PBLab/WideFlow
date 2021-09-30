from core.abstract_pipeline import AbstractPipeLine
from core.processes import *
from core.metrics import ROIMean
import cupy as cp
import numpy as np

import os
import h5py
from utils.load_matlab_vector_field import load_extended_rois_list
from Imaging.utils.create_matching_points import *

from pyvcam.constants import PARAM_LAST_MUXED_SIGNAL


class HemoDynamicsDFF(AbstractPipeLine):
    def __init__(self, camera, save_path, new_shape, capacity, rois_dict_path, mask_path, rois_names,
                 regression_n_samples, match_p_src=None, match_p_dst=None, regression_map_path=""):
        self.camera = camera
        self.save_path = save_path
        self.new_shape = new_shape
        self.capacity = capacity + capacity % 2  # make sure capacity is an even number
        self.rois_dict_path = rois_dict_path
        self.mask_path = mask_path
        self.mask, self.map, self.rois_dict = self.load_datasets()
        self.rois_names = rois_names
        self.regression_n_samples = int(np.floor(regression_n_samples / (capacity * 2)) * (capacity * 2))
        self.match_p_src, self.match_p_dst, self.mapping_coordinates = self.find_mapping_coordinates(match_p_src,
                                                                                                     match_p_dst)
        self.regression_map_path = regression_map_path

        self.input_shape = (self.camera.shape[1], self.camera.shape[0])

        # allocate memory
        self.frame = np.ndarray(self.input_shape)
        self.input = cp.ndarray(self.input_shape)
        self.warped_input = cp.ndarray((self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.warped_buffer = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.warped_buffer_ch2 = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.dff_buffer = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.dff_buffer_ch2 = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)

        self.regression_buffer = np.ndarray((self.regression_n_samples, self.new_shape[0], self.new_shape[1], 2),
                                            dtype=np.float32)

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
        rois_pixels_list = []
        for roi_name in self.rois_names:
            roi_pixels_list = self.rois_dict[roi_name]["PixelIdxList"]
            for roi_pixels in roi_pixels_list:
                rois_pixels_list.append(roi_pixels)

        self.metric = ROIMean(self.dff_buffer, rois_pixels_list, ptr=0)

        self.camera.set_param(PARAM_LAST_MUXED_SIGNAL,
                              2)  # setting camera active output wires to 2 - strobbing of two LEDs
        self.ptr = self.capacity - 1
        self.ptr_2c = 2 * self.capacity - 1

    def fill_buffers(self):
        # initialize buffers
        self.camera.start_live()
        for i in range(self.capacity * 2):
            self.get_input()
            if not i % 2:
                self.processes_list[0].process()
                self.processes_list[1].process()
            else:
                self.processes_list_ch2[0].process()
                self.processes_list_ch2[1].process()

        self.processes_list[2].initialize_buffers()
        self.processes_list_ch2[2].initialize_buffers()

        if os.path.exists(self.regression_map_path):
            # collect data to calculate regression coefficient for the hemodynamic correction
            print("\nCollecting data to calculate regression coefficient for hemodynamics effects attenuation...")
            ch1i, ch2i = 0, 0
            for i in range(self.regression_n_samples * 2):
                if self.ptr == self.capacity - 1:
                    self.ptr = 0
                else:
                    self.ptr += 1


                self.get_input()
                if not i % 2:
                    for process in self.processes_list[:3]:
                        process.initialize_buffers()
                    self.regression_buffer[ch1i, :, :, 0] = cp.asnumpy(self.dff_buffer[self.ptr, :, :])
                    ch1i += 1

                else:
                    for process in self.processes_list_ch2[:3]:
                        process.initialize_buffers()
                    self.regression_buffer[ch2i, :, :, 1] = cp.asnumpy(self.dff_buffer_ch2[self.ptr, :, :])
                    ch2i += 1

            self.camera.stop_live()
            print("Done collecting the data\n")
            print("Calculating the regression coefficients...", end="\t")
            self.processes_list_ch2[3].initialize_buffers(
                self.regression_buffer[:, :, :, 0],
                self.regression_buffer[:, :, :, 1]
            )
            self.save_regression_buffers()
            del self.regression_buffer

        else:
            print("\nLoading regression coefficient maps for hemodynamics effects attenuation...")
            reg_map = self.load_regression_map()
            self.processes_list_ch2.regression_coeff[0] = reg_map[0]
            self.processes_list_ch2.regression_coeff[0] = reg_map[1]
            del reg_map

        print("Done")
        self.camera.start_live()
        for i in range(self.capacity * 2):
            self.get_input()
            if not i % 2:
                self.processes_list[0].process()
                self.processes_list[1].process()
            else:
                self.processes_list_ch2[0].process()
                self.processes_list_ch2[1].process()

        self.camera.stop_live()

        self.processes_list[2].initialize_buffers()
        self.processes_list_ch2[2].initialize_buffers()
        self.processes_list[3].initialize_buffers()

        self.metric.initialize_buffers()
        self.ptr = self.capacity - 1
        self.ptr_2c = 2 * self.capacity - 1

    def clear_buffers(self):
        self.input = None
        self.warped_input = None
        self.warped_buffer = None
        self.warped_buffer_ch2 = None
        self.dff_buffer = None
        self.dff_buffer_ch2 = None

        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    def get_input(self):
        self.frame = self.camera.get_live_frame()
        self.input[:] = cp.asanyarray(self.frame)

    def process(self):
        if self.ptr_2c == 2 * self.capacity - 1:
            self.ptr_2c = 0
        else:
            self.ptr_2c += 1

        self.get_input()
        if not self.ptr_2c % 2:  # first channel processing
            self.ptr = int(self.ptr_2c / 2)
            for process in self.processes_list:
                process.process()

        else:  # second channel processing
            for process in self.processes_list_ch2:
                process.process()

    def evaluate(self):
        self.metric.evaluate()
        return self.metric.result

    def update_config(self, config):
        config["analysis_pipeline_config"]["match_p_src"] = self.match_p_src.tolist()
        config["analysis_pipeline_config"]["match_p_dst"] = self.match_p_dst.tolist()
        config["camera_config"]["core_attr"]["roi"] = self.camera.roi
        return config

    def load_datasets(self):
        with h5py.File(self.mask_path, 'r') as f:
            mask = np.transpose(f["mask"][()])
            map = np.transpose(f["map"][()])
        mask = cp.asanyarray(mask, dtype=cp.float32)

        rois_dict = load_extended_rois_list(self.rois_dict_path)

        return mask, map, rois_dict

    def find_mapping_coordinates(self, match_p_src, match_p_dst):
        if match_p_src is not None:
            match_p_src = np.array(match_p_src)
        if match_p_dst is not None:
            match_p_dst = np.array(match_p_dst)
        frame = self.camera.get_frame()
        mps = MatchingPointSelector(frame, self.map * np.random.random(self.map.shape),
                                    match_p_src,
                                    match_p_dst,
                                    25)
        src_cols = mps.src_cols
        src_rows = mps.src_rows
        mapping_coordinates = cp.asanyarray([src_cols, src_rows])
        return mps.match_p_src, mps.match_p_dst, mapping_coordinates

    def save_regression_buffers(self):
        with open(self.save_path + "regression_coeff_map.npy", "wb") as f:
            np.save(f, np.stack((
                self.processes_list_ch2[3].regression_coeff[0].get(),
                self.processes_list_ch2[3].regression_coeff[1].get()
                                ))
                    )

    def load_regression_map(self):
        reg_map = np.load(self.regression_map_path)
        return reg_map

