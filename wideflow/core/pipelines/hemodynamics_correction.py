from wideflow.core.abstract_pipeline import AbstractPipeLine
from wideflow.core.processes import *
from wideflow.core.metrics.ROI_Contrast import ROIContrast

import cupy as cp
import numpy as np
from skimage.transform import AffineTransform, warp_coords


class HemoDynamicsDFF(AbstractPipeLine):
    def __init__(self, camera, save_path, new_shape, capacity, mask, map, rois_dict, rois_names,
                 match_p_src, match_p_dst, regression_map, regression_n_samples=512):
        self.camera = camera
        self.save_path = save_path
        self.new_shape = new_shape
        self.capacity = capacity + capacity % 2  # make sure capacity is an even number
        self.mask = mask
        self.map = map
        self.rois_dict = rois_dict  # metric rois name
        self.rois_names = rois_names
        self.match_p_src = match_p_src
        self.match_p_dst = match_p_dst
        self.regression_map = regression_map
        self.regression_n_samples = int(np.floor(regression_n_samples / (capacity * 2)) * (capacity * 2))

        self.mapping_coordinates = self.find_mapping_coordinates(match_p_src, match_p_dst)

        self.input_shape = (self.camera.shape[1], self.camera.shape[0])

        # allocate memory
        self.frame = np.ndarray(self.input_shape, dtype=np.uint16)
        self.input = cp.ndarray(self.input_shape, dtype=cp.uint16)
        self.warped_input = cp.ndarray((self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.warped_buffer = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.warped_buffer_ch2 = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.dff_buffer = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.dff_buffer_ch2 = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)

        self.regression_buffer = np.ndarray((self.regression_n_samples, self.new_shape[0], self.new_shape[1], 2),
                                            dtype=np.float32)

        map_coord = MapCoordinates(self.input, self.warped_input, self.mapping_coordinates, self.new_shape)
        # set processes for channel 1
        masking = Mask(self.warped_input, self.mask, self.warped_buffer, ptr=self.capacity-1)
        dff = DFF(self.dff_buffer, self.warped_buffer, ptr=0)
        hemo_subtract = HemoSubtraction(self.dff_buffer, self.dff_buffer_ch2, ptr=0)
        self.processes_list = [map_coord, masking, dff, hemo_subtract]

        # set processes for channel 2
        masking_ch2 = Mask(self.warped_input, self.mask, self.warped_buffer_ch2, ptr=self.capacity-1)
        dff_ch2 = DFF(self.dff_buffer_ch2, self.warped_buffer_ch2, ptr=0)
        hemo_correct = HemoCorrect(self.dff_buffer_ch2, ptr=0)
        self.processes_list_ch2 = [map_coord, masking_ch2, dff_ch2, hemo_correct]

        # set metric
        rois_pixels_list = []
        for roi_name in self.rois_names:
            roi_pixels_list = self.rois_dict[roi_name]["PixelIdxList"]
            for roi_pixels in roi_pixels_list:
                rois_pixels_list.append(roi_pixels)

        self.metric = ROIContrast(self.dff_buffer, self.rois_dict, rois_pixels_list, self.mask, 0)

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

        if self.regression_map is None:
            # collect data to calculate regression coefficient for the hemodynamic correction
            print("\nCollecting data to calculate regression coefficients for hemodynamics correction...")
            ch1i, ch2i = 0, 0
            for i in range(self.regression_n_samples * 2):
                if self.ptr == self.capacity - 1:
                    self.ptr = 0
                else:
                    self.ptr += 1

                self.get_input()
                if not i % 2:
                    for process in self.processes_list[:3]:
                        process.process()
                    self.regression_buffer[ch1i, :, :, 0] = cp.asnumpy(self.dff_buffer[self.ptr, :, :])
                    ch1i += 1

                else:
                    for process in self.processes_list_ch2[:3]:
                        process.process()
                    self.regression_buffer[ch2i, :, :, 1] = cp.asnumpy(self.dff_buffer_ch2[self.ptr, :, :])
                    ch2i += 1

            self.camera.stop_live()
            print("Done collecting data\n")
            print("Calculating regression coefficients...", end="\t")
            self.processes_list_ch2[3].initialize_buffers(
                self.regression_buffer[:, :, :, 0],
                self.regression_buffer[:, :, :, 1]
            )
            self.save_regression_buffers()
            del self.regression_buffer

        else:
            self.processes_list_ch2[3].regression_coeff[0] = cp.asanyarray(self.regression_map[0])
            self.processes_list_ch2[3].regression_coeff[1] = cp.asanyarray(self.regression_map[1])
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
        self.frame[:] = self.camera.get_live_frame()
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
        if not self.ptr_2c % 2:
            self.metric.evaluate()
        return self.metric.result

    def find_mapping_coordinates(self, match_p_src, match_p_dst):
        tform = AffineTransform()
        tform.estimate(np.roll(match_p_src, 1, axis=1), np.roll(match_p_dst, 1, axis=1))

        warp_coor = warp_coords(tform.inverse, (self.new_shape[0], self.new_shape[1]))
        src_cols = np.reshape(warp_coor[0], (self.new_shape[0] * self.new_shape[1], 1))
        src_rows = np.reshape(warp_coor[1], (self.new_shape[0] * self.new_shape[1], 1))
        mapping_coordinates = cp.asanyarray([src_cols, src_rows])

        return mapping_coordinates

    def save_regression_buffers(self):
        with open(self.save_path + "regression_coeff_map.npy", "wb") as f:
            np.save(f, np.stack((
                self.processes_list_ch2[3].regression_coeff[0].get(),
                self.processes_list_ch2[3].regression_coeff[1].get()
                                ))
                    )

