from core.abstract_pipeline import AbstractPipeLine
from core.metrics.ROI_Diff import ROIDiff
from core.metrics.training_metric import Training
from core.processes import AffineTrans, Mask, DFF, HemoSubtraction, HemoCorrect

import cupy as cp
import numpy as np


class HemoDynamicsDFF(AbstractPipeLine):
    def __init__(self, camera,
                 mask, map, rois_dict,
                 affine_matrix, hemispheres,
                 regression_map,
                 capacity, metric_args, regression_n_samples=512):
        self.camera = camera
        self.capacity = capacity + capacity % 2  # make sure capacity is an even number
        self.mask = mask
        self.map = map
        self.rois_dict = rois_dict  # metric rois name
        self.metric_args = metric_args
        self.affine_matrix = affine_matrix
        self.hemispheres = hemispheres
        self.regression_map = regression_map
        self.regression_n_samples = int(np.floor(regression_n_samples / (capacity * 2)) * (capacity * 2))

        # crop captured frame
        if self.hemispheres == 'left':
            self.cortex_roi = [0, self.camera.shape()[1], 0, int(self.camera.shape()[0] / 2)]
            self.mask = self.mask[:, :int(self.mask.shape[1] / 2)]
            self.map = map[:, :int(self.map.shape[1] / 2)]
        elif self.hemispheres == 'right':
            self.cortex_roi = [0, self.camera.shape()[1], int(self.camera.shape()[0] / 2), self.camera.shape()[0]]
            self.mask = self.mask[:, int(self.mask.shape[1] / 2):]
            self.map = map[:, int(self.map.shape[1] / 2):]
        elif self.hemispheres == 'both':
            self.cortex_roi = [0, self.camera.shape()[1], 0, self.camera.shape()[0]]
        else:
            raise NameError('pipeline hemisphere keyword unrecognized')

        self.new_shape = self.map.shape
        self.frame_shape = (self.camera.shape()[1], self.camera.shape()[0])
        self.input_shape = (self.cortex_roi[1] - self.cortex_roi[0], self.cortex_roi[3] - self.cortex_roi[2])

        # allocate memory
        self.frame = np.ndarray(self.frame_shape, dtype=np.uint16)
        self.input = cp.ndarray(self.input_shape, dtype=cp.uint16)
        self.warped_input = cp.ndarray((self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.warped_buffer = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.warped_buffer_ch2 = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.dff_buffer = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)
        self.dff_buffer_ch2 = cp.ndarray((self.capacity, self.new_shape[0], self.new_shape[1]), dtype=cp.float32)

        affine_transform = AffineTrans(self.input, self.warped_input, self.affine_matrix, self.new_shape)
        # set processes for channel 1
        masking = Mask(self.warped_input, self.mask, self.warped_buffer)
        dff = DFF(self.dff_buffer, self.warped_buffer)
        hemo_subtract = HemoSubtraction(self.dff_buffer, self.dff_buffer_ch2)
        self.processes_list = [affine_transform, masking, dff, hemo_subtract]

        # set processes for channel 2
        masking_ch2 = Mask(self.warped_input, self.mask, self.warped_buffer_ch2)
        dff_ch2 = DFF(self.dff_buffer_ch2, self.warped_buffer_ch2)
        hemo_correct = HemoCorrect(self.dff_buffer_ch2, self.regression_map)
        self.processes_list_ch2 = [affine_transform, masking_ch2, dff_ch2, hemo_correct]

        if metric_args[0] == 'ROIDiff':
            self.metric = ROIDiff(self.dff_buffer, self.rois_dict,
                                  self.metric_args[1], self.metric_args[2])
        elif metric_args[0] == 'Training':
            self.metric = Training(self.metric_args[1], self.metric_args[2])

        self.ptr = self.capacity - 1
        self.ptr_2c = 2 * self.capacity - 1

    def fill_buffers(self):
        # fill warped_buffer of both channels
        self.camera.start_live(buffer_frame_count=self.camera.circ_buffer_count)
        self.processes_list[1].initialize_buffers()
        self.processes_list_ch2[1].initialize_buffers()
        for i in range(self.capacity * 2):
            self.get_input()
            if not i % 2:
                self.processes_list[0].process()
                self.processes_list[1].process()
            else:
                self.processes_list_ch2[0].process()
                self.processes_list_ch2[1].process()

        self.camera.finish()
        # initialize the following process
        self.processes_list[1].initialize_buffers()
        self.processes_list_ch2[1].initialize_buffers()
        self.processes_list[2].initialize_buffers()
        self.processes_list_ch2[2].initialize_buffers()
        self.processes_list_ch2[3].initialize_buffers()
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
        self.frame[:] = self.camera.poll_frame()[0]['pixel_data']
        self.input[:] = cp.asanyarray(self.frame[self.cortex_roi[0]: self.cortex_roi[1], self.cortex_roi[2]: self.cortex_roi[3]])

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

    def calculate_hemodynamics_regression_map(self):
        print("\nCollecting data hemodynamics regression maps calculation...")
        regression_buffer = np.ndarray((self.regression_n_samples, self.new_shape[0], self.new_shape[1], 2),
                                       dtype=np.float32)
        self.camera.start_live(buffer_frame_count=self.camera.circ_buffer_count)
        # fill warped_buffer of both channels
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
                regression_buffer[ch1i, :, :, 0] = cp.asnumpy(self.dff_buffer[self.ptr, :, :])
                ch1i += 1

            else:
                for process in self.processes_list_ch2[:3]:
                    process.process()
                regression_buffer[ch2i, :, :, 1] = cp.asnumpy(self.dff_buffer_ch2[self.ptr, :, :])
                ch2i += 1

        self.camera.finish()
        print("Done collecting data\n")
        print("Calculating regression coefficients...", end="\t")

        regression_coeff0 = np.zeros(self.new_shape)
        regression_coeff1 = np.zeros(self.new_shape)
        for i in range(self.new_shape[0]):
            for j in range(self.new_shape[1]):
                [theta, _, _, _] = np.linalg.lstsq(
                    np.stack((regression_buffer[:, i, j, 1], np.ones((self.regression_n_samples,))), axis=1),
                    regression_buffer[:, i, j, 0],
                    rcond=None)
                regression_coeff0[i, j] = theta[0]
                regression_coeff1[i, j] = theta[1]

        self.processes_list_ch2[3].regression_coeff[0] = cp.asanyarray(regression_coeff0)
        self.processes_list_ch2[3].regression_coeff[1] = cp.asanyarray(regression_coeff1)
        self.regression_map = [regression_coeff0, regression_coeff1]
