from core.abstract_session import AbstractSession
from core.pipelines.hemodynamics_correction import HemoDynamicsDFF
from core.pipelines.training_pipeline import TrainingPipe

from devices.mock_devices.mock_PVCam import MockPVCamera
from devices.mock_devices.mock_serial_controller import MockSerialControler

from Imaging.utils.interactive_affine_transform import InteractiveAffineTransform

from utils.load_rois_data import load_rois_data

from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
from analysis.utils.vid_pstr import vid_pstr

import numpy as np
import cupy as cp
from skimage.transform import resize

import os
import h5py
from tifffile import TiffWriter

from datetime import datetime


class PostAnalysisNeuroFeedbackSession(AbstractSession):
    def __init__(self, config, crop_sensor=False):
        """

        Args:
            config_path: str - path to directory where session config and all other data is located
            crop_sensor: bool - crop image using roi config or not
        """

        self.config = config
        self.crop_sensor = crop_sensor

        # use the same config used for running the live neurofeedback session
        self.config = config
        self.base_path = config["base_path"]
        self.date = config["date"]
        self.mouse_id = config["mouse_id"]
        self.session_name = config["session_name"]
        self.session_path = f"{self.base_path}/{self.date}/{self.mouse_id}/{self.session_name}/"

        self.regression_map_path = self.session_path + 'regression_coeff_map.npy'

        self.camera_config = config["camera_config"]

        self.serial_config = config["serial_port_config"]
        self.behavioral_camera_config = config["behavioral_camera_config"]
        self.acquisition_config = config["acquisition_config"]
        self.feedback_config = config["feedback_config"]
        self.registration_config = config["registration_config"]
        self.supplementary_data_config = config["supplementary_data_config"]
        self.analysis_pipeline_config = config["analysis_pipeline_config"]

        self.camera = self.set_imaging_camera()
        self.metadata = self.set_metadata_writer()
        self.serial_controller = self.set_serial_controler()
        # self.metadata = self.set_metadata_writer()

        # load supplementary data
        self.cortex_mask, self.cortex_map, self.cortex_rois_dict = self.load_datasets()

        self.analysis_pipeline = None

        # self.results_dataset_path = '/data/Rotem/WideFlow prj/results/sessions_20220320.h5'
        #self.results_dataset_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
        self.results_dataset_path = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'

    def set_imaging_camera(self):
        cam = MockPVCamera(self.camera_config, self.session_path, self.crop_sensor)
        return cam

    def set_behavioral_camera(self):
        pass

    def set_serial_controler(self):
        serial_controller = MockSerialControler(self.metadata["serial_readout"])

        return serial_controller

    def set_metadata_writer(self):
        start_frame = 0
        timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(f'{self.session_path}/metadata.txt')
        metadata = {"timestamp": timestamp[start_frame:], "cue": cue[start_frame:], "metric_result": metric_result[start_frame:], "threshold": threshold[start_frame:],
                    "serial_readout": serial_readout[start_frame:]}
        return metadata

    def session_preparation(self):
        regression_map = self.load_regression_map()
        affine_matrix = np.loadtxt(self.session_path + 'affine_matrix.txt')

        # initialzie analysis pipeline
        if self.analysis_pipeline_config["pipeline"] == "HemoDynamicsDFF":
            self.analysis_pipeline = HemoDynamicsDFF(
                self.camera,
                self.cortex_mask, self.cortex_map, self.cortex_rois_dict,
                affine_matrix, self.analysis_pipeline_config["args"]["hemispheres"],
                regression_map, self.analysis_pipeline_config["args"]["capacity"],
                self.analysis_pipeline_config["args"]["metric_args"]
            )
        else:
            raise NameError(f"{self.analysis_pipeline_config['pipeline']} pipeline class doesn't exist")

    def run_session_pipeline(self):
        frame_counter = 0
        frame_counter_ch = 0
        self.analysis_pipeline.fill_buffers()
        self.analysis_pipeline.camera.frame_idx = -1  # back to first frame to compensate for frames used to fill the buffers
        self.analysis_pipeline.camera.total_cap_frames = 0
        rois_traces_ch1 = {}
        rois_traces_ch2 = {}
        for roi_key in self.cortex_rois_dict:
            rois_traces_ch1[roi_key] = np.zeros(
                (int(self.acquisition_config["num_of_frames"] / self.camera_config['attr']['channels']),)
                , dtype=np.float32)
            rois_traces_ch2[roi_key] = np.zeros(
                (int(self.acquisition_config["num_of_frames"] / self.camera_config['attr']['channels']),)
                , dtype=np.float32)

        metric_result = np.zeros((self.acquisition_config["num_of_frames"],))
        dff_movie = np.zeros((int(self.acquisition_config["num_of_frames"] / self.camera_config['attr']['channels']),
                              int(self.analysis_pipeline.new_shape[0] / 4), int(self.analysis_pipeline.new_shape[1] / 4)),
                             dtype=np.float32)
        print(f'starting session at {datetime.now()}')
        # start session
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        while self.camera.total_cap_frames < self.acquisition_config["num_of_frames"]:
            self.analysis_pipeline.process()

            # metric results
            metric_result[frame_counter] = self.analysis_pipeline.evaluate()

            # rois traces
            if not self.analysis_pipeline.ptr_2c % 2:
                for roi_key in rois_traces_ch1:
                    rois_traces_ch1[roi_key][frame_counter_ch] = cp.asnumpy(
                        cp.mean(self.analysis_pipeline.dff_buffer[self.analysis_pipeline.ptr,
                                                                  self.cortex_rois_dict[roi_key]['unravel_index'][1],
                                                                  self.cortex_rois_dict[roi_key]['unravel_index'][0]])
                    )
            else:
                for roi_key in rois_traces_ch2:
                    rois_traces_ch2[roi_key][frame_counter_ch] = cp.asnumpy(
                        cp.mean(self.analysis_pipeline.dff_buffer_ch2[self.analysis_pipeline.ptr,
                                                                      self.cortex_rois_dict[roi_key]['unravel_index'][1],
                                                                      self.cortex_rois_dict[roi_key]['unravel_index'][0]])
                    )
                frame_counter_ch += 1

            # dff movie
            if not self.analysis_pipeline.ptr_2c % 2:
                dff_movie[frame_counter_ch] = resize(
                    cp.asnumpy(self.analysis_pipeline.dff_buffer[self.analysis_pipeline.ptr]),
                    output_shape=dff_movie.shape[-2:])

            frame_counter += 1
            print(f'frame: {frame_counter:06d}', end='\r')

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # end session
        self.session_termination(rois_traces_ch1, rois_traces_ch2, metric_result, dff_movie)
        print(f'session endded at {datetime.now()}')

    def session_termination(self, rois_traces_ch1, rois_traces_ch2, metric_result, dff_movie):
        self.analysis_pipeline.clear_buffers()
        with h5py.File(self.results_dataset_path, 'a') as f:
            main_group = f[self.mouse_id]
            session_group = main_group.create_group(self.session_name)
            rois_traces_group = session_group.create_group('rois_traces')
            #rois_traces_group = session_group.create_group('rois_traces')
            ch0_grp = rois_traces_group.create_group('channel_0')
            ch1_grp = rois_traces_group.create_group('channel_1')
            for roi_key, roi_trace in rois_traces_ch1.items():
                ch0_grp.create_dataset(roi_key, data=roi_trace)

            for roi_key, roi_trace in rois_traces_ch2.items():
                ch1_grp.create_dataset(roi_key, data=roi_trace)

            session_group.create_dataset('metric_results', data=metric_result)

        if not os.path.exists(self.session_path + 'post_analysis_results'):
            os.mkdir(self.session_path + 'post_analysis_results')
        cue = self.metadata['cue'].copy()
        reward = cue[::2]
        reward = np.maximum(reward, np.array(cue)[1::2])
        dff_movie_pstr = vid_pstr(dff_movie, reward, 20)
        with TiffWriter(self.session_path + 'post_analysis_results/dff_blue.tif') as tif:
            tif.write(dff_movie, contiguous=True)
        with TiffWriter(self.session_path + 'post_analysis_results/dff_blue_pstr.tif') as tif:
            tif.write(dff_movie_pstr, contiguous=True)

    def find_affine_mapping_coordinates(self, frame, match_p_src=None):
        iat = InteractiveAffineTransform(frame, self.cortex_map, match_p_src)
        return iat.tform._inv_matrix, iat.trans_points_pos, iat.fixed_points_pos

    def load_datasets(self):
        with h5py.File(self.supplementary_data_config["mask_path"], 'r') as f:
            mask = np.transpose(f["mask"][()])
            map = np.transpose(f["map"][()])
        mask = cp.asanyarray(mask, dtype=cp.float32)

        rois_dict = load_rois_data(self.supplementary_data_config["rois_dict_path"])
        for key in self.supplementary_data_config["closest_rois"]:
            del rois_dict[key]

        if self.supplementary_data_config["rois_dict_path"].endswith('rois.h5'):
            shape = (297, 337)
        else:
            shape = (297, 168)

        for i, (roi_key, roi_dict) in enumerate(rois_dict.items()):
            rois_dict[roi_key]['unravel_index'] = np.unravel_index(roi_dict['PixelIdxList'], (shape[0], shape[1]))

        return mask, map, rois_dict

    def load_regression_map(self):
        print("\nLoading regression coefficients for hemodynamics correction...")
        if os.path.exists(self.regression_map_path):
            reg_map = np.load(self.regression_map_path)
            return reg_map
        else:
            return None
