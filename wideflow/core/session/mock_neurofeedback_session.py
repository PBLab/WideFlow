from core.abstract_session import AbstractSession
from core.pipelines.hemodynamics_correction import HemoDynamicsDFF
from core.pipelines.training_pipeline import TrainingPipe

from devices.mock_devices.mock_PVCam import MockPVCamera
from devices.mock_devices.mock_serial_controller import MockSerialControler

from Imaging.utils.interactive_affine_transform import InteractiveAffineTransform
from Imaging.utils.create_matching_points import MatchingPointSelector

from utils.load_tiff import load_tiff
from utils.load_bbox import load_bbox
from utils.load_matching_points import load_matching_points
from utils.matplotlib_rectangle_selector_events import *
from utils.find_2d_max_correlation_coordinates import find_2d_max_correlation_coordinates
from utils.load_matlab_vector_field import load_extended_rois_list

from analysis.utils.load_session_metadata import load_session_metadata

import numpy as np
import cupy as cp

import os
import h5py

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

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
        self.mouse_id = config["mouse_id"]
        self.session_name = config["session_name"]
        self.session_path = f"{self.base_path}/{self.mouse_id}/{self.session_name}/"
        # this correction is used since imaging and analyzed data is done at different computers
        self.session_path = '/data/Rotem/WideFlow prj' + self.session_path[21:]
        self.regression_map_path = self.session_path + 'regression_coeff_map.npy'

        self.camera_config = config["camera_config"]

        self.serial_config = config["serial_port_config"]
        self.behavioral_camera_config = config["behavioral_camera_config"]
        self.acquisition_config = config["acquisition_config"]
        self.feedback_config = config["feedback_config"]
        self.registration_config = config["registration_config"]
        self.supplementary_data_config = config["supplementary_data_config"]
        self.analysis_pipeline_config = config["analysis_pipeline_config"]

        # this correction is used since imaging and analyzed data is done at different computers
        self.supplementary_data_config["mask_path"] = '/data/Rotem/Wide Field/WideFlow/data/cortex_map/allen_2d_cortex.h5'
        if self.analysis_pipeline_config['args']['hemispheres'] == 'left':
            self.supplementary_data_config["rois_dict_path"] = '/data/Rotem/Wide Field/WideFlow/data/cortex_map/allen_2d_cortex_rois_left_hemi.h5'
        elif self.analysis_pipeline_config['args']['hemispheres'] == 'right':
            self.supplementary_data_config[ "rois_dict_path"] = '/data/Rotem/Wide Field/WideFlow/data/cortex_map/allen_2d_cortex_rois_right_hemi.h5'
        elif self.analysis_pipeline_config['args']['hemispheres'] == 'both':
            self.supplementary_data_config["rois_dict_path"] = '/data/Rotem/Wide Field/WideFlow/data/cortex_map/allen_2d_cortex_rois_extended.h5'
        else:
            raise NameError('pipeline hemisphere keyword unrecognized')

        self.camera = self.set_imaging_camera()
        self.serial_controller = self.set_serial_controler()
        self.metadata = self.set_metadata_writer()

        # load supplementary data
        self.cortex_mask, self.cortex_map, self.cortex_rois_dict = self.load_datasets()

        self.analysis_pipeline = None

        self.results_dataset_path = '/data/Rotem/WideFlow prj/results/sessions_dataset.h5'

    def set_imaging_camera(self):
        cam = MockPVCamera(self.camera_config, self.session_path, self.crop_sensor)
        return cam

    def set_behavioral_camera(self):
        pass

    def set_serial_controler(self):
        metadata, config = load_session_metadata(self.session_path)
        serial_controller = MockSerialControler(metadata["serial_readout"])

        return serial_controller

    def set_metadata_writer(self):
        pass

    def session_preparation(self):
        # select roi
        regression_map = self.load_regression_map()
        match_p_src, match_p_dst = load_matching_points()

        frame = self.camera.get_frame()
        if not os.path.exists(self.registration_config["matching_point_path"]):
            affine_matrix, match_p_src, match_p_dst = self.find_affine_mapping_coordinates(frame, match_p_src)

        # initialzie analysis pipeline
        if self.analysis_pipeline_config["pipeline"] == "HemoDynamicsDFF":
            self.analysis_pipeline = HemoDynamicsDFF(
                self.camera, self.session_path,
                self.cortex_mask, self.cortex_map, self.cortex_rois_dict,
                affine_matrix, self.analysis_pipeline_config["args"]["hemispheres"],
                regression_map, self.analysis_pipeline_config["args"]["diff_metric_delta"],
                self.analysis_pipeline_config["args"]["capacity"],  self.analysis_pipeline_config["args"]["rois_names"]
            )
        elif self.analysis_pipeline_config["pipeline"] == "TrainingPipe":
            self.analysis_pipeline = TrainingPipe(
                self.camera, self.session_path,
                self.analysis_pipeline_config["args"]["min_frame_count"],
                self.analysis_pipeline_config["args"]["max_frame_count"],
                self.cortex_mask, self.cortex_map,
                match_p_src, match_p_dst,
                regression_map,
                self.analysis_pipeline_config["args"]["capacity"],
            )

    def run_session_pipeline(self):

        frame_counter = 0
        frame_counter_ch = 0
        self.analysis_pipeline.fill_buffers()
        self.analysis_pipeline.camera.start_live()
        rois_traces_ch1 = {}
        rois_traces_ch2 = {}
        for roi_key in self.cortex_rois_dict:
            rois_traces_ch1[roi_key] = np.zeros(
                (int(self.acquisition_config["num_of_frames"] / self.camera_config['attr']['channels']), )
                , dtype=np.float32)
            rois_traces_ch2[roi_key] = np.zeros(
                (int(self.acquisition_config["num_of_frames"] / self.camera_config['attr']['channels']),)
                , dtype=np.float32)

        print(f'starting session at {datetime.now()}')
        # start session
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        while self.camera.total_cap_frames < self.acquisition_config["num_of_frames"]:
            self.analysis_pipeline.process()

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

            frame_counter += 1
            print(f'frame: {frame_counter:06d}', end='\r')

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # end session
        self.session_termination(rois_traces_ch1, rois_traces_ch2)
        print(f'session endded at {datetime.now()}')

    def session_termination(self, rois_traces_ch1, rois_traces_ch2):
        self.analysis_pipeline.clear_buffers()
        with h5py.File(self.results_dataset_path, 'a') as f:
            main_group = f[self.mouse_id]
            session_group = main_group.create_group(self.session_name)
            rois_traces_group = session_group.create_group('rois_traces')
            ch0_grp = rois_traces_group.create_group('channel_0')
            ch1_grp = rois_traces_group.create_group('channel_1')
            for roi_key, roi_trace in rois_traces_ch1.items():
                ch0_grp.create_dataset(roi_key, data=roi_trace)

            for roi_key, roi_trace in rois_traces_ch2.items():
                ch1_grp.create_dataset(roi_key, data=roi_trace)

    def select_camera_sensor_roi(self, frame):
        fig, ax = plt.subplots()
        ax.imshow(cp.asnumpy(frame))
        toggle_selector = RectangleSelector(ax, onselect, drawtype='box')
        fig.canvas.mpl_connect('key_press_event', toggle_selector)
        plt.show()
        bbox = toggle_selector._rect_bbox
        if np.sum(bbox) > 1:
            # convert to PyVcam format
            #  camera ROI is defined as: (x_min, x_max, y_min, y_max)
            #  bbox is defined (before conversion) as: (x_min, x_width, y_min, y_width)
            bbox = (int(bbox[0]), int(bbox[0] + bbox[2]), int(bbox[1]), int(bbox[1] + bbox[3]))

        return bbox

    def find_affine_mapping_coordinates(self, frame, match_p_src=None):
        iat = InteractiveAffineTransform(frame, self.cortex_map, match_p_src)
        return iat.tform._inv_matrix, iat.trans_points_pos, iat.fixed_points_pos

    def find_piecewise_affine_mapping_coordinates(self, frame, match_p_src, match_p_dst):
        if match_p_src is not None:

            match_p_src = np.array(match_p_src)
        if match_p_dst is not None:
            match_p_dst = np.array(match_p_dst)
        mps = MatchingPointSelector(frame, self.cortex_map * np.random.random(self.cortex_map.shape),
                                    match_p_src,
                                    match_p_dst,
                                    25)

        return mps.match_p_src, mps.match_p_dst

    def load_datasets(self):
        with h5py.File(self.supplementary_data_config["mask_path"], 'r') as f:
            mask = np.transpose(f["mask"][()])
            map = np.transpose(f["map"][()])
        mask = cp.asanyarray(mask, dtype=cp.float32)

        rois_dict = load_extended_rois_list(self.supplementary_data_config["rois_dict_path"])

        return mask, map, rois_dict

    def load_regression_map(self):
        print("\nLoading regression coefficients for hemodynamics correction...")
        reg_map = np.load(self.regression_map_path)
        return reg_map