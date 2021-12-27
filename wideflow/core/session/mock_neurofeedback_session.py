from wideflow.core.abstract_session import AbstractSession
from wideflow.core.pipelines.hemodynamics_correction import HemoDynamicsDFF

from wideflow.devices.mock_devices.mock_PVCam import MockPVCamera
from wideflow.devices.mock_devices.mock_serial_controller import MockSerialControler

from wideflow.Imaging.utils.acquisition_metadata import AcquisitionMetaData
from wideflow.Imaging.utils.memmap_process import MemoryHandler
from wideflow.Imaging.utils.adaptive_staircase_procedure import fixed_step_staircase_procedure
from wideflow.Imaging.visualization.live_video_and_metric import LiveVideoMetric
from wideflow.Imaging.utils.interactive_affine_transform import InteractiveAffineTransform
from wideflow.Imaging.utils.create_matching_points import MatchingPointSelector

from wideflow.utils.load_tiff import load_tiff
from wideflow.utils.load_bbox import load_bbox
from wideflow.utils.load_matching_points import load_matching_points
from wideflow.utils.matplotlib_rectangle_selector_events import *
from wideflow.utils.find_2d_max_correlation_coordinates import find_2d_max_correlation_coordinates
from wideflow.utils.convert_dat_to_tif import convert_dat_to_tif
from wideflow.utils.load_matlab_vector_field import load_extended_rois_list

import numpy as np
import cupy as cp

import os
import sys
import json
import h5py

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

import multiprocessing as mp
from multiprocessing import shared_memory, Queue

from time import perf_counter
from datetime import datetime

from wideflow.analysis.utils.load_session_metadata import load_session_metadata


class PostAnalysisNeuroFeedbackSession(AbstractSession):
    def __init__(self, config):
        # use the same config used for running the live neurofeedback session
        self.config = config
        self.mouse_id = config['name']
        self.base_path = config["path"]
        self.camera_config = config["camera_config"]
        self.serial_config = config["serial_port_config"]
        self.behavioral_camera_config = config["behavioral_camera_config"]
        self.acquisition_config = config["acquisition_config"]
        self.feedback_config = config["feedback_config"]
        self.registration_config = config["registration_config"]
        self.supplementary_data_config = config["supplementary_data_config"]
        self.analysis_pipeline_config = config["analysis_pipeline_config"]

        self.session_name = self.base_path.split('/')[-2]

        self.camera = self.set_imaging_camera()
        self.serial_controller = self.set_serial_controler()
        self.metadata = self.set_metadata_writer()

        # load supplementary data
        self.cortex_mask, self.cortex_map, self.cortex_rois_dict = self.load_datasets()

        self.analysis_pipeline = None

        # shared memory to save imaging data
        self.data_dtype = np.uint16
        self.frame_shm, self.memq, self.mem_process = None, None, None

        # visualization shared memory and query
        self.vis_shm, self.vis_query, self.vis_processes = {}, {}, {}

    def set_imaging_camera(self):
        cam = MockPVCamera(self.camera_config, self.base_path, self.camera_config["crop_sensor"])
        return cam

    def set_behavioral_camera(self):
        pass

    def set_serial_controler(self):
        metadata, config = load_session_metadata(self.base_path)
        serial_controller = MockSerialControler(metadata["serial_readout"])

        return serial_controller

    def set_metadata_writer(self):
        metadata = AcquisitionMetaData(session_config_path=None, config=self.config)
        return metadata

    def session_preparation(self):
        # select roi
        regression_map = None
        if self.camera_config["crop_sensor"]:
            self.camera.binning = (1, 1)  # set no binning for ROI selection
            frame = self.camera.get_frame()
            if self.registration_config["automatic_cropping"]:
                ref_image = load_tiff(self.registration_config["reference_image_path"])
                ref_bbox = load_bbox(self.config["cropping_bbox_path"])
                ref_image_roi = ref_image[ref_bbox[2]: ref_bbox[3], ref_bbox[0]: ref_bbox[1]]
                xi, yi = find_2d_max_correlation_coordinates(frame, ref_image_roi)
                ref_bbox = self.select_camera_sensor_roi(frame)
                bbox = (int(xi), int(xi + (ref_bbox[1] - ref_bbox[0])), int(yi), int(yi + (ref_bbox[3] - ref_bbox[2])))
                self.camera.roi = bbox
                if os.path.exists(self.registration_config["matching_point_path"]):
                    match_p_src, match_p_dst = load_matching_points(self.config["matching_point_path"])
                if os.path.exists(self.analysis_pipeline_config["args"]["regression_map_path"]):
                    regression_map = self.load_regression_map()

            else:
                bbox = self.select_camera_sensor_roi(frame)
                self.camera.roi = bbox

            self.camera.binning = tuple(self.camera_config["core_attr"]["binning"])  # restore configuration binning

        frame = self.camera.get_frame()
        if not os.path.exists(self.registration_config["matching_point_path"]):
            match_p_src, match_p_dst = self.find_affine_mapping_coordinates(frame)

        self.analysis_pipeline_config['args']["match_p_src"] = match_p_src
        self.analysis_pipeline_config['args']["match_p_dst"] = match_p_dst

        # initialzie analysis pipeline
        self.analysis_pipeline = HemoDynamicsDFF(
            self.camera, self.analysis_pipeline_config["args"]["save_path"],
            self.analysis_pipeline_config["args"]["new_shape"], self.analysis_pipeline_config["args"]["capacity"],
            self.cortex_mask, self.cortex_map, self.cortex_rois_dict, self.analysis_pipeline_config["args"]["rois_names"],
            match_p_src, match_p_dst,
            regression_map, self.analysis_pipeline_config["args"]["regression_n_samples"]
        )

    def run_session_pipeline(self):

        frame_counter = 0
        self.analysis_pipeline.fill_buffers()
        self.analysis_pipeline.camera.start_live()
        rois_traces = {}
        for roi_key in self.cortex_rois_dict:
            rois_traces[roi_key] = np.zeros((int(self.acquisition_config["num_of_frames"] / 2), ), dtype=np.float32)

        print(f'starting session at {datetime.now()}')
        # start session
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        while frame_counter < self.acquisition_config["num_of_frames"]:
            self.analysis_pipeline.process()
            for roi_key in rois_traces:
                rois_traces[roi_key][frame_counter] = cp.asnumpy(
                    cp.mean(self.analysis_pipeline.dff_buffer[self.analysis_pipeline.ptr,
                                                              self.cortex_rois_dict[roi_key]['unravel_index'][1],
                                                              self.cortex_rois_dict[roi_key]['unravel_index'][0]])
                )

            frame_counter += 1
            print(f'frame: {frame_counter:06d}', end='\r')

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # end session
        self.session_termination(rois_traces)
        print(f'session endded at {datetime.now()}')

    def session_termination(self, rois_traces):
        self.analysis_pipeline.clear_buffers()
        with h5py.File(self.results_dataset_path, 'a') as f:
            main_group = f[self.mouse_id]
            session_group = main_group.create_group(self.session_name)
            rois_traces_group = session_group.create_group('rois_traces')
            for roi_key, roi_trace in rois_traces.items():
                rois_traces_group.create_dataset(roi_key, roi_trace)

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

    def find_affine_mapping_coordinates(self, frame):
        iat = InteractiveAffineTransform(frame, self.cortex_map)
        return iat.trans_points_pos, iat.fixed_points_pos

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