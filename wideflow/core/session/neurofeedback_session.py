from core.abstract_session import AbstractSession
from core.pipelines.hemodynamics_correction import HemoDynamicsDFF
from core.pipelines.training_pipeline import TrainingPipe

from pyvcam import pvc
from pyvcam.constants import PARAM_LAST_MUXED_SIGNAL
from devices.PVCam import PVCamera
from devices.FLIRCam import FLIRCam
from devices.serial_port import SerialControler

from Imaging.utils.acquisition_metadata import AcquisitionMetaData
from Imaging.utils.memmap_process import MemoryHandler
from Imaging.utils.adaptive_staircase_procedure import percentile_update_procedure, binary_fixed_step_staircase_procedure
from Imaging.visualization.live_video_and_metric import LiveVideoMetric
from Imaging.visualization.live_video import LiveVideo
from Imaging.utils.interactive_affine_transform import InteractiveAffineTransform
from Imaging.utils.interactive_bandpass_selector import InteractiveBandPassSelector
from Imaging.utils.create_matching_points import MatchingPointSelector
from Imaging.utils.behavioral_camera_process import run_triggered_behavioral_camera

from utils.load_tiff import load_tiff
from utils.load_bbox import load_bbox
from utils.load_matching_points import load_matching_points
from utils.matplotlib_rectangle_selector_events import *
from utils.find_2d_max_correlation_coordinates import find_2d_max_correlation_coordinates
from utils.convert_dat_to_tif import convert_dat_to_tif
from utils.load_rois_data import load_rois_data
from utils.write_bbox_file import write_bbox_file
from utils.write_matching_point_file import write_matching_point_file

from DeepLabCut.DLCLProcess import BehavioralMonitoring

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


class NeuroFeedbackSession(AbstractSession):
    def __init__(self, config):
        self.config = config
        self.base_path = config["base_path"]
        self.mouse_id = config["mouse_id"]
        self.session_name = config["session_name"]
        self.session_path = f"{self.base_path}/{self.mouse_id}/{self.session_name}/"
        if not os.path.exists(self.session_path):
            os.mkdir(self.session_path)

        self.camera_config = config["camera_config"]
        self.serial_config = config["serial_port_config"]
        self.behavioral_camera_config = config["behavioral_camera_config"]
        self.acquisition_config = config["acquisition_config"]
        self.feedback_config = config["feedback_config"]
        self.registration_config = config["registration_config"]
        self.supplementary_data_config = config["supplementary_data_config"]
        self.analysis_pipeline_config = config["analysis_pipeline_config"]
        self.visualization_config = config["visualization_config"]
        self.deep_lab_cut_config = config["deep_lab_cut_config"]

        self.camera = self.set_imaging_camera()
        self.serial_controller = self.set_serial_controler()
        self.metadata = self.set_metadata_writer()

        # load supplementary data
        self.cortex_mask, self.cortex_map, self.cortex_rois_dict = self.load_datasets()

        self.analysis_pipeline = None

        # shared memory to save imaging data
        self.data_dtype = 'uint16'
        self.data_shm, self.frame_shm, self.memq, self.mem_process = None, None, None, None

        # visualization shared memory and query
        self.vis_shm_obj, self.vis_shm, self.vis_query, self.vis_processes = {}, {}, {}, {}

        # set behavioral camera process
        self.behavioral_camera_process, self.behavioral_camera_q, self.bcam_frame_shm_name, self.bcam_frame_shm, self.bcam_frame = None, None, None, None, None
        if self.behavioral_camera_config["activate"]:
            self.set_behavioral_camera()

        self.behavioral_monitoring_process, self.behavioral_monitoring_q = None, None
        if self.deep_lab_cut_config["activate"]:
            self.set_behavioral_monitoring()

    def set_imaging_camera(self):
        pvc.init_pvcam()
        cam = next(PVCamera.detect_camera())
        cam.open()

        for key, value in self.camera_config["attr"].items():
            setattr(cam, key, value)

        for key, value in self.camera_config["core_attr"].items():
            if type(getattr(cam, key)) == type(value):
                setattr(cam, key, value)
            else:
                setattr(cam, key, type(getattr(cam, key))(value))

        cam.start_up()

        if self.camera_config["splice_plugins_enable"]:
            for plugin_dict in self.camera_config["splice_plugins_settings"]:
                cam.set_splice_post_processing_attributes(plugin_dict["name"], plugin_dict["parameters"])

        if self.camera_config["splice_plugins_enable"]:
            for plugin_dict in self.camera_config["splice_plugins_settings"]:
                cam.set_splice_post_processing_attributes(plugin_dict["name"], plugin_dict["parameters"])

        # setting camera active output wires equals the number of channels - strobbing of illumination LEDs
        cam.set_param(PARAM_LAST_MUXED_SIGNAL, self.camera_config["attr"]["channels"])
        return cam

    def set_behavioral_camera(self):
        sensor_roi = self.behavioral_camera_config["attr"]["roi_bbox"]
        shape = (sensor_roi[2], sensor_roi[3])
        self.bcam_frame_shm = shared_memory.SharedMemory(create=True, size=np.ndarray(shape, dtype=np.uint8).nbytes)  # dtype is compatible to FLIRcam output
        self.bcam_frame_shm_name = self.bcam_frame_shm.name
        self.bcam_frame = np.ndarray(shape, dtype=np.uint8, buffer=self.bcam_frame_shm.buf)

        self.behavioral_camera_q = Queue(3)
        self.behavioral_camera_process = mp.Process(target=run_triggered_behavioral_camera,
                               args=(self.behavioral_camera_q, self.session_path + self.behavioral_camera_config["file_name"],
                                     self.bcam_frame_shm_name),
                               kwargs=self.behavioral_camera_config["attr"])

    def set_behavioral_monitoring(self):
        self.behavioral_monitoring_q = Queue(3)
        target = BehavioralMonitoring(self.behavioral_monitoring_q,
            self.deep_lab_cut_config["model_path"], self.deep_lab_cut_config["model_config"],
            self.deep_lab_cut_config["processor_config"])

        pose_shape = self.deep_lab_cut_config["processor_config"]["pose_shape"]
        pose_dtype = self.deep_lab_cut_config["processor_config"]["pose_dtype"]
        pose_shm = shared_memory.SharedMemory(create=True, size=np.ndarray(pose_shape, dtype=pose_dtype).nbytes)

        self.behavioral_monitoring_pose_shm = np.ndarray(pose_shape, dtype=pose_dtype, buffer=pose_shm.buf)
        self.behavioral_monitoring_process = mp.Process(target=target, args=(pose_shm.name, self.bcam_frame_shm_name))

    def set_serial_controler(self):
        serial_controller = SerialControler(port=self.serial_config["port_id"],
                              baudrate=self.serial_config["baudrate"],
                              timeout=self.serial_config["timeout"])

        return serial_controller

    def set_metadata_writer(self):
        metadata = AcquisitionMetaData(self.config)
        return metadata

    def session_preparation(self):
        regression_map = None
        # select roi
        self.camera.binning = (1, 1)  # set no binning for ROI selection
        frame = self.camera.get_frame()
        if self.registration_config["automatic_cropping"]:
            ref_image = load_tiff(self.registration_config["reference_image_path"])
            ref_bbox = load_bbox(self.config["cropping_bbox_path"])
            ref_image_roi = ref_image[ref_bbox[1]: ref_bbox[1] + ref_bbox[3], ref_bbox[0]: ref_bbox[0] + ref_bbox[2]]
            xi, yi = find_2d_max_correlation_coordinates(frame, ref_image_roi)
            bbox = (int(xi), int(yi), int(ref_bbox[2]), int(ref_bbox[3]))
            self.camera.set_roi(bbox[0], bbox[2], bbox[2], bbox[3])
            if os.path.exists(self.registration_config["matching_point_path"]):
                match_p_src, match_p_dst = load_matching_points(self.config["matching_point_path"])
            if os.path.exists(self.analysis_pipeline_config["args"]["regression_map_path"]):
                regression_map = self.load_regression_map()

        else:
            bbox = self.select_camera_sensor_roi(frame)
            print(bbox)
            self.camera.set_roi(bbox[0], bbox[1], bbox[2], bbox[3])
            self.sensor_roi = bbox

        self.camera.binning = tuple(self.camera_config["core_attr"]["binning"])  # restore configuration binning
        frame = self.camera.get_frame()
        if not os.path.exists(self.registration_config["matching_point_path"]):
            affine_matrix, match_p_src, match_p_dst = self.find_affine_mapping_coordinates(frame)

        # instantiate analysis pipeline
        if self.analysis_pipeline_config["pipeline"] == "HemoDynamicsDFF":
            self.analysis_pipeline = HemoDynamicsDFF(
                self.camera,
                self.cortex_mask, self.cortex_map, self.cortex_rois_dict,
                affine_matrix, self.analysis_pipeline_config["args"]["hemispheres"],
                regression_map,
                self.analysis_pipeline_config["args"]["capacity"],
                self.analysis_pipeline_config["args"]["metric_args"],
                self.analysis_pipeline_config["args"]["regression_n_samples"]
            )
        else:
            raise NameError(f"{self.analysis_pipeline_config['pipeline']} pipeline class doesn't exist")

        if regression_map is None:
            self.analysis_pipeline.calculate_hemodynamics_regression_map()
            regression_coeff0, regression_coeff1 = self.fix_regression_coefficients(
                self.analysis_pipeline.regression_map[0],
                self.analysis_pipeline.regression_map[1]
            )
            self.analysis_pipeline.processes_list_ch2[3].regression_coeff[0] = cp.asanyarray(regression_coeff0)
            self.analysis_pipeline.processes_list_ch2[3].regression_coeff[1] = cp.asanyarray(regression_coeff1)
            self.save_regression_coefficients(regression_coeff0, regression_coeff1)

        # imaging data memory handler
        data_shape = (self.acquisition_config["num_of_frames"], self.camera.shape()[1], self.camera.shape()[0])
        self.data_shm = shared_memory.SharedMemory(create=True, size=np.ndarray(data_shape[-2:], dtype=frame.dtype).nbytes)
        shm_name = self.data_shm.name
        self.frame_shm = np.ndarray(data_shape[-2:], dtype=frame.dtype, buffer=self.data_shm.buf)
        self.memq = Queue(1)
        memory_handler = MemoryHandler(self.memq, self.session_path + self.acquisition_config["vid_file_name"], data_shape,
                                       frame.dtype.name)
        self.mem_process = mp.Process(target=memory_handler, args=(shm_name,))
        self.mem_process.start()

        # initialzie behavioral monitoring
        if self.behavioral_camera_config["activate"]:
            self.behavioral_camera_process.start()
            self.behavioral_camera_q.put("start")
        if self.deep_lab_cut_config["activate"]:
            self.behavioral_monitoring_process.start()

        self.initialize_visualization()

        self.update_config(bbox, match_p_src, match_p_dst, affine_matrix)

    def run_session_pipeline(self):
        # set feedback properties
        feedback_threshold = np.float32(self.feedback_config["metric_threshold"])
        inter_feedback_delay = self.feedback_config["inter_feedback_delay"]
        update_frames = self.feedback_config["update_frames"]
        threshold_eval_frames = self.feedback_config["eval_frames"]
        update_every = self.feedback_config["update_every"]

        typical_count = self.feedback_config["typical_count"]
        count_band = self.feedback_config["count_band"]
        step = self.feedback_config["step"]

        # threshold_percentile = np.float32(self.feedback_config["percentile"])
        # threshold_nbins = np.int32(self.feedback_config["nbins"])

        results_seq = np.zeros((self.acquisition_config["num_of_frames"], ), dtype=np.float32)
        frame_counter = 0
        feedback_time = 0
        self.analysis_pipeline.fill_buffers()
        self.analysis_pipeline.camera.start_live(buffer_frame_count=self.analysis_pipeline.camera.circ_buffer_count)

        print(f'starting session at {datetime.now()}')
        # start session
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        while frame_counter < self.acquisition_config["num_of_frames"]:
            frame_clock_start = perf_counter()
            # run analysis pipline processing loop
            self.analysis_pipeline.process()

            # get behavioral state
            if self.behavioral_camera_config["activate"]:
                if not self.behavioral_camera_q.full():
                    self.behavioral_camera_q.put('grab')
                if self.deep_lab_cut_config["activate"]:
                    if not self.behavioral_monitoring_q.full():
                        self.behavioral_monitoring_q.put('estimate')

            # evaluate metric and give reward if metric above threshold
            cue = 0
            result = self.analysis_pipeline.evaluate()
            self.serial_controller.sendToArduino(f'{(1 + np.clip(result / feedback_threshold, -1, 1)) / 2:3f}')  # map result to [0, 1]
            if int(cp.asnumpy(result) > feedback_threshold) and \
                    (frame_clock_start - feedback_time) * 1000 > inter_feedback_delay and \
                    frame_counter > update_frames[0]:
                self.serial_controller.sendFeedback()
                feedback_time = perf_counter()
                cue = 1
                print('_________________________FEEDBACK HAS BEEN SENT____________________________\n'
                      '___________________________________________________________________________')

            # update threshold
            results_seq[frame_counter] = result
            if not frame_counter % update_every and frame_counter > update_frames[0] and frame_counter < update_frames[1]:
                # feedback_threshold = percentile_update_procedure(feedback_threshold,
                #         results_seq[np.min((0, frame_counter - threshold_eval_frames)):frame_counter: self.camera_config["attr"]["channels"]],
                #         threshold_percentile, threshold_nbins)
                feedback_threshold = binary_fixed_step_staircase_procedure(feedback_threshold,
                            results_seq[np.max((0, frame_counter - threshold_eval_frames)):frame_counter: self.camera_config["attr"]["channels"]],
                            threshold_eval_frames, typical_count, count_band, step)

            # save Wide Filed data
            self.frame_shm[:] = self.analysis_pipeline.frame
            self.memq.put("flush")

            # write frame metadata
            serial_readout = self.serial_controller.getReadout()
            self.metadata.write_frame_metadata(frame_clock_start, cue, result, feedback_threshold, serial_readout)

            # update visualization
            self.update_visualiztion(feedback_threshold, result)

            frame_counter += 1
            frame_clock_stop = perf_counter()
            # print status
            print(f'frame: {frame_counter:06d} '
                  f'elapsed time:{frame_clock_stop - frame_clock_start:.3f} '
                  f'threshold: {feedback_threshold:.3f} '
                  f'metric_results: {result:.3f} '
                  f'serial_readout: {serial_readout}', end='\r')

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # end session
        self.session_termination()

    def session_termination(self):
        print(f'session endded at {datetime.now()}')
        self.metadata.save_file()

        self.analysis_pipeline.camera.finish()
        self.analysis_pipeline.camera.close()
        self.analysis_pipeline.clear_buffers()

        self.serial_controller.sendToArduino('0.0')  # make sure cueig LED is off
        self.serial_controller.close()

        try:
            self.terminate_visualiztion()
        except RuntimeError:
            print("something went wrong while terminating visualization processes")

        if self.behavioral_camera_config["activate"]:
            self.behavioral_camera_q.put('finish')
            self.behavioral_camera_process.join()
            self.behavioral_camera_process.terminate()

        if self.deep_lab_cut_config["activate"]:
            self.behavioral_monitoring_q.put('finish')
            self.behavioral_monitoring_process.join()
            self.behavioral_monitoring_process.terminate()

        if self.bcam_frame_shm is not None:
            self.bcam_frame_shm.close()
            self.bcam_frame_shm.unlink()

        with open(self.session_path + "session_config.json", 'w') as fp:
            json.dump(self.config, fp)


        try:
            self.memq.put("terminate")
            self.mem_process.join()
            self.mem_process.terminate()
            self.data_shm.close()
            self.data_shm.unlink()
        except RuntimeError:
            print("something went wrong while terminating Wide Field memory handler")

        if self.acquisition_config["convert_to_tiff"]:
            try:
                print("converting imaging dat file into tiff, this might take few minutes")
                data_shape = (self.acquisition_config["num_of_frames"], self.camera.shape()[1], self.camera.shape()[0])
                frame_offset = self.analysis_pipeline.frame.nbytes
                frame_shape = data_shape[-2:]

                convert_dat_to_tif(self.session_path + self.acquisition_config["vid_file_name"], frame_offset,
                                   (2000, frame_shape[0], frame_shape[1]),
                                   # ~2000 frames is the maximum amount of frames readable using Fiji imagej using standard image size
                                   self.data_dtype, self.acquisition_config["num_of_frames"])
                os.remove(self.session_path + self.acquisition_config["vid_file_name"])
            except RuntimeError:
                print("something went wrong while converting to tiff. dat file still exist in folder")
                print("Unexpected error:", sys.exc_info()[0])
                raise

            finally:
                print("done")

        print(f"ready for another?")

    def initialize_visualization(self):
        config = self.visualization_config["show_live_stream"]
        if config["status"]:
            image = np.ndarray(config["size"], dtype=np.dtype(config["dtype"]))
            vshm = shared_memory.SharedMemory(create=True, size=image.nbytes)
            image_shm = np.ndarray(image.shape, dtype=image.dtype, buffer=vshm.buf)
            image_shm_name = vshm.name

            live_stream_que = Queue(5)
            target = LiveVideo(live_stream_que, config["params"]["image_shape"], config["params"]["image_dtype"])
            live_stream_process = mp.Process(target=target, args=(image_shm_name, ))
            live_stream_process.start()

            self.vis_shm_obj["live_stream"] = {"image": vshm}
            self.vis_shm["live_stream"] = {"image": image_shm}
            self.vis_query["live_stream"] = live_stream_que
            self.vis_processes["live_stream"] = live_stream_process

        config = self.visualization_config["show_behavioral_monitoring"]
        if config["status"]:
            live_stream_queb = Queue(5)
            target = LiveVideo(live_stream_queb, config["params"]["image_shape"], config["params"]["image_dtype"], vmin=config["params"]["vmin"], vmax=config["params"]["vmax"])
            live_stream_processb = mp.Process(target=target, args=(self.bcam_frame_shm_name, ))
            live_stream_processb.start()

            self.vis_query["behavioral_monitoring"] = live_stream_queb
            self.vis_processes["behavioral_monitoring"] = live_stream_processb

    def update_visualiztion(self, feedback_threshold, metric):
        if self.visualization_config["show_live_stream"]["status"]:
            self.vis_shm["live_stream"]["image"][:] = cp.asnumpy(self.analysis_pipeline.dff_buffer[self.analysis_pipeline.processes_list[2].ptr, :, :])
            if not self.vis_query["live_stream"].full():
                self.vis_query["live_stream"].put("draw")

        if self.visualization_config["show_behavioral_monitoring"]["status"]:
            if not self.vis_query["behavioral_monitoring"].full():
                self.vis_query["behavioral_monitoring"].put("draw")

    def terminate_visualiztion(self):
        if self.visualization_config["show_live_stream"]["status"]:
            if self.vis_query["live_stream"].full():
                self.vis_query["live_stream"].get()
            self.vis_query["live_stream"].put("terminate")
            self.vis_processes["live_stream"].join()
            self.vis_processes["live_stream"].terminate()
            del self.vis_shm["live_stream"]["image"]
            self.vis_shm_obj["live_stream"]["image"].close()

        if self.visualization_config["show_behavioral_monitoring"]["status"]:
            if self.vis_query["behavioral_monitoring"].full():
                self.vis_query["behavioral_monitoring"].get()
            self.vis_query["behavioral_monitoring"].put("terminate")
            self.vis_processes["behavioral_monitoring"].join()
            self.vis_processes["behavioral_monitoring"].terminate()

    def select_camera_sensor_roi(self, frame):
        fig, ax = plt.subplots()
        ax.imshow(cp.asnumpy(frame))
        toggle_selector = RectangleSelector(ax, onselect, drawtype='box')
        fig.canvas.mpl_connect('key_press_event', toggle_selector)
        plt.show()
        bbox = toggle_selector._rect_bbox
        # if np.sum(bbox) > 1:
        #     # convert to PyVcam format
        #     #  PyVCAM: camera ROI is defined as: (x_min, y_min, x_width, y_height)
        #     #  bbox is defined (before conversion) as: (x_min, x_width, y_min, y_width)
        bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

        return bbox

    def find_affine_mapping_coordinates(self, frame):
        iat = InteractiveAffineTransform(frame, self.cortex_map)
        return iat.tform._inv_matrix, iat.trans_points_pos, iat.fixed_points_pos

    def find_piecewise_affine_mapping_coordinates(self, frame, match_p_src, match_p_dst):
        if match_p_src is not None:
            match_p_src = np.array(match_p_src)
        if match_p_dst is not None:
            match_p_dst = np.array(match_p_dst)
        mps = MatchingPointSelector(frame, self.map * np.random.random(self.map.shape),
                                    match_p_src,
                                    match_p_dst,
                                    25)

        return mps.match_p_src, mps.match_p_dst

    def load_datasets(self):
        with h5py.File(self.supplementary_data_config["mask_path"], 'r') as f:
            mask = np.transpose(f["mask"][()])
            map = np.transpose(f["map"][()])
        mask = cp.asanyarray(mask, dtype=cp.float32)

        print(self.supplementary_data_config["rois_dict_path"])
        rois_dict = load_rois_data(self.supplementary_data_config["rois_dict_path"])

        return mask, map, rois_dict

    def load_regression_map(self):
        print("\nLoading regression coefficients for hemodynamics correction...")
        reg_map = np.load(self.regression_map_path)
        return reg_map

    def fix_regression_coefficients(self, regression_coeff0, regression_coeff1):
        ibp = InteractiveBandPassSelector(regression_coeff0)
        reg_map0_fft = np.fft.fftshift((np.fft.fft2(regression_coeff0)))
        reg_map1_fft = np.fft.fftshift((np.fft.fft2(regression_coeff1)))
        for bbox in ibp.bbox_list:
            reg_map0_fft[bbox[2]: bbox[3], bbox[0]: bbox[1]] = 1
            reg_map1_fft[bbox[2]: bbox[3], bbox[0]: bbox[1]] = 1

        regression_coeff0 = abs(np.fft.ifft2(reg_map0_fft))
        regression_coeff1 = abs(np.fft.ifft2(reg_map1_fft))
        return regression_coeff0, regression_coeff1

    def save_regression_coefficients(self, regression_coeff0, regression_coeff1):
        with open(self.session_path + "regression_coeff_map.npy", "wb") as f:
            np.save(f, np.stack((regression_coeff0, regression_coeff1)))

    def update_config(self, bbox, match_p_src, match_p_dst, affine_matrix):
        self.config["registration_config"]["cropping_bbox_path"] = self.session_path + 'bbox.txt'
        self.config["registration_config"]["matching_point_path"] = self.session_path + 'matching_points.txt'

        write_bbox_file(self.config["registration_config"]["cropping_bbox_path"], bbox)
        write_matching_point_file(self.config["registration_config"]["matching_point_path"], match_p_src.tolist(), match_p_dst.tolist())
        np.savetxt(self.session_path + 'affine_matrix.txt', affine_matrix)


