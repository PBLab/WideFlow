from devices.PVCam import PVCamera
from pyvcam import pvc

from devices.serial_port import ser
from devices.cupy_cuda_kernels import *
# from devices.numba_cuda_kernels import *

from core.processing import Processing
from core.metric import Metric

from utils.imaging_utils import *
# from devices.parallel_port import port

import pathlib
import cupy as cp
import numpy as np
threadsperblock = 32


if __name__ == "__main__":

    # imaging_config_path = str(
    #     pathlib.Path('/home') / 'pb' / 'PycharmProjects' / 'WideFlow' / 'Imaging' / 'imaging_config_template.json')
    # config = load_config(imaging_config_path)

    # allocate processing station in device
    # if config["acquisition_config"]["splice_plugin_parameters"][0]["dstIndex"]:
    #     process_station = cp.ndarray((config["camera_config"]["n_rows"], config["camera_config"]["n_columns"]),
    #                                  data=config["acquisition_config"]["splice_plugin_parameters"][0]["dstIndex"],
    #                                  dtype=np.dtype("uint16"))
    # else:
    #     process_station = cp.ndarray((config["camera_config"]["n_rows"], config["camera_config"]["n_columns"]),
    #                                  dtype=np.dtype("uint16"))
    #     config["acquisition_config"]["splice_plugin_parameters"][0]["dstIndex"] = process_station.data.mem.ptr
    # allocate circular buffer in device
    capacity = cp.asanyarray(np.ones((1, )*32, dtype=np.int8()))
    buffer = cp.asanyarray(np.zeros((capacity, 128, 128)))
    pointer = cp.asanyarray(np.zeros((1, ), dtype=np.int8()))

    pvc.init_pvcam()
    cam = next(PVCamera.detect_camera())
    cam.open()

    # cam.set_post_processing_attributes(config["acquisition_config"])
    #
    # process = Processing.get_child_from_str(config["process_config"]["method"], ["process_config"]["attributes"])
    # metric = Metric.get_child_from_str(config["metric_config"]["method"], ["process_config"]["attributes"])

    frame_counter = 0
    cam.start_live()
    while True:
        frame = cam.get_live_frame()
        # print(frame[0, :5])
        d_frame = cp.asanyarray(frame)
        d_frame_rs = cp.ndarray((128, 128), dtype=cp.float32)
        resize(d_frame, d_frame_rs, 128, 128)
        update_buffer(buffer, d_frame_rs, pointer, capacity)

        frame_counter += 1
        # process_result = process.cal_process(frame)
        # cue = metric.calc_metric(process_result)
        #
        # if cue:
        #     ser.write(b'on')