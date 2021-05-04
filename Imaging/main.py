from pyvcam import pvc
from pyvcam import constants as const

from devices.PVCam import PVCamera
from devices.cupy_cuda_kernels import *
# from devices.parallel_port import port

from utils.imaging_utils import *

import pathlib
import cupy as cp
import numpy as np
import cv2

from time import perf_counter
import matplotlib.pyplot as plt



if __name__ == "__main__":

    # load session configurations
    imaging_config_path = str(
        pathlib.Path('/home') / 'pb' / 'PycharmProjects' / 'WideFlow' / 'Imaging' / 'imaging_config_template.json')
    config = load_config(imaging_config_path)

    # allocate circular buffer in device
    capacity = config["acquisition_config"]["buffer"]["nframes"]
    nrows = config["acquisition_config"]["buffer"]["nrows"]
    ncols = config["acquisition_config"]["buffer"]["ncols"]
    buffer_rs = cp.asanyarray(np.zeros((capacity, nrows, ncols)))
    buffer_dff = cp.asanyarray(np.zeros((capacity, nrows, ncols)))
    buffer_th = cp.asanyarray(np.zeros((capacity, nrows, ncols)))

    # allocate preprocessing buffer in device
    d_frame_rs = cp.ndarray((nrows, ncols), dtype=cp.float32)
    d_std_map = cp.ndarray((nrows, ncols), dtype=cp.float32)

    # video writer settings
    fourcc = cv2.VideoWriter_fourcc('t', 'i', 'f', 'f')
    writer = cv2.VideoWriter(config["acquisition_config"]["save_path"], fourcc, 25, (ncols, nrows))

    # open camera and set camera settings
    pvc.init_pvcam()
    cam = next(PVCamera.detect_camera())
    cam.open()
    if config["acquisition_config"]["splice_plugins_enable"]:
        for plugin_dict in config["acquisition_config"]["splice_plugins_settings"]:
            cam.set_splice_post_processing_attributes(plugin_dict["name"], plugin_dict["parameters"])
    cam.start_live()

    # select rou
    if config["camera_config"]["roi"] is None:
        frame = cam.get_live_frame()
        bbox = cv2.selectROI('choose roi', frame)
        if len(bbox):
            # convert to PyVcam format
            bbox = [int(bbox[1]), int(bbox[1]+bbox[3]), int(bbox[0]), int(bbox[0]+bbox[2])]
            cam.set_param(const.PARAM_ROI_COUNT, bbox)



    frame_counter = 0
    ptr = capacity - 1
    while frame_counter < capacity:
        frame = cam.get_live_frame()
        d_frame = cp.asanyarray(frame)

        zoom(d_frame, d_frame_rs)
        buffer_rs[ptr, :, :] = d_frame_rs

    # start session

    frame_counter = 0
    ptr = capacity - 1
    while frame_counter < config["acquisition_config"]["num_of_frames"]:
        t1_start = perf_counter()

        if ptr == capacity - 1:
            ptr = 0
        else:
            ptr += 1

        frame = cam.get_live_frame()
        d_frame = cp.asanyarray(frame)

        zoom(d_frame, d_frame_rs)
        buffer_rs[ptr, :, :] = d_frame_rs

        bs = baseline_calc_carbox(buffer_rs)
        buffer_dff[ptr, :, :] = dff(d_frame_rs, bs)

        d_std_map = nd_std(buffer_dff, ax=0)
        buffer_th = std_threshold(buffer_dff, d_std_map, 2)


        print(cp.asnumpy(buffer_th[:, 10, 10]))
        frame_counter += 1

        frame_out = cp.asnumpy(d_frame_rs)

        cv2.imshow('preprocessd frame', frame_out)
        if cv2.waitKey(1) == 27:
            break
        # writer.write(frame_out)

        t1_stop = perf_counter()
        print("Elapsed time:", t1_stop - t1_start)

        # process_result = process.cal_process(frame)
        # cue = metric.calc_metric(process_result)
        #
        # if cue:
        #     ser.write(b'on')