from pyvcam import pvc

from devices.PVCam import PVCamera
from devices.cupy_cuda_kernels import *
# from devices.parallel_port import port

from utils.imaging_utils import *

import pathlib
import cupy as cp
import numpy as np
import cv2

from time import perf_counter

fourcc = cv2.VideoWriter_fourcc('t', 'i', 'f', 'f')

if __name__ == "__main__":

    # load session configurations
    imaging_config_path = str(
        pathlib.Path('/home') / 'pb' / 'PycharmProjects' / 'WideFlow' / 'Imaging' / 'imaging_config_template.json')
    config = load_config(imaging_config_path)

    # allocate circular buffer in device
    capacity = config["acquisition_config"]["buffer"]["nframes"]
    nrows = config["acquisition_config"]["buffer"]["nrows"]
    ncols = config["acquisition_config"]["buffer"]["ncols"]
    buffer = cp.asanyarray(np.zeros((capacity, nrows, ncols)))

    # allocate preprocessing buffer in device
    d_frame_rs = cp.ndarray((nrows, ncols), dtype=cp.float32)

    # open camera and set camera settings
    pvc.init_pvcam()
    cam = next(PVCamera.detect_camera())
    cam.open()
    if config["acquisition_config"]["splice_plugins_enable"]:
        for plugin_dict in config["acquisition_config"]["splice_plugins_settings"]:
            cam.set_splice_post_processing_attributes(plugin_dict)

    # start session
    writer = cv2.VideoWriter(config["acquisition_config"]["save_path"], fourcc, 25, (nrows, ncols))
    frame_counter = 0
    ptr = capacity - 1
    cam.start_live()

    while True:
        t1_start = perf_counter()

        frame = cam.get_live_frame()
        print(frame[0, :5])
        d_frame = cp.asanyarray(frame)

        resize(d_frame, d_frame_rs)
        bs = cp.mean(buffer, axis=0)
        d_frame_rs = dff(d_frame_rs, bs)
        if ptr == capacity - 1:
            ptr = 0
        else:
            ptr += 1
        buffer[ptr, :, :] = d_frame_rs

        frame_counter += 1

        frame_out = cp.asnumpy(d_frame_rs)
        cv2.imshow('preprocessd frame', frame_out)
        # writer.write(frame_out)

        t1_stop = perf_counter()
        print("Elapsed time:", t1_stop - t1_start)

        # process_result = process.cal_process(frame)
        # cue = metric.calc_metric(process_result)
        #
        # if cue:
        #     ser.write(b'on')