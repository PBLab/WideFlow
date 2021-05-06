from pyvcam import pvc
from pyvcam import constants as const

from devices.PVCam import PVCamera
from devices.cupy_cuda_kernels import *
import serial
# from devices.parallel_port import port

from utils.imaging_utils import *

import pathlib
import cupy as cp
import numpy as np
import cv2

from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


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

    # # video writer settings
    # fourcc = cv2.VideoWriter_fourcc('t', 'i', 'f', 'f')
    # writer = cv2.VideoWriter(config["acquisition_config"]["save_path"], fourcc, 25, (ncols, nrows))

    # open camera and set camera settings
    pvc.init_pvcam()
    cam = next(PVCamera.detect_camera())
    cam.open()
    if config["acquisition_config"]["splice_plugins_enable"]:
        for plugin_dict in config["acquisition_config"]["splice_plugins_settings"]:
            cam.set_splice_post_processing_attributes(plugin_dict["name"], plugin_dict["parameters"])
    cam.start_live()

    # serial port
    ser = serial.Serial('/dev/ttyS0')

    # select roi
    if config["camera_config"]["roi"] is None:
        frame = cam.get_live_frame()
        fig, ax = plt.subplots()
        ax.imshow(cp.asnumpy(frame))
        toggle_selector = RectangleSelector(ax, onselect, drawtype='box')
        fig.canvas.mpl_connect('key_press_event', toggle_selector)
        plt.show()
        bbox = toggle_selector._rect_bbox
        if np.sum(bbox) > 1:
            # convert to PyVcam format
            bbox = (int(bbox[1]), int(bbox[1]+bbox[3]), int(bbox[0]), int(bbox[0]+bbox[2]))
            cam.stop_live()
            cam.roi = bbox
            cam.start_live()

    # fill buffers before session starts
    frame_counter = 0
    ptr = capacity - 1
    while frame_counter < capacity:
        if ptr == capacity - 1:
            ptr = 0
        else:
            ptr += 1

        frame = cam.get_live_frame()
        d_frame = cp.asanyarray(frame)

        zoom(d_frame, d_frame_rs)
        buffer_rs[ptr, :, :] = d_frame_rs
        frame_counter += 1

    # start session
    frames_seq = np.zeros((nrows, ncols, 100))
    live_win = plt.imshow(cp.asnumpy(d_frame_rs))
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

        frame_out = cp.asnumpy(d_frame_rs)
        #
        # # writer.write(frame_out)
        plt.imshow(frame_out)
        t1_stop = perf_counter()
        print("Elapsed time:", t1_stop - t1_start)
        frame_counter += 1
        # frames_seq[:, :, frame_counter-1] = frame_out
        # if frame_counter == 50:
        #     ser.close()  # led is on
        #     t2_start = perf_counter()
        #     print('________________LED ON___________________')
        #
        # if buffer_dff[ptr, :, :].mean() > 0.03:
        #     t2_stop = perf_counter()
        #     break

    cam.stop_live()
    cam.close()
    # print("__________________________delay time:", t2_stop - t2_start)



