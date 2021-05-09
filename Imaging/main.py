from pyvcam import pvc
from devices.PVCam import PVCamera
from devices.cupy_cuda_kernels import *
import serial

from utils.imaging_utils import *
from utils.load_tiff import load_tiff

import pathlib
import cupy as cp
import numpy as np
import cv2

from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


def run_session(config, cam):

    # process & metric config
    process_func = config["process_config"]["attributes"]["function_name"]
    process_params = config["process_config"]["attributes"]["parameters"]
    metric_func = config["metric_config"]["attributes"]["function_name"]
    metric_params = config["metric_config"]["attributes"]["parameters"]

    # allocate circular buffer in device
    capacity = config["acquisition_config"]["buffer"]["nframes"]
    nrows = config["acquisition_config"]["buffer"]["nrows"]
    ncols = config["acquisition_config"]["buffer"]["ncols"]
    buffer_rs = cp.asanyarray(np.zeros((capacity, nrows, ncols)))
    buffer_dff = cp.asanyarray(np.zeros((capacity, nrows, ncols)))
    buffer_th = cp.asanyarray(np.zeros((capacity, nrows, ncols)))
    pattern = cp.asanyarray(load_tiff(metric_params["pattern_path"]))
    metric_threshold = metric_params["threshold"]

    # allocate preprocessing buffer in device
    d_frame_rs = cp.ndarray((nrows, ncols), dtype=cp.float32)
    d_std_map = cp.ndarray((nrows, ncols), dtype=cp.float32)

    # video writer settings
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(config["acquisition_config"]["vid_save_path"], fourcc, 25, (ncols, nrows))
    metadata = AcquisitionMetaData(imaging_config_path)

    # open camera and set camera settings
    cam.open()
    if config["acquisition_config"]["splice_plugins_enable"]:
        for plugin_dict in config["acquisition_config"]["splice_plugins_settings"]:
            cam.set_splice_post_processing_attributes(plugin_dict["name"], plugin_dict["parameters"])

    # serial port
    ser = serial.Serial('/dev/ttyS0')

    # select roi
    if config["camera_config"]["roi"] is None:
        frame = cam.get_frame()
        fig, ax = plt.subplots()
        ax.imshow(cp.asnumpy(frame))
        toggle_selector = RectangleSelector(ax, onselect, drawtype='box')
        fig.canvas.mpl_connect('key_press_event', toggle_selector)
        plt.show()
        bbox = toggle_selector._rect_bbox
        if np.sum(bbox) > 1:
            # convert to PyVcam format
            bbox = (int(bbox[1]), int(bbox[1]+bbox[3]), int(bbox[0]), int(bbox[0]+bbox[2]))
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

        # evaluate metric
        # result = eval(metric_func + "(" + "result" + ",".join(list(metric_params.values())) + ")")
        results = cross_corr(cp.roll(buffer_th, ptr), pattern)

        if results > metric_threshold:
            cue = 1
            ser.close()  # led is on
            t2_start = perf_counter()
            print('________________TTL SENT___________________')

        frame_out = cp.asnumpy(d_frame_rs)
        writer.write(frame_out)
        metadata.write_frame_metadata(t1_start, cue)
        cue = 0

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


if __name__ == "__main__":
    imaging_config_path = str(
        pathlib.Path('/home') / 'pb' / 'PycharmProjects' / 'WideFlow' / 'Imaging' / 'imaging_config_template.json')
    session_config = load_config(imaging_config_path)
    pvc.init_pvcam()
    cam = next(PVCamera.detect_camera())
    run_session(session_config, cam)