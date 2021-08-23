#!/usr/bin/env python

import sys
from core.pipelines import *
from devices.serial_port import SerialControler

from utils.imaging_utils import load_config
from utils.convert_dat_to_tif import convert_dat_to_tif
from Imaging.utils.memmap_process import MemoryHandler
from Imaging.utils.acquisition_metadata import AcquisitionMetaData
from Imaging.utils.roi_select import *
from Imaging.visualization import *
from Imaging.utils.create_matching_points import *
from Imaging.utils.behavioral_camera_process import run_triggered_behavioral_camera
from utils.load_tiff import load_tiff
from utils.load_bbox import load_bbox
from utils.load_matching_points import load_matching_points

import cupy as cp
import numpy as np
from scipy.signal import fftconvolve
import os

import time
from time import perf_counter
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

import json

import multiprocessing as mp
from multiprocessing import shared_memory, Queue


def run_session(config, cam):
    # session config
    base_path = config["path"]
    camera_config = config["camera_config"]
    serial_config = config["serial_port_config"]
    behavioral_camera_config = config["behavioral_camera_config"]
    acquisition_config = config["acquisition_config"]
    feedback_config = config["feedback_config"]
    analysis_pipeline_config = config["analysis_pipeline_config"]
    visualization_config = config["visualization_config"]

    # set feedback metric
    feedback_threshold = feedback_config["metric_threshold"]
    inter_feedback_delay = feedback_config["inter_feedback_delay"]

    # serial port
    ser = SerialControler(port=serial_config["port_id"],
                          baudrate=serial_config["baudrate"],
                          timeout=serial_config["timeout"])

    # open camera and set camera settings
    cam.open()
    for key, value in camera_config["attr"].items():
        setattr(cam, key, value)

    for key, value in camera_config["core_attr"].items():
        if type(getattr(cam, key)) == type(value):
            setattr(cam, key, value)
        else:
            setattr(cam, key, type(getattr(cam, key))(value))

    cam.start_up()

    if camera_config["splice_plugins_enable"]:
        for plugin_dict in camera_config["splice_plugins_settings"]:
            cam.set_splice_post_processing_attributes(plugin_dict["name"], plugin_dict["parameters"])

    # select roi
    cam.binning = (1, 1)  # set no binning for ROI selection
    frame = cam.get_frame()
    if not os.path.exists(config["reference_image_path"]):
        fig, ax = plt.subplots()
        ax.imshow(cp.asnumpy(frame))
        toggle_selector = RectangleSelector(ax, onselect, drawtype='box')
        fig.canvas.mpl_connect('key_press_event', toggle_selector)
        plt.show()
        bbox = toggle_selector._rect_bbox
        if np.sum(bbox) > 1:
            # convert to PyVcam format
            bbox = (int(bbox[0]), int(bbox[0] + bbox[2]), int(bbox[1]), int(bbox[1] + bbox[3]))
            cam.roi = bbox
            #  camera ROI is defined as: (x_min, x_max, y_min, y_max)
            #  bbox is defined (before conversion) as: (x_min, x_width, y_min, y_width)

    else:  # if a reference image exist, use it to select roi and matching points
        ref_image = load_tiff(config["reference_image_path"] + "reference_image.tif")
        ref_bbox = load_bbox(config["reference_image_path"] + "bbox.txt")
        match_p_src, match_p_dst = load_matching_points(config["reference_image_path"] + "matching_points.txt")
        if 'match_p_src' and 'match_p_dst' in analysis_pipeline_config['args'].keys():
            analysis_pipeline_config['args']["match_p_src"] = match_p_src
            analysis_pipeline_config['args']["match_p_dst"] = match_p_dst

        ref_image_roi = ref_image[ref_bbox[2]: ref_bbox[3], ref_bbox[0]: ref_bbox[1]]

        corr = fftconvolve(frame, np.fliplr(np.flipud(ref_image_roi)))
        (yi, xi) = np.unravel_index(np.argmax(corr), corr.shape)
        yi = yi - (corr.shape[0] - frame.shape[0])
        xi = xi - (corr.shape[1] - frame.shape[1])
        bbox = (int(xi), int(xi + (ref_bbox[1] - ref_bbox[0])), int(yi), int(yi + (ref_bbox[3] - ref_bbox[2])))
        cam.roi = bbox
    cam.binning = tuple(camera_config["core_attr"]["binning"])

    # video writer settings and metadata handler
    metadata = AcquisitionMetaData(session_config_path=None, config=config)

    data_shape = (acquisition_config["num_of_frames"], cam.shape[1], cam.shape[0])
    temp_arr = np.ndarray(data_shape[-2:], dtype=frame.dtype)
    data_shm = shared_memory.SharedMemory(create=True, size=temp_arr.nbytes)
    shm_name = data_shm.name
    frame_shm = np.ndarray(data_shape[-2:], dtype=frame.dtype, buffer=data_shm.buf)
    memq = Queue(1)
    memory_handler = MemoryHandler(memq, base_path + acquisition_config["vid_file_name"], data_shape, frame.dtype)
    mem_process = mp.Process(target=memory_handler, args=(shm_name,))
    mem_process.start()

    # start behavioral camera process
    bcam_q = Queue(10)
    bcam_process = mp.Process(target=run_triggered_behavioral_camera,
               args=(bcam_q, base_path + behavioral_camera_config["vid_file_name"]), kwargs=behavioral_camera_config["attr"])
    bcam_process.start()

    # set pipeline
    pipeline = eval(analysis_pipeline_config["pipeline"] + "(cam, **analysis_pipeline_config['args'])")

    # initialize visualization processes
    vis_shm, vis_processes, vis_qs, vis_buffers = [], [], [], []
    for key, vis_config in visualization_config.items():
        if vis_config["status"]:
            temp_arr = np.ndarray(vis_config["size"], dtype=np.dtype(vis_config["dtype"]))
            shm = shared_memory.SharedMemory(create=True, size=temp_arr.nbytes)
            shm_name = shm.name
            vis_shm.append(np.ndarray(vis_config["size"], dtype=np.dtype(vis_config["dtype"]), buffer=shm.buf))

            vis_qs.append(Queue(5))
            params = [key + '=' + str(val) for key, val in vis_config["params"].items()]
            target = eval(vis_config["class"] + '(vis_qs[-1], ' + ','.join(params) + ')')  # LiveVideo(vis_qs[-1])
            vis_processes.append(mp.Process(target=target, args=(shm_name,)))
            vis_processes[-1].start()

            vis_buffers.append(vis_config["buffer"])
    del temp_arr

    # start session
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print(f'starting session at {time.localtime().tm_hour}:{time.localtime().tm_min}:{time.localtime().tm_sec}')
    frame_counter = 0
    feedback_time = 0
    pipeline.fill_buffers()
    pipeline.camera.start_live()
    while frame_counter < acquisition_config["num_of_frames"]:
        frame_clock_start = perf_counter()
        pipeline.process()
        bcam_q.put('grab')

        # evaluate metric and send TTL if metric above threshold
        cue = 0
        result = pipeline.evaluate()
        if cp.asnumpy(result) > feedback_threshold and (
                frame_clock_start - feedback_time) * 1000 > inter_feedback_delay:
            feedback_time = perf_counter()
            cue = 1
            ser.sendFeedback()
            print('________________FEEDBACK HAS BEEN SENT___________________\n'
                  '_________________________________________________________')

        # save data
        frame_shm[:] = pipeline.frame
        memq.put("flush")

        serial_readout = ser.getReadout()
        metadata.write_frame_metadata(frame_clock_start, cue, result, serial_readout)

        # update visualization
        for i in range(len(vis_processes)):
            if not vis_qs[i].full():
                vis_qs[i].put("draw")
            vis_shm[i][:] = cp.asnumpy(getattr(pipeline, vis_buffers[i])[pipeline.ptr, :, :])

        frame_counter += 1
        frame_clock_stop = perf_counter()
        print(f'frame: {frame_counter:06d} '
              f'metric results: {result:.3f} '
              f'Elapsed time:{frame_clock_stop - frame_clock_start:.3f} '
              f'serial_readout: {serial_readout}', end='\r')

    ###########################################################################################################
    ###########################################################################################################
    bcam_q.put("finish")
    metadata.save_file()

    pipeline.camera.stop_live()
    pipeline.camera.close()
    pipeline.clear_buffers()
    config = pipeline.update_config(config)

    ser.close()

    now = datetime.now()
    with open(config["path"] + config["name"] + "_" + now.strftime("%m_%d_%Y__%H_%M_%S") + '.json', 'w') as fp:
        json.dump(config, fp)

    print("converting imaging dat file into tiff, this might take few minutes")
    try:
        frame_offset = pipeline.frame.nbytes
        frame_shape = data_shape[-2:]
        memq.put("terminate")  # closes the dat file
        convert_dat_to_tif(base_path + acquisition_config["vid_file_name"], frame_offset,
                           (2000, frame_shape[0], frame_shape[1]),  # ~2000 frames is the maximum amount of frames readable using Fiji imagej
                           str(frame.dtype), acquisition_config["num_of_frames"])
        os.remove(base_path + acquisition_config["vid_file_name"])

    except RuntimeError:
        print("something went wrong while converting to tiff. dat file still exist in folder")
        print("Unexpected error:", sys.exc_info()[0])
        raise

    finally:
        print("done")

    # terminate visualization processes
    for i in range(len(vis_processes)):
        if vis_qs[i].full():
            vis_qs[i].get()
        vis_qs[i].put("terminate")
        vis_processes[i].join()
        vis_processes[i].terminate()

    print(f"session finished successfully at: "
          f"{time.localtime().tm_hour}:{time.localtime().tm_min}:{time.localtime().tm_sec}")


if __name__ == "__main__":
    from pyvcam import pvc
    from devices.PVCam import PVCamera
    import pathlib

    imaging_config_path = str(
        pathlib.Path(
            '/home') / 'pb' / 'PycharmProjects' / 'WideFlow' / 'wideflow' / 'Imaging' / 'imaging_configurations' / 'training_config.json')
    session_config = load_config(imaging_config_path)

    pvc.init_pvcam()
    cam = next(PVCamera.detect_camera())
    run_session(session_config, cam)