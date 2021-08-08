# from wideflow.core.pipelines.hemodynamics_correction import HemoDynamicsDFF as PipeLine
from wideflow.core.pipelines import *
from wideflow.devices.serial_port import SerialControler

from wideflow.utils.imaging_utils import load_config
from wideflow.utils.convert_dat_to_tif import convert_dat_to_tif
from wideflow.Imaging.utils.acquisition_metadata import AcquisitionMetaData
from wideflow.Imaging.utils.roi_select import *
from wideflow.Imaging.visualization import *
from wideflow.Imaging.utils.create_matching_points import *
from wideflow.utils.load_tiff import load_tiff
from wideflow.utils.load_bbox import load_bbox
from wideflow.utils.load_matching_points import load_matching_points

import cupy as cp
import numpy as np
from scipy.signal import fftconvolve
import cv2
import os
import sys

import time
from time import perf_counter
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

import h5py
import json

import multiprocessing as mp
from multiprocessing import shared_memory, Queue


def run_session(config, cam):
    # free gpu memory
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    # session config
    camera_config = config["camera_config"]
    serial_config = config["serial_port_config"]
    cortex_config = config["rois_data_config"]
    acquisition_config = config["acquisition_config"]
    feedback_config = config["feedback_config"]
    analysis_pipeline_config = config["analysis_pipeline_config"]
    visualization_config = config["visualization_config"]

    # load roi data file
    with h5py.File(config["rois_data_config"]["cortex_file_path"], 'r') as f:
        cortex_map = np.transpose(f["map"][()])

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
    if not os.path.exists(config["rois_data_config"]["reference_image_path"]):
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

    else:  # if a reference image exist, use
        ref_image = load_tiff(config["rois_data_config"]["reference_image_path"] + "reference_image.tif")
        ref_bbox = load_bbox(config["rois_data_config"]["reference_image_path"] + "bbox.txt")
        match_p_src, match_p_dst = load_matching_points(config["rois_data_config"]["reference_image_path"] + "matching_points.txt")
        cortex_config["cortex_matching_point"]["match_p_src"] = match_p_src
        cortex_config["cortex_matching_point"]["match_p_dst"] = match_p_dst

        ref_image_roi = ref_image[ref_bbox[2]: ref_bbox[3], ref_bbox[0]: ref_bbox[1]]

        corr = fftconvolve(frame, np.fliplr(np.flipud(ref_image_roi)))
        (yi, xi) = np.unravel_index(np.argmax(corr), corr.shape)
        yi = yi - (corr.shape[0] - frame.shape[0])
        xi = xi - (corr.shape[1] - frame.shape[1])
        bbox = (int(xi), int(xi + (ref_bbox[1] - ref_bbox[0])), int(yi), int(yi + (ref_bbox[3] - ref_bbox[2])))
        cam.roi = bbox
    cam.binning = tuple(camera_config["core_attr"]["binning"])

    # select matching points for allen atlas alignment
    frame = cam.get_frame()
    match_p_src = np.array(cortex_config["cortex_matching_point"]["match_p_src"])
    match_p_dst = np.array(cortex_config["cortex_matching_point"]["match_p_dst"])
    mps = MatchingPointSelector(frame, cortex_map * np.random.random(cortex_map.shape),
                                match_p_src,
                                match_p_dst,
                                cortex_config["cortex_matching_point"]["minimal_n_points"])

    src_cols = mps.src_cols
    src_rows = mps.src_rows
    coordinates = cp.asanyarray([src_cols, src_rows])

    # update config for metadata file
    config["rois_data_config"]["cortex_matching_point"]["match_p_src"] = mps.match_p_src.tolist()
    config["rois_data_config"]["cortex_matching_point"]["match_p_dst"] = mps.match_p_dst.tolist()
    config["camera_config"]["core_attr"]["roi"] = cam.roi

    # video writer settings
    metadata = AcquisitionMetaData(session_config_path=None, config=config)
    dat_shape = (acquisition_config["num_of_frames"], frame.shape[0], frame.shape[1])
    vid_mem = np.memmap(acquisition_config["vid_save_path"], dtype='uint16', mode='w+',
                        shape=dat_shape)

    # initialize visualization processes
    vis_shm, vis_processes, vis_qs, vis_buffers = [], [], [], []
    for key, vis_config in visualization_config.items():
        if vis_config["status"]:
            a = np.ndarray(vis_config["size"], dtype=np.dtype(vis_config["dtype"]))
            shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
            shm_name = shm.name
            vis_shm.append(np.ndarray(vis_config["size"], dtype=np.dtype(vis_config["dtype"]), buffer=shm.buf))

            vis_qs.append(Queue(5))
            params = [key + '=' + str(val) for key, val in vis_config["params"].items()]
            target = eval(vis_config["class"] + '(vis_qs[-1], ' + ','.join(params) + ')')  # LiveVideo(vis_qs[-1])
            vis_processes.append(mp.Process(target=target, args=(shm_name,)))
            vis_processes[-1].start()

            vis_buffers.append(vis_config["buffer"])

    # set pipeline
    # pipeline = PipeLine(cam, coordinates, **analysis_pipeline_config["args"])
    pipeline = eval(analysis_pipeline_config["pipeline"] + "(cam, coordinates, **analysis_pipeline_config['args'])")
    pipeline.camera.start_live()
    pipeline.fill_buffers()

    # start session
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    time.sleep(1)
    print(f'starting session at {time.localtime().tm_hour}:{time.localtime().tm_min}:{time.localtime().tm_sec}')
    frame_counter = 0
    feedback_time = 0
    while frame_counter < acquisition_config["num_of_frames"]:
        frame_clock_start = perf_counter()
        pipeline.process()

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
        # vid_mem[frame_counter] = getattr(pipeline, acquisition_config["frame_var"])
        vid_mem[frame_counter] = pipeline.frame
        vid_mem.flush()
        serial_readout = ser.getReadout()

        metadata.write_frame_metadata(frame_clock_start, cue, result, serial_readout)

        # update visualization
        for i in range(len(vis_processes)):
            if not vis_qs[i].full():
                vis_qs[i].put("draw")
            vis_shm[i][:] = cp.asnumpy(getattr(pipeline, vis_buffers[i])[pipeline.ptr, :, :])

        frame_counter += 1
        frame_clock_stop = perf_counter()
        print(f'frame: {frame_counter}      metric results: {result}')
        print("Elapsed time:", frame_clock_stop - frame_clock_start)
        print(f'serial_readout: {serial_readout}')

    ###########################################################################################################
    ###########################################################################################################
    metadata.save_file()

    pipeline.camera.stop_live()
    pipeline.camera.close()
    pipeline.clear_buffers()

    ser.close()

    now = datetime.now()
    with open(config["path"] + config["name"] + "_" + now.strftime("%m_%d_%Y__%H_%M_%S") + '.json', 'w') as fp:
        json.dump(config, fp)

    print("converting imaging dat file into tiff, this might take few minutes")
    try:
        frame_offset = frame.nbytes
        frame_shape = frame.shape
        del vid_mem  # closes the dat file
        convert_dat_to_tif(acquisition_config["vid_save_path"], frame_offset, frame_shape,
                           str(frame.dtype), acquisition_config["num_of_frames"])
        os.remove(acquisition_config["vid_save_path"])

    except:
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

    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    print(f"session finished successfully at: "
          f"{time.localtime().tm_hour}:{time.localtime().tm_min}:{time.localtime().tm_sec}")


if __name__ == "__main__":
    from pyvcam import pvc
    from wideflow.devices.PVCam import PVCamera
    import pathlib

    # imaging_config_path = str(
    #     pathlib.Path('/home') / 'pb' / 'PycharmProjects' / 'WideFlow' / 'Imaging' / 'imaging_configurations'/ 'training_config.json')
    imaging_config_path = str(
        pathlib.Path(
            '/home') / 'pb' / 'PycharmProjects' / 'WideFlow' / 'wideflow' / 'Imaging' / 'imaging_configurations' / 'neurofeedback_3422_config.json')
    session_config = load_config(imaging_config_path)

    pvc.init_pvcam()
    cam = next(PVCamera.detect_camera())
    run_session(session_config, cam)
