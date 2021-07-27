from devices.cupy_cuda_kernels import *
from devices.serial_port import SerialControler

from utils.imaging_utils import load_config
from utils.gen_utils import extract_rois_data
from Imaging.utils.acquisition_metadata import AcquisitionMetaData
from Imaging.utils.roi_select import *
from Imaging.utils.multiple_trace_plot import TracePlot
from Imaging.utils.live_video import LiveVideo
from Imaging.utils.create_matching_points import *

from utils.load_tiff import load_tiff
from utils.load_matlab_vector_field import load_extended_rois_list

import cupy as cp
import cupyx.scipy.ndimage as csn
import numpy as np
import cv2
from skvideo.io import FFmpegWriter

import time
from time import perf_counter
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
import h5py

import multiprocessing as mp
from multiprocessing import shared_memory, Queue


def run_session(config, cam):

    # session config
    camera_config = config["camera_config"]
    serial_config = config["serial_port_config"]
    cortex_config = config["rois_data"]
    acquisition_config = config["acquisition_config"]
    feedback_config = config["feedback_config"]
    allocation_config = config["allocation_config"]
    pipeline_config = config["preprocess_pipeline_config"]
    metric_config = config["metric_config"]
    visualization_config = config["visualization_config"]

    # circular buffers capacity
    capacity = acquisition_config["capacity"]

    # load roi data file
    rois_dict = load_extended_rois_list(config["rois_data"]["file_path"])
    with h5py.File(config["rois_data"]["cortex_file_path"], 'r') as f:
        cortex_mask = np.transpose(f["mask"][()])
        cortex_map = np.transpose(f["map"][()])
    d_mask = cp.asanyarray(cortex_mask)
    d_mask_stack = cp.stack((d_mask, ) * capacity, 0)

    # set feedback metric
    pattern = cp.asanyarray(load_tiff(feedback_config["pattern_path"])) / 65535  # convert back from uint16 to original range
    pattern = cp.multiply(pattern, d_mask_stack)
    feedback_threshold = feedback_config["metric_threshold"]

    # allocate memory in device - first stage allocation
    for allc_params in allocation_config["first_stage"]:
        locals()[allc_params["name"]] = \
            cp.asanyarray(np.empty(shape=allc_params["size"], dtype=np.dtype(allc_params["dtype"])))
        if allc_params["init"] is not None:
            locals()[allc_params["name"]] = \
                eval(allc_params["init"]["function"] + "(" + ",".join(allc_params["init"]["args"]) + ")")

    # serial port
    ser = SerialControler(port=serial_config["port_id"],
                          baudrate=serial_config["baudrate"],
                          timeout=serial_config["timeout"])

    # open camera and set camera settings
    cam.open()
    for key, value in camera_config["core_attr"].items():
        if type(getattr(cam, key)) == type(value):
            setattr(cam, key, value)
        else:
            setattr(cam, key, type(getattr(cam, key))(value))

    if camera_config["splice_plugins_enable"]:
        for plugin_dict in camera_config["splice_plugins_settings"]:
            cam.set_splice_post_processing_attributes(plugin_dict["name"], plugin_dict["parameters"])

    # select roi
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

    # select matching points for allen atlas alignment
    frame = cam.get_frame()
    match_p_src = np.array(
        [[0, 0], [0, 297], [337, 297], [337, 0], [401., 264.], [182., 438.], [191., 834.], [395., 822.], [453., 750.],
         [518., 827.], [756., 820.],
         [744., 443.], [573., 259.], [448., 254.], [455., 501.], [436., 389.], [450., 622.]])
    match_p_dst = np.array(
        [[0, 0], [0, 297], [337, 297], [337, 0], [155., 6.], [13., 118.], [17., 287.], [121., 287.], [167., 237.],
         [214., 287.], [326., 286.], [324., 114.],
         [242., 13.], [182., 5.], [167., 124.], [169., 64.], [167., 181.]])
    appf = ApprovalFigure(frame, cortex_map * np.random.random(cortex_map.shape),
                          match_p_src,
                          match_p_dst,
                          cortex_config["cortex_matching_point"]["minimal_n_points"])
    # appf = ApprovalFigure(frame, cortex_map * np.random.random(cortex_map.shape),
    #                       cortex_config["cortex_matching_point"]["match_p_src"],
    #                       cortex_config["cortex_matching_point"]["match_p_dst"],
    #                       cortex_config["cortex_matching_point"]["minimal_n_points"])

    src_cols = appf.src_cols
    src_rows = appf.src_rows
    coordinates = cp.asanyarray([src_cols, src_rows])
    if int(cp.__version__[0]) < 9:
        coordinates = np.squeeze(coordinates)


    # update config for metadata file
    config["rois_data"]["cortex_matching_point"]["match_p_src"] = appf.match_p_src
    config["rois_data"]["cortex_matching_point"]["match_p_dst"] = appf.match_p_dst
    config["camera_config"]["core_attr"]["roi"] = cam.roi

    # video writer settings
    vid_write_config = acquisition_config["vid_writer"]
    frame_out_str = vid_write_config["frame_var"]
    metadata = AcquisitionMetaData(session_config_path=None, config=config)
    writer = FFmpegWriter(acquisition_config["vid_save_path"])

    # fill buffers before session starts
    cam.start_live()
    frame_counter = 0
    ptr = capacity - 1
    while frame_counter < capacity:
        if ptr == capacity - 1:
            ptr = 0
        else:
            ptr += 1

        frame = cam.get_live_frame()
        d_frame = cp.asanyarray(frame)

        for process in itemgetter(*acquisition_config["init_process_list"])(pipeline_config):
            if process["return"]["buffer_type"] == "buffer":
                locals()[process["return"]["to"]] = \
                    eval(process["function"] + "(" + ",".join(process["args"]) + ")")

            elif process["return"]["buffer_type"] == "circ_buffer":
                locals()[process["return"]["to"]][ptr, :, :] = \
                    eval(process["function"] + "(" + ",".join(process["args"]) + ")")

            else:
                eval(process["function"] + "(" + ",".join(process["args"]) + ")")

        frame_counter += 1

    # allocate memory in device - second stage allocation
    for allc_params in allocation_config["second_stage"]:
        locals()[allc_params["name"]] = \
            cp.asanyarray(np.empty(shape=allc_params["size"], dtype=np.dtype(allc_params["dtype"])))
        if allc_params["init"] is not None:
            locals()[allc_params["name"]] = \
                eval(allc_params["init"]["function"] + "(" + ",".join(allc_params["init"]["args"]) + ")")

    # initialize visualization processes
    if visualization_config["show_live_stream"]["status"]:
        live_stream_config = visualization_config["show_live_stream"]
        a = np.ndarray(live_stream_config["size"],  dtype=np.dtype(live_stream_config["dtype"]))
        live_vid_shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
        shm_name = live_vid_shm.name
        live_vid_frame = np.ndarray(live_stream_config["size"],  dtype=np.dtype(live_stream_config["dtype"]), buffer=live_vid_shm.buf)

        live_vid_q = Queue(5)
        live_vid = LiveVideo(live_vid_q)

        live_vid_process = mp.Process(target=live_vid, args=(shm_name, ))
        live_vid_process.start()

    if visualization_config["show_rois_trace"]["status"]:
        trace_plot_config = visualization_config["show_rois_trace"]
        a = np.ndarray(trace_plot_config["size"], dtype=np.dtype(trace_plot_config["dtype"]))
        trace_plot_shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
        shm_name = trace_plot_shm.name
        trace_plot_samples = np.ndarray(trace_plot_config["size"], dtype=np.dtype(trace_plot_config["dtype"]),
                                    buffer=trace_plot_shm.buf)

        rois_trace_q = Queue(5)
        trace_plot = TracePlot(rois_trace_q, len(rois_dict), 0.05, list(rois_dict.keys()), capacity)

        trace_plot_process = mp.Process(target=trace_plot, args=(shm_name,))
        trace_plot_process.start()

    if visualization_config["show_pattern_weight"]["status"]:
        pattern_weight_config = visualization_config["show_pattern_weight"]
        a = np.ndarray(pattern_weight_config["size"], dtype=np.dtype(pattern_weight_config["dtype"]))
        pattern_corr_shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
        shm_name = pattern_corr_shm.name
        pattern_corr_sample = np.ndarray(pattern_weight_config["size"], dtype=np.dtype(pattern_weight_config["dtype"]),
                                         buffer=pattern_corr_shm.buf)

        pattern_corr_q = Queue(5)
        pattern_corr_plot = TracePlot(pattern_corr_q, 1, 1, ['pattern corr'], capacity)

        pattern_corr_process = mp.Process(target=pattern_corr_plot, args=(shm_name,))
        pattern_corr_process.start()

    # start session
    print(f'starting session at {time.localtime().tm_hour}:{time.localtime().tm_min}:{time.localtime().tm_sec}')
    frame_counter = 0
    ptr = capacity - 1
    while frame_counter < acquisition_config["num_of_frames"]:
        if ptr == capacity - 1:
            ptr = 0
        else:
            ptr += 1

        frame_clock_start = perf_counter()
        frame = cam.get_live_frame()
        d_frame = cp.asanyarray(frame)

        # preprocessing pipeline
        for process in pipeline_config:
            if process["return"]["buffer_type"] == "buffer":
                locals()[process["return"]["to"]] = \
                    eval(process["function"] + "(" + ",".join(process["args"]) + ")")

            elif process["return"]["buffer_type"] == "circ_buffer":
                locals()[process["return"]["to"]][ptr, :, :] = \
                    eval(process["function"] + "(" + ",".join(process["args"]) + ")")

            else:
                eval(process["function"] + "(" + ",".join(process["args"]) + ")")

        # evaluate metric
        result = eval(metric_config["function"] + "(" + ",".join(metric_config["args"]) + ")")

        # send TTL if metric above threshold
        cue = 0
        if cp.asnumpy(result) > feedback_threshold:
            cue = 1
            ser.sendTTL()
            print('________________TTL SENT___________________')

        # frame_out = np.array(cv2.normalize(cp.asnumpy(locals()["buffer_dff"][ptr, :, :]), None, 0, 255, cv2.NORM_MINMAX), dtype=np.uint8)
        frame_out = np.array(cv2.normalize(cp.asnumpy(locals()[frame_out_str][ptr, :, :]), None, 0, 255, cv2.NORM_MINMAX), dtype=np.uint8)
        writer.writeFrame(frame_out)
        serial_readout = ser.readSerial()
        metadata.write_frame_metadata(frame_clock_start, cue, result, serial_readout)

        if visualization_config["show_live_stream"]["status"]:
            if not live_vid_q.full():

                live_vid_q.put("draw")
            live_vid_frame[:] = cp.asnumpy(locals()[live_stream_config["var_name"]][ptr, :, :])

        if visualization_config["show_rois_trace"]["status"]:
            if not rois_trace_q.full():
                rois_trace_q.put("draw")
            rois_trace = extract_rois_data(cp.asnumpy(locals()["buffer_dff"][ptr, :, :]), rois_dict)
            rois_trace = np.reshape(rois_trace, (56, 1))
            trace_plot_samples[:] = rois_trace

        if visualization_config["show_pattern_weight"]["status"]:
            if not pattern_corr_q.full():
                pattern_corr_q.put("draw")
            pattern_corr_sample[:] = result


        frame_counter += 1
        frame_clock_stop = perf_counter()
        print(f'frame: {frame_counter}      metric results: {result}')
        print("Elapsed time:", frame_clock_stop - frame_clock_start)

    if visualization_config["show_live_stream"]["status"]:
        if live_vid_q.full():
            live_vid_q.get()
        live_vid_q.put("terminate")
    if visualization_config["show_rois_trace"]["status"]:
        if rois_trace_q.full():
            rois_trace_q.get()
        rois_trace_q.put("terminate")
    if visualization_config["show_pattern_weight"]["status"]:
        if pattern_corr_q.full():
            pattern_corr_q.get()
        pattern_corr_q.put("terminate")

    metadata.save_file()
    cam.stop_live()
    cam.close()
    ser.close()
    writer.close()
    print("finished session")


if __name__ == "__main__":
    from pyvcam import pvc
    from devices.PVCam import PVCamera
    import pathlib

    imaging_config_path = str(
        pathlib.Path('/home') / 'pb' / 'PycharmProjects' / 'WideFlow' / 'Imaging' / 'imaging_config_template.json')
    session_config = load_config(imaging_config_path)

    # from devices.mock_cam import Camera
    # cam = Camera()
    pvc.init_pvcam()
    cam = next(PVCamera.detect_camera())
    run_session(session_config, cam)

    # import cProfile
    # cProfile.run('run_session(session_config, cam)')
