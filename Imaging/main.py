from core.pipelines.patterns_detection import PatternsDetection as Pipeline
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
import numpy as np
from skvideo.io import FFmpegWriter

import time
from time import perf_counter
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
    visualization_config = config["visualization_config"]

    # circular buffers capacity
    capacity = acquisition_config["capacity"]

    # load roi data file
    rois_dict = load_extended_rois_list(config["rois_data"]["file_path"])
    with h5py.File(config["rois_data"]["cortex_file_path"], 'r') as f:
        cortex_mask = np.transpose(f["mask"][()])
        cortex_map = np.transpose(f["map"][()])
    d_mask = cp.asanyarray(cortex_mask)

    # set feedback metric
    pattern = cp.asanyarray(load_tiff(feedback_config["pattern_path"])) / 65535  # convert back from uint16 to original range
    pattern = cp.multiply(pattern, d_mask)
    feedback_threshold = feedback_config["metric_threshold"]

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
    appf = ApprovalFigure(frame, cortex_map * np.random.random(cortex_map.shape),
                          cortex_config["cortex_matching_point"]["match_p_src"],
                          cortex_config["cortex_matching_point"]["match_p_dst"],
                          cortex_config["cortex_matching_point"]["minimal_n_points"])

    src_cols = appf.src_cols
    src_rows = appf.src_rows
    coordinates = cp.asanyarray([src_cols, src_rows])

    # update config for metadata file
    config["rois_data"]["cortex_matching_point"]["match_p_src"] = appf.match_p_src
    config["rois_data"]["cortex_matching_point"]["match_p_dst"] = appf.match_p_dst
    config["camera_config"]["core_attr"]["roi"] = cam.roi

    # video writer settings
    metadata = AcquisitionMetaData(session_config_path=None, config=config)
    writer = FFmpegWriter(acquisition_config["vid_save_path"])

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
        trace_plot = TracePlot(rois_trace_q, len(rois_dict), 0.05, list(rois_dict.keys()), 30)

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

    # set pipeline
    pipeline = Pipeline(cam, cortex_map.shape, capacity, coordinates, pattern)
    pipeline.camera.start_live()
    pipeline.fill_buffers()

    # start session
    print(f'starting session at {time.localtime().tm_hour}:{time.localtime().tm_min}:{time.localtime().tm_sec}')
    frame_counter = 0
    while frame_counter < acquisition_config["num_of_frames"]:
        frame_clock_start = perf_counter()
        pipeline.process()

        # evaluate metric
        result = pipeline.evaluate()

        # send TTL if metric above threshold
        cue = 0
        if cp.asnumpy(result) > feedback_threshold:
            cue = 1
            ser.sendTTL()
            print('________________TTL SENT___________________')

        writer.writeFrame(pipeline.frame)
        serial_readout = ser.readSerial()
        metadata.write_frame_metadata(frame_clock_start, cue, result, serial_readout)

        if visualization_config["show_live_stream"]["status"]:
            if not live_vid_q.full():
                live_vid_q.put("draw")
            live_vid_frame[:] = cp.asnumpy(getattr(pipeline, live_stream_config["attr"])[pipeline.ptr, :, :])

        if visualization_config["show_rois_trace"]["status"]:
            if not rois_trace_q.full():
                rois_trace_q.put("draw")
            rois_trace = extract_rois_data(cp.asnumpy(getattr(pipeline, pattern_weight_config["attr"])[pipeline.ptr, :, :]), rois_dict)
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

    pipeline.camera.stop_live()
    pipeline.camera.close()
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

    pvc.init_pvcam()
    cam = next(PVCamera.detect_camera())

    run_session(session_config, cam)
    # import cProfile
    # cProfile.run('run_session(session_config, cam)')
