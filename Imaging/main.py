from core.pipelines.patterns_detection import PatternsDetection as PipeLine
from devices.serial_port import SerialControler

from utils.imaging_utils import load_config
from Imaging.utils.acquisition_metadata import AcquisitionMetaData
from Imaging.utils.roi_select import *
from Imaging.visualization import *
from Imaging.utils.create_matching_points import *

import cupy as cp
import numpy as np
from skvideo.io import FFmpegWriter

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

    # session config
    camera_config = config["camera_config"]
    serial_config = config["serial_port_config"]
    cortex_config = config["rois_data_config"]
    acquisition_config = config["acquisition_config"]
    feedback_config = config["feedback_config"]
    analysis_pipeline_config = config["analysis_pipeline_config"]
    visualization_config = config["visualization_config"]

    # load roi data file
    with h5py.File(config["rois_data"]["cortex_file_path"], 'r') as f:
        cortex_map = np.transpose(f["map"][()])

    # set feedback metric
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
    vis_shm, vis_processes, vis_qs, vis_buffers = [], [], [], []
    for key, vis_config in visualization_config.items():
        if vis_config["status"]:
            a = np.ndarray(vis_config["size"], dtype=np.dtype(vis_config["dtype"]))
            shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
            shm_name = shm.name
            vis_shm.append(np.ndarray(vis_config["size"], dtype=np.dtype(vis_config["dtype"]), buffer=shm.buf))

            vis_qs.append(Queue(5))
            params = [key + '=' + str(val) for key, val in vis_config["params"].items()]
            target = eval(vis_config["class"] + '(vis_qs[-1], ' + ','.join(params) + ')')       # LiveVideo(vis_qs[-1])
            vis_processes.append(mp.Process(target=target, args=(shm_name,)))
            vis_processes[-1].start()

            vis_buffers.append(vis_config["buffer"])

    # set pipeline
    pipeline = PipeLine(cam, coordinates, **analysis_pipeline_config["args"])
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

        # save data
        writer.writeFrame(pipeline.frame)
        serial_readout = ser.readSerial()
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

    # terminate visualization processes
    for i in range(len(vis_processes)):
        if vis_qs[i].full():
            vis_qs[i].get()
        vis_qs[i].put("terminate")

    metadata.save_file()
    pipeline.camera.stop_live()
    pipeline.camera.close()
    ser.close()
    writer.close()
    now = datetime.now()
    with open(config["path"][:-6] + now.strftime("%m_%d_%Y__%H_%M_%S") + '.json', 'w') as fp:
        json.dump(config, fp)

    print(f"session finished successfully at {time.localtime().tm_hour}:{time.localtime().tm_min}:{time.localtime().tm_sec}")


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
