from devices.cupy_cuda_kernels import *
from devices.serial_port import SerialControler

from utils.imaging_utils import load_config
from Imaging.utils.acquisition_metadata import AcquisitionMetaData
from Imaging.utils.roi_select import *
from utils.load_tiff import load_tiff
from utils.load_matlab_vector_field import load_allen_2d_cortex_rois

import cupy as cp
import numpy as np
from skvideo.io import FFmpegWriter

import time
from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.animation import FuncAnimation


def run_session(config, cam):

    # process & metric config
    camera_config = config["camera_config"]
    serial_config = config["serial_port_config"]
    acquisition_config = config["acquisition_config"]
    pipeline_config = config["preprocess_pipeline_config"]
    metric_config = config["metric_config"]
    visualization_config = config["visualization_config"]

    # load roi data file
    rois_dict = load_allen_2d_cortex_rois(config["rois_data"]["file_path"])

    # allocate circular buffer in device
    for process_config in pipeline_config:
        if process_config["circ_buffer"] is not None:
            capacity = process_config["circ_buffer"]["size"][0]
            break

    for process_config in pipeline_config:
        if process_config["circ_buffer"] is not None:
            name = process_config["circ_buffer"]["name"]
            size = process_config["circ_buffer"]["size"]
            locals()[name] = cp.asanyarray(np.empty(size))
        if process_config["buffer"] is not None:
            name = process_config["buffer"]["name"]
            size = process_config["buffer"]["size"]
            locals()[name] = cp.asanyarray(np.empty(size))

    # set feedback metric
    pattern = cp.asanyarray(load_tiff(metric_config["attributes"]["pattern_path"]))
    metric_threshold = metric_config["attributes"]["parameters"]["threshold"]

    # video writer settings
    vid_write_config = acquisition_config["vid_writer"]
    frame_out_str = vid_write_config["frame_var"]
    metadata = AcquisitionMetaData(session_config_path=None, config=config)
    writer = FFmpegWriter(acquisition_config["vid_save_path"])

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

    if camera_config["splice_plugins_enable"]:
        for plugin_dict in camera_config["splice_plugins_settings"]:
            cam.set_splice_post_processing_attributes(plugin_dict["name"], plugin_dict["parameters"])

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

        process = pipeline_config[0]
        eval(process["function"] + "(" + ",".join(process["args"]) + ")")
        locals()[process["circ_buffer"]["name"]][ptr, :, :] = locals()[process["buffer"]["name"]]

        frame_counter += 1

    # start session
    print(f'starting session at {time.localtime().tm_hour}:{time.localtime().tm_min}:{time.localtime().tm_sec}')
    if visualization_config["show_live_stream"]:
        fig = plt.figure()
        ax = plt.gca()
        im = ax.imshow(cp.asnumpy(np.array(cp.asnumpy(locals()[frame_out_str]), dtype=np.uint8)), animated=True)

    frame_counter = 0
    ptr = capacity - 1
    while frame_counter < acquisition_config["num_of_frames"]:
        if ptr == capacity - 1:
            ptr = 0
        else:
            ptr += 1

        t1_start = perf_counter()
        frame = cam.get_live_frame()
        d_frame = cp.asanyarray(frame)

        # preprocessing pipeline
        for process in pipeline_config:
            if process["return"] == "buffer":
                locals()[process["buffer"]["name"]] = \
                    eval(process["function"] + "(" + ",".join(process["args"]) + ")")

            elif process["return"] == "circ_buffer":
                eval(process["function"] + "(" + ",".join(process["args"]) + ")")
                locals()[process["circ_buffer"]["name"]][ptr, :, :] = locals()[process["buffer"]["name"]]

            else:
                eval(process["function"] + "(" + ",".join(process["args"]) + ")")

        # evaluate metric
        result = eval(metric_config["function"] + "(" + ",".join(process["args"]) + ")")

        # send TTL if
        cue = 0
        if cp.asnumpy(result) > metric_threshold:
            cue = 1
            ser.sendTTL()
            t2_start = perf_counter()
            print('________________TTL SENT___________________')

        frame_out = np.array(cp.asnumpy(locals()[frame_out_str]), dtype=np.uint8)
        writer.writeFrame(frame_out)
        metadata.write_frame_metadata(t1_start, cue)

        if visualization_config["show_live_stream"]:
            im.set_data(frame_out)
            plt.pause(0.0001)
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
    ser.close()
    writer.close()
    metadata.save_file()
    # print("__________________________delay time:", t2_stop - t2_start)


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