from devices.cupy_cuda_kernels import *
from devices.serial_port import SerialControler

from utils.imaging_utils import load_config
from utils.gen_utils import extract_rois_data
from Imaging.utils.acquisition_metadata import AcquisitionMetaData
from Imaging.utils.roi_select import *
from Imaging.utils.multiple_trace_plot import TracePlot
from utils.load_tiff import load_tiff
from utils.load_matlab_vector_field import load_extended_rois_list

import cupy as cp
import numpy as np
from skvideo.io import FFmpegWriter

import time
from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


def run_session(config, cam):

    # process & metric config
    camera_config = config["camera_config"]
    serial_config = config["serial_port_config"]
    acquisition_config = config["acquisition_config"]
    feedback_config = config["feedback_config"]
    pipeline_config = config["preprocess_pipeline_config"]
    metric_config = config["metric_config"]
    visualization_config = config["visualization_config"]

    # load roi data file
    rois_dict = load_extended_rois_list(config["rois_data"]["file_path"])
    # circular buffers capacity
    capacity = acquisition_config["capacity"]

    # allocate memory in device
    for process_config in pipeline_config:
        if len(process_config["allocate"]) != 0:
            for settings in process_config["allocate"]:
                locals()[settings["name"]] = \
                    cp.asanyarray(np.empty(shape=settings["size"], dtype=np.dtype(settings["dtype"])))


    # set feedback metric
    pattern = cp.asanyarray(load_tiff(feedback_config["pattern_path"]))
    feedback_threshold = feedback_config["metric_threshold"]

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
        # eval(process["function"] + "(" + ",".join(process["args"]) + ")")
        # locals()[process["return"]["to"]][ptr, :, :] = locals()[process["buffer"]["name"]]
        locals()[process["return"]["to"]][ptr, :, :] = eval(process["function"] + "(" + ",".join(process["args"]) + ")")

        frame_counter += 1

    # start session
    print(f'starting session at {time.localtime().tm_hour}:{time.localtime().tm_min}:{time.localtime().tm_sec}')
    if visualization_config["show_live_stream"]:
        fig = plt.figure()
        ax = plt.gca()
        im = ax.imshow(cp.asnumpy(np.array(cp.asnumpy(locals()[frame_out_str]), dtype=np.uint8)), animated=True)

    if visualization_config["show_rois_trace"]:
        plt.ion()
        trace_plot = TracePlot(len(rois_dict), 0.05, list(rois_dict.keys()), capacity)

    if visualization_config["show_pattern_weight"]:
        plt.ion()
        pattern_corr_plot = TracePlot(1, 1, 'pattern corr', capacity)

    frame_counter = 0
    ptr = capacity - 1
    while frame_counter < acquisition_config["num_of_frames"]:
        if ptr == capacity - 1:
            ptr = 0
        else:
            ptr += 1

        frame_clock_start = perf_counter()
        frame = cam.get_live_frame()
        print(frame[150,200])
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
        # if cp.mean(locals()["d_frame_rs"])>1000:
        if cp.asnumpy(result) > feedback_threshold:
            cue = 1
            ser.sendTTL()
            # ttl_clock_stop = perf_counter()
            # print(f'time elapsed between signal to ttl output: {ttl_clock_stop-ttl_clock_start}')
            print('________________TTL SENT___________________')

        frame_out = np.array(cp.asnumpy(locals()[frame_out_str]), dtype=np.uint8)
        writer.writeFrame(frame_out)
        metadata.write_frame_metadata(frame_clock_start, cue)

        if visualization_config["show_live_stream"]:
            im.set_data(frame_out)
            plt.pause(0.0001)

        if visualization_config["show_rois_trace"]:
            rois_trace = extract_rois_data(cp.asnumpy(locals()["buffer_dff"][ptr, :, :]), rois_dict)
            trace_plot.update_plot(np.array(rois_trace))

        if visualization_config["show_pattern_weight"]:
            pattern_corr_plot.update_plot(result)

        frame_counter += 1
        frame_clock_stop = perf_counter()
        print("Elapsed time:", frame_clock_stop - frame_clock_start)
        # if frame_counter == 80:
        #     ser.sendTTL()  # led is on
        #     ttl_clock_start = perf_counter()
        #     print('________________LED ON___________________')

    metadata.save_file()
    cam.stop_live()
    cam.close()
    ser.close()
    writer.close()


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