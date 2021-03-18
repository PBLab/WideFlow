from devices.PVCam import PVCamera
from pyvcam import pvc

from devices.serial_port import ser

from core.processing import Processing
from core.metric import Metric

from utils.imaging_utils import *
# from devices.parallel_port import port

import pathlib


if __name__ == "__main__":
    imaging_config_path = str(pathlib.Path('/home') / 'pb' / 'PycharmProjects' / 'WideFlow' / 'Imaging' / 'imaging_config_template.json')
    config = load_config(imaging_config_path)

    pvc.init_pvcam()
    cam = next(PVCamera.detect_camera())
    cam.open()
    cam.set_post_processing_attributes(config["acquisition_config"])

    # process = Processing.get_child_from_str(config["process_config"]["method"], ["process_config"]["attributes"])
    # metric = Metric.get_child_from_str(config["metric_config"]["method"], ["process_config"]["attributes"])

    frame_counter = 0
    cam.start_live()
    while True:
        frame = cam.get_live_frame()
        print(frame[0, :5])

        # process_result = process.cal_process(frame)
        # cue = metric.calc_metric(process_result)
        #
        # if cue:
        #     ser.write(b'on')