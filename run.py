from utils.imaging_utils import load_config
from pyvcam import pvc
from devices.PVCam import PVCamera
import pathlib
from tifffile import TiffWriter


imaging_config_path = str(
    pathlib.Path(
        '/home') / 'pb' / 'PycharmProjects' / 'WideFlow' / 'Imaging' / 'imaging_configurations' / 'training_config.json')
config = load_config(imaging_config_path)
pvc.init_pvcam()
cam = next(PVCamera.detect_camera())


camera_config = config["camera_config"]
behavioral_camera_config = config["behavioral_camera_config"]
serial_config = config["serial_port_config"]
cortex_config = config["rois_data_config"]
acquisition_config = config["acquisition_config"]
feedback_config = config["feedback_config"]
analysis_pipeline_config = config["analysis_pipeline_config"]
visualization_config = config["visualization_config"]


cam.open()
cam.start_up()
for key, value in camera_config["core_attr"].items():
    if type(getattr(cam, key)) == type(value):
        setattr(cam, key, value)
    else:
        setattr(cam, key, type(getattr(cam, key))(value))


cam.binning = (1, 1)

path = "/home/pb/WideFlow_prj/3422/reference_image.tif"
frame = cam.get_frame()
with TiffWriter(path) as tif:
    tif.write(frame, contiguous=True)