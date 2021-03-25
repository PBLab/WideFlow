from pyvcam import pvc
from pyvcam.camera import Camera

# For farther reading:
# https://github.com/Photometrics/PyVCAM/blob/master/Documents/PyVCAM%20Wrapper.md
from matplotlib import pyplot as plt

class PVCamera(Camera):
    def __init__(self, exp_time=10, binning=(0, 0)):
        super().__init__(exp_time, binning)




pvc.my_set_callback()
# pvc.init_pvcam()
# cam = next(PVCamera.detect_camera())
# cam.open()


# cam_settings = {
#     "binning": None,  # (binx, biny)
#     "exp_time": 10,
#     "roi": None  # (x_start, x_end, y_start, y_end)
# }
#
# for key, value in cam_settings.items():
#     if hasattr(cam, key) and value:
#         setattr(cam, key, value)