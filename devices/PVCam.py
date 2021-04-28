# For farther reading:
# https://github.com/Photometrics/PyVCAM/blob/master/Documents/PyVCAM%20Wrapper.md

from pyvcam.camera import Camera
from pyvcam import constants as const


class PVCamera(Camera):
    def __init__(self, exp_time=10, binning=(0, 0)):
        super().__init__(exp_time, binning)
        self.attr_count = const.ATTR_COUNT

    def set_post_processing_attributes(self, plugin_name, plugin_features_list):
        pp_index = None
        for i in range(self.attr_count):
            self.set_param(const.PARAM_PP_INDEX, i)
            if self.get_param(const.PARAM_PP_FEAT_NAME) == plugin_name:
                pp_index = i  # pp_index is the index of the post processing from the dict cam.pp_table
                break

        if pp_index != None:
            for feat in plugin_features_list:
                self.set_param(const.PARAM_PP_PARAM_INDEX, feat[0])
                self.set_param(const.PARAM_PP_PARAM, feat[1])
                # param_name = self.get_param(const.PARAM_PP_PARAM_NAME)



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