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

        else:
            raise ValueError(f"plugin {plugin_name} doesn't exist in the available post processing list.")


 # cam.pp_table
# const.PARAM_PP_FEAT_NAME - post processing plugin name
# const.PARAM_PP_PARAM_INDEX - indicate plugin dictionary key value


# {
#     'DESPECKLE BRIGHT LOW': {
#          'ENABLED': (140432545677313, 140432545677312, 140432545677313),
#          'THRESHOLD': (140432545677437, 140432545677412, 140432545677612)
#     },
#
#     'DESPECKLE BRIGHT HIGH': {
#         'ENABLED': (140432545677313, 140432545677312, 140432545677313),
#         'THRESHOLD': (140432545677452, 140432545677412, 140432545677612),
#         'MIN ADU AFFECTED': (140432545677512, 140432545677512, 140432545679359)
#     },
#
#     'DESPECKLE DARK LOW': {
#         'ENABLED': (140432545677312, 140432545677312, 140432545677313),
#         'THRESHOLD': (140432545677387, 140432545677312, 140432545677412)
#     },
#
#     'DESPECKLE DARK HIGH': {
#         'ENABLED': (140432545677313, 140432545677312, 140432545677313),
#         'THRESHOLD': (140432545677372, 140432545677312, 140432545677412),
#         'MIN ADU AFFECTED': (140432545677512, 140432545677512, 140432545679359)
#     },
#
#     'QUANTVIEW':{
#      'ENABLED': (140432545677312, 140432545677312, 140432545677313)
#     },
#
#     'GPU-TOPLOCK': {
#         'ENABLED': (140432545677312, 140432545677312, 140432545677313),
#         'WHITE CLIP': (140432545677462, 140432545677312, 140432545742847),
#         'MODULE_CALL_ORDER': (140432545677313, 140432545677312, 140432545677320)
#     }
# }


# cam_settings = {
#     "binning": None,  # (binx, biny)
#     "exp_time": 10,
#     "roi": None  # (x_start, x_end, y_start, y_end)
# }
#
# for key, value in cam_settings.items():
#     if hasattr(cam, key) and value:
#         setattr(cam, key, value)