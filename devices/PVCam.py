# For farther reading:
# https://github.com/Photometrics/PyVCAM/blob/master/Documents/PyVCAM%20Wrapper.md

from pyvcam.camera import Camera
from pyvcam import constants as const


class PVCamera(Camera):
    def __init__(self, exp_time=10, binning=(1, 1)):
        super().__init__(exp_time, binning)
        self.attr_count = const.ATTR_COUNT

    def set_splice_post_processing_attributes(self, plugin_name, plugin_parameters_list):
        # search for plugin index in pp_table
        pp_index = None
        for i in range(self.attr_count):
            self.set_param(const.PARAM_PP_INDEX, i)
            if self.get_param(const.PARAM_PP_FEAT_NAME) == plugin_name:
                pp_index = i  # pp_index is the index of the post processing from the dict cam.pp_table
                break

        # set plugin parameters
        if pp_index is not None:  # TODO: fix for new configuration file structure
            for param in plugin_parameters_list:
                self.set_param(const.PARAM_PP_PARAM_INDEX, param[0])
                self.set_param(const.PARAM_PP_PARAM, param[1])
                # param_name = self.get_param(const.PARAM_PP_PARAM_NAME)
