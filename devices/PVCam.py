# For farther reading:
# https://github.com/Photometrics/PyVCAM/blob/master/Documents/PyVCAM%20Wrapper.md

from pyvcam.camera import Camera
from pyvcam import constants as const
import ctypes


class PVCamera(Camera):
    def __init__(self, exp_time=10, binning=(2, 2), channels=2):
        super().__init__(exp_time, binning)
        self.channels = channels
        self.start_up()

    def start_up(self):
        self.clear_mode("Pre-Sequence")
        self.set_param(const.PARAM_CLEAR_CYCLES, 2)
        self.exp_out_mode("All Rows")
        self.set_param(const.PARAM_LAST_MUXED_SIGNAL, self.channels)

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

    def set_smart_stream_acquisition(self, exposures_list):
        n = len(exposures_list)
        entries = ctypes.c_uint16(n)

        c_array = ctypes.c_uint32 * n
        params = c_array()
        params[0:n] = exposures_list[0:n]

        smrt_stream = smart_stream_type(entries, params)

        self.set_param(const.PARAM_LAST_MUXED_SIGNAL, n)
        self.set_param(const.PARAM_SMART_STREAM_MODE_ENABLED, True)
        self.set_param(const.PARAM_SMART_STREAM_MODE, const.SMTMODE_ARBITRARY_ALL)
        self.set_param(const.PARAM_SMART_STREAM_EXP_PARAMS, id(smrt_stream))


class smart_stream_type(ctypes.Structure):  # TODO: consider changing the source code of "smart_stream_type" at constant file to this one
    _fields_ = [
                ('entries', ctypes.c_uint16),
                ('params', ctypes.c_void_p),
               ]