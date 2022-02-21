# For farther reading:
# https://github.com/Photometrics/PyVCAM/blob/master/Documents/PyVCAM%20Wrapper.md

from pyvcam.camera import Camera
from pyvcam import constants as const
from pyvcam import pvc
import ctypes


class PVCamera(Camera):
    def __init__(self, name, exp_time=10, binning=(2, 2), channels=2, circ_buffer_count=16, sensor_roi=[0, 2048, 0, 2048]):
        super().__init__(name)
        self.__exp_time = exp_time
        self.__binning = binning
        self.channels = channels
        self.circ_buffer_count = circ_buffer_count
        self.sensor_roi = sensor_roi

    @classmethod
    def detect_camera(cls, **kwargs):
        """Detects and creates a new Camera object.

        Returns:
            A Camera object.
        """
        cam_count = 0
        total = pvc.get_cam_total()

        while cam_count < total:
            try:
                yield PVCamera(pvc.get_cam_name(cam_count), **kwargs)
                cam_count += 1
            except RuntimeError:
                raise RuntimeError('Failed to create a detected camera.')

    def start_up(self):
        print("setting camera startup config")
        self.clear_mode = "Pre-Sequence"
        self.set_param(const.PARAM_CLEAR_CYCLES, 2)
        self.exp_out_mode = "All Rows"
        self.set_param(const.PARAM_LAST_MUXED_SIGNAL, self.channels)
        self.set_roi(self.sensor_roi[0], self.sensor_roi[2], self.sensor_roi[1], self.sensor_roi[3])

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