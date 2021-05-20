import time
from utils.imaging_utils import load_config
import numpy as np


class AcquisitionMetaData:
    def __init__(self, session_config_path=None, config=None):
        self.datetime = time.localtime()
        self.session_config_path = session_config_path
        self.config = config or load_config(session_config_path)
        self.write_metafile_header()

    def write_frame_metadata(self, timestemp, cue, result):
        self.metatext += "timestemp:{:0.5f}    cue:{}   metric result:{:0.5f}\n".format(timestemp, cue, result.item())

    def write_metafile_header(self):
        self.metatext = str(self.datetime.tm_year) + '/' + str(self.datetime.tm_mon) + '/' + str(self.datetime.tm_mday) + ' - ' + str(self.datetime.tm_hour) + ':' + str(self.datetime.tm_min) + ':' + str(self.datetime.tm_sec) + '\n'
        for key, value in self.config.items():
            self.metatext += key + '\n'
            if isinstance(value, dict):
                self.metatext += self.dict_to_text(value)
            elif isinstance(value, list):
                for d in value:
                    self.metatext += self.dict_to_text(d)
            else:
                self.metatext += value

        self.metatext += "\nframes metadata:\n"

    def dict_to_text(self, dictionary):
        return "{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in dictionary.items()) + "}"

    def save_file(self):
        path = self.config["acquisition_config"]["meta_save_path"]
        with open(path, 'w') as f:
            f.write(self.metatext)