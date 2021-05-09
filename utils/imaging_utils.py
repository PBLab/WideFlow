import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import time


def load_config(path, handler=None):
    import os
    import json

    if not os.path.exists(path):
        raise RuntimeError('could not find path: ' + path)

    if os.path.isdir(path):
        path = os.path.join(path, 'config.json')

    with open(path, 'r') as f:
        data = f.read()
        config = json.loads(data)

    return config


def onselect(eclick, erelease):
    "eclick and erelease are matplotlib events at press and release."
    print('startposition: (%f, %f)' % (eclick.xdata, eclick.ydata))
    print('endposition  : (%f, %f)' % (erelease.xdata, erelease.ydata))
    print('used button  : ', eclick.button)


def toggle_selector(event):
    print('Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print('RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print('RectangleSelector activated.')
        toggle_selector.RS.set_active(True)


class AcquisitionMetaData:
    def __init__(self, session_config_path):
        self.datetime = time.localtime()
        self.session_config_path = session_config_path
        self.config = load_config(session_config_path)
        self.metatext = self.write_metafile_header

    def write_frame_metadata(self, timestemp, cue):
        self.metatext += f"timestemp:{timestemp}    cue:{cue}"

    def write_metafile_header(self):
        self.metatext = \
            str(self.datetime.tm_year) + '/' + str(self.datetime.tm_month) + '/' + str(self.datetime.tm_mday) + ' - ' +\
            str(self.datetime.tm_hour) + ':' + str(self.datetime.tm_min) + ':' + str(self.datetime.tm_sec) + '\n'

        for key, value in self.config.items():
            self.metatext += key + '\n'
            self.metatext += self.dict_to_text(value)

        self.metatext += "frames metadata:\n"

    def dict_to_text(self, dictionary):
        return "{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in dictionary.items()) + "}"

    def save_file(self):
        path = self.config["acquisition_config"]["meta_save_path"]
        with open(path, 'w') as f:
            f.write(self.metatext)
