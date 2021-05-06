import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


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


