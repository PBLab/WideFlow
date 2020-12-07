import pathlib
from skimage import io
import numpy as np


# vid_path = str(pathlib.Path('C:/') / 'Users' / 'motar' / 'PycharmProjects' / 'wf_opticflow' / 'data' / 'ImgSeq.tif')
def load_tiff(vid_path):
    vid = io.imread(vid_path)
    vid = np.array(vid)
    return vid
