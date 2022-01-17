from skimage import io
import numpy as np


def load_tiff(vid_path):
    vid = io.imread(vid_path)
    vid = np.array(vid)
    return vid

