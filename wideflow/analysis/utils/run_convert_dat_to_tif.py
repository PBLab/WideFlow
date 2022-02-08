from utils.convert_dat_to_tif import convert_dat_to_tif
from utils.load_bbox import load_bbox
from utils.imaging_utils import load_config
import numpy as np


def run_converter(session_path):
    bbox = load_bbox(session_path + '/bbox.txt')
    config = load_config(session_path + '/session_config.json')

    binning = config["camera_config"]["core_attr"]["binning"]
    nframes = config['acquisition_config']['num_of_frames']
    dtype = np.uint16
    shape = (int((bbox[3] - bbox[2]) / binning[1]), int((bbox[1] - bbox[0]) / binning[0]))
    frame = np.zeros(shape, dtype=dtype)
    nbytes = frame.nbytes

    tiff_shape = (2000, shape[0], shape[1])
    convert_dat_to_tif(session_path + '/wf_raw_data.dat', nbytes, tiff_shape, dtype, nframes)
