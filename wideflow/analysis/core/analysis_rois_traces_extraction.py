import numpy as np


def extract_roi_traces(wf_data, rois_dict, shape):
    rois_pixels_inds = {}
    for key, val in rois_dict.items():
        rois_pixels_inds[key] = np.unravel_index(val['PixelIdxList'], shape)

    rois_traces = {}
    for ch, ch_wf_data in wf_data.items():
        rois_traces[ch] = {}
        for key, val in rois_pixels_inds.items():
            rois_traces[ch][key] = np.mean(ch_wf_data[:, val[0], val[1]], axis=1)

    return rois_traces