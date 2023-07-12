import numpy as np
from scipy.signal import find_peaks, peak_prominences


def find_trace_peaks(x, height, distance, prominence=False):
    '''

    Args:
        x: 1D numpy array
        height: scipy.signal find_peaks height argument - minimal required height for peaks
        distance: scipy.signal find_peaks distance argument - minimal horizontal distance between neighbouring peaks
        prominence_th: bool - filter peaks using typical peaks prominence

    Returns:
        peaks:
        peaks_inds:
        peaks_props:

    '''
    if type(prominence) == int or type(prominence) == float:
        peaks_inds, peaks_props = find_peaks(x, height=height, distance=distance, width=0.0, prominence=prominence)
    else:
        peaks_inds, peaks_props = find_peaks(x, height=height, distance=distance, width=0.0)

    if type(prominence) == bool and prominence:
        prominence = peak_prominences(x, peaks_inds)
        prominence_mean = np.mean(prominence[0])
        prominence_sd = np.std(prominence[0])
        prominence_th = prominence[0] > prominence_mean + prominence_sd
        peaks_inds = peaks_inds[prominence_th]
        for key, val in peaks_props.items():
            peaks_props[key] = val[prominence_th]

    peaks = np.zeros(x.shape, dtype=np.bool_)
    peaks[peaks_inds] = 1
    return peaks, peaks_inds, peaks_props
