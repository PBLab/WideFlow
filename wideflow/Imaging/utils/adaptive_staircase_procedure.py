from itertools import groupby

from Imaging.utils.numba_histogram import numba_histogram
import numpy as np


def binary_fixed_step_staircase_procedure(threshold, results_seq, num_frames, typical_count, count_band, step):
    """

    Args:
        threshold: float - current threshold
        results_seq: list - a list of all previous metric results
        num_frames: int - number of frames to evaluate the metric
        typical_count: int - expected typical number of times metric is above threshold in typical_n frames
        count_band: int - a band of typical count for which threshold doesn't change
        step: float - threshold update delta

    Returns: threshold: float - updated threshold

    """
    n = sum(1 for k, _ in groupby(np.array(results_seq) > threshold) if k)
    n = n * (num_frames / len(results_seq))  # a fix for when the function is provided with less frames then num_frames
    if n < typical_count - count_band:
        return threshold - step
    elif n > typical_count + count_band:
        return threshold + step
    else:
        return threshold


def percentile_update_procedure(threshold, samples, percentile, nbins):
    """

    Args:
        threshold: float32: default threshold
        samples: 1D numpy array of float32
        percentile: float32: samples threshold percentile
        nbins: int: number of bins to compose the histogram

    Returns: float: samples threshold

    """
    hist, bins = numba_histogram(samples, nbins, density=True)
    bins_width = np.diff(bins)
    p_density = hist * bins_width
    prob = np.cumsum(p_density)
    percentile_inds = np.where(prob > np.percentile(prob, percentile))[0]
    if len(percentile_inds):
        return bins[percentile_inds[0]]
    else:
        return threshold

