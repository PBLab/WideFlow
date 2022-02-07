from itertools import groupby
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
    n = sum(1 for k, _ in groupby(np.array(results_seq[-num_frames:]) > threshold) if k)
    if n < typical_count - count_band:
        return threshold - step
    elif n > typical_count + count_band:
        return threshold + step
    else:
        return threshold


def percentile_fixed_step_staircase_procedure(threshold, samples, num_frames, percent, perc_band, step):
    p = np.mean(samples[-num_frames:] > threshold)
    if p < percent - perc_band:
        return threshold - step
    elif p > percent + perc_band:
        return threshold + step
    else:
        return threshold


def percentile_update_procedure(threshold, samples, percentile, nbins):
    hist, bins = np.histogram(samples, nbins, density=True)
    bins_width = np.diff(bins)
    p_density = hist * bins_width
    percentile_inds = np.where(bins > np.percentile(p_density, percentile))[0]
    if not len(percentile_inds):
        return threshold
    else:
        return bins[percentile_inds[0]] * (nbins/100)


def bayesian_update():
    pass
