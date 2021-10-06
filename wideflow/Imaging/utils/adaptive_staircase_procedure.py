from itertools import groupby
import numpy as np


def fixed_step_staircase_procedure(threshold, results_seq, num_frames, typical_count, count_band, step):
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



