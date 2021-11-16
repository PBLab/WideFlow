import numpy as np


def dynamic_threshold(traces, n_std):
    """

    Args:
        traces: 2d numpy array, each row is a sample time series
        n_std: float, number of standard deviation from to threshold

    Returns: binary 2d numpy array, each row is a sample time series

    """

    mn = np.mean(traces, axis=0)
    band = np.std(traces, axis=0) * n_std
    th = np.ones(traces.shape) * (mn + band)
    return traces > th
