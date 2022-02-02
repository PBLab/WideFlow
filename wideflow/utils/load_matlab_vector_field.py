import h5py
import numpy as np


def load_matlab_OF(mat_path):
    """
    load optical fow vector field calculated with Matlab OFAMM optic flow toolbox
    """

    arrays = {}
    with h5py.File(mat_path) as f:
        for k, v in f.items():
            arrays[k] = np.array(v)

    [t, m, n] = arrays["uvHS"].shape
    uxy = np.swapaxes(arrays["uvHS"].view(np.double).reshape((t, m, n, 2)), 1, 2)
    return uxy


