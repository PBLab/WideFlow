import h5py


def load_analysis_results(path):
    traces = {}
    with h5py.File(path, 'r') as f:
        for ch_key in f.keys():
            traces[ch_key] = {}
            for roi_key in f[ch_key].keys():
                traces[ch_key][roi_key] = f[ch_key][roi_key][()]

    return traces

