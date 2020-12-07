import numpy as np
import h5py


def load_matlab_OF(mat_path):
    """
    load optical fow vector field calculated with Matlab OFAMM optic flow toolbox
    """

    arrays = {}
    f = h5py.File(mat_path)
    for k, v in f.items():
        arrays[k] = np.array(v)

    [t, m, n] = arrays["uvHS"].shape
    uxy = arrays["uvHS"].view(np.double).reshape((t, m, n, 2))
    return uxy


def load_allen_2d_cortex_rois(file_path):
    """
    load the allen cortex map roi data set
    """

    with h5py.File(file_path) as f:
        roi_list = {}
        for key, grp in f.items():
            roi_list[key] = {}
            roi_list[key]['Index'] = int(key.split('_')[1])
            roi_list[key]['Area'] = grp['Area'].value
            roi_list[key]['Centroid'] = grp['Centroid'].value
            roi_list[key]['PixelIdxList'] = grp['PixelIdxList'].value - 1  # -1 to convert from matlab to python
            roi_list[key]['PixelIdxList'] = roi_list[key]['PixelIdxList'][0].astype(np.int32)

    return roi_list


# from utils.gen_utils import *
# import pathlib
# pth = str(pathlib.Path('C:/') / 'Users' / 'motar' / 'PycharmProjects' / 'wf_opticflow' / 'data' / 'cortex_map' / 'allen_2d_cortex_rois.h5')
# roi_list = load_allen_2d_cortex_rois(pth)
# roi_list = add_properties_to_roi_list(roi_list, (297, 337))
# z = 3