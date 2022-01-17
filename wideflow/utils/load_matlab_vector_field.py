import h5py
import numpy as np
from wideflow.utils.gen_utils import add_properties_to_roi_list
# from wideflow.utils.gen_utils import add_morphological_adjacent_rois_to_roi_list


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


def load_allen_2d_cortex_rois(file_path):
    """
    load the allen cortex map roi data set
    """

    with h5py.File(file_path) as f:
        roi_list = {}
        for key, grp in f.items():
            roi_list[key] = {}
            roi_list[key]['Index'] = int(key.split('_')[1])
            roi_list[key]['Area'] = grp['Area'][()]
            roi_list[key]['Centroid'] = grp['Centroid'][()]
            roi_list[key]['PixelIdxList'] = grp['PixelIdxList'][()] - 1  # -1 to convert from matlab to python
            roi_list[key]['PixelIdxList'] = roi_list[key]['PixelIdxList'][0].astype(np.int32)

    return roi_list


def extend_rois_list(file_path, save_path, order='F'):
    shape = (297, 337)
    roi_list = load_allen_2d_cortex_rois(file_path)
    roi_list = add_properties_to_roi_list(roi_list, shape, order)
    # roi_list = add_morphological_adjacent_rois_to_roi_list(roi_list, shape)

    with h5py.File(save_path, 'w') as f:
        for key, rdict in roi_list.items():
            f.create_group(key)
            for rkey, rval in rdict.items():
                f.create_dataset(f'{key}/{rkey}', data=rval)


def load_extended_rois_list(file_path):
    with h5py.File(file_path, 'r') as f:
        roi_list = {}
        for key, grp in f.items():
            roi_list[key] = {}
            roi_list[key]['Index'] = int(key.split('_')[1])
            roi_list[key]['Area'] = grp['Area'][()]
            roi_list[key]['Centroid'] = grp['Centroid'][()]
            roi_list[key]['PixelIdxList'] = grp['PixelIdxList'][()] - 1  # -1 to convert from matlab to python
            roi_list[key]['outline'] = grp['outline'][()]
            roi_list[key]['top_left_bottom_rigth'] = grp['top_left_bottom_rigth'][()]
            roi_list[key]['name'] = grp['name'][()]


    # keep keys order as index order
    roi_list = {k: v for k, v in sorted(roi_list.items(), key=lambda item: item[1]['Index'])}
    return roi_list



# import pathlib
# file_path = '/home/pb/PycharmProjects/WideFlow/data/cortex_map/allen_2d_cortex_rois.h5'
# save_path = '/home/pb/PycharmProjects/WideFlow/data/cortex_map/allen_2d_cortex_rois_extended.h5'
# extend_rois_list(file_path, save_path)
# roi_list = load_extended_rois_list(save_path)
# # z=3