from mat4py import loadmat
import h5py
import numpy as np


def convert_matlab_functional_parcellation_struct_to_dict(rois_struct_path, rois_dict_path):
    data = loadmat(rois_struct_path)
    rois_struct = data['ROI_list']
    rois_dict = {}
    for i in range(len(rois_struct['name'])):
        rois_dict[f'roi_{i+1:02d}'] = {}
        rois_dict[f'roi_{i+1:02d}']['Index'] = i
        rois_dict[f'roi_{i+1:02d}']['Area'] = len(rois_struct['pixel_list'][i])
        rois_dict[f'roi_{i+1:02d}']['Centroid'] = np.array(rois_struct['centerPos'][i]) - 1  # shift=1 to convert from matlab to python
        rois_dict[f'roi_{i+1:02d}']['PixelIdxList'] = np.array(rois_struct['pixel_list'][i]) - 1
        rois_dict[f'roi_{i+1:02d}']['outline'] = np.array(rois_struct['boundary_list'][i]) - 1
        rois_dict[f'roi_{i + 1:02d}']['top_left_bottom_rigth'] = []
        rois_dict[f'roi_{i+1:02d}']['name'] = rois_struct['name'][i]

    with h5py.File(rois_dict_path, 'w') as f:
        for roi_key, roi_dict in rois_dict.items():
            grp = f.create_group(roi_key)
            for key, val in roi_dict.items():
                grp.create_dataset(key, data=val)


