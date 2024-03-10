import matplotlib.pyplot as plt
import numpy as np
import h5py
from utils.paint_roi import paint_roi

from utils.load_rois_data import load_rois_data
from untracked_files.matlab_functional_parcellation_to_roi_dict import convert_matlab_functional_parcellation_struct_to_dict

from wideflow.utils.paint_roi import paint_roi

mouse_id = '64ML'
map_size = [297, 337]
rois_matlab_path = f'/data/Lena/WideFlow_prj/{mouse_id}/ROI_list_left.mat'  # path to the functional parcellation results, containing only ROI_list variable
mouse_base_path = f'/data/Lena/WideFlow_prj/{mouse_id}/'  # path to the directory where to save parcellation rois data and map
rois_data_path = mouse_base_path + 'functional_parcellation_rois_dict.h5'
parcellation_map_path = mouse_base_path + 'functional_parcellation_cortex_map.h5'

convert_matlab_functional_parcellation_struct_to_dict(rois_matlab_path, rois_data_path)
rois_data = load_rois_data(rois_data_path)

cortex_map = np.zeros(map_size)
for name, roi in rois_data.items():
    roi_outline_pixels = roi["outline"]
    pixels_inds = np.unravel_index(roi_outline_pixels, (cortex_map.shape[1], cortex_map.shape[0]))
    cortex_map[pixels_inds[1], pixels_inds[0]] = 1

with h5py.File('/home/perelrot/WideFlow/data/cortex_map/allen_2d_cortex.h5', 'r') as f:
    cortex_mask = np.transpose(f["mask"][()])

with h5py.File(parcellation_map_path, 'w') as f:
    f.create_dataset("mask", data=cortex_mask)
    f.create_dataset("map", data=cortex_map)



