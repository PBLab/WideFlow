from utils.load_rois_data import load_rois_data
from wideflow.analysis.utils.rois_proximity import calc_rois_proximity

import matplotlib.pyplot as plt
from utils.paint_roi import paint_roi
import numpy as np


mouse_base_path = '/data/Lena/WideFlow_prj/MNL/'  # path to the directory where to save parcellation rois data and map
rois_data_path = mouse_base_path + '20221122_MNL_CRC3functional_parcellation_rois_dict.h5'
#20221122_MR_CRC3functional_parcellation_rois_dict.h5
#FLfunctional_parcellation_rois_dict_CRC3.h5

roi_list = load_rois_data(rois_data_path)
prox = calc_rois_proximity (roi_list, 'roi_41')
sorted_prox = sorted(prox.items(), key=lambda x: x[1])
#closest_rois = [key for key, _ in sorted_prox[1:6]]
#closest_rois = []
closest_rois = ['roi_65', 'roi_71', 'roi_43', 'roi_36','roi_50', 'roi_45']

for key in closest_rois:
    del roi_list[key]


a=5