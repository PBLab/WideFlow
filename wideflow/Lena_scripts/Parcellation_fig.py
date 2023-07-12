# After running parcellation full process run this in the console to see an image of the parcellation

from utils.load_rois_data import load_rois_data
import matplotlib.pyplot as plt
from utils.paint_roi import paint_roi
import numpy as np

mouse_id = '64ML'
mouse_base_path = f'/data/Lena/WideFlow_prj/{mouse_id}/'  # path to the directory where to save parcellation rois data and map
rois_data_path = mouse_base_path + 'functional_parcellation_rois_dict.h5'
#rois_data_path = mouse_base_path + 'functional_parcellation_cortex_map.h5'
#20221122_MR_CRC3functional_parcellation_rois_dict.h5
#FLfunctional_parcellation_rois_dict_CRC3.h5

roi_list = load_rois_data(rois_data_path)
fig, ax = plt.subplots()
ax, im, paint_map = paint_roi (roi_list, np.zeros((297,168)),list(roi_list.keys()), ax=ax, annotate=True)
# to show all rois: list(roi_list.keys())
# to show specific rois: ['roi_', 'roi_']
#['roi_15', 'roi_86','roi_92','roi_88','roi_66','roi_67','roi_69' ]
#MNL metric_ROI - 57

fig.suptitle(f'{mouse_id}')
plt.imshow(paint_map)
plt.show()

#fig.suptitle(f'{mouse_id}')
