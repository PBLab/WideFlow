# After running parcellation full process run this in the console to see an image of the parcellation

from utils.load_rois_data import load_rois_data
import matplotlib.pyplot as plt
from utils.paint_roi import paint_roi
import numpy as np

mouse_id = '54MRL'
mouse_base_path = f'/data/Lena/WideFlow_prj/{mouse_id}/'  # path to the directory where to save parcellation rois data and map
rois_data_path = mouse_base_path + 'functional_parcellation_rois_dict.h5'
#rois_data_path = mouse_base_path + 'functional_parcellation_cortex_map.h5'


roi_list = load_rois_data(rois_data_path)
fig, ax = plt.subplots()
ax, im, paint_map = paint_roi (roi_list, np.zeros((297,168)),list(roi_list.keys()), ax=ax, annotate=False)
# to show all rois: list(roi_list.keys())
# to show specific rois: ['roi_', 'roi_']



fig.suptitle(f'{mouse_id}')

plt.show()
#
#plt.rcParams['svg.fonttype'] = 'none'  # or 'path' or 'none'
#plt.savefig( f'/data/Lena/WideFlow_prj/Figs_for_paper/parcellation_{mouse_id}_outlineTRY2.svg',format='svg',dpi=500)

