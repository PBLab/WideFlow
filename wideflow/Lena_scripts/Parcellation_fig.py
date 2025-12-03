# After running parcellation full process run this in the console to see an image of the parcellation

from utils.load_rois_data import load_rois_data
import matplotlib.pyplot as plt
from utils.paint_roi import paint_roi
import numpy as np

#base_path = '/data/Lena/WideFlow_prj'
base_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj'
mouse_id = '277FRL'
#mouse_base_path = f'/data/Lena/WideFlow_prj/{mouse_id}/'  # path to the directory where to save parcellation rois data and map
mouse_base_path = f'/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/{mouse_id}/'  # path to the directory where to save parcellation rois data and map
rois_data_path = mouse_base_path + 'functional_parcellation_rois_dict_NEW.h5'
#rois_data_path = mouse_base_path + 'functional_parcellation_cortex_map.h5'


roi_list = load_rois_data(rois_data_path)
fig, ax = plt.subplots()
ax, im, paint_map = paint_roi (roi_list, np.zeros((297,168)),
                               ['roi_01', 'roi_131','roi_118', 'roi_60', 'roi_112','roi_127',
                                'roi_69','roi_101','roi_94','roi_38']
                               #  list(roi_list.keys())
                                ,ax=ax, annotate=True)
# to show all rois: list(roi_list.keys())
# to show specific rois: ['roi_', 'roi_']



fig.suptitle(f'{mouse_id}')

plt.show()
#plt.savefig(f'{base_path}/{mouse_id}/Parcellation with closest NEW.png', format = 'png')
#
# plt.rcParams['svg.fonttype'] = 'none'  # or 'path' or 'none'
# plt.savefig( f'/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/{mouse_id}/parcellation_{mouse_id}.svg',format='svg',dpi=500)

