import numpy as np
from skimage.transform import resize
from skimage.morphology import skeletonize
from scipy.ndimage.filters import maximum_filter1d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py
import seaborn as sns
import matplotlib.gridspec as gridspec
import scipy.cluster.hierarchy as sch

from utils.load_tiff import load_tiff
from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict
import pandas as pd

from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
from analysis.utils.peristimulus_time_response import calc_pstr
from utils.load_config import load_config
from utils.load_rois_data import load_rois_data
from analysis.plots import plot_traces, wf_imshow
from analysis.utils.rois_proximity import calc_rois_proximity
from utils.paint_roi import paint_roi

def calc_rois_corr(rois_dict, data, data_chosen_roi):
    rois_corr = {}
    for i, roi_key in enumerate (rois_dict.keys()):
        corr=(np.corrcoef(data[roi_key],data_chosen_roi)[0,1])
        rois_corr[roi_key] = corr
    return rois_corr




base_path = '/data/Lena/WideFlow_prj'
dates_vec = ['20230608','20230614']
mouse_id = '64ML'
#sess_name_vec = 'spont_mockNF_NOTexcluded_closest'
sess_name_vec = ['CRC4','NF4']



session_id0 = f'{dates_vec[0]}_{mouse_id}_{sess_name_vec[0]}'
session_id1 = f'{dates_vec[1]}_{mouse_id}_{sess_name_vec[1]}'
#metric_index = 105 #21ML - 134, 31MN - 105, 54MRL - 85, 63MR - 52, 64ML - 71 (those are the indexes, the actual ROI numbers are this +1)

title = f'{mouse_id}_{sess_name_vec}'

results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
results_path_CRC = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'


# timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')
# serial_readout = 1 - np.array(serial_readout)
# serial_readout = maximum_filter1d(serial_readout, 2)[::2]
# cue = maximum_filter1d(cue, 2)[::2]

data0 = {}
if sess_name_vec[0] == 'CRC4' or sess_name_vec[0] == 'CRC3':
     with h5py.File(results_path_CRC, 'r') as f:
         decompose_h5_groups_to_dict(f, data0, f'/{mouse_id}/{session_id0}/')
else:
    with h5py.File(results_path, 'r') as f:
        decompose_h5_groups_to_dict(f, data0, f'/{mouse_id}/{session_id0}/')

#traces_dff_delta5 = data['post_session_analysis']['dff_delta5']['traces']
traces0 = data0['rois_traces']['channel_0']
#traces_dff = data['post_session_analysis']['dff']['traces']
#correlation_matrix_dff_delta5 = np.corrcoef(traces_dff_delta5)
#correlation_matrix_dff = np.corrcoef(traces_dff)
#correlation_matrix_dff0 = np.corrcoef(traces0)
# distance_matrix = np.sqrt((1 - correlation_matrix_dff_delta5) / 2.0)
# linkage_matrix = sch.linkage(distance_matrix, method='ward')

functional_cortex_map_path = f'{base_path}/{mouse_id}/functional_parcellation_cortex_map.h5'
functional_rois_dict_path = f'{base_path}/{mouse_id}/functional_parcellation_rois_dict.h5'
with h5py.File(functional_cortex_map_path, 'r') as f:
    functional_cortex_mask = f["mask"][()]
    functional_cortex_map = f["map"][()]
functional_cortex_mask = functional_cortex_mask[:, :168]
functional_cortex_map = functional_cortex_map[:, :168]
functional_cortex_map = skeletonize(functional_cortex_map)
functional_rois_dict = load_rois_data(functional_rois_dict_path)


correlation_dict_dff0 = {}
for i, (key, val) in enumerate(functional_rois_dict.items()):
    # metric_corr[key] = np.corrcoef(pstr_cat[key], pstr_cat[metric_roi])[0, 1]  # correlation with metric ROI
    # dff_corr[key] = np.corrcoef (sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i], sessions_data[sess_id]['post_session_analysis']['dff']['traces'][105])[0,1]
    # metric_corr[key] = np.corrcoef(metric_trace, sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i])[0, 1]

    #metric_corr[key] = np.corrcoef(traces[key], traces[f'roi_{metric_index + 1}'])[0, 1]
    # metric_corr[key] = np.corrcoef(traces[key], traces[f'roi_01'])[0, 1]

    correlation_dict_dff0[key] = calc_rois_corr(functional_rois_dict, traces0, traces0[key])


a=5

vmin = 0
vmax = 1
clustermap0 = sns.clustermap(correlation_dict_dff0,cmap='inferno', vmin=vmin,vmax=vmax)

# Extract the row and column linkage information
row_linkage0 = clustermap0.dendrogram_row.linkage
col_linkage0 = clustermap0.dendrogram_col.linkage

plt.title(f'{mouse_id} {sess_name_vec[0]}')

plt.rcParams['svg.fonttype'] = 'none'  # or 'path' or 'none'
plt.savefig(f'{base_path}/Figs_for_paper/{session_id0} correlation map.svg',format='svg',dpi=500)



### Afetr learning
data1 = {}
if sess_name_vec[1] == 'CRC4':
     with h5py.File(results_path_CRC, 'r') as f:
         decompose_h5_groups_to_dict(f, data1, f'/{mouse_id}/{session_id1}/')
else:
    with h5py.File(results_path, 'r') as f:
        decompose_h5_groups_to_dict(f, data1, f'/{mouse_id}/{session_id1}/')

#traces_dff_delta5 = data['post_session_analysis']['dff_delta5']['traces']
traces1 = data1['rois_traces']['channel_0']
#traces_dff = data['post_session_analysis']['dff']['traces']
#correlation_matrix_dff_delta5 = np.corrcoef(traces_dff_delta5)
#correlation_matrix_dff = np.corrcoef(traces_dff)
#correlation_matrix_dff0 = np.corrcoef(traces0)
# distance_matrix = np.sqrt((1 - correlation_matrix_dff_delta5) / 2.0)
# linkage_matrix = sch.linkage(distance_matrix, method='ward')


correlation_dict_dff1 = {}
for i, (key, val) in enumerate(functional_rois_dict.items()):
    # metric_corr[key] = np.corrcoef(pstr_cat[key], pstr_cat[metric_roi])[0, 1]  # correlation with metric ROI
    # dff_corr[key] = np.corrcoef (sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i], sessions_data[sess_id]['post_session_analysis']['dff']['traces'][105])[0,1]
    # metric_corr[key] = np.corrcoef(metric_trace, sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i])[0, 1]

    #metric_corr[key] = np.corrcoef(traces[key], traces[f'roi_{metric_index + 1}'])[0, 1]
    # metric_corr[key] = np.corrcoef(traces[key], traces[f'roi_01'])[0, 1]

    correlation_dict_dff1[key] = calc_rois_corr(functional_rois_dict, traces1, traces1[key])


# Create a new figure for both matrices
#fig, (ax1, ax2) = plt.subplots(1, 2) #figsize=(12, 6))

# data1 = pd.DataFrame(correlation_dict_dff0)
# data2 = pd.DataFrame(correlation_dict_dff1)


clustermap1 = sns.clustermap(correlation_dict_dff1, row_linkage=row_linkage0, col_linkage=col_linkage0,cmap='inferno',
                            vmin=vmin,vmax=vmax)

plt.title(f'{mouse_id} {sess_name_vec[1]}')

plt.rcParams['svg.fonttype'] = 'none'  # or 'path' or 'none'
plt.savefig(f'{base_path}/Figs_for_paper/{session_id1} correlation map with {session_id0} dendrogram.svg',format='svg',dpi=500)

#plt.show()


