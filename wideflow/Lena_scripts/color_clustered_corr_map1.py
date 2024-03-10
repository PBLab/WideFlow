import numpy as np
from skimage.transform import resize
from skimage.morphology import skeletonize
from scipy.ndimage.filters import maximum_filter1d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py
import seaborn as sns
import scipy.cluster.hierarchy as sch

from utils.load_tiff import load_tiff
from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict

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
        corr=(np.corrcoef(data[i],data_chosen_roi)[0,1])
        rois_corr[roi_key] = corr
    return rois_corr




base_path = '/data/Lena/WideFlow_prj'
date = '20230614'
mouse_id = '31MN'
sess_name = 'spont_mockNF_NOTexcluded_closest'
sess_name = 'NF4'

session_id = f'{date}_{mouse_id}_{sess_name}'
metric_index = 105 #21ML - 134, 31MN - 105, 54MRL - 85, 63MR - 52, 64ML - 71 (those are the indexes, the actual ROI numbers are this +1)

title = f'{mouse_id}_{sess_name}'

results_path = '/data/Lena/WideFlow_prj/Results/results_exp2.h5'

# timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')
# serial_readout = 1 - np.array(serial_readout)
# serial_readout = maximum_filter1d(serial_readout, 2)[::2]
# cue = maximum_filter1d(cue, 2)[::2]

data = {}
with h5py.File(results_path, 'r') as f:
    decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/')

traces_dff_delta5 = data['post_session_analysis']['dff_delta5']['traces']
traces_dff = data['post_session_analysis']['dff']['traces']
correlation_matrix_dff_delta5 = np.corrcoef(traces_dff_delta5)
correlation_matrix_dff = np.corrcoef(traces_dff)
distance_matrix = np.sqrt((1 - correlation_matrix_dff_delta5) / 2.0)
linkage_matrix = sch.linkage(distance_matrix, method='ward')

functional_cortex_map_path = f'{base_path}/{mouse_id}/functional_parcellation_cortex_map.h5'
functional_rois_dict_path = f'{base_path}/{mouse_id}/functional_parcellation_rois_dict.h5'
with h5py.File(functional_cortex_map_path, 'r') as f:
    functional_cortex_mask = f["mask"][()]
    functional_cortex_map = f["map"][()]
functional_cortex_mask = functional_cortex_mask[:, :168]
functional_cortex_map = functional_cortex_map[:, :168]
functional_cortex_map = skeletonize(functional_cortex_map)
functional_rois_dict = load_rois_data(functional_rois_dict_path)


# #next 5 lines are for the mock perfect data
# traces_to_copy = traces_dff_delta5[metric_index]
# traces = np.tile(traces_to_copy,(traces_dff_delta5.shape[0],1))
# metric_prox = calc_rois_proximity(functional_rois_dict,f'roi_{metric_index+1}')
# for i in range(len(metric_prox)):
#     traces[i] = (traces[i]+(list(metric_prox.values())[i])*1000)
# correlation_matrix_dff_delta5 = np.corrcoef(traces)

# config = load_config(f'{base_path}/{date}/{mouse_id}/{session_id}/session_config.json')
# closest = config["supplementary_data_config"]["closest_rois"]
# for key in closest:
#     del functional_rois_dict[key]

a=5

# metric_corr = {}
# rois_proximity = {}
# corr_all_rois = {}
# for i, (key, val) in enumerate(functional_rois_dict.items()):
#     # metric_corr[key] = np.corrcoef(pstr_cat[key], pstr_cat[metric_roi])[0, 1]  # correlation with metric ROI
#     # dff_corr[key] = np.corrcoef (sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i], sessions_data[sess_id]['post_session_analysis']['dff']['traces'][105])[0,1]
#     # metric_corr[key] = np.corrcoef(metric_trace, sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i])[0, 1]
#     metric_corr[key] = np.corrcoef(data['post_session_analysis']['dff_delta5']['traces'][i],
#                                    data['post_session_analysis']['dff_delta5']['traces'][metric_index])[0, 1]
#
#     corr_all_rois[key] = calc_rois_corr(functional_rois_dict,data['post_session_analysis']['dff_delta5']['traces'],data['post_session_analysis']['dff_delta5']['traces'][i])
#     rois_proximity[key] = calc_rois_proximity(functional_rois_dict, key)

vmin = 0
vmax = 1
sns.clustermap(correlation_matrix_dff_delta5,cmap='inferno', vmin=vmin,vmax=vmax)



plt.title(title)
plt.show()
#plt.savefig(f'{base_path}/Figs_for_paper/{title}_color_clustered_corr_mat.svg',format='svg',dpi=500)

#plt.savefig(f'{base_path}/Figures_exp2_all_mice_compare/{session_id}_corr_dendrogram.pdf', format="pdf",dpi=500)










#rois_proximity = calc_rois_proximity(functional_rois_dict, f'roi_{metric_index+1}')

# f = plt.figure(constrained_layout=True, figsize=(11, 6))
# gs = f.add_gridspec(2,2)
# ax_left0 = f.add_subplot(gs[0, 0])
# _, _, pmap0 = paint_roi(functional_rois_dict,
#                       functional_cortex_map,
#                       list(functional_rois_dict.keys()),
#                       metric_corr) #LK
# pmap0[functional_cortex_mask==0] = None
# im, _ = wf_imshow(ax_left0, pmap0, mask=None, map=None, conv_ker=None, show_cb=False, cm_name='inferno', vmin=None, vmax=None, cb_side='right')
# #ax_left0.set_ylabel("PSTR", fontsize=12)
# #ax_left0.scatter(metric_outline[1], metric_outline[0], marker='.', s=0.5, c='k')
# ax_left0.axis('off')
#
# ax_med0 = f.add_subplot(gs[0, 1])
#
# #ax_med0.imshow(sns.clustermap(correlation_matrix_dff_delta5, cmap = 'inferno'))
# clustermap = sns.clustermap(correlation_matrix_dff_delta5,cmap='inferno')
# clustermap_fig = clustermap.fig
# clustermap_ax = clustermap.ax_heatmap
# clustermap_ax.set_xticks([])
# clustermap_ax.set_yticks([])
# clustermap_ax.set_xticklabels([])
# clustermap_ax.set_yticklabels([])
#
# ax_med0.imshow(np.zeros_like(correlation_matrix_dff_delta5), cmap="coolwarm")  # Create a dummy image with the same size
# ax_position = ax_med0.get_position()
# clustermap_ax_position = clustermap_ax.get_position()
# clustermap_ax.set_position([ax_position.x0, ax_position.y0, ax_position.width, ax_position.height])
# #clustermap_ax.sca(clustermap_ax)

#ax_med0.imshow(sns.heatmap(correlation_matrix_dff_delta5, cmap = 'inferno'))
#ax_med0.imshow(sch.dendrogram(linkage_matrix))
#sch.dendrogram(linkage_matrix)
#plt.imshow(correlation_matrix_dff_delta5, cmap='inferno', interpolation='nearest')
#plt.colorbar(label='Correlation')

# ax_right0 = f.add_subplot(gs[0, 2])
# plt.scatter(list(rois_proximity.values()), list(metric_corr.values()))
# plt.ylim(0.5,1)
# plt.ylabel("Correlation")
# plt.xlabel("Proximity")

# ax_right0 = f.add_subplot(gs[1,:])
# for key in functional_rois_dict:
#     plt.scatter(list(rois_proximity[key].values()), list(corr_all_rois[key].values()), label=key)
#     #plt.plot(set_threshold,i, color =color_palette(sessions_vec.index(session_id)), label = f'{session_id}')
# #plt.scatter(list(rois_proximity.values()), list(metric_corr.values()))
# #plt.ylim(0.5,1)
# plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
# plt.ylabel("Correlation")
# plt.xlabel("Proximity")
#
# ax4 = f.add_subplot(gs[1, 1])
# ax4.axis('off')


# plt.figure(figsize=(8, 6))
# plt.imshow(correlation_matrix_dff_delta5, cmap='plasma', interpolation='none')
# plt.colorbar(label='Correlation')
# plt.title('Correlation Between Rows')



a=5
