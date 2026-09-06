import numpy as np
from skimage.transform import resize
from skimage.morphology import skeletonize
from scipy.ndimage.filters import maximum_filter1d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import cut_tree
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

from wideflow.config import DATA_STAGING_PATH  #added by Claude 20260906
def calc_rois_corr(rois_dict, data, data_chosen_roi):
    rois_corr = {}
    for i, roi_key in enumerate (rois_dict.keys()):
        corr=(np.corrcoef(data[roi_key],data_chosen_roi)[0,1])
        rois_corr[roi_key] = corr
    return rois_corr




# base_path = '/data/Lena/WideFlow_prj'
base_path = DATA_STAGING_PATH  #added by Claude 20260906
#date = '20230614'
mouse_id = '64ML'
#sess_name = 'spont_mockNF_NOTexcluded_closest'
sess_name = 'NF4'
metric_index = 71 #21ML - 134, 31MN - 105, 54MRL - 85, 63MR - 52, 64ML - 71 (those are the indexes, the actual ROI numbers are this +1)
cut_height = 0.7  # Specify the height at which to cut the dendrogram

if sess_name == 'NF4':
    date = '20230614'
elif sess_name == 'CRC4':
    date = '20230608'

session_id = f'{date}_{mouse_id}_{sess_name}'
if mouse_id == '63MR' and sess_name == 'CRC4':
    session_id = '20230607_63MR_CRC3'


# results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
results_path = DATA_STAGING_PATH + '/Results/results_exp2_noMH.h5'  #added by Claude 20260906
if sess_name == 'CRC4' or sess_name == 'CRC3':
    # results_path = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'
    results_path = DATA_STAGING_PATH + '/Results/Results_exp2_CRC_sessions.h5'  #added by Claude 20260906

# timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')
# serial_readout = 1 - np.array(serial_readout)
# serial_readout = maximum_filter1d(serial_readout, 2)[::2]
# cue = maximum_filter1d(cue, 2)[::2]


data = {}
with h5py.File(results_path, 'r') as f:
    decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/')

traces = data['rois_traces']['channel_0']
# traces_dff_delta5 = data['post_session_analysis']['dff_delta5']['traces']
# traces_dff = data['post_session_analysis']['dff']['traces']
# correlation_matrix_dff_delta5 = np.corrcoef(traces_dff_delta5)
# correlation_matrix_dff = np.corrcoef(traces_dff)
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
metric_outline = np.unravel_index(functional_rois_dict[f'roi_{metric_index+1}']['outline'], (functional_cortex_map.shape[1], functional_cortex_map.shape[0]))


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

correlation_dict_dff = {}
for i, (key, val) in enumerate(functional_rois_dict.items()):
    # metric_corr[key] = np.corrcoef(pstr_cat[key], pstr_cat[metric_roi])[0, 1]  # correlation with metric ROI
    # dff_corr[key] = np.corrcoef (sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i], sessions_data[sess_id]['post_session_analysis']['dff']['traces'][105])[0,1]
    # metric_corr[key] = np.corrcoef(metric_trace, sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i])[0, 1]

    #metric_corr[key] = np.corrcoef(traces[key], traces[f'roi_{metric_index + 1}'])[0, 1]
    # metric_corr[key] = np.corrcoef(traces[key], traces[f'roi_01'])[0, 1]

    correlation_dict_dff[key] = calc_rois_corr(functional_rois_dict, traces, traces[key])

vmin = 0
vmax = 1
clustermap = sns.clustermap(correlation_dict_dff,cmap='inferno', vmin=vmin,vmax=vmax)
plt.close()
############ This cuts the dendrogram at threshold
# # Get the dendrogram axes
# ax_row_dendrogram = cluster_map.ax_row_dendrogram
# ax_col_dendrogram = cluster_map.ax_col_dendrogram
#
# # Set the threshold on the dendrogram plot
# threshold = 0.65  # Set your desired threshold here
# #ax_row_dendrogram.axhline(y=threshold, color='r', linestyle='--')
# ax_col_dendrogram.axhline(y=threshold, color='r', linestyle='--')
############3



# Get the linkage matrix from the clustermap result
linkage_matrix = clustermap.dendrogram_row.linkage



# Cut the dendrogram to obtain clusters
clusters = cut_tree(linkage_matrix, height=cut_height)
# Flatten the cluster assignments
flat_clusters = clusters.flatten()

# Map the clusters to the keys in the original dictionary
clustered_dict = {}
for i, key in enumerate(correlation_dict_dff.keys()):
    clustered_dict[key] = flat_clusters[i]
    # cluster = clusters[i]
    # if cluster not in clustered_dict:
    #     clustered_dict[cluster] = [key]
    # else:
    #     clustered_dict[cluster].append(key)


# Plot the dendrogram with the cut height indicated

f = plt.figure(constrained_layout=True, figsize=(11, 6))
gs = f.add_gridspec(1,2)
ax_left0 = f.add_subplot(gs[0, 0])
#plt.figure(figsize=(10, 5))
hierarchy.dendrogram(linkage_matrix)
plt.axhline(y=cut_height, color='r', linestyle='--')
plt.title(f'{session_id}')
plt.xlabel('Sample Index')
plt.ylabel('Distance')



ax_right0 = f.add_subplot(gs[0, 1])
_, _, pmap0 = paint_roi(functional_rois_dict,
                      functional_cortex_map,
                      list(functional_rois_dict.keys()),
                      clustered_dict) #LK
pmap0[functional_cortex_mask==0] = None
im, _ = wf_imshow(ax_right0, pmap0, mask=None, map=None, conv_ker=None, show_cb=False, cm_name='nipy_spectral', vmin=-1, vmax=1, cb_side='left')
cbar = plt.colorbar(im, ax=ax_right0)#, ticks=[np.arange(0,max(flat_clusters))])
im.set_clim(0,max(flat_clusters))
#cbar.set_ticks(ticks=[np.arange(0,18)])
#ax_left0.set_ylabel("PSTR", fontsize=12)
ax_right0.scatter(metric_outline[0], metric_outline[1], marker='.', s=8, color='white')
plt.title(f'num of clusters = {max(flat_clusters)+1}')
ax_right0.axis('off')

plt.show()
#
# plt.rcParams['svg.fonttype'] = 'none'  # or 'path' or 'none'
# plt.savefig(f'{base_path}/Figs_for_paper/{session_id} corr dendrogram cut at {cut_height} with clustering map.svg',format='svg',dpi=500)

# Output the grouping according to the cut dendrogram
#print("Grouping according to the cut dendrogram:")
#print(clustered_dict)


#plt.title(title)



#plt.show()



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
