import numpy as np
from skimage.transform import resize
from skimage.morphology import skeletonize
from scipy.ndimage.filters import maximum_filter1d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import h5py
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.optimize import curve_fit

from wideflow.utils.load_tiff import load_tiff
from wideflow.utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict

from wideflow.analysis.utils.extract_from_metadata_file import extract_from_metadata_file
from wideflow.analysis.utils.peristimulus_time_response import calc_pstr
from wideflow.utils.load_config import load_config
from wideflow.utils.load_rois_data import load_rois_data
from wideflow.analysis.plots import plot_traces, wf_imshow
from wideflow.analysis.utils.rois_proximity import calc_rois_proximity
from wideflow.utils.paint_roi import paint_roi

def exponential_func(x, a, b):
    return a * np.exp(b * x)

def linear_function(x, m, b):
    return m * x + b

def exponential_decay(x, a,b,c,d):
    return b / np.exp(a * x + c) + d

def one_over_x(x,a):
    return (1/(x)) +a

def calc_rois_corr(rois_dict, data, data_chosen_roi):
    rois_corr = {}
    for i, roi_key in enumerate (rois_dict.keys()):
        corr=(np.corrcoef(data[roi_key],data_chosen_roi)[0,1])
        rois_corr[roi_key] = corr
    return rois_corr




base_path = '/data/Lena/WideFlow_prj'
date = '20230604'
mouse_id = '31MN'
sess_name= 'NF5'
sess_name = 'spont_mockNF_NOTexcluded_closest'
session_id = f'{date}_{mouse_id}_{sess_name}'
title = f'{mouse_id}_{sess_name}_noMH_allROIS_corr_graph_smalldots_mm_colormap_text'
metric_index = 105 #21ML - 134, 31MN - 105, 54MRL - 85, 63MR - 52, 64ML - 71 (those are the indexes, the actual ROI numbers are this +1)

results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'

# timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')
# serial_readout = 1 - np.array(serial_readout)
# serial_readout = maximum_filter1d(serial_readout, 2)[::2]
# cue = maximum_filter1d(cue, 2)[::2]

data = {}
with h5py.File(results_path, 'r') as f:
    decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/')

# traces_dff_delta5 = data['post_session_analysis']['dff_delta5']['traces']
# traces_dff = data['post_session_analysis']['dff']['traces']
# correlation_matrix_dff_delta5 = np.corrcoef(traces_dff_delta5)
# correlation_matrix_dff = np.corrcoef(traces_dff)
# distance_matrix = np.sqrt((1 - correlation_matrix_dff_delta5) / 2.0)
# #linkage_matrix = sch.linkage(distance_matrix, method='ward')

functional_cortex_map_path = f'{base_path}/{mouse_id}/functional_parcellation_cortex_map.h5'
functional_rois_dict_path = f'{base_path}/{mouse_id}/functional_parcellation_rois_dict.h5'
with h5py.File(functional_cortex_map_path, 'r') as f:
    functional_cortex_mask = f["mask"][()]
    functional_cortex_map = f["map"][()]
functional_cortex_mask = functional_cortex_mask[:, :168]
functional_cortex_map = functional_cortex_map[:, :168]
functional_cortex_map = skeletonize(functional_cortex_map)
functional_rois_dict = load_rois_data(functional_rois_dict_path)

# config = load_config(f'{base_path}/{date}/{mouse_id}/{session_id}/session_config.json')
# closest = config["supplementary_data_config"]["closest_rois"]
# for key in closest:
#     del functional_rois_dict[key]

a=5

metric_corr = {}
rois_proximity = {}
corr_all_rois = {}
#traces = data['post_session_analysis']['dff_delta5']['traces']

# #next 5 lines are for the mock perfect data
# traces_to_copy = traces[metric_index]
# traces = np.tile(traces_to_copy,(traces.shape[0],1))
# metric_prox = calc_rois_proximity(functional_rois_dict,f'roi_{metric_index+1}')
# for i in range(len(metric_prox)):
#     traces[i] = (traces[i]+(list(metric_prox.values())[i])*1000)
# correlation_matrix_dff_delta5 = np.corrcoef(traces)
traces = data['rois_traces']['channel_0']
metric_outline = np.unravel_index(functional_rois_dict[f'roi_{metric_index+1}']['outline'], (functional_cortex_map.shape[1], functional_cortex_map.shape[0]))


for i, (key, val) in enumerate(functional_rois_dict.items()):
    # metric_corr[key] = np.corrcoef(pstr_cat[key], pstr_cat[metric_roi])[0, 1]  # correlation with metric ROI
    # dff_corr[key] = np.corrcoef (sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i], sessions_data[sess_id]['post_session_analysis']['dff']['traces'][105])[0,1]
    # metric_corr[key] = np.corrcoef(metric_trace, sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i])[0, 1]
    metric_corr[key] = np.corrcoef(traces[key],traces[f'roi_{metric_index+1}'])[0, 1]

    corr_all_rois[key] = calc_rois_corr(functional_rois_dict,traces,traces[key])
    rois_proximity[key] = calc_rois_proximity(functional_rois_dict, key)


rois_proximity_metric = calc_rois_proximity(functional_rois_dict, f'roi_{metric_index+1}')

#colormap:
# # Number of random colors
# num_colors = len(list(functional_rois_dict.keys()))
#
# # Generate random RGB values
# random_colors = np.random.rand(num_colors, 3)
#
# # Create a ListedColormap
# custom_cmap = ListedColormap(random_colors)
# #custom_cmap = 'nipy_spectral'

# Define your custom colors
colors = [
    #(0.0,0.0,0.0), #Black
    #(0.8,0.8,0.8),#light grey
    (0.0, 0.0, 0.5),  # Dark blue
    (1.0, 0.0, 0.0),  # Red
    (0.0, 1.0, 1.0),  # Cyan
     (0.75,0.25,1), #Purple
    (1,0.6,0),#Orange
    (1,0.1,0.3),#Crimson

    (1.0, 1.0, 0.0),  # Yellow
    (0.5, 0.0, 0.0),   # Dark red
    (0.0,0.5,0.0),
     (0.0,1.0,0.0), #green
     (0.0,0.0,1.0)
]
#positions = [0,0.1,0.2 ,0.3,0.5,0.6,0.8, 1.0]
positions = np.linspace(0.0,1.0,len(colors))

# Create a custom colormap
custom_cmap = LinearSegmentedColormap.from_list('custom_colormap',list(zip(positions, colors)) , N=len(list(functional_rois_dict.keys())))

rois_values = np.random.rand(len(functional_rois_dict))
rois_values_dict = dict(zip(list(functional_rois_dict.keys()),rois_values))

f = plt.figure(constrained_layout=True, figsize=(11, 6))
gs = f.add_gridspec(1,2)
ax_left0 = f.add_subplot(gs[0, 0])
_, _, pmap0 = paint_roi(functional_rois_dict,
                      functional_cortex_map,
                      list(functional_rois_dict.keys()),
                      rois_values_dict) #LK

pmap0[functional_cortex_mask==0] = None
im, _ = wf_imshow(ax_left0, pmap0, mask=None, map=None, conv_ker=None, show_cb=False, cm_name=custom_cmap, vmin=None, vmax=None, cb_side='right')
#ax_left0.set_ylabel("PSTR", fontsize=12)
#ax_left0.scatter(metric_outline[0], metric_outline[1], marker='.', s=3, c='k')
ax_left0.axis('off')

#ax_med0 = f.add_subplot(gs[0, 1])

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

# ax_med0.imshow(sns.heatmap(correlation_matrix_dff_delta5, cmap = 'inferno'))
# ax_med0.imshow(sch.dendrogram(linkage_matrix))
# sch.dendrogram(linkage_matrix)
# plt.imshow(correlation_matrix_dff_delta5, cmap='inferno', interpolation='nearest')
# plt.colorbar(label='Correlation')

# ax_right0 = f.add_subplot(gs[0, 2])
# plt.scatter(list(rois_proximity.values()), list(metric_corr.values()))
# plt.ylim(0.5,1)
# plt.ylabel("Correlation")
# plt.xlabel("Proximity")

##########3ax_right0 = f.add_subplot(gs[0,1])
#plt.scatter(list(metric_prox.values()), list(metric_corr.values()))
# plt.scatter(list(rois_proximity_metric.values()), list(metric_corr.values()))
# plt.ylim(0.2,1.02)
# plt.ylabel("Correlation")
# plt.xlabel("Proximity")

######     To plot multiple ROIs prox vs. correlation use the following loop. To plot a single one use plt.scatter(lis(rois_proximity_metric.values()), list(metric_corr.values())) and the plot settings under it (above this line)


#ax_left0 = f.add_subplot(gs[0, 0])

ax_right0 = f.add_subplot(gs[0, 1])
#colormap = cm.inferno

#rois_to_plot = ['roi_106','roi_01']

for key in functional_rois_dict:
#for key in rois_to_plot:
    # params, covariance = curve_fit(exponential_func, list(rois_proximity[key].values()),list(corr_all_rois[key].values()))
    # a, b = params
    # x_fit = np.linspace(min(rois_proximity[key].values()), max(rois_proximity[key].values()), 100)
    # y_fit = exponential_func(x_fit, a, b)
    sm = ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    color = sm.to_rgba(rois_values_dict[key])
    prox_microns = [value * 0.029 for value in list(rois_proximity[key].values())]
    ax_right0.scatter(prox_microns, list(corr_all_rois[key].values()),  label=key, s=2, color=color)
    ax_right0.set_ylim(0.55, 1.02)
    ax_right0.set_ylabel("Correlation [A.U.]")
    ax_right0.set_xlabel("Proximity [mm]")
    #ax_right0.legend()
#    ax, im, paint_map = paint_roi(functional_rois_dict, functional_cortex_map,[key], ax=ax_left0,rois_vals={key:metric_corr[key]})

    # plt.plot(x_fit, y_fit, label=key)

########## To add fit: uncomment from here to  plt.plot(x_fit, y_fit, label=f'Fitted Curve: y(x) = (1/(x) +{a})', color='red')

# x_data = np.array(list(rois_proximity_metric.values()))
# y_data = np.array(list(metric_corr.values()))
#
# # # Linear fit
# # coefficients = np.polyfit(x_data, y_data, 1)
# # linear_fit = np.poly1d(coefficients)
# # y_fit = linear_fit(x_data)
#
# # Exp fit
# # params, covariance = curve_fit(exponential_func, x_data, y_data)
# # a, b = params
# # y_fit = exponential_func(x_data, a, b)
# params, covariance = curve_fit(one_over_x, x_data, y_data)
# a = params
# x_fit = np.linspace(min(x_data), max(x_data), 100)
# y_fit = one_over_x(x_fit, a)
#
# #plt.scatter(list(rois_proximity_metric.values()), list(metric_corr.values()))
# plt.scatter(x_data, y_data, label=f'Data')
# plt.ylim(bottom=0)
# # plt.plot(x_fit, y_fit, label=f'Fitted Curve: y(x) = {b:.4f}/e^({a:.4f}x + {c:.4f}) + {d:.4f}', color='red')
# plt.plot(x_fit, y_fit, label=f'Fitted Curve: y(x) = (1/(x) +{a})', color='red')

# #To save value to MATLAB:
# import scipy.io
# scipy.io.savemat('x_data_64_NF5', {'variable_name': x_data})
# scipy.io.savemat('y_data_64_NF5', {'variable_name': y_data})


#plt.plot(x_data, y_fit, label=f'Exponential Fit a={a}', color='red')
#plt.plot(x_data, y_fit, color='r', label=f'Slope = {coefficients[0]}') #for linear

#plt.ylim(0.5,1)
#plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
# plt.ylabel("Correlation")
# plt.xlabel("Proximity")
#
# ax4 = f.add_subplot(gs[1, 1])
# ax4.axis('off')


# plt.figure(figsize=(8, 6))
# plt.imshow(correlation_matrix_dff_delta5, cmap='plasma', interpolation='none')
# plt.colorbar(label='Correlation')
plt.title(f'{title}')

plt.show()
# plt.rcParams['svg.fonttype'] = 'none'  # or 'path' or 'none'
# plt.savefig(f'{base_path}/Figs_for_paper/{title}.svg',format='svg',dpi=500)



#print(popt)
#plt.savefig(f'{base_path}/Figures_exp2_all_mice_compare/{title}.png', format="png")

a=5
