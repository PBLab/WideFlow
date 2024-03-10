import numpy as np
from skimage.transform import resize
from skimage.morphology import skeletonize
from scipy.ndimage.filters import maximum_filter1d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel

from utils.load_tiff import load_tiff
from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict

from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
from analysis.utils.peristimulus_time_response import calc_pstr
from utils.load_config import load_config
from utils.load_rois_data import load_rois_data
from analysis.plots import plot_traces, wf_imshow
from analysis.utils.rois_proximity import calc_rois_proximity
from utils.paint_roi import paint_roi

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

def find_closest_key(dict1, dict2):
    result = {}

    for key1, value1 in dict1.items():
        closest_key = None
        min_difference = float('inf')

        for key2, value2 in dict2.items():
            difference = abs(value1[0] - value2[0]) + abs(value1[1] - value2[1])

            if difference < min_difference:
                min_difference = difference
                closest_key = key2

        result[key1] = closest_key

    return result

def calculate_zscore(data_point, mean, std_dev):
    return (data_point - mean) / std_dev

def cohens_d(group1, group2):
    mean_diff = np.mean(group1) - np.mean(group2)
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)

    cohen_d = mean_diff / pooled_std
    return cohen_d





base_path = '/data/Lena/WideFlow_prj'
dates_vec = ['20230608','20230614']
mice_id = ['21ML','31MN','54MRL','63MR','64ML']
colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'
sessions_vec = ['CRC4','NF4']
#sess_name = 'spont_mockNF_NOTexcluded_closest'
#session_id = f'{date}_{mouse_id}_{sess_name}'
#title = f'{mouse_id}_{sess_name}_noMH_allROIS_corr_graph_smalldots'
#metric_index = 58 #21ML - 134, 31MN - 105, 54MRL - 85, 63MR - 52, 64ML - 71 (those are the indexes, the actual ROI numbers are this +1)
indexes_vec = [134, 105, 85, 52, 71 ]#(those are the indexes of ROI1, the actual ROI numbers are this +1)


results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'

roi_lists = {}
centroid_lists = {}
for mouse in mice_id:
    roi_lists[mouse] = load_rois_data(f'/data/Lena/WideFlow_prj/{mouse}/functional_parcellation_rois_dict.h5')
    centroid_lists[mouse] = {}
    for key in roi_lists[mouse]:
        centroid_lists[mouse][key] = roi_lists[f'{mouse}'][key]['Centroid']

a=5

closest_rois = {}
for mouse in mice_id:
    closest_rois[mouse] = find_closest_key(centroid_lists['54MRL'], centroid_lists[mouse])

slopes = {}

for mouse_id, metric_index in zip(mice_id, indexes_vec):
    for date,sess_name in zip(dates_vec,sessions_vec):
        session_id = f'{date}_{mouse_id}_{sess_name}'
        if sess_name == 'CRC4' and mouse_id == '63MR':
            session_id = '20230607_63MR_CRC3'

        results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
        if sess_name == 'CRC4':
            results_path = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'
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

        traces = data['rois_traces']['channel_0']
        # traces_long = data['rois_traces']['channel_0']
        # traces = {}
        # for a, b in traces_long.items():
        #     shortened_list = b[:14000]
        #     traces[a] = shortened_list

        #traces = data['post_session_analysis_LK2']['diff5']
        #traces = data['post_session_analysis_LK2']['zsores_MH_diff5']
        #traces = data['post_session_analysis_LK2']['zsores_MH']
        traces_choice = 'traces'
    #metric_outline = np.unravel_index(functional_rois_dict[f'roi_{metric_index+1}']['outline'], (functional_cortex_map.shape[1], functional_cortex_map.shape[0]))


        slopes[f'{sess_name}_{mouse_id}']={}
        for i, (key, val) in enumerate(functional_rois_dict.items()):
            # metric_corr[key] = np.corrcoef(pstr_cat[key], pstr_cat[metric_roi])[0, 1]  # correlation with metric ROI
            # dff_corr[key] = np.corrcoef (sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i], sessions_data[sess_id]['post_session_analysis']['dff']['traces'][105])[0,1]
            # metric_corr[key] = np.corrcoef(metric_trace, sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i])[0, 1]
            #metric_corr[key] = np.corrcoef(traces[key],traces[f'roi_{metric_index+1}'])[0, 1]

            corr_all_rois[key] = calc_rois_corr(functional_rois_dict,traces,traces[key])
            rois_proximity[key] = calc_rois_proximity(functional_rois_dict, key)
            ### Remove ROIs further than 2mm
            for key2 in functional_rois_dict:
                dist = 2
                dist_relative = f'dist over {dist}'
                if rois_proximity[key][key2] < dist / 0.029:
                    del rois_proximity[key][key2]
                    del corr_all_rois[key][key2]
            #############
            x_data = [value * 0.029 for value in list(rois_proximity[key].values())]
            y_data = list(corr_all_rois[key].values())
            coefficients = np.polyfit(x_data, y_data, 1)
            slopes[f'{sess_name}_{mouse_id}'][key] = (coefficients[0])
            a=5


    #rois_proximity_metric = calc_rois_proximity(functional_rois_dict, f'roi_{metric_index+1}')

a=5
title = f'{sessions_vec[0]} vs {sessions_vec[1]} {dist_relative} {traces_choice} all mice barplot'

# slopes_pre_nested = []
# for mouse in mice_id:
#     slopes_pre_nested.append(list(slopes[f'{sessions_vec[0]}_{mouse}'].values()))
#
# slopes_pre = [item for sublist in slopes_pre_nested for item in sublist]
#
#
# slopes_post_nested = []
# for mouse in mice_id:
#     slopes_post_nested.append(list(slopes[f'{sessions_vec[1]}_{mouse}'].values()))
#
# slopes_post = [item for sublist in slopes_post_nested for item in sublist]

slopes_pre_dict = {}
mean_slopes_pre = []
for mouse in mice_id:
    slopes_pre_dict[mouse] = []
    slopes_pre_dict[mouse].append(list(slopes[f'{sessions_vec[0]}_{mouse}'].values()))
    mean_slopes_pre.append(np.mean(list(slopes[f'{sessions_vec[0]}_{mouse}'].values())))

slopes_post_dict = {}
mean_slopes_post = []
for mouse in mice_id:
    slopes_post_dict[mouse] = []
    slopes_post_dict[mouse].append(list(slopes[f'{sessions_vec[1]}_{mouse}'].values()))
    mean_slopes_post.append(np.mean(list(slopes[f'{sessions_vec[1]}_{mouse}'].values())))

t_statistic, p_value_ttset = ttest_rel(mean_slopes_pre, mean_slopes_post)
effect_size = cohens_d(mean_slopes_pre, mean_slopes_post)

mean_pre = np.mean(mean_slopes_pre)
mean_post = np.mean(mean_slopes_post)

# plt.bar(['pre','post'], [mean_pre, mean_post], color=['blue', 'orange'])
# # for i in range(len(mice_id)):
# #     plt.plot([mice_id[i], mice_id[i]], [slopes_pre[i], slopes_post[i]], color='gray', linestyle='--', marker='o', markersize=8)
#
# for pre, post, mouse_id in zip(slopes_pre, slopes_post, mice_id):
#     plt.plot([mice_id.index(mouse_id), mice_id.index(mouse_id) ], [pre, post], color='gray', linestyle='--', marker='o', markersize=8)

bar_width = 0.001
bar_positions = [0,0.0015]

plt.bar(bar_positions, [mean_pre, mean_post], width=bar_width, color=['red', 'blue'])

# Plot individual data points on top of each bar
plt.scatter(np.full_like(mean_slopes_pre, bar_positions[0], dtype=float), mean_slopes_pre, color='black', marker='o')
plt.scatter(np.full_like(mean_slopes_post, bar_positions[1], dtype=float), mean_slopes_post, color='black', marker='o')
#plt.ylim(bottom=-0.105)


for val1, val2, color in zip(mean_slopes_pre, mean_slopes_post,colors):
    plt.plot(bar_positions, [val1, val2], color=color, linestyle='--', label=mice_id[list(mean_slopes_pre).index(val1)])

#title = f'{sessions_vec[0]} vs {sessions_vec[1]} pv={p_value_ttset}'

plt.legend()
plt.ylabel('Slope [1/mm]')
#print(p_value_ttset)
plt.title(f'{title} pv = {p_value_ttset} Cohens d = {effect_size}')
#plt.show()
plt.rcParams['svg.fonttype'] = 'none'  # or 'path' or 'none'
plt.savefig(f'{base_path}/Figs_for_paper/{sessions_vec[0]} vs {sessions_vec[1]}  {dist_relative} mean slope all ROIs all mice {traces_choice}.svg',format='svg',dpi=500)
# a=5