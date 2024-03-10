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




base_path = '/data/Lena/WideFlow_prj'
dates_vec = ['20230604','20230608']
mice_id = ['21ML','31MN','54MRL','63MR','64ML']
colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'
sessions_vec = ['spont_mockNF_NOTexcluded_closest','CRC4']
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
    #metric_outline = np.unravel_index(functional_rois_dict[f'roi_{metric_index+1}']['outline'], (functional_cortex_map.shape[1], functional_cortex_map.shape[0]))


        slopes[f'{sess_name}_{mouse_id}']={}
        for i, (key, val) in enumerate(functional_rois_dict.items()):
            # metric_corr[key] = np.corrcoef(pstr_cat[key], pstr_cat[metric_roi])[0, 1]  # correlation with metric ROI
            # dff_corr[key] = np.corrcoef (sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i], sessions_data[sess_id]['post_session_analysis']['dff']['traces'][105])[0,1]
            # metric_corr[key] = np.corrcoef(metric_trace, sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i])[0, 1]
            #metric_corr[key] = np.corrcoef(traces[key],traces[f'roi_{metric_index+1}'])[0, 1]

            corr_all_rois[key] = calc_rois_corr(functional_rois_dict,traces,traces[key])
            rois_proximity[key] = calc_rois_proximity(functional_rois_dict, key)
            x_data = [value * 0.029 for value in list(rois_proximity[key].values())]
            y_data = list(corr_all_rois[key].values())
            coefficients = np.polyfit(x_data, y_data, 1)
            slopes[f'{sess_name}_{mouse_id}'][key] = (coefficients[0])
            a=5


    #rois_proximity_metric = calc_rois_proximity(functional_rois_dict, f'roi_{metric_index+1}')


slopes_pre = {}
for key in closest_rois['54MRL']:
    slopes_pre[key] = []
    for mouse in mice_id:
        key2 = closest_rois[mouse][key]
        slopes_pre[key].append(slopes[f'{sessions_vec[0]}_{mouse}'][key2])


slopes_post = {}
for key in closest_rois['54MRL']:
    slopes_post[key] = []
    for mouse in mice_id:
        key2 = closest_rois[mouse][key]
        slopes_post[key].append(slopes[f'{sessions_vec[1]}_{mouse}'][key2])

p_values = {}
for key in closest_rois['54MRL']:
    t_statistic, p_value_ttset = ttest_rel(slopes_pre[key], slopes_post[key])
    p_values[key] = p_value_ttset

# Sort the dictionary by values
sorted_items = sorted(p_values.items(), key=lambda x: x[1])
# Convert the sorted list of tuples back to a dictionary
sorted_dict = dict(sorted_items)



a=5
p_values_list = list(p_values.values())
hist, bin_edges = np.histogram(p_values_list, bins=30)
mode_bin_index = np.argmax(hist)
mode_values = [(bin_edges[mode_bin_index] + bin_edges[mode_bin_index + 1]) / 2]
mean_slopes = np.mean(p_values_list)


#ax_edge_right0 = f.add_subplot(gs[0,2])
plt.hist(p_values_list, bins=30, edgecolor = 'black')
plt.xlabel('Slope')
plt.ylabel('Frequency')
plt.axvline(x=mode_values[0], color='r', linestyle='--', label=f'Mode = {mode_values[0]:.2f}')
# plt.axvline(x=mean_slopes,color = 'darkred', linestyle='--', label =f'Mean = {mean_slopes:.2f}' )
plt.legend()

#plt.show()
plt.rcParams['svg.fonttype'] = 'none'  # or 'path' or 'none'
plt.savefig(f'{base_path}/Figs_for_paper/hist_pvalues_change_in_slope_spont_to_{sessions_vec[1]}.svg',format='svg',dpi=500)

