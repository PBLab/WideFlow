import numpy as np
from skimage.transform import resize
from skimage.morphology import skeletonize
from scipy.ndimage.filters import maximum_filter1d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py
import pandas as pd
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
dates_vec = ['20230604','20230611','20230612','20230613','20230614','20230615']
mice_id = ['21ML','31MN','54MRL','63MR','64ML']
colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'
sessions_vec = ['spont_mockNF_NOTexcluded_closest','NF1','NF2','NF3','NF4','NF5']
#sess_name = 'spont_mockNF_NOTexcluded_closest'
#session_id = f'{date}_{mouse_id}_{sess_name}'
#title = f'{mouse_id}_{sess_name}_noMH_allROIS_corr_graph_smalldots'
#metric_index = 58 #21ML - 134, 31MN - 105, 54MRL - 85, 63MR - 52, 64ML - 71 (those are the indexes, the actual ROI numbers are this +1)
indexes_vec = [134, 105, 85, 52, 71 ]#(those are the indexes of ROI1, the actual ROI numbers are this +1)

results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'

# roi_lists = {}
# centroid_lists = {}
# for mouse in mice_id:
#     roi_lists[mouse] = load_rois_data(f'/data/Lena/WideFlow_prj/{mouse}/functional_parcellation_rois_dict.h5')
#     centroid_lists[mouse] = {}
#     for key in roi_lists[mouse]:
#         centroid_lists[mouse][key] = roi_lists[f'{mouse}'][key]['Centroid']
#
# a=5
#
# closest_rois = {}
# for mouse in mice_id:
#     closest_rois[mouse] = find_closest_key(centroid_lists['54MRL'], centroid_lists[mouse])

slopes = {}
slope_zscores = {}

for mouse_id, metric_index in zip(mice_id, indexes_vec):
    for date,sess_name in zip(dates_vec,sessions_vec):
        session_id = f'{date}_{mouse_id}_{sess_name}'


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

        config = load_config(f'{base_path}/{date}/{mouse_id}/{session_id}/session_config.json')
        closest = config["supplementary_data_config"]["closest_rois"]
        for key in closest:
            del functional_rois_dict[key]

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
            #rois_proximity[key] = calc_rois_proximity(functional_rois_dict, key)
            rois_prox_pixels = calc_rois_proximity(functional_rois_dict, key)
            #rois_proximity[key] = [value * 0.029 for value in list(rois_prox_pixels)]
            x_data = [value * 0.029 for value in list(rois_proximity[key].values())]
            #x_data = list(rois_proximity[key].values())
            y_data = list(corr_all_rois[key].values())
            coefficients = np.polyfit(x_data, y_data, 1)
            slopes[f'{sess_name}_{mouse_id}'][key] = (coefficients[0])
            a=5


        ###########Save excels of prox and corr for each mouse and each session, edit paths if needed:

        # df = pd.DataFrame(corr_all_rois)
        # # Specify the file path for saving the Excel file
        # excel_file_path = f'/data/Lena/WideFlow_prj/Raw data correlations and proximities/{mouse_id}_{session_id}_corr_all_rois.xlsx'
        # # Save the DataFrame to an Excel file
        # df.to_excel(excel_file_path, index=False)
        #
        # df = pd.DataFrame(rois_proximity)
        # # Specify the file path for saving the Excel file
        # excel_file_path = f'/data/Lena/WideFlow_prj/Raw data correlations and proximities/{mouse_id}_{session_id}_rois_prox_in_mm.xlsx'
        # # Save the DataFrame to an Excel file
        # df.to_excel(excel_file_path, index=False)
        #####################################################

        #slope_zscores[f'{sess_name}_{mouse_id}'] = {}
        for i, (key, val) in enumerate(functional_rois_dict.items()):
            mean_value = np.mean(list(slopes[f'{sess_name}_{mouse_id}'].values()))
            std_dev_value = np.std(list(slopes[f'{sess_name}_{mouse_id}'].values()))
            #slope_zscores[f'{sess_name}_{mouse_id}'][key] = calculate_zscore(slopes[f'{sess_name}_{mouse_id}'][key], mean_value, std_dev_value)

            zscore = calculate_zscore(slopes[f'{sess_name}_{mouse_id}'][key], mean_value, std_dev_value)
            if f'{mouse_id}' in slope_zscores:
                if f'{key}' in slope_zscores[f'{mouse_id}']:
                    slope_zscores[f'{mouse_id}'][key].append(zscore)
                else:
                    slope_zscores[f'{mouse_id}'][key] = [(zscore)]
            else:
                slope_zscores[f'{mouse_id}']={}
                slope_zscores[f'{mouse_id}'][key] = [(zscore)]

    #rois_proximity_metric = calc_rois_proximity(functional_rois_dict, f'roi_{metric_index+1}')

# config = load_config(f'{base_path}/{date}/31MN/20230615_31MN_NF5/session_config.json')
# closest = config["supplementary_data_config"]["closest_rois"]
# for key in closest:
#     del slope_zscores['31MN'][key]
#
# for key in slope_zscores['31MN']:
#     if key=='roi_106':
#         plt.plot(sessions_vec,slope_zscores['31MN'][key], color = 'black', linewidth=3)
#     else:
#         plt.plot(sessions_vec, slope_zscores['31MN'][key])


for metric_ind, mouse_id1,color_name in zip(indexes_vec, mice_id,colors):
    plt.plot(sessions_vec, slope_zscores[mouse_id1][f'roi_{metric_ind+1}'], label = f'{mouse_id1}', color = color_name)


plt.plot(sessions_vec,[0,0,0,0,0,0], color='black', linestyle='--')
plt.legend()
plt.ylabel('Z-score of target ROI slope [A.U.]')

plt.rcParams['svg.fonttype'] = 'none'  # or 'path' or 'none'
plt.savefig( f'/data/Lena/WideFlow_prj/Figs_for_paper/zscores_of_slopes_all_mice_targetROI.svg',format='svg',dpi=500)
#plt.show()


a=5