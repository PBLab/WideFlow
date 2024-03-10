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
from scipy.stats import wilcoxon

from utils.load_tiff import load_tiff
from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict

from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
from analysis.utils.peristimulus_time_response import calc_pstr
from utils.load_config import load_config
from utils.load_rois_data import load_rois_data
from analysis.plots import plot_traces, wf_imshow
from analysis.utils.rois_proximity import calc_rois_proximity
from utils.paint_roi import paint_roi
from scipy.stats import linregress


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
dates_vec = ['20230604','20230608','20230611','20230614']
mice_id = ['31MN']
colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'
sessions_vec = ['spont_mockNF_NOTexcluded_closest','CRC4','NF1','NF4']
#session_id = f'{date}_{mouse_id}_{sess_name}'
#title = f'{mouse_id}_{sess_name}_noMH_allROIS_corr_graph_smalldots'
indexes_vec = [105]#(those are the indexes, the actual ROI numbers are this +1)[134, 105, 85, 52, 71 ]
#indexes_vec = [65] #ROI2(motor)
#indexes_vec = [58] #retrosplenial
#indexes_vec = [85] #bottom of somatosensory
#indexes_vec = [47] #v1


#title = f'{mice_id}_{sessions_vec}_roi_{indexes_vec[0]+1}_corr_vs_prox_linear_fit'

results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
CRC_res_path = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'

corr_all_sess = []
prox_all_sess = []
for mouse_id,metric_index in zip(mice_id,indexes_vec):
    for date, sess_name in zip(dates_vec, sessions_vec):
        session_id = f'{date}_{mouse_id}_{sess_name}'


        data = {}
        results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
        if sess_name == 'CRC4':
            results_path = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'

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




        # traces = data['rois_traces']['channel_0']
        traces = data['post_session_analysis_LK2']['diff5']
        #traces = data['post_session_analysis_LK2']['zsores_MH_diff5']
        #traces = data['post_session_analysis_LK2']['zsores_MH']
        traces_choice = 'diff5'
        #metric_outline = np.unravel_index(functional_rois_dict[f'roi_{metric_index+1}']['outline'], (functional_cortex_map.shape[1], functional_cortex_map.shape[0]))


        for i, (key, val) in enumerate(functional_rois_dict.items()):
            # metric_corr[key] = np.corrcoef(pstr_cat[key], pstr_cat[metric_roi])[0, 1]  # correlation with metric ROI
            # dff_corr[key] = np.corrcoef (sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i], sessions_data[sess_id]['post_session_analysis']['dff']['traces'][105])[0,1]
            # metric_corr[key] = np.corrcoef(metric_trace, sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i])[0, 1]


            metric_corr[key] = np.corrcoef(traces[key],traces[f'roi_{metric_index+1}'])[0, 1]
            #metric_corr[key] = np.corrcoef(traces[key], traces[f'roi_01'])[0, 1]

            corr_all_rois[key] = calc_rois_corr(functional_rois_dict,traces,traces[key])
            rois_proximity[key] = calc_rois_proximity(functional_rois_dict, key)




        rois_proximity_metric = calc_rois_proximity(functional_rois_dict, f'roi_{metric_index+1}')
        #rois_proximity_metric = calc_rois_proximity(functional_rois_dict, f'roi_01')
        ### Remove ROIs further than 2mm
        for key in functional_rois_dict:
            dist = 2
            if rois_proximity_metric[key] > dist/0.029:
                del rois_proximity_metric[key]
                del metric_corr[key]
        ############
        prox_all_sess.append(rois_proximity_metric)
        corr_all_sess.append(metric_corr)
        a=5
a=5

title = f'{mice_id} {sessions_vec} roi {indexes_vec[0]+1} dist under {dist} {traces_choice} corr  vs prox linear fit'

for i in range(len(sessions_vec)):
    #x_data = list(prox_all_sess[i].values())
    x_data = [value * 0.029 for value in list(prox_all_sess[i].values())]
    y_data = list(corr_all_sess[i].values())
    # coefficients = np.polyfit(x_data, y_data, 1)
    # linear_fit = np.poly1d(coefficients)
    # y_fit = linear_fit(x_data)
    # plt.scatter(x_data, y_data, s=2.5)
    # plt.ylim(bottom=0.35)
    # plt.plot(x_data, y_fit, label=sessions_vec[i])  # for linear

    # # Perform linear regression
    # slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)

    # # Calculate the y-values for the regression line
    # line = slope * np.array(x_data) + intercept

    # # Calculate confidence intervals for the regression line
    # ci = 1.96 * std_err  # 95% confidence interval

    # # Calculate the predicted values and confidence intervals
    # x_range = np.linspace(min(x_data), max(x_data), 100)  # Create a range of x values
    # predicted_values = slope * x_range + intercept
    # ci = 1.96 * std_err  # 95% confidence interval

    # Plot the data points and linear regression line with confidence intervals
    coefficients = np.polyfit(x_data, y_data, 1)
    sns.regplot(x=x_data, y=y_data, ci=95, label=f'{sessions_vec[i]} slope {coefficients[0]}', scatter_kws={'s': 2.5})

    # # Plot the data points
    # plt.scatter(x_data, y_data, label=sessions_vec[i])
    #
    # # Plot the regression line
    # plt.plot(x_range, predicted_values, label='Linear Fit')
    #
    # # Plot the confidence intervals
    # plt.fill_between(x_range, predicted_values - ci, predicted_values + ci, color='red', alpha=0.2, label='95% CI')

plt.ylabel("Correlation [A.U.]")
plt.xlabel("Proximity [mm]")
plt.title(f'{title}')
plt.legend()


#plt.show()
plt.rcParams['svg.fonttype'] = 'none'  # or 'path' or 'none'
plt.savefig(f'{base_path}/Figs_for_paper/{title}.svg',format='svg',dpi=500)

