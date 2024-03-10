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
from scipy.stats import ttest_rel


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
mice_id = ['21ML',
     '31MN','54MRL','63MR','64ML']
colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'
sessions_vec = [#'spont_mockNF_NOTexcluded_closest',
    'CRC4','NF4']
#session_id = f'{date}_{mouse_id}_{sess_name}'
#title = f'{mouse_id}_{sess_name}_noMH_allROIS_corr_graph_smalldots'

indexes_vec = [134,
               105, 85, 52, 71 ]#(those are the indexes of ROI1, the actual ROI numbers are this +1)
#indexes_vec = [148, 65, 56, 71, 46] #indexes of ROI2, to check changes in untargeted area
#indexes_vec = [162, 58, 42, 21, 59] # indexes for retrosplenial ROIs
#indexes_vec = [28, 85, 54, 38, 37] #somatosensory boarder, below target ROI
#indexes_vec = [189, 47, 21, 58, 80] #V1 ROIs
#title = f'pre_post_corr_slopes_target_ROI_{sessions_vec[0]}_{sessions_vec[1]}'

#results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
results_path = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'

slopes = []
for mouse_id,metric_index in zip(mice_id,indexes_vec):
    for date, sess_name in zip(dates_vec, sessions_vec):
        session_id = f'{date}_{mouse_id}_{sess_name}'
        if sess_name == 'CRC4' and mouse_id == '63MR':
            session_id = '20230607_63MR_CRC3'



        if sess_name == 'CRC4':
            results_path = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'

        else:
            results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'


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
        # traces_long = data['rois_traces']['channel_0']
        # traces = {}
        # for a, b in traces_long.items():
        #     shortened_list = b[:14000]
        #     traces[a] = shortened_list
        # traces = data['post_session_analysis_LK2']['diff5']
        # traces = data['post_session_analysis_LK2']['zsores_MH_diff5']
        # traces = data['post_session_analysis_LK2']['zsores_MH']
        traces_choice = 'traces'


        #metric_outline = np.unravel_index(functional_rois_dict[f'roi_{metric_index+1}']['outline'], (functional_cortex_map.shape[1], functional_cortex_map.shape[0]))


        for i, (key, val) in enumerate(functional_rois_dict.items()):
            # metric_corr[key] = np.corrcoef(pstr_cat[key], pstr_cat[metric_roi])[0, 1]  # correlation with metric ROI
            # dff_corr[key] = np.corrcoef (sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i], sessions_data[sess_id]['post_session_analysis']['dff']['traces'][105])[0,1]
            # metric_corr[key] = np.corrcoef(metric_trace, sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i])[0, 1]
            metric_corr[key] = np.corrcoef(traces[key],traces[f'roi_{metric_index+1}'])[0, 1]

            corr_all_rois[key] = calc_rois_corr(functional_rois_dict,traces,traces[key])
            rois_proximity[key] = calc_rois_proximity(functional_rois_dict, key)


        rois_proximity_metric = calc_rois_proximity(functional_rois_dict, f'roi_{metric_index+1}')

        ### Remove ROIs further than 2mm
        for key in functional_rois_dict:
            dist = 2
            relative = 'over' #this is a string used for the title to write whether dist under or over the dist was chosen
            if rois_proximity_metric[key] < dist/0.029:
                del rois_proximity_metric[key]
                del metric_corr[key]
        ############

# f = plt.figure(constrained_layout=True, figsize=(11, 6))
# gs = f.add_gridspec(1,3)
# ax_left0 = f.add_subplot(gs[0, 0])
# _, _, pmap0 = paint_roi(functional_rois_dict,
#                       functional_cortex_map,
#                       list(functional_rois_dict.keys()),
#                       metric_corr) #LK
# pmap0[functional_cortex_mask==0] = None
# im, _ = wf_imshow(ax_left0, pmap0, mask=None, map=None, conv_ker=None, show_cb=False, cm_name='inferno', vmin=None, vmax=None, cb_side='right')
# #ax_left0.set_ylabel("PSTR", fontsize=12)
# ax_left0.scatter(metric_outline[0], metric_outline[1], marker='.', s=3, c='k')
# ax_left0.axis('off')

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

# ax_middle0 = f.add_subplot(gs[0,1])
# #plt.scatter(list(metric_prox.values()), list(metric_corr.values()))
# plt.scatter(list(rois_proximity_metric.values()), list(metric_corr.values()))
# plt.ylim(0.2,1.02)
# plt.ylabel("Correlation")
# plt.xlabel("Proximity")
        #x_data = np.array(list(rois_proximity_metric.values()))
        x_data = np.array([value * 0.029 for value in list(rois_proximity_metric.values())])
        y_data = np.array(list(metric_corr.values()))
        coefficients = np.polyfit(x_data, y_data, 1)
        slopes.append(coefficients[0])


######     To plot multiple ROIs prox vs. correlation use the following loop. To plot a single one use plt.scatter(lis(rois_proximity_metric.values()), list(metric_corr.values())) and the plot settings under it (above this line)

        #
        # for key in functional_rois_dict:
        #     # params, covariance = curve_fit(exponential_func, list(rois_proximity[key].values()),list(corr_all_rois[key].values()))
        #     # a, b = params
        #     # x_fit = np.linspace(min(rois_proximity[key].values()), max(rois_proximity[key].values()), 100)
        #     # y_fit = exponential_func(x_fit, a, b)
        #     x_data = list(rois_proximity[key].values())
        #     y_data = list(corr_all_rois[key].values())
        #     coefficients = np.polyfit(x_data, y_data, 1)
        #     slopes.append(coefficients[0])
        #     # plt.scatter(list(rois_proximity[key].values()), list(corr_all_rois[key].values()), label=key, s=2)
        #     # plt.ylim(0.5, 1.02)
        #     # plt.ylabel("Correlation")
        #     # plt.xlabel("Proximity")
        #     # plt.plot(x_fit, y_fit, label=key)

        a=5

a=5

title = f'pre_post_corr_slopes_target_ROI_{sessions_vec[0]}_{sessions_vec[1]}_dist {relative} {dist} {traces_choice}'

slopes_pre = [slopes[0], slopes[2],slopes[4],slopes[6],slopes[8]]
slopes_post = [slopes[1],slopes[3],slopes[5],slopes[7],slopes[9]]
# slopes_pre = [slopes[0], slopes[2],slopes[4],slopes[6]]
# slopes_post = [slopes[1],slopes[3],slopes[5],slopes[7]]
statistic_wil, p_value_wil = wilcoxon(slopes_pre, slopes_post) #paired t-test gave p=0.076, so we went with wilcoxon (which is the substitute for paired t test when data is not compatible with a t test)
t_statistic, p_value_ttset = ttest_rel(slopes_pre, slopes_post)
effect_size = cohens_d(slopes_pre, slopes_post)

mean_pre = np.mean(slopes_pre)
mean_post = np.mean(slopes_post)

# plt.bar(['pre','post'], [mean_pre, mean_post], color=['blue', 'orange'])
# # for i in range(len(mice_id)):
# #     plt.plot([mice_id[i], mice_id[i]], [slopes_pre[i], slopes_post[i]], color='gray', linestyle='--', marker='o', markersize=8)
#
# for pre, post, mouse_id in zip(slopes_pre, slopes_post, mice_id):
#     plt.plot([mice_id.index(mouse_id), mice_id.index(mouse_id) ], [pre, post], color='gray', linestyle='--', marker='o', markersize=8)

bar_width = 0.03
bar_positions = [0,0.035]

plt.bar(bar_positions, [mean_pre, mean_post], width=bar_width, color=['red', 'blue'])

# Plot individual data points on top of each bar
plt.scatter(np.full_like(slopes_pre, 0, dtype=float), slopes_pre, color='black', marker='o')
plt.scatter(np.full_like(slopes_post, 0.035, dtype=float), slopes_post, color='black', marker='o')
#plt.ylim(bottom=-0.105)
for val1, val2, color in zip(slopes_pre, slopes_post,colors):
    plt.plot([0 , 0.035], [val1, val2], color=color, linestyle='--', label=mice_id[list(slopes_pre).index(val1)])



plt.legend()
plt.ylabel('Slope [1/mm]')
#print(p_value_ttset)
plt.title(f'{title} pv={p_value_ttset} cohens d = {effect_size}')
#plt.show()
plt.rcParams['svg.fonttype'] = 'none'  # or 'path' or 'none'
plt.savefig(f'{base_path}/Figs_for_paper/{title}.svg',format='svg',dpi=500)
a=5
