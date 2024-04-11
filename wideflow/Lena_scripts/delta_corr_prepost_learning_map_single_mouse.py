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




base_path = '/data/Lena/WideFlow_prj'
dates_vec = ['20230608','20230614']
mice_id = ['64ML']
colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'
sessions_vec = ['CRC4','NF4']
indexes_vec = [71]#(those are the indexes, the actual ROI numbers are this +1)[134, 105, 85, 52, 71 ]
#indexes_vec = [65] #ROI2(motor)
#indexes_vec = [58] #retrosplenial
#indexes_vec = [85] #bottom of somatosensory
#indexes_vec = [47] #v1


title = f'Delta corr {mice_id} {sessions_vec[1]}-{sessions_vec[0]} roi {indexes_vec[0]+1} colorbar -0.2 to 0.2 blue to red'

results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
CRC_res_path = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'

corr_all_sess = []
prox_all_sess = []
for mouse_id,metric_index in zip(mice_id,indexes_vec):
    for date, sess_name in zip(dates_vec, sessions_vec):
        session_id = f'{date}_{mouse_id}_{sess_name}'
        if mouse_id == '63MR' and sess_name == 'CRC4':
            session_id = '20230607_63MR_CRC3'


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




        traces = data['rois_traces']['channel_0']
        #traces = data['post_session_analysis_LK2']['diff5']
        #traces = data['post_session_analysis_LK2']['zsores_MH_diff5']
        #traces = data['post_session_analysis_LK2']['zsores_MH']
        traces_choice = 'traces'
        metric_outline = np.unravel_index(functional_rois_dict[f'roi_{metric_index+1}']['outline'], (functional_cortex_map.shape[1], functional_cortex_map.shape[0]))


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
        # for key in functional_rois_dict:
        #     dist = 2
        #     if rois_proximity_metric[key] > dist/0.029:
        #         del rois_proximity_metric[key]
        #         del metric_corr[key]
        ############
        prox_all_sess.append(rois_proximity_metric)
        corr_all_sess.append(metric_corr)
        a=5




a=5

delta_corr = {}
for i, (key1, val1) in enumerate(corr_all_sess[0].items()):
    delta_corr[key1] = corr_all_sess[1][key1]-val1


f = plt.figure(constrained_layout=True, figsize=(11, 6))
gs = f.add_gridspec(1,2)
ax_left0 = f.add_subplot(gs[0, 0])
_, _, pmap0 = paint_roi(functional_rois_dict,
                      functional_cortex_map,
                      list(functional_rois_dict.keys()),
                      delta_corr) #LK
pmap0[functional_cortex_mask==0] = None
im, _ = wf_imshow(ax_left0, pmap0, mask=None, map=None, conv_ker=None, show_cb=False, cm_name='seismic', vmin=-1, vmax=1, cb_side='left')
cbar = plt.colorbar(im, ax=ax_left0, label=f'Corr {sessions_vec[1]} - corr {sessions_vec[0]}')#, ticks=[-1, 0, 1]
#im.set_clim(min(list(delta_corr.values())),max(list(delta_corr.values())))
im.set_clim(-0.2,0.2)
#plt.colorbar(label=f'Corr {sessions_vec[1]} - corr {sessions_vec[0]}')
#ax_left0.set_ylabel("PSTR", fontsize=12)
ax_left0.scatter(metric_outline[0], metric_outline[1], marker='.', s=3, c='k')
ax_left0.axis('off')


plt.title(f'Delta corr {mice_id} {sessions_vec[1]}-{sessions_vec[0]} roi {indexes_vec[0]+1}')


plt.show()

# plt.rcParams['svg.fonttype'] = 'none'  # or 'path' or 'none'
# plt.savefig(f'{base_path}/Figs_for_paper/{title}.svg',format='svg',dpi=500)







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

# ax_right0 = f.add_subplot(gs[0,1])
# #plt.scatter(list(metric_prox.values()), list(metric_corr.values()))
# #[value * 29 for value in list(rois_proximity_metric.values())]
# plt.scatter([value * 0.029 for value in list(rois_proximity_metric.values())], list(metric_corr.values()))
# plt.ylim(0.55,1.02)
# plt.ylabel("Correlation [A.U.]")
# plt.xlabel("Proximity [mm]")

######     To plot multiple ROIs prox vs. correlation use the following loop. To plot a single one use plt.scatter(lis(rois_proximity_metric.values()), list(metric_corr.values())) and the plot settings under it (above this line)

# for key in functional_rois_dict:
#     # params, covariance = curve_fit(exponential_func, list(rois_proximity[key].values()),list(corr_all_rois[key].values()))
#     # a, b = params
#     # x_fit = np.linspace(min(rois_proximity[key].values()), max(rois_proximity[key].values()), 100)
#     # y_fit = exponential_func(x_fit, a, b)
#     plt.scatter(list(rois_proximity[key].values()), list(corr_all_rois[key].values()), label=key, s=2)
#     plt.ylim(0.5, 1.02)
#     plt.ylabel("Correlation")
#     plt.xlabel("Proximity")
#     # plt.plot(x_fit, y_fit, label=key)

########## To add fit: uncomment from here to  plt.plot(x_fit, y_fit, label=f'Fitted Curve: y(x) = (1/(x) +{a})', color='red')

# x_data = np.array([value * 0.029 for value in list(rois_proximity_metric.values())])
# y_data = np.array(list(metric_corr.values()))
#
# # # Linear fit
# coefficients = np.polyfit(x_data, y_data, 1)
# linear_fit = np.poly1d(coefficients)
# y_fit = linear_fit(x_data)

# Exp fit
# params, covariance = curve_fit(exponential_func, x_data, y_data)
# a, b = params
# y_fit = exponential_func(x_data, a, b)
# params, covariance = curve_fit(one_over_x, x_data, y_data)
# a = params
# x_fit = np.linspace(min(x_data), max(x_data), 100)
# y_fit = one_over_x(x_fit, a)

#plt.scatter(list(rois_proximity_metric.values()), list(metric_corr.values()))
# plt.scatter(x_data, y_data)
# plt.ylim(bottom=0.55)
# plt.plot(x_fit, y_fit, label=f'Fitted Curve: y(x) = {b:.4f}/e^({a:.4f}x + {c:.4f}) + {d:.4f}', color='red')
# plt.plot(x_fit, y_fit, label=f'Fitted Curve: y(x) = (1/(x) +{a})', color='red')

# #To save value to MATLAB:
# import scipy.io
# scipy.io.savemat('x_data_64_NF5', {'variable_name': x_data})
# scipy.io.savemat('y_data_64_NF5', {'variable_name': y_data})


#plt.plot(x_data, y_fit, label=f'Exponential Fit a={a}', color='red')
# plt.plot(x_data, y_fit, color='r', label=f'Slope = {coefficients[0]}') #for linear
#
# #plt.ylim(0.5,1)
# plt.legend()
# plt.ylabel("Correlation")
# plt.xlabel("Proximity")
#
# ax4 = f.add_subplot(gs[1, 1])
# ax4.axis('off')


# plt.figure(figsize=(8, 6))
# plt.imshow(correlation_matrix_dff_delta5, cmap='plasma', interpolation='none')
#plt.colorbar(label=f'Corr {sessions_vec[1]} - corr {sessions_vec[0]}')





#print(popt)

#plt.savefig(f'{base_path}/Figures_exp2_all_mice_compare/{title}.png', format="png")
