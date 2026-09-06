import numpy as np
from skimage.transform import resize
from skimage.morphology import skeletonize
from scipy.ndimage.filters import maximum_filter1d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py
import pingouin as pg
import pandas as pd
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from scipy.stats import linregress
from scipy.ndimage import convolve
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import ListedColormap, BoundaryNorm , to_rgba

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

def calc_z_score(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    return (x - x_mean) / (x_std + np.finfo(np.float32).eps)


# def calc_z_score(x):
#     x_mean = np.mean(x, axis=0)-np.mean(x, axis=0)
#     x_std = np.std(x, axis=0)
#     return (x - x_mean) / (x_std + np.finfo(np.float32).eps)

def average_five_tuples(t1, t2, t3, t4, t5):
    # Find the minimum length among the tuples
    min_length = min(len(t1), len(t2), len(t3), len(t4), len(t5))

    # Initialize an empty list to store the averages
    avg_tuple = []

    # Iterate over the tuples element-wise and calculate the average based on the minimum length
    for i in range(min_length):
        avg_element = (t1[i] + t2[i] + t3[i] + t4[i] + t5[i]) / 5
        avg_tuple.append(avg_element)

    return tuple(avg_tuple)

def average_two_tuples(t1, t2):
    # Find the minimum length among the tuples
    min_length = min(len(t1), len(t2))

    # Initialize an empty list to store the averages
    avg_tuple = []

    # Iterate over the tuples element-wise and calculate the average based on the minimum length
    for i in range(min_length):
        avg_element = (t1[i] + t2[i]) / 2
        avg_tuple.append(avg_element)

    return tuple(avg_tuple)

def average_four_tuples(t1, t2, t3, t4):
    # Find the minimum length among the tuples
    min_length = min(len(t1), len(t2), len(t3), len(t4))

    # Initialize an empty list to store the averages
    avg_tuple = []

    # Iterate over the tuples element-wise and calculate the average based on the minimum length
    for i in range(min_length):
        avg_element = (t1[i] + t2[i] + t3[i] + t4[i]) / 4
        avg_tuple.append(avg_element)

    return tuple(avg_tuple)

def low_pass_filter(image, kernel_size=10):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    return convolve(image, kernel)

def bin_ndarray(array, new_shape):
    shape = (new_shape[0], array.shape[0] // new_shape[0],
             new_shape[1], array.shape[1] // new_shape[1])
    return array.reshape(shape).sum(-1).sum(1)




# base_path = '/data/Lena/WideFlow_prj'
base_path = DATA_STAGING_PATH  #added by Claude 20260906
dates_vec = ['20241126','20241203']
mice_id = [
    #'21ML'
     '31MN'
    , '54MRL'
    #, '63MR'
    #, '64ML'
    ,'187FN'
    ,'203MN'
    ,'204FR'
    ,'206FRL'
    ,'211MRR'
    ,'218MN'
    ]
colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'
sessions_vec = [
    #'spont_mockNF_NOTexcluded_closest',
    'CRC4',
    'NF5'
    ]
indexes_vec = [
   # 134  # 21
     105  # 31
    , 85  # 54
    #, 52  # 63
    #, 71  # 64
    ,56  #187
    ,41  #203
    ,69  #204
     ,50  #206
     ,53  #211
     ,46  #218
    ]#(those are the indexes, the actual ROI numbers are this +1)[134, 105, 85, 52, 71 ]
#indexes_vec = [65] #ROI2(motor)
#indexes_vec = [58] #retrosplenial
#indexes_vec = [85] #bottom of somatosensory
#indexes_vec = [47] #v1
num_frames_21ML = 10000


title = (f'Average zscores of delta corr {mice_id} {sessions_vec[1]}-{sessions_vec[0]} '
         f'colormap and scatter max shortened for 21ML with avg ROI')

# results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
# CRC_res_path = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'
# results_path = '/data/Lena/WideFlow_prj/Results/results_exp2.1.h5'
results_path = DATA_STAGING_PATH + '/Results/results_exp2.1.h5'  #added by Claude 20260906
# CRC_res_path = '/data/Lena/WideFlow_prj/Results/results_exp2.1.h5'
CRC_res_path = DATA_STAGING_PATH + '/Results/results_exp2.1.h5'  #added by Claude 20260906


prox_all_sess = []
zscores_delta_all_mice = {}
pmaps = {}
pmaps_all_sessions = {}
rois_dicts = {}
delta_corr_all_mice = {}
metric_outlines = {}
for mouse_id,metric_index in zip(mice_id,indexes_vec):
    corr_all_sess = []
    pmaps_all_sessions[f'{mouse_id}'] = {}
    if mouse_id == '187FN' or mouse_id == '203MN' or mouse_id == '204FR' or mouse_id == '206FRL' or mouse_id == '211MRR' or mouse_id == '218MN':
        dates_vec = ['20241129','20241130','20241201','20241202'
            ,'20241203'
                     ]
        sessions_vec = ['NF1','NF2','NF3','NF4'
            , 'NF5'
                        ]
        spont_sess_length_frames = 60000
        CRC_sess_length_frames = 60000
        NF_sess_length_frames = 65000
        # dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/results_exp2.1.h5'
        dataset_path_noMH = DATA_STAGING_PATH + '/Results/results_exp2.1.h5'  #added by Claude 20260906
    else:
        dates_vec = ['20230611','20230612','20230613','20230614'
            ,  '20230615'
                     ]
        sessions_vec = ['NF1','NF2','NF3','NF4'
            , 'NF5'
                        ]
        spont_sess_length_frames = 50000
        CRC_sess_length_frames = 50000
        NF_sess_length_frames = 65000
        #dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
    for date, sess_name in zip(dates_vec, sessions_vec):
        session_id = f'{date}_{mouse_id}_{sess_name}'
        if mouse_id == '63MR' and sess_name == 'CRC4':
            session_id = '20230607_63MR_CRC3'
        if sess_name == 'CRC4' and mouse_id == '203MN':
            session_id = '20241125_203MN_CRC3'
        if sess_name == 'CRC4' and mouse_id == '204FR':
            session_id = '20241125_204FR_CRC3'
        if (mouse_id == '21ML' or mouse_id == '31MN' or mouse_id == '54MRL' or mouse_id == '63MR' or mouse_id == '64ML') and sess_name == 'CRC4':
            # dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'
            dataset_path_noMH = DATA_STAGING_PATH + '/Results/Results_exp2_CRC_sessions.h5'  #added by Claude 20260906
        if (mouse_id == '21ML' or mouse_id == '31MN' or mouse_id == '54MRL' or mouse_id == '63MR' or mouse_id == '64ML') and sess_name != 'CRC4':
            # dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
            dataset_path_noMH = DATA_STAGING_PATH + '/Results/results_exp2_noMH.h5'  #added by Claude 20260906


        data = {}
        # results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
        # if sess_name == 'CRC4':
        #     results_path = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'

        with h5py.File(dataset_path_noMH, 'r') as f:
            decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/')

        functional_cortex_map_path = f'{base_path}/{mouse_id}/functional_parcellation_cortex_map.h5'
        functional_rois_dict_path = f'{base_path}/{mouse_id}/functional_parcellation_rois_dict.h5'
        with h5py.File(functional_cortex_map_path, 'r') as f:
            functional_cortex_mask = f["mask"][()]
            functional_cortex_map = f["map"][()]
        functional_cortex_mask = functional_cortex_mask[:, :168]
        functional_cortex_map = functional_cortex_map[:, :168]
        functional_cortex_map = skeletonize(functional_cortex_map)
        functional_rois_dict = load_rois_data(functional_rois_dict_path)
        rois_dicts[f'{mouse_id}'] = functional_rois_dict

        # config = load_config(f'{base_path}/{date}/{mouse_id}/{session_id}/session_config.json')
        # closest = config["supplementary_data_config"]["closest_rois"]
        # for key in closest:
        #     del functional_rois_dict[key]

        a=5

        metric_corr = {}
        #rois_proximity = {}
        #corr_all_rois = {}
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
        #metric_outline = np.unravel_index(functional_rois_dict[f'roi_{metric_index+1}']['outline'], (functional_cortex_map.shape[1], functional_cortex_map.shape[0]))
        metric_outline = np.unravel_index(functional_rois_dict[f'roi_{metric_index + 1}']['PixelIdxList'],
                                          (functional_cortex_map.shape[1], functional_cortex_map.shape[0]))
        metric_outlines[f'{mouse_id}'] = metric_outline


        for i, (key, val) in enumerate(functional_rois_dict.items()):
            # metric_corr[key] = np.corrcoef(pstr_cat[key], pstr_cat[metric_roi])[0, 1]  # correlation with metric ROI
            # dff_corr[key] = np.corrcoef (sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i], sessions_data[sess_id]['post_session_analysis']['dff']['traces'][105])[0,1]
            # metric_corr[key] = np.corrcoef(metric_trace, sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i])[0, 1]


            metric_corr[key] = np.corrcoef(traces[key][:],traces[f'roi_{metric_index+1}'][:])[0, 1]
            if (session_id == '20230604_21ML_spont_mockNF_NOTexcluded_closest' or session_id == '20230613_21ML_NF3'
                    or session_id == '20241126_187FN_CRC4' or session_id == '20241126_218MN_CRC4' or session_id=='20241203_206FRL_NF5'
                    or session_id=='20241201_211MRR_NF3' or session_id=='20241129_204FR_NF1'):
                metric_corr[key] = np.corrcoef(traces[key][:num_frames_21ML], traces[f'roi_{metric_index + 1}'][:num_frames_21ML])[0, 1]
            a=5
            #metric_corr[key] = np.corrcoef(traces[key], traces[f'roi_01'])[0, 1]

            #corr_all_rois[key] = calc_rois_corr(functional_rois_dict,traces,traces[key])
            #rois_proximity[key] = calc_rois_proximity(functional_rois_dict, key)




        #rois_proximity_metric = calc_rois_proximity(functional_rois_dict, f'roi_{metric_index+1}')
        #rois_proximity_metric = calc_rois_proximity(functional_rois_dict, f'roi_01')
        ### Remove ROIs further than 2mm
        # for key in functional_rois_dict:
        #     dist = 2
        #     if rois_proximity_metric[key] > dist/0.029:
        #         del rois_proximity_metric[key]
        #         del metric_corr[key]
        ############
        #prox_all_sess.append(rois_proximity_metric)
        corr_all_sess.append(metric_corr)
        _, _, pmaps_all_sessions[f'{mouse_id}'][f'{sess_name}'] = paint_roi(functional_rois_dict,
                                               functional_cortex_map,
                                               list(functional_rois_dict.keys()),
                                               # delta_corr)
                                               # zscores_delta_corr_dict)
                                               metric_corr)
        a=5

    # delta_corr = {}
    #
    # for i, (key1, val1) in enumerate(corr_all_sess[0].items()):
    #     delta_corr[key1] = corr_all_sess[1][key1]-val1
    #
    # delta_corr_all_mice[f'{mouse_id}'] = delta_corr
    # zscores_delta_corr_list = calc_z_score(list(delta_corr.values()))
    # zscores_delta_corr_dict = {}
    # for key in delta_corr:
    #     zscores_delta_corr_dict[key] = zscores_delta_corr_list[list(delta_corr.keys()).index(key)]
    #
    # zscores_delta_all_mice[f'{mouse_id}'] = zscores_delta_corr_dict
    _, _, pmaps[f'{mouse_id}'] = paint_roi(functional_rois_dict,
                            functional_cortex_map,
                            list(functional_rois_dict.keys()),
                            #delta_corr)
                            #zscores_delta_corr_dict)
                            metric_corr)
    a=5



groups = {'31MN':'NF','54MRL':'NF','187FN':'NF','203MN':'control','204FR':'NF','206FRL':'control','211MRR':'control','218MN':'control'}
n_rows, n_cols = 297, 168
# Initialize arrays to store p-values for each pixel
pvals_group = np.zeros((n_rows, n_cols))
pvals_session = np.zeros((n_rows, n_cols))
pvals_interaction = np.zeros((n_rows, n_cols))

# Iterate through each pixel
for i in range(n_rows):
    for j in range(n_cols):
        # Prepare data for this pixel
        pixel_data = []
        for subject, sessions in pmaps_all_sessions.items():
            for session, score_map in sessions.items():
                pixel_data.append({
                    'Subject': subject,
                    'Group': groups[subject],
                    'Session': session,
                    'Score': score_map[i, j]
                })

        # Convert to a DataFrame
        pixel_df = pd.DataFrame(pixel_data)
        if (pixel_df.Score == 0).any():
            continue

        # Clean and check data
        pixel_df['Subject'] = pixel_df['Subject'].astype(str)
        pixel_df['Group'] = pixel_df['Group'].astype('category')
        pixel_df['Session'] = pixel_df['Session'].astype('category')
        pixel_df['Score'] = pixel_df['Score'].astype(float)

        # Drop any problematic rows
        #pixel_df = pixel_df.dropna().drop_duplicates()

        a=5

        # Run Mixed ANOVA for this pixel
        anova = pg.mixed_anova(dv='Score', between='Group', within='Session', subject='Subject', data=pixel_df)

        # Store p-values for Group, Session, and Interaction
        pvals_group[i, j] = anova.loc[anova['Source'] == 'Group', 'p-unc'].values[0]
        pvals_session[i, j] = anova.loc[anova['Source'] == 'Session', 'p-unc'].values[0]
        pvals_interaction[i, j] = anova.loc[anova['Source'] == 'Interaction', 'p-unc']
        print(f'pv saving for pixel {i,j} finished')



# Apply FDR correction to all pixels
pvals_group_flat = pvals_group.flatten()
pvals_session_flat = pvals_session.flatten()
pvals_interaction_flat = pvals_interaction.flatten()

# FDR correction
_, pvals_group_fdr, _, _ = multipletests(pvals_group_flat, method='fdr_bh')
_, pvals_session_fdr, _, _ = multipletests(pvals_session_flat, method='fdr_bh')
_, pvals_interaction_fdr, _, _ = multipletests(pvals_interaction_flat, method='fdr_bh')

# Reshape corrected p-values back to the original map shape
pvals_group_fdr = pvals_group_fdr.reshape(n_rows, n_cols)
pvals_session_fdr = pvals_session_fdr.reshape(n_rows, n_cols)
pvals_interaction_fdr = pvals_interaction_fdr.reshape(n_rows, n_cols)

# Visualize significant results (optional)
import matplotlib.pyplot as plt

plt.imshow(pvals_group_fdr < 0.05, cmap='hot', interpolation='nearest')
plt.title('Significant Group Effects (FDR < 0.05)')
plt.colorbar()
plt.show()










# pmap_avg_NF = ((
#             #pmaps['21ML']
#             pmaps['54MRL']
#             #+pmaps['63MR']
#             #+pmaps['64ML']
#             +pmaps['31MN']
#             +pmaps['187FN']
#             +pmaps['204FR']
#             # pmaps['203MN']
#             #  +pmaps['206FRL']
#             #  +pmaps['211MRR']
#             #  +pmaps['218MN']
#             )
#             /4)
#
# #pmap_avg = low_pass_filter(pmap_avg)
# #pmap_avg = bin_ndarray(pmap_avg, (5, 5))
#
# avg_tuple_outline_x_NF = average_four_tuples(
#              #metric_outlines['21ML'][0]
#             metric_outlines['54MRL'][0]
#             #,metric_outlines['63MR'][0]
#             #,metric_outlines['64ML'][0]
#             ,metric_outlines['31MN'][0]
#             ,metric_outlines['187FN'][0]
#            ,metric_outlines['204FR'][0]
#      #        metric_outlines['203MN'][0]
#      #         ,metric_outlines['206FRL'][0]
#      #         ,metric_outlines['211MRR'][0]
#      #         ,metric_outlines['218MN'][0]
# )
# avg_tuple_outline_y_NF = average_four_tuples(
#              #metric_outlines['21ML'][1]
#             metric_outlines['54MRL'][1]
#             #,metric_outlines['63MR'][1]
#             #,metric_outlines['64ML'][1]
#             ,metric_outlines['31MN'][1]
#              ,metric_outlines['187FN'][1]
#             ,metric_outlines['204FR'][1]
#     #         metric_outlines['203MN'][1]
#     #          ,metric_outlines['206FRL'][1]
#     #          ,metric_outlines['211MRR'][1]
#     #          ,metric_outlines['218MN'][1]
# )
#
# max_value_NF = np.nanmax(pmap_avg_NF)
# max_indices_NF = np.argwhere(pmap_avg_NF == max_value_NF)
# max_index_NF = [np.mean(max_indices_NF[:,0]), np.mean(max_indices_NF[:,1])]
#
# min_value_NF = np.nanmin(pmap_avg_NF)
# min_indices_NF = np.argwhere(pmap_avg_NF == min_value_NF)
# min_index_NF = [np.mean(min_indices_NF[:,0]), np.mean(min_indices_NF[:,1])]
#
#
#
# pmap_avg_control = ((
#             #pmaps['21ML']
#             #pmaps['54MRL']
#             #+pmaps['63MR']
#             #+pmaps['64ML']
#             #+pmaps['31MN']
#             #+pmaps['187FN']
#             #+pmaps['204FR']
#              pmaps['203MN']
#               +pmaps['206FRL']
#               +pmaps['211MRR']
#               +pmaps['218MN']
#             )
#             /4)
#
# #pmap_avg = low_pass_filter(pmap_avg)
# #pmap_avg = bin_ndarray(pmap_avg, (5, 5))
#
# avg_tuple_outline_x_control = average_four_tuples(
#              #metric_outlines['21ML'][0]
#             #metric_outlines['54MRL'][0]
#             #,metric_outlines['63MR'][0]
#             #,metric_outlines['64ML'][0]
#             #,metric_outlines['31MN'][0]
#             #,metric_outlines['187FN'][0]
#            #,metric_outlines['204FR'][0]
#              metric_outlines['203MN'][0]
#               ,metric_outlines['206FRL'][0]
#               ,metric_outlines['211MRR'][0]
#               ,metric_outlines['218MN'][0]
# )
# avg_tuple_outline_y_control = average_four_tuples(
#              #metric_outlines['21ML'][1]
#             #metric_outlines['54MRL'][1]
#             #,metric_outlines['63MR'][1]
#             #,metric_outlines['64ML'][1]
#             #,metric_outlines['31MN'][1]
#              #,metric_outlines['187FN'][1]
#             #,metric_outlines['204FR'][1]
#              metric_outlines['203MN'][1]
#               ,metric_outlines['206FRL'][1]
#               ,metric_outlines['211MRR'][1]
#               ,metric_outlines['218MN'][1]
# )
#
# max_value_control = np.nanmax(pmap_avg_control)
# max_indices_control = np.argwhere(pmap_avg_control == max_value_control)
# max_index_control = [np.mean(max_indices_control[:,0]), np.mean(max_indices_control[:,1])]
#
# min_value_control = np.nanmin(pmap_avg_control)
# min_indices_control = np.argwhere(pmap_avg_control == min_value_control)
# min_index_control = [np.mean(min_indices_control[:,0]), np.mean(min_indices_control[:,1])]
#
# ##STATS for NF vs control
#
# a=5
# NF_mice = ['31MN','54MRL','187FN','204FR']
# NF_pmaps = {key: pmaps[key] for key in NF_mice if key in pmaps}
# arrays_list_NF = list(NF_pmaps.values())  # Get the 2D arrays from the dict
# stacked_array_NF = np.stack(arrays_list_NF, axis=0)  # Shape will be (4, 64, 64)
#
# control_mice = ['203MN','206FRL','211MRR','218MN']
# control_pmaps = {key: pmaps[key] for key in control_mice if key in pmaps}
# arrays_list_control = list(control_pmaps.values())  # Get the 2D arrays from the dict
# stacked_array_control = np.stack(arrays_list_control, axis=0)  # Shape will be (4, 64, 64)
#
# a=5
#
# # Dimensions
# n_samples, x_dim, y_dim = stacked_array_control.shape
#
# # Initialize array for p-values
# p_values = np.zeros((x_dim, y_dim))
#
# # Perform pixel-wise t-tests
# for i in range(x_dim):
#     for j in range(y_dim):
#         _, p_values[i, j] = ttest_ind(stacked_array_control[:, i, j], stacked_array_NF[:, i, j])
#
#
# # # Multiple comparisons correction (FDR or Bonferroni)
# # # Method options: 'bonferroni', 'fdr_bh' (Benjamini-Hochberg), etc.
# flat_p_values = p_values.flatten()
# flat_p_values_nonan = np.nan_to_num(flat_p_values, nan=0)
#
# reject, corrected_p_values, _, _ = multipletests(flat_p_values_nonan, alpha=0.05, method='fdr_bh')
# corrected_p_values = corrected_p_values.reshape(x_dim, y_dim)
# significant_mask = reject.reshape(x_dim, y_dim)
# #significant_mask[np.isnan(corrected_p_values)] = np.nan
# # significant_mask_int = significant_mask.astype(int)
# # significant_mask_switched = 1 - significant_mask_int
#
#
#
#
#
#
# # all_dists_from_max_pix = []
# # all_dists_from_min_pix = []
# # all_delta_corr_flat = []
# # all_zscores_delta_corr_flat = []
# # for mouse_id in mice_id:
# #     for key in list(rois_dicts[f'{mouse_id}'].keys()):
# #         dist = np.sqrt((max_index[0] - rois_dicts[f'{mouse_id}'][key]['Centroid'][0]) ** 2
# #                        + (max_index[1] - rois_dicts[f'{mouse_id}'][key]['Centroid'][1]) ** 2)
# #         dist_min = np.sqrt((min_index[0] - rois_dicts[f'{mouse_id}'][key]['Centroid'][0]) ** 2
# #                        + (min_index[1] - rois_dicts[f'{mouse_id}'][key]['Centroid'][1]) ** 2)
# #         all_dists_from_max_pix.append(dist)
# #         all_dists_from_min_pix.append(dist_min)
# #         all_delta_corr_flat.append(delta_corr_all_mice[f'{mouse_id}'][key])
# #         all_zscores_delta_corr_flat.append(zscores_delta_all_mice[f'{mouse_id}'][key])
# #
# # multiplier = 0.029
# # all_dists_from_max_mm = [value * multiplier for value in all_dists_from_max_pix]
# # all_dists_from_min_mm = [value * multiplier for value in all_dists_from_min_pix]
# # correlation_coefficient, p_value = pearsonr(all_dists_from_max_mm, all_zscores_delta_corr_flat)
# # correlation_coefficient_min, p_value_min = pearsonr(all_dists_from_min_mm, all_zscores_delta_corr_flat)
# #
# # pmaps_avg_nan = pmap_avg.copy()
# # pmaps_avg_nan[pmaps_avg_nan == 0] = np.nan
# # avg_zscore_flat = []
# # prox_avg_pmaps_flat_pix = []
# # prox_min_avg_pmaps_flat_pix = []
# # for x in range(np.shape(pmaps_avg_nan)[0]):
# #     for y in range(np.shape(pmaps_avg_nan)[1]):
# #         if np.isnan(pmaps_avg_nan[x,y]):
# #             continue
# #         elif pmaps_avg_nan[x,y] in avg_zscore_flat:
# #             continue
# #         else:
# #             dist = np.sqrt((max_index[0] - x) ** 2
# #                                + (max_index[1] - y) ** 2)
# #             dist_min = np.sqrt((min_index[0] - x) ** 2
# #                                + (min_index[1] - y) ** 2)
# #             avg_zscore_flat.append(pmaps_avg_nan[x,y])
# #             prox_avg_pmaps_flat_pix.append(dist)
# #             prox_min_avg_pmaps_flat_pix.append(dist_min)
# #
# # prox_avg_pmaps_flat_mm = [value * multiplier for value in prox_avg_pmaps_flat_pix]
# # prox_min_avg_pmaps_flat_mm = [value * multiplier for value in prox_min_avg_pmaps_flat_pix]
# # correlation_coefficient_avg, p_value_avg = pearsonr(prox_avg_pmaps_flat_mm, avg_zscore_flat)
# # correlation_coefficient_avg_min, p_value_avg_min = pearsonr(prox_min_avg_pmaps_flat_mm, avg_zscore_flat)
#
#
#
# ##### Plotting
# f = plt.figure(constrained_layout=True, figsize=(17, 6))
# gs = f.add_gridspec(1,3)
# ax_left0 = f.add_subplot(gs[0, 0])
# pmap_avg_NF[functional_cortex_mask==0] = None
# im, _ = wf_imshow(ax_left0, pmap_avg_NF, mask=None, map=None, conv_ker=None, show_cb=False, cm_name='inferno', cb_side='left') #cm_name='seismic'
# #cbar = plt.colorbar(im, ax=ax_left0, label=f'Z-score of delta corr {sessions_vec[1]} - corr {sessions_vec[0]}')
# cbar = plt.colorbar(im, ax=ax_left0, label=f'Correlations to target ROI {sessions_vec[0]}')
# #im.set_clim(np.nanmin(pmap_avg),np.nanmax(pmap_avg))
# #im.set_clim(-2,2) #(0,1)
# im.set_clim(0,1)
# #plt.colorbar(label=f'Corr {sessions_vec[1]} - corr {sessions_vec[0]}')
# #ax_left0.set_ylabel("PSTR", fontsize=12)
# ax_left0.scatter(avg_tuple_outline_x_NF, avg_tuple_outline_y_NF, marker='.', s=10, c='grey')
# ax_left0.scatter(max_index_NF[1], max_index_NF[0], marker='.', s=200, c='green') #it's plotted max_indices[1] and then
#                                                                        # [0] because [1] is the column (x) and [0] is the row (y)
# #ax_left0.scatter(min_indices[:,1], min_indices[:,0], marker='.', s=14, c='green') #it's plotted max_indices[1] and then
#                                                                        # [0] because [1] is the column (x) and [0] is the row (y)
# plt.title(#f'Z-score of delta '
# f'corr. to target ROI NF group, sessions: \n'
#           #f'{sessions_vec[1]}-'
#           f'{sessions_vec[0]}'
# )
# ax_left0.axis('off')
#
#
#
# ax_right0 = f.add_subplot(gs[0, 1])
# pmap_avg_control[functional_cortex_mask==0] = None
# im, _ = wf_imshow(ax_right0, pmap_avg_control, mask=None, map=None, conv_ker=None, show_cb=False, cm_name='inferno', cb_side='left')#cm_name='seismic'
# #cbar = plt.colorbar(im, ax=ax_right0, label=f'Z-score of delta corr {sessions_vec[1]} - corr {sessions_vec[0]}')
# #im.set_clim(np.nanmin(pmap_avg),np.nanmax(pmap_avg))
# #im.set_clim(-2,2) #(0,1)
# im.set_clim(0,1) #(0,1)
# #plt.colorbar(label=f'Corr {sessions_vec[1]} - corr {sessions_vec[0]}')
# #ax_right0.set_ylabel("PSTR", fontsize=12)
# ax_right0.scatter(avg_tuple_outline_x_control, avg_tuple_outline_y_control, marker='.', s=10, c='grey')
# ax_right0.scatter(max_index_control[1], max_index_control[0], marker='.', s=200, c='green') #it's plotted max_indices[1] and then
#                                                                        # [0] because [1] is the column (x) and [0] is the row (y)
# #ax_right0.scatter(min_indices[:,1], min_indices[:,0], marker='.', s=14, c='green') #it's plotted max_indices[1] and then
#                                                                        # [0] because [1] is the column (x) and [0] is the row (y)
# plt.title(#f'Z-score of delta '
# f'corr. to target ROI control group, sessions: \n'
#           #f'{sessions_vec[1]}-'
#           f'{sessions_vec[0]}'
# )
# ax_right0.axis('off')
#
#
#
# # #ax_right0.scatter(all_dists_from_max_mm,all_delta_corr_flat)
# # ax_right0.scatter(prox_avg_pmaps_flat_mm, avg_zscore_flat,color = 'grey', alpha=0.3, s=10, edgecolor = 'none')
# # # coefficients = np.polyfit(prox_avg_pmaps_flat_mm, avg_zscore_flat, 1)
# # # sns.regplot(x=prox_avg_pmaps_flat_mm, y=avg_zscore_flat, ci=95, label=f'slope {coefficients[0]}', scatter_kws={'s': 2.5})
# # #slope, intercept, r_value, p_value_linear, std_err = linregress(prox_avg_pmaps_flat_mm, avg_zscore_flat)
# # ax_right0.scatter(all_dists_from_max_mm,all_zscores_delta_corr_flat, s=10)
# # #ax_right0.scatter(prox_avg_pmaps_flat_mm, avg_zscore_flat, color = 'black')
# # # ax_right0.text(min(all_dists_from_max_mm),min(all_zscores_delta_corr_flat)+0.2,
# # #                f'Correlation coefficient = {correlation_coefficient}, P-value = {p_value}')
# # # ax_right0.text(min(all_dists_from_max_mm),min(all_zscores_delta_corr_flat),
# # #                f'Correlation coefficient of avg = {correlation_coefficient_avg}, P-value of avg = {p_value_avg}')
# # ax_right0.set_title(f'Correlation coefficient = {correlation_coefficient}, P-value = {p_value} '
# #                     f'Correlation coefficient of avg = {correlation_coefficient_avg}, P-value of avg = {p_value_avg}')
# # ax_right0.set_xlabel('Proximity to max z-score [mm]')
# # ax_right0.set_ylabel('Z-score of delta corr [A.U.]')
#
#
#
# # Define a threshold
# threshold = 0.05
# special_color = 'red'  # Color for values below the threshold
#
# # Define the colormap and modify it
# base_cmap = plt.cm.viridis  # Base colormap (e.g., viridis)
# new_colors = base_cmap(np.linspace(0, 1, 256))  # Extract colors from the base colormap
#
# # Replace all colors below the threshold with the special color
# special_rgb = to_rgba(special_color)  # Convert the special color to RGBA
# new_colors[:int(threshold * 256)] = special_rgb  # Replace colors below the threshold
#
# # Create the new colormap
# custom_cmap = ListedColormap(new_colors)
#
#
#
#
# ax_right01 = f.add_subplot(gs[0, 2])
# corrected_p_values[functional_cortex_mask==0] = None
# im, _ = wf_imshow(ax_right01, corrected_p_values, mask=None, map=None, conv_ker=None, show_cb=False, cm_name=custom_cmap, cb_side='left')
# cbar = plt.colorbar(im, ax=ax_right01, label=f'corrected pv of t-test comparing NF and Control')
# #im.set_clim(np.nanmin(pmap_avg),np.nanmax(pmap_avg))
# im.set_clim(0,1)
# #plt.colorbar(label=f'Corr {sessions_vec[1]} - corr {sessions_vec[0]}')
# #ax_right01.set_ylabel("PSTR", fontsize=12)
# # ax_right01.scatter(avg_tuple_outline_x_control, avg_tuple_outline_y_control, marker='.', s=3, c='k')
# # ax_right01.scatter(max_index_control[1], max_index_control[0], marker='.', s=14, c='green') #it's plotted max_indices[1] and then
# #                                                                        # [0] because [1] is the column (x) and [0] is the row (y)
# #ax_right01.scatter(min_indices[:,1], min_indices[:,0], marker='.', s=14, c='green') #it's plotted max_indices[1] and then
#                                                                        # [0] because [1] is the column (x) and [0] is the row (y)
# #plt.title(f'Z-score of delta corr over mice:{mice_id}, sessions: {sessions_vec[1]}-{sessions_vec[0]}')
# ax_right01.axis('off')
#
#
# # ax_right02 = f.add_subplot(gs[0, 3])
# # #significant_mask[functional_cortex_mask==0] = np.nan
# # significant_mask = np.ma.masked_where(functional_cortex_mask == 0, significant_mask)
# # im, _ = wf_imshow(ax_right02, significant_mask, mask=None, map=None, conv_ker=None, show_cb=False, cm_name='Set3', cb_side='left')
# # cbar = plt.colorbar(im, ax=ax_right02, label=f'Z-score of delta corr {sessions_vec[1]} - corr {sessions_vec[0]}')
# # #im.set_clim(np.nanmin(pmap_avg),np.nanmax(pmap_avg))
# # im.set_clim(0,1)
# # #plt.colorbar(label=f'Corr {sessions_vec[1]} - corr {sessions_vec[0]}')
# # #ax_right02.set_ylabel("PSTR", fontsize=12)
# # # ax_right02.scatter(avg_tuple_outline_x_control, avg_tuple_outline_y_control, marker='.', s=3, c='k')
# # # ax_right02.scatter(max_index_control[1], max_index_control[0], marker='.', s=14, c='green') #it's plotted max_indices[1] and then
# # #                                                                        # [0] because [1] is the column (x) and [0] is the row (y)
# # #ax_right02.scatter(min_indices[:,1], min_indices[:,0], marker='.', s=14, c='green') #it's plotted max_indices[1] and then
# #                                                                        # [0] because [1] is the column (x) and [0] is the row (y)
# # #plt.title(f'Z-score of delta corr over mice:{mice_id}, sessions: {sessions_vec[1]}-{sessions_vec[0]}')
# # ax_right02.axis('off')
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # #ax_right0.scatter(all_dists_from_max_mm,all_delta_corr_flat)
# # ax_right01.scatter(prox_min_avg_pmaps_flat_mm, avg_zscore_flat,color = 'grey', alpha=0.3, s=10, edgecolor = 'none')
# # # coefficients = np.polyfit(prox_avg_pmaps_flat_mm, avg_zscore_flat, 1)
# # # sns.regplot(x=prox_avg_pmaps_flat_mm, y=avg_zscore_flat, ci=95, label=f'slope {coefficients[0]}', scatter_kws={'s': 2.5})
# # #slope, intercept, r_value, p_value_linear, std_err = linregress(prox_avg_pmaps_flat_mm, avg_zscore_flat)
# # ax_right01.scatter(all_dists_from_min_mm,all_zscores_delta_corr_flat, s=10)
# # #ax_right0.scatter(prox_avg_pmaps_flat_mm, avg_zscore_flat, color = 'black')
# # # ax_right0.text(min(all_dists_from_max_mm),min(all_zscores_delta_corr_flat)+0.2,
# # #                f'Correlation coefficient = {correlation_coefficient}, P-value = {p_value}')
# # # ax_right0.text(min(all_dists_from_max_mm),min(all_zscores_delta_corr_flat),
# # #                f'Correlation coefficient of avg = {correlation_coefficient_avg}, P-value of avg = {p_value_avg}')
# # ax_right01.set_title(f'Correlation coefficient = {correlation_coefficient_min}, P-value = {p_value_min} '
# #                     f'Correlation coefficient of avg = {correlation_coefficient_avg_min}, P-value of avg = {p_value_avg_min}')
# # ax_right01.set_xlabel('Proximity to min z-score [mm]')
# # ax_right01.set_ylabel('Z-score of delta corr [A.U.]')
#
#
#
#
#
# plt.show()
#
# # plt.rcParams['svg.fonttype'] = 'none'  # or 'path' or 'none'
# # plt.savefig(f'{base_path}/Figures_exp2.1/corr_NFandControl_with_pvmap_corrected_multiplecomparison'
# #             #f'{sessions_vec[1]}-'
# #             f'-{sessions_vec[0]}'
# #             f'.svg',format='svg',dpi=500)
