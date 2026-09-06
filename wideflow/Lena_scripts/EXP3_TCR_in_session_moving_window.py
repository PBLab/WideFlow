import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from skimage.morphology import skeletonize
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import mixedlm
from Imaging.utils.adaptive_staircase_procedure import percentile_update_procedure, \
    binary_fixed_step_staircase_procedure

from wideflow.analysis.utils.extract_from_metadata_file import extract_from_metadata_file
from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict
from utils.paint_roi import paint_roi
from utils.load_rois_data import load_rois_data
from analysis.plots import *


from wideflow.config import BASE_PATH, DATA_STAGING_PATH  #added by Claude 20260906
def count_threshold_crossings(vector, threshold, window_size, step_size):
    """
    Counts the rate of threshold crossings in a moving window.

    Parameters:
    - vector: 1D array of values.
    - threshold: The threshold value to detect crossings.
    - window_size: Number of elements in each window.
    - step_size: Step size for moving the window.

    Returns:
    - crossing_rates: List of crossing rates for each window.
    """

    crossing_rates = []

    # Iterate over windows with the given step size
    for i in range(0, len(vector) - window_size + 1, step_size):
        window = vector[i: i + window_size]  # Extract the window
        crossings = np.sum((window[:-1] < threshold) & (window[1:] >= threshold))  # Count upward crossings
        rate = crossings / 1500  # Normalize by 1500 frames ~1min
        crossing_rates.append(rate)

    return crossing_rates

def moving_average_custom(vector, window_size, step_size):
    """
    Computes a moving average with a specified window size and step size.

    Parameters:
    - vector: List or 1D NumPy array of values.
    - window_size: Number of elements in each window.
    - step_size: Step size for moving the window.

    Returns:
    - List of averaged values.
    """
    averages = []
    for i in range(0, len(vector) - window_size + 1, step_size):
        window = vector[i : i + window_size]
        averages.append(np.mean(window))  # Compute mean of the window
    return averages

def percentage_increasing(vector):
    vector = np.array(vector)  # Ensure it's a NumPy array
    increases = np.sum(np.diff(vector) > 0)  # Count positive differences
    total_changes = len(vector) - 1  # Total number of changes
    return (increases / total_changes) * 100 if total_changes > 0 else 0  # Convert to percentage



#base_path = '/data/Lena/WideFlow_prj'
# base_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj'
base_path = BASE_PATH  #added by Claude 20260906
mice_id = [
   # '21ML',
   # '31MN',
   # '54MRL',
   # '63MR',
   # '64ML',
   # '187FN',
   # '203MN',
   # '204FR',
   # '206FRL',
   # '211MRR',
   # '218MN'
    '226MR',
    '228MN',
    '232FN',
    '241FRLL'

]

indexes_vec = [ #somatosensory target ROI
   # 134,  # 21
   # 105,  # 31
   #  85,  # 54
   #  52,  # 63
   #  71,  # 64
   #  56,  #187
   #  41,  #203
   #  69,  #204
   #  50,  #206
   #  53,  #211
   #  46  #218
    '36_11',#226
    '47_44', #228
    '47_49',#232
    '68_53'#241
    ]#(those are the indexes, the actual ROI numbers are this +1)[134, 105, 85, 52, 71 ]

# line_styles = [
#     'solid',#21
#     'solid',#31
#     'solid',#54
#     'solid',#63
#     'solid',#64
#     'solid',#187
#     'dashed',#203
#     'solid',#204
#     'dashed',#206
#     'dashed',#211
#     'dashed'#218
# ]



colors = [
#     'grey', #21
#     'cyan',#31
#     'orange', #54
#     'green',#64
#     'aquamarine',#64
#     'purple', #187
#     'chartreuse', #203
    'magenta',#204
    'blue',#206
    'red',#211
    'olivedrab'#218
 ]



#NF_indexes = [0,1,2,3,4,5,7] #indexes of NF mice in mice vector
#NF_indexes = [0,1,2,4] #indexes of NF mice in mice vector
# NF_indexes = [0]
# #control_indexes = [6,8,9,10] #indexes of control mice in mice vector
# #control_indexes = [3,5,6,7] #indexes of control mice in mice vector
# control_indexes = [1]
#line_styles = ['solid','solid','solid','solid','solid','solid','dashed','solid','dashed','dashed','dashed']
#line_styles = ['solid','solid','solid','dashed','solid','dashed','dashed','dashed']
#line_styles = ['solid','dashed','solid','dashed','dashed','dashed']
#markers = ['o','o','o','H','o','H','H','H']
#facecolors = ['cyan', 'orange', 'purple','none', 'none','none','none']
#facecolors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta','blue','none','none','none','none']
# facecolors = [
#     #'cyan' #31
#     # 'orange' #54
#      'purple' #187
#     #'none' #203
#     #,'magenta' #204
#     # 'none' #206
#     #'none' #211
#     #'none' #218
# ]

    #['21ML','31MN','54MRL', '63MR', '64ML']
#colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta','blue','red','olivedrab','grey','green','aquamarine'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'
# sessions_vec = [
#     #'spont','CRC4',
#     'NF1'
# #    ,'NF2','NF3','NF4','NF5'
#         ]


crossing_rates = {}
plotting_allROIs = {}
for mouse_id, metric_index in zip(mice_id,indexes_vec):
    crossing_rates[f'{mouse_id}']={}
    plotting_allROIs[f'{mouse_id}'] = {}
    if mouse_id == '187FN' or mouse_id == '203MN' or mouse_id == '204FR' or mouse_id == '206FRL' or mouse_id == '211MRR' or mouse_id == '218MN':
        dates_vec = [
            #'20241121', #spont
            # '20241126', #crc4
            '20241129' #nf1
             #'20241130' #nf2
            # '20241201', #nf3
            #'20241202', #nf4
            #'20241203' #nf5
        ]
        sessions_vec = [
            #'spont',
            # 'CRC4',
            'NF1'
            #'NF2'
             #'NF3',
             #'NF4',
            #'NF5'
        ]
        # dates_vec = [
        #     #'20241121','20241126',
        # '20241123', '20241124', '20241125']
        #sessions_vec = ['CRC1', 'CRC2', 'CRC3']
        # spont_sess_length_frames = 60000
        # CRC_sess_length_frames = 60000
        # NF_sess_length_frames = 65000
        # dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/results_exp2.1.h5'
        # results_path = '/data/Lena/WideFlow_prj/Results/results_exp2.1.h5'
        results_path = DATA_STAGING_PATH + '/Results/results_exp2.1.h5'  #added by Claude 20260906
    else:
        dates_vec = [
            #'20230604', #spont
            # '20230608', #crc4
            '20250718' #nf1
            #'20230612' #nf2
            #'20230613', #nf3
            #'20230614', #nf4
            #'20230615' #nf5
            ]
        sessions_vec = [
         #   'spont_mockNF_NOTexcluded_closest',
            #   'CRC4',
             'NF1.3'
            #'NF2'
             #'NF3'
            #'NF4',
            #'NF5'
        ]
        #dates_vec = ['20230605','20230607','20230608']
        #sessions_vec = ['CRC1','CRC3','CRC4']
        #results_path = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'
        #results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
        # results_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp3.3.h5'
        results_path = BASE_PATH + '/Results/results_exp3.3.h5'  #added by Claude 20260906

    for date, session_name in zip(dates_vec, sessions_vec):
        session_id = f'{date}_{mouse_id}_{session_name}'
        # if session_name == 'CRC4' and mouse_id == '63MR':
        #     session_id = '20230607_63MR_CRC3'
        # if session_name == 'CRC4' and mouse_id == '203MN':
        #     session_id = '20241125_203MN_CRC3'
        # if session_name == 'CRC4' and mouse_id == '204FR':
        #     session_id = '20241125_204FR_CRC3'
        # if mouse_id =='31MN' or mouse_id=='54MRL':
        #     results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
        # else:
        #     results_path = '/data/Lena/WideFlow_prj/Results/results_exp2.1.h5'

        [timestamp, cue, metric_result, threshold, serial_readout, trial_number] = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')
        serial_readout_correct = [1-x for x in serial_readout]
        # if mice_id.index(f'{mouse_id}') in control_indexes:
        #     cue = [2 if x == 1 else 1 if x == 2 else x for x in cue]
        if mouse_id=='203MN' or mouse_id=='206FRL' or mouse_id=='211MRR' or mouse_id=='218MN':
            cue = [2 if x == 1 else 1 if x == 2 else x for x in cue]

        data = {}
        #results_path = '/data/Lena/WideFlow_prj/Results/results_exp2.1.h5'
        if (mouse_id == '31MN' or mouse_id == '54MRL') and session_name == 'CRC4':
            # results_path = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'
            results_path = DATA_STAGING_PATH + '/Results/Results_exp2_CRC_sessions.h5'  #added by Claude 20260906

        with h5py.File(results_path, 'r') as f:
            decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/post_session_analysis_LK2/zsores_MH_diff10_exc_top15/')
            #decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/rois_traces/channel_0/')

        ### correction for channel switch
        if (session_id == '20241129_204FR_NF1'
                or session_id == '20230604_21ML_spont_mockNF_NOTexcluded_closest' or session_id == '20230613_21ML_NF3'
                or session_id == '20241126_187FN_CRC4' or session_id == '20241126_218MN_CRC4' or session_id == '20241203_206FRL_NF5'
                or session_id == '20241201_211MRR_NF3' or session_id == '20241123_203MN_CRC1' or session_id == '20230613_21ML_NF3'
        ):
            crossing_rates[f'{mouse_id}'] = ((),())
            continue
        repeated_diff5_dict = {key: np.repeat(val, 2) for key, val in data.items()}

        set_threshold = 0.5  # Define threshold
        window_size = 3000 #(frames)  # Window size ###1500 frames is ~1min
        step_size = 750 #(frames)  # Move window by 1 step at a time

        #rates = count_threshold_crossings(np.array(metric_result), set_threshold, window_size, step_size)
        #rates = moving_average_custom(np.array(metric_result), window_size, step_size)
        rates = count_threshold_crossings(np.array(repeated_diff5_dict[f'roi_{metric_index}']), set_threshold, window_size, step_size)
        percent_up = percentage_increasing(rates)
        a=5
        crossing_rates[f'{mouse_id}'] = (list(range(0, len(rates))), rates)


#plt.figure(figsize=(4,8))
# for (label, (x, y)),color,linestyle in zip(crossing_rates.items(), colors,line_styles):
#for (label, (x, y)) in zip(crossing_rates.items()):
for (label, (x, y)) in crossing_rates.items():
    #plt.scatter(x, y, label=label, edgecolors=color, facecolors=FC)
    if not x or not y:  # Skip if x or y is empty
        continue
    # else:
    #     plt.plot(x,y, label=label, color=color, linestyle=linestyle)
    else:
        plt.plot(x,y, label=label)
#
# for (label, (x, y)),color, FC in zip(plotting_allROIs.items(), colors, facecolors):
#     plt.scatter(x, y, label=f'{label} all ROIs avg', edgecolors=color, facecolors=FC, alpha=0.3)


plt.legend()
plt.ylabel('crosses/min')
#plt.ylim(-0.02,0.05)
plt.title(f'{session_name} TCR for threshold={set_threshold}')
plt.show()

