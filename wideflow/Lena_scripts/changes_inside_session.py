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

from wideflow.config import DATA_STAGING_PATH  #added by Claude 20260906
# base_path = '/data/Lena/WideFlow_prj'
base_path = DATA_STAGING_PATH  #added by Claude 20260906
mice_id = [
   ## '21ML',
   '31MN',
   #'54MRL',
   ##'63MR',
   ##'64ML',
   #'187FN',
   #'203MN',
   #'204FR',
   #'206FRL',
   #'211MRR',
    #'218MN'
]

indexes_vec = [ #somatosensory target ROI
   #134,  # 21
   105,  # 31
   #85,  # 54
   #52,  # 63
   #71,  # 64
   #56,  #187
   #41,  #203
   #69,  #204
   #50,  #206
   #53,  #211
    #46  #218
    ]#(those are the indexes, the actual ROI numbers are this +1)[134, 105, 85, 52, 71 ]

facecolors = [
    'cyan', #31
    #'orange', #54
    #'purple', #187
    #'none', #203
    #'magenta', #204
    #'none', #206
    #'none', #211
    #'none' #218
]
colors = [
    'cyan',#31
    #'orange', #54
    #'purple', #187
    #'chartreuse', #203
    #'magenta',#204
    #'blue',#206
    #'red',#211
    #'olivedrab'#218
]



#NF_indexes = [0,1,2,3,4,5,7] #indexes of NF mice in mice vector
#NF_indexes = [0,1,2,4] #indexes of NF mice in mice vector
# NF_indexes = [0]
# #control_indexes = [6,8,9,10] #indexes of control mice in mice vector
# #control_indexes = [3,5,6,7] #indexes of control mice in mice vector
# control_indexes = [1]
#line_styles = ['solid','solid','solid','solid','solid','solid','dashed','solid','dashed','dashed','dashed']
line_styles = ['solid','solid','solid','dashed','solid','dashed','dashed','dashed']
#line_styles = ['solid','dashed','solid','dashed','dashed','dashed']
#markers = ['o','o','o','H','o','H','H','H']
#facecolors = ['cyan', 'orange', 'purple','none', 'none','none','none']
#facecolors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta','blue','none','none','none','none']

    #['21ML','31MN','54MRL', '63MR', '64ML']
#colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta','blue','red','olivedrab','grey','green','aquamarine'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'
# sessions_vec = [
#     #'spont','CRC4',
#     'NF1'
# #    ,'NF2','NF3','NF4','NF5'
#         ]
#sessions_vec = ['CRC1', 'CRC2', 'CRC3']

plotting = {}
plotting_allROIs = {}
for mouse_id, metric_index in zip(mice_id,indexes_vec):
    #set_threshold = set_threshold_vec[mice_id.index(mouse_id)]
    plotting[f'{mouse_id}']={}
    plotting_allROIs[f'{mouse_id}'] = {}
    if mouse_id == '187FN' or mouse_id == '203MN' or mouse_id == '204FR' or mouse_id == '206FRL' or mouse_id == '211MRR' or mouse_id == '218MN':
        dates_vec = [
            #'20241121', #spont
            # '20241126', #crc4
            #'20241129' #nf1
             #'20241130' #nf2
             #'20241201', #nf3
            #'20241202', #nf4
            '20241203' #nf5
        ]
        sessions_vec = [
            #'spont',
            # 'CRC4',
            #'NF1'
             #'NF2'
             #'NF3',
             #'NF4',
            'NF5'
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
            #'20230611' #nf1
            #'20230612' #nf2
            # '20230613', #nf3
            #'20230614', #nf4
            '20230615' #nf5
            ]
        sessions_vec = [
         #   'spont_mockNF_NOTexcluded_closest',
            #   'CRC4',
              # 'NF1'
            #'NF2'
             #'NF3'
            #'NF4',
            'NF5'
        ]
        #dates_vec = ['20230605','20230607','20230608']
        #sessions_vec = ['CRC1','CRC3','CRC4']
        #results_path = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'
        # results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
        results_path = DATA_STAGING_PATH + '/Results/results_exp2_noMH.h5'  #added by Claude 20260906

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

        [timestamp, cue, metric_result, threshold, serial_readout] = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')
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
            decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/post_session_analysis_LK2/diff5/')
            #decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/rois_traces/channel_0/')

        # ### correction for channel switch
        # if (session_id == '20241129_204FR_NF1'
        #         or session_id == '20230604_21ML_spont_mockNF_NOTexcluded_closest' or session_id == '20230613_21ML_NF3'
        #         or session_id == '20241126_187FN_CRC4' or session_id == '20241126_218MN_CRC4' or session_id == '20241203_206FRL_NF5'
        #         or session_id == '20241201_211MRR_NF3' or session_id == '20241123_203MN_CRC1' or session_id == '20230613_21ML_NF3'
        # ):
        #     threshold = np.tile(threshold[20000:22000], (len(timestamp) // 2000 + 1))[:len(timestamp)]
        #     cue = np.tile(cue[20000:22000], (len(timestamp) // 2000 + 1))[:len(timestamp)]
        #     metric_result = np.tile(metric_result[20000:22000], (len(timestamp) // 2000 + 1))[:len(timestamp)]
        #     for (key, value) in data.items():
        #         data[key] = np.tile(data[key][20000:22000], ((len(timestamp)//2) // 2000 + 1))[:(len(timestamp)//2)]
        #crosses = np.zeros(len(cue))



        # crosses = np.full(len(cue), np.nan)
        # crosses_allROIs = np.full(len(cue), np.nan)
        crosses = []
        crosses_allROIs = []
        repeated_diff5_dict = {key: np.repeat(val, 2) for key, val in data.items()}
        indices_cues = np.where(np.array(cue) == 1)[0]
        for i in indices_cues:
            # crosses[i] = repeated_diff5_dict[f'roi_{str(metric_index + 1).zfill(2)}'][i]
            # crosses_allROIs[i] = sum(vec[i] for vec in repeated_diff5_dict.values()) / len(repeated_diff5_dict)
            crosses.append(repeated_diff5_dict[f'roi_{str(metric_index + 1).zfill(2)}'][i])
            crosses_allROIs.append( sum(vec[i] for vec in repeated_diff5_dict.values()) / len(repeated_diff5_dict))
        #crosses = [repeated_diff5_dict[f'roi_{str(metric_index + 1).zfill(2)}'][i] for i, v in enumerate(cue) if v == 1]
        #crosses[crosses == 0] = np.nan
        # plotting[f'{mouse_id}'] = (list(range(0,len(crosses))), crosses)
        # plotting_allROIs[f'{mouse_id}'] = (list(range(0,len(crosses_allROIs))), crosses_allROIs)
        plotting[f'{mouse_id}'] = (indices_cues, crosses)
        plotting_allROIs[f'{mouse_id}'] = (indices_cues, crosses_allROIs)
        a=5
#x = list(range(0,len(crosses)))
# plt.plot(x, crosses, marker='o')
# plt.plot([1,2,3],[0.01,0.01,0.01], marker='s')
plt.figure(figsize=(4,8))
for (label, (x, y)),color, FC in zip(plotting.items(), colors, facecolors):
    plt.scatter(x, y, label=label, edgecolors=color, facecolors=FC)

for (label, (x, y)),color, FC in zip(plotting_allROIs.items(), colors, facecolors):
    plt.scatter(x, y, label=f'{label} all ROIs avg', edgecolors=color, facecolors=FC, alpha=0.3)

for (label, (xt, yt)),(label1,(xa,ya)),color, FC in zip(plotting.items(),plotting_allROIs.items(), colors, facecolors):
    plt.plot(x, abs(np.array(yt))-abs(np.array(ya)), label=f'{label} delta', color='red')#edgecolors='red', facecolors='red')


plt.legend()
plt.ylim(-0.02,0.05)
plt.xlim(0,65000)
plt.title(f'{session_name}')
plt.show()








        #
        # if (mouse_id == '187FN' or mouse_id == '203MN' or mouse_id == '204FR' or mouse_id == '206FRL'
        #         or mouse_id == '211MRR' or mouse_id == '218MN') and session_id == 'NF1':
        #     feedback_threshold = 1.0
        # else:
        #     feedback_threshold = 2.8
        #
        # threshold = []
        # #metric_result = np.repeat(data[f'roi_{metric_index+1}'], 2)
        # metric_result = np.repeat(data[f'roi_{str(metric_index + 1).zfill(2)}'], 2)
        # feedback_time = 0
        # cue = np.zeros(len(metric_result))
        #
        #
        # for frame_counter in range(0, len(timestamp)):
        #     if (metric_result[frame_counter] > feedback_threshold) and \
        #             (timestamp[frame_counter] - feedback_time) * 1000 > 1000 and \
        #             frame_counter > 1000:
        #         cue[frame_counter] = 1
        #         feedback_time = timestamp[frame_counter]
        #
        #     if not frame_counter % 10 and frame_counter > 1000 and frame_counter < len(timestamp):
        #         # feedback_threshold = percentile_update_procedure(feedback_threshold,
        #         #         results_seq[np.min((0, frame_counter - threshold_eval_frames)):frame_counter: self.camera_config["attr"]["channels"]],
        #         #         threshold_percentile, threshold_nbins)
        #         feedback_threshold = binary_fixed_step_staircase_procedure(feedback_threshold, metric_result[np.max(
        #             (0, frame_counter - 20000)):(frame_counter):2], 20000, 10, 2, 0.02)
        #     threshold.append(float(f'{feedback_threshold:.2f}'))
        #
        #
        # ### correction for channel switch
        # if (session_id == '20241129_204FR_NF1'
        #         or session_id == '20230604_21ML_spont_mockNF_NOTexcluded_closest' or session_id == '20230613_21ML_NF3'
        #         or session_id == '20241126_187FN_CRC4' or session_id == '20241126_218MN_CRC4' or session_id == '20241203_206FRL_NF5'
        #         or session_id == '20241201_211MRR_NF3' or session_id == '20241123_203MN_CRC1' or session_id == '20230613_21ML_NF3'
        # ):
        #     threshold = np.tile(threshold[20000:22000], (len(timestamp) // 2000 + 1))[:len(timestamp)]
        #     cue = np.tile(cue[20000:22000], (len(timestamp) // 2000 + 1))[:len(timestamp)]
        #     metric_result = np.tile(metric_result[20000:22000], (len(timestamp) // 2000 + 1))[:len(timestamp)]
        # #     threshold = np.full(len(threshold), np.nan)
        # #     cue = np.full(len(threshold), np.nan)
        # #     metric_result = np.full(len(threshold), np.nan)
        #
        # repeated_traces_dict = {key: np.repeat(val, 2) for key, val in data.items()}
        # cue_indices = np.where(cue == 1)[0]
        # average_dict1[f'{mouse_id}'][f'{session_name}'] = {}
        # for key1 in repeated_traces_dict.keys():
        #     # Select only relevant indices and compute mean
        #     selected_values = [repeated_traces_dict[key1][k] for k in cue_indices]
        #     selected_mean = np.mean(selected_values)
        #     average_dict1[f'{mouse_id}'][f'{session_name}'][key1] = selected_mean