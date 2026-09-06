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
def convert_input(input_value, led_baseline, adj_led_analog_val_max):
    """Applies the same transformation as the Arduino LED logic."""
    input_value = np.clip(input_value, 0.0, 1.0)  # Ensure input is in [0,1]
    led_analog_val = (np.power(input_value, 2) + led_baseline) * adj_led_analog_val_max
    #led_analog_val = input_value * adj_led_analog_val_max
    speaker_value = (led_analog_val / adj_led_analog_val_max) * ((adj_led_analog_val_max/70)*10000 - 1000) + 1000
    return np.clip(led_analog_val, 0, 255),speaker_value  # Ensure valid PWM range




# base_path = '/data/Lena/WideFlow_prj'
base_path = DATA_STAGING_PATH  #added by Claude 20260906
mice_id = [
    #'21ML',
   #'31MN',
   #'54MRL',
   #'63MR',
    #'64ML',
   #'187FN',
   #'203MN',
    '204FR',
    #'206FRL',
    #'211MRR',
    #'218MN'
]

indexes_vec = [ #somatosensory target ROI
    #134,  # 21
    #105,  # 31
    # 85,  # 54
    # 52,  # 63
    #71,  # 64
     #56,  #187
    # 41,  #203
     69,  #204
    #  50,  #206
    #  53,  #211
     # 46  #218
    ]#(those are the indexes, the actual ROI numbers are this +1)[134, 105, 85, 52, 71 ]

line_styles = [
     #'solid',#21
    # 'solid',#31
    #  'solid',#54
    #  'solid',#63
    #  #'solid',#64
     'solid',#187
    #  'dashed',#203
    #  'solid',#204
    #  'dashed',#206
    #  'dashed',#211
    #  'dashed'#218
]



colors = [
     #'grey', #21
     #'cyan',#31
    #  'orange', #54
    #  'green',#63
    #  #'aquamarine',#64
    'purple', #187
    #  'chartreuse', #203
    #  'magenta',#204
    #  'blue',#206
    #  'red',#211
    #  'olivedrab'#218
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

#colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta','blue','red','olivedrab','grey','green','aquamarine'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'



crossing_rates = {}
plotting_allROIs = {}
for mouse_id, metric_index in zip(mice_id,indexes_vec):
    crossing_rates[f'{mouse_id}']={}
    plotting_allROIs[f'{mouse_id}'] = {}
    if mouse_id == '187FN' or mouse_id == '203MN' or mouse_id == '204FR' or mouse_id == '206FRL' or mouse_id == '211MRR' or mouse_id == '218MN':
        dates_vec = [
            '20241121', #spont
            # '20241126', #crc4
            #'20241129' #nf1
           # '20241130' #nf2
            # '20241201', #nf3
            #'20241202', #nf4
            #'20241203' #nf5
        ]
        sessions_vec = [
            'spont',
            # 'CRC4',
            #'NF1'
            #'NF2'
            # 'NF3',
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
             #'20230608', #crc4
            #'20230611' #nf1
            '20230612' #nf2
            #'20230613', #nf3
            #'20230614', #nf4
            #'20230615' #nf5
            ]
        sessions_vec = [
            #'spont_mockNF_NOTexcluded_closest',
              #'CRC4',
            # 'NF1'
            'NF2'
            #'NF3'
            #'NF4',
            #'NF5'
        ]
        #dates_vec = ['20230605','20230607','20230608']
        #sessions_vec = ['CRC1','CRC3','CRC4']
        #results_path = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'
        # results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
        results_path = DATA_STAGING_PATH + '/Results/results_exp2_noMH.h5'  #added by Claude 20260906

    for date, session_name in zip(dates_vec, sessions_vec):
        session_id = f'{date}_{mouse_id}_{session_name}'
        if session_name == 'CRC4' and mouse_id == '63MR':
            session_id = '20230607_63MR_CRC3'
        if session_name == 'CRC4' and mouse_id == '203MN':
            session_id = '20241125_203MN_CRC3'
        if session_name == 'CRC4' and mouse_id == '204FR':
            session_id = '20241125_204FR_CRC3'
        # if mouse_id =='31MN' or mouse_id=='54MRL':
        #     results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
        # else:
        #     results_path = '/data/Lena/WideFlow_prj/Results/results_exp2.1.h5'

        [timestamp, cue1, metric_result1, threshold1, serial_readout] = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')
        serial_readout_correct = [1-x for x in serial_readout]
        # # if mice_id.index(f'{mouse_id}') in control_indexes:
        # #     cue = [2 if x == 1 else 1 if x == 2 else x for x in cue]
        # if mouse_id=='203MN' or mouse_id=='206FRL' or mouse_id=='211MRR' or mouse_id=='218MN':
        #     cue = [2 if x == 1 else 1 if x == 2 else x for x in cue]

        data = {}
        #results_path = '/data/Lena/WideFlow_prj/Results/results_exp2.1.h5'
        if (mouse_id == '21ML' or mouse_id == '31MN' or mouse_id == '54MRL'
            or mouse_id == '63MR' or mouse_id == '64ML') and session_name == 'CRC4':
            # results_path = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'
            results_path = DATA_STAGING_PATH + '/Results/Results_exp2_CRC_sessions.h5'  #added by Claude 20260906
        with h5py.File(results_path, 'r') as f:
            decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/post_session_analysis_LK2/zsores_MH_diff5/')

        if (mouse_id == '187FN' or mouse_id == '203MN' or mouse_id == '204FR' or mouse_id == '206FRL'
                or mouse_id == '211MRR' or mouse_id == '218MN') and session_name == 'NF1':
            feedback_threshold = 1.0
        else:
            feedback_threshold = 4.2

        threshold = []
        #metric_result = np.repeat(data[f'roi_{metric_index+1}'], 2)
        metric_result = np.repeat(data[f'roi_{str(metric_index + 1).zfill(2)}'][:], 2)
        feedback_time = 0
        cue = np.zeros(len(metric_result))


        for frame_counter in range(0, len(metric_result)):
            if (metric_result[frame_counter] > feedback_threshold) and \
                    (timestamp[frame_counter] - feedback_time) * 1000 > 1000 and \
                    frame_counter > 1000:
                cue[frame_counter] = 1
                feedback_time = timestamp[frame_counter]

            if not frame_counter % 10 and frame_counter > 1000 and frame_counter < len(timestamp):
                # feedback_threshold = percentile_update_procedure(feedback_threshold,
                #         results_seq[np.min((0, frame_counter - threshold_eval_frames)):frame_counter: self.camera_config["attr"]["channels"]],
                #         threshold_percentile, threshold_nbins)
                feedback_threshold = binary_fixed_step_staircase_procedure(feedback_threshold, metric_result[np.max(
                    (0, frame_counter - 20000)):(frame_counter):2], 20000, 10, 2, 0)
            threshold.append(float(f'{feedback_threshold:.2f}'))

        # with h5py.File(results_path, 'r') as f:
        #     decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/post_session_analysis_LK2/diff5/')
        #     #decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/rois_traces/channel_0/')
        #
        ### correction for channel switch
        if (session_id == '20241129_204FR_NF1'
                or session_id == '20230604_21ML_spont_mockNF_NOTexcluded_closest' or session_id == '20230613_21ML_NF3'
                or session_id == '20241126_187FN_CRC4' or session_id == '20241126_218MN_CRC4' or session_id == '20241203_206FRL_NF5'
                or session_id == '20241201_211MRR_NF3' or session_id == '20241123_203MN_CRC1' or session_id == '20230613_21ML_NF3'
        ):
            crossing_rates[f'{mouse_id}']['cue'] = []
            crossing_rates[f'{mouse_id}']['converted_values'] = []
            crossing_rates[f'{mouse_id}']['metric_result'] = []
            crossing_rates[f'{mouse_id}']['metric_result1'] = []
            crossing_rates[f'{mouse_id}']['threshold'] = []
            crossing_rates[f'{mouse_id}']['threshold1'] = []
            crossing_rates[f'{mouse_id}']['delta'] = []
            crossing_rates[f'{mouse_id}']['filtered_delta'] = []
            continue
        # repeated_diff5_dict = {key: np.repeat(val, 2) for key, val in data.items()}

        led_baseline = 0
        adj_led_analog_val_max = 60
        # Step 1: Compute the transformation for each result/threshold pair
        scaled_values = (1 + np.clip(np.array(metric_result) / np.array(threshold), -1, 1)) / 2
        #scaled_values = (1 + np.clip(np.array(metric_result) , -1, 1)) / 2
        # # Compute the ratio
        # ratios = np.array(metric_result) / np.array(threshold)
        #
        # # Clip only negative values to -1, keep positive values unchanged
        # clipped_ratios = np.where(ratios < -1, -1, ratios)
        #
        # # Scale to range [0,1]
        # scaled_values = (1 + clipped_ratios) / 2

        # Step 2: Apply `convert_input` element-wise
        converted_values = np.array([convert_input(val, led_baseline, adj_led_analog_val_max) for val in scaled_values])[:,0]
        speaker_values = np.array([convert_input(val, led_baseline, adj_led_analog_val_max) for val in scaled_values])[:,1]
        sorted_speaker_values = np.sort(speaker_values)
        delta = metric_result-threshold
        filtered_delta = delta[delta>0]
        crossing_rates[f'{mouse_id}']['cue'] = cue
        crossing_rates[f'{mouse_id}']['converted_values'] = converted_values
        crossing_rates[f'{mouse_id}']['speaker_values'] = speaker_values
        crossing_rates[f'{mouse_id}']['sorted_speaker_values'] = sorted_speaker_values
        crossing_rates[f'{mouse_id}']['metric_result'] = metric_result
        crossing_rates[f'{mouse_id}']['metric_result1'] = metric_result1
        crossing_rates[f'{mouse_id}']['threshold'] = threshold
        crossing_rates[f'{mouse_id}']['threshold1'] = threshold1
        crossing_rates[f'{mouse_id}']['delta'] = delta
        crossing_rates[f'{mouse_id}']['filtered_delta'] = filtered_delta


# valid_arrays_NF = [
#     np.array(crossing_rates[key]['threshold'][7000:])
#     for key in [
#         #'21ML',
#         '31MN', '54MRL', '63MR',
#         #'64ML',
#         '187FN', '204FR']
#     if len(crossing_rates[key]['threshold'][7000:]) > 0
# ]

# mean_thresh_NF = np.mean(valid_arrays_NF, axis=0) if valid_arrays_NF else None


# mean_thresh_NF = np.mean([
#     np.array(crossing_rates['21ML']['threshold'][7000:]),
#     np.array(crossing_rates['31MN']['threshold'][7000:]),
#     np.array(crossing_rates['54MRL']['threshold'][7000:]),
#     np.array(crossing_rates['63MR']['threshold'][7000:]),
#     np.array(crossing_rates['64ML']['threshold'][7000:]),
#     np.array(crossing_rates['187FN']['threshold'][7000:]),
#     np.array(crossing_rates['204FR']['threshold'][7000:]),
# ], axis=0)


# valid_arrays_control = [
#     np.array(crossing_rates[key]['threshold'][7000:])
#     for key in ['203MN', '206FRL', '211MRR', '218MN']
#     if len(crossing_rates[key]['threshold'][7000:]) > 0
# ]
#
# mean_thresh_control = np.mean(valid_arrays_control, axis=0) if valid_arrays_control else None

# mean_thresh_control = np.mean([
#     np.array(crossing_rates['203MN']['threshold'][7000:]),
#     np.array(crossing_rates['206FRL']['threshold'][7000:]),
#     np.array(crossing_rates['211MRR']['threshold'][7000:]),
#     np.array(crossing_rates['218MN']['threshold'][7000:]),
# ], axis=0)

for color,mouse_id1,line in zip(colors,mice_id,line_styles):
    #plt.figure()
    #plt.plot(17*np.array(crossing_rates[f'{mouse_id1}']['cue']), color='black')
    #plt.plot( np.array(crossing_rates[f'{mouse_id1}']['cue'])[np.array(crossing_rates[f'{mouse_id1}']['cue']) != 2], color='black')
    # # plt.plot(17*np.array(crossing_rates[f'{mouse_id1}']['cue'])[17*np.array(crossing_rates[f'{mouse_id1}']['cue']) == 17*2],
    # #          color='purple')
    #plt.plot(crossing_rates[f'{mouse_id1}']['converted_values'], alpha=0.7)
    #plt.plot(np.array(crossing_rates[f'{mouse_id1}']['metric_result']), color='red', alpha=0.7)
    #plt.plot(np.array(crossing_rates[f'{mouse_id1}']['metric_result1']), color='green', alpha=0.7)
    #plt.plot((1/9000)*np.array(crossing_rates[f'{mouse_id1}']['speaker_values']),color = 'purple', alpha=0.7)
    plt.plot(np.array(crossing_rates[f'{mouse_id1}']['sorted_speaker_values']), color='purple', alpha=0.7)
    # plt.plot(np.array(crossing_rates[f'{mouse_id1}']['threshold']), color='green', alpha=0.7)
    #plt.plot(np.array(crossing_rates[f'{mouse_id1}']['threshold1']), color='purple', alpha=0.4)
    #plt.plot(np.array(crossing_rates[f'{mouse_id1}']['threshold'][7000:]), linestyle=line ,label=f'{mouse_id1}', color=color)
    #plt.plot(np.array(crossing_rates[f'{mouse_id1}']['metric_result1']), color='green', alpha=0.7)
    #plt.scatter(np.array(crossing_rates[f'{mouse_id1}']['metric_result']),crossing_rates[f'{mouse_id1}']['converted_values'], c = np.array(crossing_rates[f'{mouse_id1}']['threshold']))


    #plt.plot(np.array(crossing_rates[f'{mouse_id1}']['filtered_delta']))

    # plt.title(f'{mouse_id1}')
    plt.title(f'{session_name}')

# plt.plot(mean_thresh_NF, color='black',linewidth=3.5)
# plt.plot(mean_thresh_control, color='black',linestyle='dashed',linewidth=3.5)
plt.legend()
plt.show()