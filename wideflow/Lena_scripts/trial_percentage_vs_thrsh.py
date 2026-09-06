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

def convert_input(input_value, led_baseline, adj_led_analog_val_max):
    """Applies the same transformation as the Arduino LED logic."""
    input_value = np.clip(input_value, 0.0, 1.0)  # Ensure input is in [0,1]
    led_analog_val = (np.power(input_value, 8) + led_baseline) * adj_led_analog_val_max
    return np.clip(led_analog_val, 0, 255)  # Ensure valid PWM range




# base_path = '/data/Lena/WideFlow_prj'
base_path = '/storage/DataCrunching/Lena'

# base_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj'

mice_id = [
    #'21ML',
   #'31MN',
  # '54MRL',
   #'63MR',
    #'64ML',
   #'187FN',
   #'203MN',
    #'204FR',
    #'206FRL',
   #'211MRR',
    #'218MN'
  #'226MR',
  # '228MN'
  #'232FN'
 #'241FRLL'
   # '245FRL'
    #'246FN'
    #'248FL'
    #'252MR'
    #'256FLL'
   # '257FR'
   #  '228MN',
   #  '258FL',
   #     '259FRL',
   #    '260FN',
   #   '261MR',
   #   '263MRL',
   #  '266FR',
   #  '276FL',
   #  '277FRL',
   #  '281MRL'
   #  '322MR'
   #  '327FL'
   #  '329FRR'
    '331FN'
]

indexes_vec = [ #somatosensory target ROI
    #134,  # 21
    #105,  # 31
    # 85,  # 54
    # 52,  # 63
    #71,  # 64
     #56,  #187
    # 41,  #203
    #69,  #204
    #  50,  #206
      #53,  #211
     # 46  #218
    #35, #226
    #46, #228
   # 40, #229
   #46, #232
    #67, #241
    #'36_11', #226
    #'47_44', #228
    #'47_49', #232
    #'68_53' #241
    #'90_71' #245
    #'60_52' #246
    #'35_36' #248
    #'52_67' #252
    #'23_27' #256
    #'68_75' #257

     #'14_17' #245 roi2
    #'34_38' #246 roi2
     #'19_39' #248 roi2
    # '16_17' #252 roi2
    # '69_67' #256 roi2
    #'22_21' #257 roi2

    # '66_45'# '228MN' roi1
   #'66_42' # '258FL' roi1
     #'62_45'# '259FRL' roi1
     #'89_87'# '260FN' roi1
    #'26_16'# '261MR' roi1
    #'67_50'# '263MRL' roi1

    # '50_40'  # '228MN' roi2
    #  '12_18' # '258FL' roi2
    #   '26_18'# '259FRL' roi2
    #   '64_49'# '260FN' roi2
    #  '76_110'# '261MR' roi2
    #  '40_44'# '263MRL' roi2
    # '62_43', #266 roi1
    # '78_80', #276 roi1
    # '131_118', #277 roi1
    # '48_42' #281 roi1

    # '13_40' #266 roi2
    # '37_20' #276 roi2
    # '20_36' #277 roi2
    # '14_25' #281 roi2

    # '30_37' #322 roi1
    # '16_17' #327 roi1
    # '07_08' #329 roi1
    # '22_34' #331 roi1

    # '10_14' #322 roi2
    # '20_25' #327 roi2
    # '35_44' #329 roi2
    '31_39' #331 roi2
     ]#(those are the indexes, the actual ROI numbers are this +1)[134, 105, 85, 52, 71 ]

line_styles = [
     #'solid',#21
    # 'solid',#31
    #  'solid',#54
    #  'solid',#63
    #  #'solid',#64
    # 'solid',#187
    #  'dashed',#203
    #  'solid',#204
    #  'dashed',#206
      'dashed',#211
    #  'dashed'#218
]



colors = [
     #'grey', #21
     #'cyan',#31
     # 'orange', #54
    #  'green',#63
    #  #'aquamarine',#64
    #'purple', #187
    #  'chartreuse', #203
    #  'magenta',#204
    #  'blue',#206
     'red',#211
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

session_last_frame = 19000

data_all_mice = {}
#plotting_allROIs = {}
for mouse_id, metric_index in zip(mice_id,indexes_vec):
    data_all_mice[f'{mouse_id}']={}
    #plotting_allROIs[f'{mouse_id}'] = {}
    if mouse_id == '266FR' or mouse_id == '276FL' or mouse_id == '277FRL' or mouse_id == '281MRL' :
        dates_vec = [
            # '20251202', #spont
            # '20251205', #nf1
            '20251208' #nf2


        ]
        sessions_vec = [
            'NF2_p3',
            # 'CRC4',
            # 'NF1'
            #'NF2'
            # 'NF3',
            #'NF4',
            #'NF5'
        ]

        results_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp4_parc_ROI2.h5'
    elif mouse_id == '187FN' or mouse_id == '203MN' or mouse_id == '204FR' or mouse_id == '206FRL' or mouse_id == '211MRR' or mouse_id == '218MN':
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
        results_path = '/data/Lena/WideFlow_prj/Results/results_exp2.1.h5'
    else:
        dates_vec = [
            #'20230604', #spont
            #'20230608', #crc4
            #'20230611' #nf1
            #'20230612' #nf2
            #'20230613', #nf3
            #'20230614', #nf4
            #'20230615' #nf5
            #'20250608'
            #'20250610'
            #'20250612'
            #'20250622'
            #'20250718'
            #'20250727'
            #'20250728'
            #'20250731'
            # '20250905'
            '20260607'
            ]
        sessions_vec = [
           # 'spont_mockNF_NOTexcluded_closest',
            #'CRC4',
            # 'NF1'
            #'NF2'
            #'NF3'
            #'NF4',
            #'NF5'
            #'spont'
            #'CRC1'
            #'NF11_diff10'
            #'NF2.2'
            #'NF1.3'
            #'CRC1_p4'
            'NF1_p3'
            # 'spont_p3'
        ]
        #dates_vec = ['20230605','20230607','20230608']
        #sessions_vec = ['CRC1','CRC3','CRC4']
        #results_path = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'
        #results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
        #results_path = '/data/Lena/WideFlow_prj/Results/results_exp2.1.h5'
        #results_path = '/data/Lena/WideFlow_prj/Results/results_exp3.h5'
        #results_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp3.3.h5'
        #results_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp3.4.h5'
        # results_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp3.5.h5'
        # results_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp3.5_parc_NEW2.h5'
        results_path = '/storage/DataCrunching/Lena/Results/results_exp4.1_parc_ROI2.h5'
        #results_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp3.4_full_parcellations.h5'

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

        if session_id == '20250608_226MR_spont2':
            timestamp = np.arange(65000)
            cue1 = np.arange(65000)
            metric_result1 =  np.arange(65000)
            threshold1 =  np.arange(65000)
            serial_readout_correct = np.arange(65000)
        else:
            [timestamp1, cue1, metric_result1, threshold1, serial_readout, trial_number1] = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')
            serial_readout_correct = [1-x for x in serial_readout]
        # # if mice_id.index(f'{mouse_id}') in control_indexes:
        # #     cue = [2 if x == 1 else 1 if x == 2 else x for x in cue]
        # if mouse_id=='203MN' or mouse_id=='206FRL' or mouse_id=='211MRR' or mouse_id=='218MN':
        #     cue = [2 if x == 1 else 1 if x == 2 else x for x in cue]

        data = {}
        #results_path = '/data/Lena/WideFlow_prj/Results/results_exp2.1.h5'
        if (mouse_id == '21ML' or mouse_id == '31MN' or mouse_id == '54MRL'
            or mouse_id == '63MR' or mouse_id == '64ML') and session_name == 'CRC4':
            results_path = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'
        with h5py.File(results_path, 'r') as f:
            decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/post_session_analysis_LK2/zsores_MH_diff10_exc_top15/')

        # if (mouse_id == '187FN' or mouse_id == '203MN' or mouse_id == '204FR' or mouse_id == '206FRL'
        #         or mouse_id == '211MRR' or mouse_id == '218MN') and session_name == 'NF1':
        #     feedback_threshold = 1.0
        # else:
        #     feedback_threshold = 2.8

        #threshold = 3.8
        thresholds = np.arange(1, 28 + 0.1, 0.1)
        percentage_success = []
        #metric_result = np.repeat(data[f'roi_{metric_index+1}'], 2)
        #metric_result = np.repeat(data[f'roi_{str(metric_index + 1).zfill(2)}'][:], 2)
        metric_result = np.repeat(data[f'roi_{metric_index}'][:int(session_last_frame/2)], 2)
        max_trial_frames = 750
        timeout_rewarded = 200
        timeout_no_reward = 250




        # for frame_counter in range(0, len(metric_result)):
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
        for threshold in thresholds:
            #threshold = 3.8
            data_all_mice[f'{mouse_id}'][f'threshold={threshold}']={}
            feedback_time = 0
            cue = np.zeros(len(metric_result))
            trial_counter = 0
            frame_counter = 0
            trial_number = []

            timestamp = timestamp1[:session_last_frame]
            while frame_counter < len(timestamp):
                # --- Start of trial ---
                trial_counter += 1
                print(f"\n--- Starting Trial {trial_counter} at frame {frame_counter} ---")

                trial_start_frame = frame_counter
                reward_given = False

                for trial_frame in range(max_trial_frames):
                    if frame_counter >= len(timestamp):
                        break

                    frame_clock_start = timestamp[frame_counter]

                    result = metric_result[frame_counter]

                    # Check threshold crossing
                    if (metric_result[frame_counter] > threshold) and \
                        (frame_clock_start - feedback_time) * 1000 > 1000 and \
                        frame_counter > 1000:

                        cue[frame_counter] = 1
                        feedback_time=timestamp[frame_counter]
                        reward_given = True
                        print(f'>>> Reward given (NF) at frame {frame_counter}, result={result:.3f}')
                        timeout_duration = timeout_rewarded
                        frame_counter += 1
                        trial_number.append(trial_counter)
                        break  # Trial ends

                    frame_counter += 1
                    trial_number.append(trial_counter)

                # Trial ended without reward
                if not reward_given:
                    print(f'>>> No reward in Trial {trial_counter}')
                    timeout_duration = timeout_no_reward

                print(f"--- Ending Trial {trial_counter} at frame {frame_counter} ---")
                # Timeout period (no processing or reward)
                for _ in range(timeout_duration):
                    if frame_counter >= len(timestamp):
                        break
                    frame_clock_start = timestamp[frame_counter]
                    frame_counter += 1
                    trial_number.append(trial_counter)

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
                data_all_mice[f'{mouse_id}'][f'threshold={threshold}']['cue'] = []
                data_all_mice[f'{mouse_id}'][f'threshold={threshold}']['converted_values'] = []
                data_all_mice[f'{mouse_id}'][f'threshold={threshold}']['metric_result'] = []
                data_all_mice[f'{mouse_id}'][f'threshold={threshold}']['metric_result1'] = []
                data_all_mice[f'{mouse_id}'][f'threshold={threshold}']['threshold'] = []
                data_all_mice[f'{mouse_id}'][f'threshold={threshold}']['threshold1'] = []
                data_all_mice[f'{mouse_id}'][f'threshold={threshold}']['delta'] = []
                data_all_mice[f'{mouse_id}'][f'threshold={threshold}']['filtered_delta'] = []
                continue
            # repeated_diff5_dict = {key: np.repeat(val, 2) for key, val in data.items()}

            # led_baseline = 0
            # adj_led_analog_val_max = 15
            # # Step 1: Compute the transformation for each result/threshold pair
            # scaled_values = (1 + np.clip(np.array(metric_result) / np.array(threshold), -1, 1)) / 2
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
            # converted_values = np.array([convert_input(val, led_baseline, adj_led_analog_val_max) for val in scaled_values])
            # delta = metric_result-threshold
            # filtered_delta = delta[delta>0]
            data_all_mice[f'{mouse_id}'][f'threshold={threshold}']['cue'] = cue
            #data_all_mice[f'{mouse_id}'][f'threshold={threshold}']['converted_values'] = converted_values
            data_all_mice[f'{mouse_id}'][f'threshold={threshold}']['metric_result'] = metric_result
            # data_all_mice[f'{mouse_id}'][f'threshold={threshold}']['metric_result1'] = metric_result1
            data_all_mice[f'{mouse_id}'][f'threshold={threshold}']['threshold'] = threshold
            data_all_mice[f'{mouse_id}'][f'threshold={threshold}']['threshold1'] = threshold1
            data_all_mice[f'{mouse_id}'][f'threshold={threshold}']['trial_number'] = trial_number
            percentage_success.append(np.sum(cue)/trial_number[-1])
            #data_all_mice[f'{mouse_id}'][f'threshold={threshold}']['delta'] = delta
            #data_all_mice[f'{mouse_id}'][f'threshold={threshold}']['filtered_delta'] = filtered_delta

        data_all_mice[f'{mouse_id}']['percentage_success'] = percentage_success
        data_all_mice[f'{mouse_id}']['thresholds'] = thresholds

# valid_arrays_NF = [
#     np.array(data_all_mice[key]['threshold'][7000:])
#     for key in [
#         #'21ML',
#         '31MN', '54MRL', '63MR',
#         #'64ML',
#         '187FN', '204FR']
#     if len(data_all_mice[key]['threshold'][7000:]) > 0
# ]

# mean_thresh_NF = np.mean(valid_arrays_NF, axis=0) if valid_arrays_NF else None


# mean_thresh_NF = np.mean([
#     np.array(data_all_mice['21ML']['threshold'][7000:]),
#     np.array(data_all_mice['31MN']['threshold'][7000:]),
#     np.array(data_all_mice['54MRL']['threshold'][7000:]),
#     np.array(data_all_mice['63MR']['threshold'][7000:]),
#     np.array(data_all_mice['64ML']['threshold'][7000:]),
#     np.array(data_all_mice['187FN']['threshold'][7000:]),
#     np.array(data_all_mice['204FR']['threshold'][7000:]),
# ], axis=0)


# valid_arrays_control = [
#     np.array(data_all_mice[key]['threshold'][7000:])
#     for key in ['203MN', '206FRL', '211MRR', '218MN']
#     if len(data_all_mice[key]['threshold'][7000:]) > 0
# ]
#
# mean_thresh_control = np.mean(valid_arrays_control, axis=0) if valid_arrays_control else None

# mean_thresh_control = np.mean([
#     np.array(data_all_mice['203MN']['threshold'][7000:]),
#     np.array(data_all_mice['206FRL']['threshold'][7000:]),
#     np.array(data_all_mice['211MRR']['threshold'][7000:]),
#     np.array(data_all_mice['218MN']['threshold'][7000:]),
# ], axis=0)

for color,mouse_id1,line in zip(colors,mice_id,line_styles):
    # #plt.figure()
    # plt.plot(68*np.array(data_all_mice[f'{mouse_id1}'][f'threshold={threshold}']['cue']), color='black')
    # # plt.plot( 17*np.array(data_all_mice[f'{mouse_id1}']['cue'])[np.array(data_all_mice[f'{mouse_id1}']['cue']) != 2], color='black')
    # # # plt.plot(17*np.array(data_all_mice[f'{mouse_id1}']['cue'])[17*np.array(data_all_mice[f'{mouse_id1}']['cue']) == 17*2],
    # # #          color='purple')
    # # plt.plot(data_all_mice[f'{mouse_id1}']['converted_values'], alpha=0.7)
    # plt.plot(np.array(data_all_mice[f'{mouse_id1}'][f'threshold={threshold}']['metric_result']), color='red', alpha=0.7)
    # #plt.plot(np.array(data_all_mice[f'{mouse_id1}']['metric_result1']), color='green', alpha=0.7)
    # plt.plot(np.array(data_all_mice[f'{mouse_id1}'][f'threshold={threshold}']['trial_number']), color='grey', alpha=0.7)

    #######plot succes rate vs thresholds
    x=np.array(data_all_mice[f'{mouse_id1}']['thresholds'])
    y=np.array(data_all_mice[f'{mouse_id1}']['percentage_success'])
    idx = np.argmin(np.abs(y - 0.3))
    # idx = np.argmin(np.abs(x - 3.9))
    x_point = x[idx]
    y_point = y[idx]
    plt.plot(x,y, color='grey', alpha=0.7)
    plt.plot(x_point, y_point, 'ro')  # red dot
    plt.text(x_point, y_point, f'({x_point:.2f}, {y_point:.2f})', fontsize=10, ha='left', va='bottom')
    plt.xlabel('threshold')
    plt.ylabel('percentage of success trials')

    # plt.plot(np.array(data_all_mice[f'{mouse_id1}']['threshold']), color='green', alpha=0.7)
    #plt.plot(np.array(data_all_mice[f'{mouse_id1}']['threshold1']), color='purple', alpha=0.4)
    #plt.plot(np.array(data_all_mice[f'{mouse_id1}']['threshold'][7000:]), linestyle=line ,label=f'{mouse_id1}', color=color)
    #plt.plot(np.array(data_all_mice[f'{mouse_id1}']['metric_result1']), color='green', alpha=0.7)
    #plt.scatter(np.array(data_all_mice[f'{mouse_id1}']['metric_result']),data_all_mice[f'{mouse_id1}']['converted_values'], c = np.array(data_all_mice[f'{mouse_id1}']['threshold']))


    #plt.plot(np.array(data_all_mice[f'{mouse_id1}']['filtered_delta']))

    # plt.title(f'{mouse_id1}')
    plt.title(f'{mouse_id} {session_name}')

# plt.plot(mean_thresh_NF, color='black',linewidth=3.5)
# plt.plot(mean_thresh_control, color='black',linestyle='dashed',linewidth=3.5)
plt.legend()
plt.show()