from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM
from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict

from wideflow.config import BASE_PATH, DATA_STAGING_PATH  #added by Claude 20260906
# base_path_qnap = '/data/Lena/WideFlow_prj'
base_path_qnap = DATA_STAGING_PATH  #added by Claude 20260906
# base_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj'
base_path = BASE_PATH  #added by Claude 20260906
# results_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp4_parc_ROI1.h5'
results_path = BASE_PATH + '/Results/results_exp4_parc_ROI1.h5'  #added by Claude 20260906
#dates_vec = ['20250731','20250731','20250731', '20250801','20250801','20250801','20250802','20250802','20250802']
mice_id = [
    # '245FRL',
    # '246FN',
    # '228MN',
    # '259FRL',
    # '252MR',
    # '257FR',
    # '258FL',
    # '260FN',
    # '261MR',
    # '248FL',
    # '256FLL',
    # '263MRL'
    '266FR',
'276FL',
'277FRL',
'281MRL',

]
# control_indexes = [0,1,2,3]
# NF_indexes = [4,5,6,7,8]
control_indexes = []
NF_indexes = [0,1,2]

mice_and_roi = 'EXP4_ROI1_norm_to_NF1_LK2_metric'

colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta','green','blue','red','mediumslateblue','olive', 'dodgerblue','maroon'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'
# lines = ['dashed','dashed','dashed','dashed','solid','solid','solid','solid','solid','solid','solid','solid']
lines = ['solid','solid','solid','solid']
#sessions_vec = ['spont_mockNF_ROI2_excluded_closest','NF21', 'NF22', 'NF23', 'NF24', 'NF25']
sessions_vec = [#'NF1_p1', 'NF1_p2', 'NF1_p3',
                'NF2_p1', 'NF2_p2', 'NF2_p3',
                'NF3_p1', 'NF3_p2', 'NF3_p3',
                'NF4_p1', 'NF4_p2', 'NF4_p3',
                'NF5_p1', #'NF5_p2', 'NF5_p3',
    #             'NF6_p1', 'NF6_p2', 'NF6_p3',
    # 'NF7_p1', 'NF7_p2', 'NF7_p3',
    # 'NF8_p1', 'NF8_p2', 'NF8_p3',
    # 'NF9_p1', 'NF9_p2', 'NF9_p3',
    # 'NF10_p1', 'NF10_p2', 'NF10_p3',
    # 'NF11_p1', 'NF11_p2', 'NF11_p3',

]
    #,'NF_control_p1','NF_control_p2','NF_control_p3']#sessions_vec = ['spont_mockNF_ROI2_excluded_closest', 'NF1_mock_ROI2','NF2_mock_ROI2','NF3_mock_ROI2','NF4_mock_ROI2', 'NF5_mock_ROI2']
#sessions_vec = ['NF5', 'NF21_mock_ROI1','NF22_mock_ROI1','NF23_mock_ROI1','NF24_mock_ROI1', 'NF25_mock_ROI1']
#set_threshold = 0.5


success_rates = np.zeros((len(mice_id),len(sessions_vec)))

for mouse_id in mice_id:
    if mouse_id == '248FL' or mouse_id == '257FR':
        dates_vec = [#'20250731', '20250731', '20250731',
                     '20250801', '20250801', '20250801', '20250802', '20250802','20250802', '20250803', '20250803','20250803',
                      '20250804', '20250804','20250804', '20250805', '20250805','20250805']#,'20250806', '20250806','20250806']
        sessions_vec = [# 'NF1_p1', 'NF1_p2', 'NF1_p3',
                         'NF2_p1', 'NF2_p2', 'NF2_p3', 'NF3_p1', 'NF3_p2', 'NF3_p3', 'NF4_p1', 'NF4_p2', 'NF4_p3', 'NF5_p1',
                         'NF5_p2', 'NF5_p3' , 'NF6_p1', 'NF6_p2', 'NF6_p3']
            # ,'NF_control_p1','NF_control_p2','NF_control_p3'  ]
        # results_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp3.4.h5'
        results_path = BASE_PATH + '/Results/results_exp3.4.h5'  #added by Claude 20260906
    elif mouse_id == '252MR':
        dates_vec = [#'20250731', '20250731', '20250731',
                     '20250801', '20250801', '20250801', '20250802', '20250802', '20250802',
                     '20250804', '20250804', '20250804', '20250805', '20250805', '20250805','20250806', '20250806', '20250806']#,'20250810', '20250810', '20250810']
        sessions_vec = [#'NF1_p1', 'NF1_p2', 'NF1_p3',
                        'NF2_p1', 'NF2_p2', 'NF2_p3', 'NF3_p1', 'NF3_p2', 'NF3_p3',
                        'NF5_p1', 'NF5_p2', 'NF5_p3', 'NF6_p1', 'NF6_p2', 'NF6_p3','NF7_p1', 'NF7_p2', 'NF7_p3']#,'NF_control_p1','NF_control_p2','NF_control_p3']
        # results_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp3.4.h5'
        results_path = BASE_PATH + '/Results/results_exp3.4.h5'  #added by Claude 20260906

    elif mouse_id == '245FRL' or mouse_id=='246FN':
        dates_vec = [#'20250731', '20250731', '20250731',
                     '20250731', '20250731', '20250731', '20250801', '20250801', '20250801','20250801', '20250801', '20250801',
                     '20250803', '20250803','20250803', '20250804', '20250804','20250804']#,'20250805', '20250805','20250805']
        sessions_vec = [#'NF1_p1', 'NF1_p2', 'NF1_p3',
                        'NF1_p1', 'NF1_p2', 'NF1_p3', 'NF2_p1', 'NF2_p2', 'NF2_p3','NF2_p1', 'NF2_p2', 'NF2_p3', 'NF4_p1', 'NF4_p2', 'NF4_p3', 'NF5_p1',
                         'NF5_p2', 'NF5_p3']#,'NF_control_p1','NF_control_p2','NF_control_p3']
        # results_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp3.4.h5'
        results_path = BASE_PATH + '/Results/results_exp3.4.h5'  #added by Claude 20260906

    elif (mouse_id == '258FL' or mouse_id=='260FN' or mouse_id=='261MR' or mouse_id=='263MRL'
          or mouse_id=='228MN' or mouse_id=='259FRL'):
        dates_vec = [#'20250905','20250905','20250905',
                     '20250906','20250906','20250906','20250907','20250907','20250907','20250908'
                     , '20250908','20250908','20250909','20250909','20250909','20250910','20250910','20250910']#,'20250805', '20250805','20250805']
        sessions_vec = [# 'NF1_p1', 'NF1_p2', 'NF1_p3',
                         'NF2_p1', 'NF2_p2', 'NF2_p3', 'NF3_p1', 'NF3_p2', 'NF3_p3', 'NF4_p1', 'NF4_p2', 'NF4_p3', 'NF5_p1',
                         'NF5_p2', 'NF5_p3' , 'NF6_p1', 'NF6_p2', 'NF6_p3']
        # results_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp3.5.h5'
        results_path = BASE_PATH + '/Results/results_exp3.5.h5'  #added by Claude 20260906

    elif (mouse_id == '266FR' or mouse_id=='276FL' or mouse_id=='281MRL'):
        dates_vec = [#'20251205','20251205','20251205',
                     '20251208',#'20251208','20251208',
            '20251209',#'20251209','20251209',
            '20251210', #'20251210','20251210',
            '20251211',#'20251211','20251211',
            '20251212',#'20251212','20251212',
        '20251214',#'20251214','20251214',
            '20251215',#'20251215','20251215'
        '20251216',#'20251216','20251216',
            '20251217',#'20251217','20251217',
            '20251218',#'20251218','20251218'
                 ]
        sessions_vec = [#'NF1_p1', 'NF1_p2', 'NF1_p3',
            'NF2_p1', #'NF2_p2', 'NF2_p3',
            'NF3_p1', #'NF3_p2', 'NF3_p3',
            'NF4_p1', #'NF4_p2', 'NF4_p3',
            'NF5_p1',#'NF5_p2', 'NF5_p3' ,
            'NF6_p1', #'NF6_p2', 'NF6_p3',
            'NF7_p1', #'NF7_p2', 'NF7_p3',
            'NF8_p1', #'NF8_p2', 'NF8_p3',
            'NF9_p1', #'NF9_p2', 'NF9_p3',
            'NF10_p1', #'NF10_p2', 'NF10_p3',
            'NF11_p1', #'NF11_p2', 'NF11_p3',
        ]
        # results_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp4_parc_ROI1.h5'
        results_path = BASE_PATH + '/Results/results_exp4_parc_ROI1.h5'  #added by Claude 20260906

    elif (mouse_id == '277FRL'):
        dates_vec = [#'20251205','20251205','20251205',
            '20251208',#'20251208','20251208',
            '20251209',#'20251209','20251209',
            '20251210', #'20251210','20251210',
            '20251211',#'20251211','20251211',
            '20251212',#'20251212','20251212',
        # '20251214',#'20251214','20251214',
            '20251215',#'20251215','20251215'
        '20251216',#'20251216','20251216',
            '20251217',#'20251217','20251217',
            '20251218',#'20251218','20251218'
            '20251219',  # '20251219','20251219'
             ]
        sessions_vec = [#'NF1_p1', 'NF1_p2', 'NF1_p3',
            'NF2_p1',  # 'NF2_p2', 'NF2_p3',
            'NF3_p1',  # 'NF3_p2', 'NF3_p3',
            'NF4_p1',  # 'NF4_p2', 'NF4_p3',
            'NF5_p1',  # 'NF5_p2', 'NF5_p3' ,
            'NF6_p1',  # 'NF6_p2', 'NF6_p3',
            # 'NF7_p1',  # 'NF7_p2', 'NF7_p3',
            'NF8_p1',  # 'NF8_p2', 'NF8_p3',
            'NF9_p1',  # 'NF9_p2', 'NF9_p3',
            'NF10_p1',  # 'NF10_p2', 'NF10_p3',
            'NF11_p1',  # 'NF11_p2', 'NF11_p3',
            'NF12_p1',  # 'NF12_p2', 'NF12_p3',
        ]
        # results_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp4_parc_ROI1.h5'
        results_path = BASE_PATH + '/Results/results_exp4_parc_ROI1.h5'  #added by Claude 20260906





    else:
        dates_vec = [#'20250731', '20250731', '20250731',
                     '20250731', '20250731', '20250731', '20250801', '20250801', '20250801', '20250802', '20250802','20250802',
                     '20250803', '20250803','20250803', '20250804', '20250804','20250804']#,'20250805', '20250805','20250805']
        sessions_vec = [#'NF1_p1', 'NF1_p2', 'NF1_p3',
                        'NF1_p1', 'NF1_p2', 'NF1_p3', 'NF2_p1', 'NF2_p2', 'NF2_p3', 'NF3_p1', 'NF3_p2', 'NF3_p3', 'NF4_p1', 'NF4_p2', 'NF4_p3', 'NF5_p1',
                         'NF5_p2', 'NF5_p3']#,'NF_control_p1','NF_control_p2','NF_control_p3']

    # for date, session_name in zip(dates_vec, sessions_vec):
    #     session_id = f'{date}_{mouse_id}_{session_name}'
    for sess_idx, (date, session_name) in enumerate(zip(dates_vec, sessions_vec)):
        session_id = f'{date}_{mouse_id}_{session_name}'
        if session_id == '20251211_266FR_NF5_p2':
            success_rate_sess = 15/26
            success_rates[mice_id.index(mouse_id), sess_idx] = success_rate_sess
            continue
        timestamp, cue, metric_result, threshold, serial_readout, trial_number = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')


        # if mouse_id=='245FRL' or mouse_id=='246FN' or mouse_id=='228MN' or mouse_id=='259FRL' or session_name == 'NF1_p1' or session_name== 'NF1_p2' or session_name== 'NF1_p3':
        #     #thresholds = [threshold[0]]
        if 1==1:
            if mouse_id=='245FRL':
                thresholds = [4.6]
            elif mouse_id == '246FN':
                thresholds = [3.6]
            elif mouse_id == '228MN':
                thresholds = [3.5]
            elif mouse_id == '259FRL':
                thresholds = [3.9]
            elif mouse_id == '258FL':
                thresholds = [3.9]
            elif mouse_id == '260FN':
                thresholds = [4.9]
            elif mouse_id == '261MR':
                thresholds = [3.5]
            elif mouse_id == '263MRL':
                thresholds = [4.2]
            elif mouse_id == '248FL':
                thresholds = [4.03]
            elif mouse_id == '252MR':
                thresholds = [4.1]
            elif mouse_id == '257FR':
                thresholds = [4.3]
            elif mouse_id == '256FLL':
                thresholds = [6.4]
            elif mouse_id == '266FR':
                thresholds = [3.6]
                metric_roi = ['roi_62_43']
            elif mouse_id == '276FL':
                thresholds = [6.6]
                metric_roi = ['roi_78_80']
            elif mouse_id == '277FRL':
                thresholds = [3.0]
                metric_roi = ['roi_131_118']
            elif mouse_id == '281MRL':
                thresholds = [5.0]
                metric_roi = ['roi_48_42']
            percentage_success = []
            max_trial_frames = 750
            timeout_rewarded = 200
            timeout_no_reward = 250

            data = {}
            with h5py.File(results_path, 'r') as f:
                decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/post_session_analysis_LK2/')

            metric_result = np.repeat((data['zsores_MH_diff10_exc_top15'][f'{metric_roi[0]}']),2)

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
                feedback_time = 0
                cue = np.zeros(len(metric_result))
                trial_counter = 0
                frame_counter = 0
                trial_number = []

                #timestamp = timestamp1[:session_last_frame]
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
                            feedback_time = timestamp[frame_counter]
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


        rewards = np.sum(cue)
        trials = trial_number[-1]-1
        success_rate_sess = rewards/trials

        #success_rates[mice_id.index(mouse_id), sessions_vec.index(session_name)] = success_rate_sess
        success_rates[mice_id.index(mouse_id), sess_idx] = success_rate_sess

# success_rates=success_rates.reshape(len(mice_id), 10, 3).mean(axis=2)
#
# first_column = success_rates[:, 0]
# # first_column = (NF_sess_length_frames/spont_sess_length_frames)*first_column
# success_rates[:,0] = first_column
# norm_success_rates = success_rates - first_column[:, np.newaxis]
# norm_success_rates = [[element / first_column[i] for element in row] for i, row in enumerate(norm_success_rates)]
norm_success_rates = success_rates
a=5



# #after = [np.max(crossings[i,-2:]) for i in range(len(mice_id))]
# after = [crossings[i,-3] for i in range(len(mice_id))]
# #after = [norm_crossings[i][-3] for i in range(len(mice_id))]
# #first_column1 = np.zeros(len(mice_id))
# #after = [0.2,0.1,0.5,1,0.2,0.2]
# #before = np.zeros(len(mice_id))
# t_statistic, p_value = stats.ttest_rel(first_column, after)

# df = pd.DataFrame({'Mice':np.repeat([21,31,54,63,64],6),'Sessions': np.tile([0,1,2,3,4,5],5),'Scores':[item for sublist in norm_crossings for item in sublist]})
# res = pg.rm_anova(dv = 'Scores', within = 'Sessions', subject = 'Mice', data = df)
# post_hocs = pg.pairwise_tests(dv='Scores', within='Sessions',subject='Mice', data=df)
# res_stats = AnovaRM(data = df, depvar = 'Scores', subject = 'Mice', within=['Sessions']).fit()
# posthoc_tukey = pairwise_tukeyhsd(df['Scores'], df['Sessions'])

############STATS WITH CONTROL GROUP
#norm_crossings_noSpont = [sublist[1:] for sublist in norm_success_rates]
control_norm_crossings = [norm_success_rates[i] for i in control_indexes]
NF_norm_crossings = [norm_success_rates[i] for i in NF_indexes]

# #control vs. nf
# subjects = list([245, 246, 228, 259, 252, 257, 258, 260, 261])
# #subjects = list([21,31,54,63,64,187,203,204,206,211,218])
# #groups = ['NF'] * 6 + ['control'] + ['NF'] +['control'] * 3  # Group labels
# groups = ['control'] * 4 + ['NF'] * 5  # Group labels
# #sessions = list(range(1,6)) # Session numbers (1 to 7)
# #sessions = list(range(1,4)) # Session numbers (1 to 5)
# sessions = list(range(1,6))
#
#
# # Create a long-format DataFrame
# long_data = []
# for subject, group, scores in zip(subjects, groups, norm_success_rates):
#     for session1, score1 in zip(sessions, scores):
#         long_data.append([subject, group, session1, score1])
#
# df_controlandNF = pd.DataFrame(long_data, columns=['Subject', 'Group', 'Session', 'Score'])
#
# # Convert columns to appropriate types
# df_controlandNF['Subject'] = df_controlandNF['Subject'].astype('category')
# df_controlandNF['Group'] = df_controlandNF['Group'].astype('category')
# df_controlandNF['Session'] = df_controlandNF['Session'].astype('category')
#
#
# anova = pg.mixed_anova(dv="Score", within="Session", between="Group", subject="Subject", data=df_controlandNF)
# # Fit a mixed-effects model
# # 'Score' is the dependent variable
# # 'Group' is a between-subject factor
# # 'Session' is a within-subject factor
# # model_controlandNF = mixedlm("Score ~ Group * Session", df_controlandNF, groups=df_controlandNF["Subject"], re_formula="~Session")
# # result_controlandNF = model_controlandNF.fit()
# #model_controlandNF = mixedlm("Score ~ Group * Session", df_controlandNF, groups="Subject")
# #result_controlandNF = model_controlandNF.fit(reml=True)
##############################################################################################################################


sum = []
for a,b,c in zip (norm_success_rates[0], norm_success_rates[1],norm_success_rates[2]):
    #,norm_success_rates[3]):#,
                        # norm_success_rates[8]):#,norm_success_rates[9]):#,norm_success_rates[10],norm_success_rates[11]):
    sum.append(a+b+c)

mean = [num /3 for num in sum]

##STD vector for NF group
data_NF_norm = np.array(NF_norm_crossings)             # shape (5, n)
std_vector_NF = np.std(data_NF_norm, axis=0) # std along rows for each column

# sum_control = []
# for a,b,c,d in zip (norm_success_rates[0], norm_success_rates[1], norm_success_rates[2], norm_success_rates[3]):
#     sum_control.append(a+b+c+d)
#
# mean_control = [num /4 for num in sum_control]
#
# ##STD vector for control group
# data_control_norm = np.array(control_norm_crossings)             # shape (5, n)
# std_vector_control = np.std(data_control_norm, axis=0) # std along rows for each column

sessions_vec1 = ['NF1','NF2','NF3','NF4','NF5','NF6','NF7','NF8','NF9','NF10']#,'Control']
##PLOT LINE PER MOUSE
for i, mouse_id1,color_name, linestyle in zip(norm_success_rates, mice_id,colors, lines):
    plt.plot(sessions_vec1, i, label = f'{mouse_id1}', color = color_name, linestyle = linestyle)

plt.plot(sessions_vec1, mean, color='black', linestyle='solid', label='Mean', linewidth=5)
# plt.plot(sessions_vec1, mean_control, color='grey', linestyle='--', label='Mean control', linewidth = 5)

# ##PLOT ONLY 2*STD PER GROUP
# plt.fill_between(
#     sessions_vec1,
#     mean - 2*std_vector_NF,
#     mean + 2*std_vector_NF,
#     color="black",
#     alpha=0.3,
#     label="±2 std"
# )
# plt.fill_between(
#     sessions_vec1,
#     mean_control - 2*std_vector_control,
#     mean_control + 2*std_vector_control,
#     color="grey",
#     alpha=0.3,
#     label="±2 std"
# )

plt.suptitle(f'{mice_and_roi}')
#plt.grid()
plt.legend()


plt.show()
#plt.savefig(f'{base_path_qnap}/figs_for_paper_EXP3.5/success_rates_over_sessions_{mice_and_roi}.svg',format='svg',dpi=200)



# print(anova)
#
# # Post-hoc: Interaction (Group × Session)
# print("\nPost-hoc: Group Comparisons for Each Session")
# for session in df_controlandNF["Session"].unique():
#     print(f"Session {session}:")
#     session_data = df_controlandNF[df_controlandNF["Session"] == session]
#     tukey_interaction = pg.pairwise_tukey(dv="Score", between="Group", data=session_data)
#     print(tukey_interaction)