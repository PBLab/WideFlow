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
   #'21ML',
    '31MN'
   ,'54MRL'
   #,'63MR'
    #,'64ML'
    ,'187FN'
    ,'203MN'
    ,'204FR'
    ,'206FRL'
    ,'211MRR'
    ,'218MN'
]

indexes_vec = [ #somatosensory target ROI
    #134,  # 21
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

# indexes_vec = [ #MOTOR
#    # 134  # 21
#      65  # 31
#     , 56  # 54
#     #, 52  # 63
#     #, 71  # 64
#     ,50  #187
#     ,52  #203
#     ,44  #204
#      ,51  #206
#      ,56  #211
#      ,52  #218
#     ]#(those are the indexes, the actual ROI numbers are this +1)[134, 105, 85, 52, 71 ]

# indexes_vec = [ #MOTOR2 (green dot)
#    # 134  # 21
#      91  # 31
#     , 67  # 54
#     #, 52  # 63
#     #, 71  # 64
#     ,50  #187
#     ,68  #203
#     ,44  #204
#      ,29  #206
#      ,64  #211
#      ,50  #218
#     ]#(those are the indexes, the actual ROI numbers are this +1)[134, 105, 85, 52, 71 ]

# indexes_vec = [ #retrosplenial
#    # 134  # 21
#      27  # 31
#     , 42  # 54
#     #, 52  # 63
#     #, 71  # 64
#     ,22  #187
#     ,11  #203
#     ,17  #204
#      ,12  #206
#      ,12  #211
#      ,15  #218
#     ]#(those are the indexes, the actual ROI numbers are this +1)[134, 105, 85, 52, 71 ]

# indexes_vec = [ #somatos. limb (10)
#    # 134  # 21
#      2  # 31
#     , 9  # 54
#     #, 52  # 63
#     #, 71  # 64
#     ,8  #187
#     ,2  #203
#     ,0  #204
#      ,14  #206
#      ,14  #211
#      ,1  #218
#     ]#(those are the indexes, the actual ROI numbers are this +1)[134, 105, 85, 52, 71 ]


#NF_indexes = [0,1,2,3,4,5,7] #indexes of NF mice in mice vector
NF_indexes = [0,1,2,4] #indexes of NF mice in mice vector
#NF_indexes = [0,2]
#control_indexes = [6,8,9,10] #indexes of control mice in mice vector
control_indexes = [3,5,6,7] #indexes of control mice in mice vector
#control_indexes = [1,3,4,5]
#line_styles = ['solid','solid','solid','solid','solid','solid','dashed','solid','dashed','dashed','dashed']
line_styles = ['solid','solid','solid','dashed','solid','dashed','dashed','dashed']
#line_styles = ['solid','dashed','solid','dashed','dashed','dashed']

    #['21ML','31MN','54MRL', '63MR', '64ML']
colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta','blue','red','olivedrab','grey','green','aquamarine'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'
sessions_vec = [
    #'spont','CRC4',
    'NF1','NF2','NF3','NF4','NF5']
#sessions_vec = ['CRC1', 'CRC2', 'CRC3']

average_dict1 = {}

score_values = np.zeros((len(mice_id),len(sessions_vec)))
for mouse_id, metric_index in zip(mice_id,indexes_vec):
    #set_threshold = set_threshold_vec[mice_id.index(mouse_id)]
    average_dict1[f'{mouse_id}']={}
    if mouse_id == '187FN' or mouse_id == '203MN' or mouse_id == '204FR' or mouse_id == '206FRL' or mouse_id == '211MRR' or mouse_id == '218MN':
        dates_vec = [
           # '20241121','20241126',
        '20241129', '20241130', '20241201', '20241202', '20241203']
        sessions_vec = [
            #'spont', 'CRC4',
            'NF1', 'NF2', 'NF3', 'NF4', 'NF5']
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
           # '20230604','20230608',
            '20230611', '20230612', '20230613', '20230614', '20230615']
        sessions_vec = [
            #'spont_mockNF_NOTexcluded_closest', 'CRC4',
            'NF1', 'NF2', 'NF3', 'NF4', 'NF5']
        #dates_vec = ['20230605','20230607','20230608']
        #sessions_vec = ['CRC1','CRC3','CRC4']
        #results_path = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'
        # results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
        results_path = DATA_STAGING_PATH + '/Results/results_exp2_noMH.h5'  #added by Claude 20260906


        #sessions_vec = ['spont_mockNF_NOTexcluded_closest','CRC4','NF1', 'NF2', 'NF3', 'NF4', 'NF5']
        # spont_sess_length_frames = 50000
        # CRC_sess_length_frames = 50000
        # NF_sess_length_frames = 65000
        # #dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'

    for date, session_name in zip(dates_vec, sessions_vec):
        session_id = f'{date}_{mouse_id}_{session_name}'
        if session_name == 'CRC4' and mouse_id == '63MR':
            session_id = '20230607_63MR_CRC3'
        if session_name == 'CRC4' and mouse_id == '203MN':
            session_id = '20241125_203MN_CRC3'
        if session_name == 'CRC4' and mouse_id == '204FR':
            session_id = '20241125_204FR_CRC3'
        if mouse_id =='31MN' or mouse_id=='54MRL' or mouse_id=='21ML' or mouse_id=='63MR' or mouse_id=='64ML'  :
            # results_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
            results_path = DATA_STAGING_PATH + '/Results/results_exp2_noMH.h5'  #added by Claude 20260906
        else:
            # results_path = '/data/Lena/WideFlow_prj/Results/results_exp2.1.h5'
            results_path = DATA_STAGING_PATH + '/Results/results_exp2.1.h5'  #added by Claude 20260906

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
            decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/post_session_analysis_LK2/zsores_MH_diff5/')

        if (mouse_id == '187FN' or mouse_id == '203MN' or mouse_id == '204FR' or mouse_id == '206FRL'
                or mouse_id == '211MRR' or mouse_id == '218MN') and session_name == 'NF1':
            feedback_threshold = 1.0
        else:
            feedback_threshold = 2.8

        threshold = []
        #metric_result = np.repeat(data[f'roi_{metric_index+1}'], 2)
        metric_result = np.repeat(data[f'roi_{str(metric_index + 1).zfill(2)}'], 2)
        feedback_time = 0
        cue = np.zeros(len(metric_result))


        for frame_counter in range(0, len(timestamp)):
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
                    (0, frame_counter - 20000)):(frame_counter):2], 20000, 10, 2, 0.02)
            threshold.append(float(f'{feedback_threshold:.2f}'))


        ### correction for channel switch
        if (session_id == '20241129_204FR_NF1'
                or session_id == '20230604_21ML_spont_mockNF_NOTexcluded_closest' or session_id == '20230613_21ML_NF3'
                or session_id == '20241126_187FN_CRC4' or session_id == '20241126_218MN_CRC4' or session_id == '20241203_206FRL_NF5'
                or session_id == '20241201_211MRR_NF3' or session_id == '20241123_203MN_CRC1' or session_id == '20230613_21ML_NF3'
        ):
            threshold = np.tile(threshold[20000:22000], (len(timestamp) // 2000 + 1))[:len(timestamp)]
            cue = np.tile(cue[20000:22000], (len(timestamp) // 2000 + 1))[:len(timestamp)]
            metric_result = np.tile(metric_result[20000:22000], (len(timestamp) // 2000 + 1))[:len(timestamp)]
        #     threshold = np.full(len(threshold), np.nan)
        #     cue = np.full(len(threshold), np.nan)
        #     metric_result = np.full(len(threshold), np.nan)

        # repeated_traces_dict = {key: np.repeat(val, 2) for key, val in data.items()}
        # cue_indices = np.where(cue == 1)[0]
        # average_dict1[f'{mouse_id}'][f'{session_name}'] = {}
        # for key1 in repeated_traces_dict.keys():
        #     # Select only relevant indices and compute mean
        #     selected_values = [repeated_traces_dict[key1][k] for k in cue_indices]
        #     selected_mean = np.mean(selected_values)
        #     average_dict1[f'{mouse_id}'][f'{session_name}'][key1] = selected_mean



        ind_sess = round((len(threshold))/2)
        #threshold_short = threshold[-ind_sess:]
        threshold_short = threshold[20000:]
        unq_thresh_short = sorted(set(threshold_short))
        ind = round((len(unq_thresh_short)) / 4)
        derivative = np.diff(threshold_short)
        crosses = [metric_result[20000+i] for i, v in enumerate(cue[20000:]) if v == 1]
        #crosses = [metric_result[i] for i, v in enumerate(cue) if v == 1]
        crosses_ind = round((len(crosses)) / 4)
        crosses_short = crosses[-crosses_ind:]
        #score = (np.sum(derivative > 0) / len(derivative)) * 100
        #score = np.mean(unq_thresh_short[-ind:]) #mean of top quarter values of threshold
        #max_thresh_sess = np.max(threshold_short)
        #score = np.max(threshold_short)
        #score = threshold_short[-1]
        #min_thresh = np.min(threshold_short)
        #score = max_thresh_sess-min_thresh #this is delta thresh per session, named max to not edit the whole script
        #score = np.mean(threshold[20000:]) #this is the mean thresh per session, named max to not edit the whole script
        score = np.mean(threshold_short)
        #score = np.mean (crosses)
        score_values[mice_id.index(mouse_id), sessions_vec.index(session_name)] = score
        title = 'Mean threshold value (starting from frame 20000 for each session)'

a=5

# # After the loop, replace the score for 204FR during NF1 with the mean of other NF mice scores
# nf1_index = sessions_vec.index('NF1')
# nf3_index = sessions_vec.index('NF3')
# nf5_index = sessions_vec.index('NF5')
# exclude_204FR_index = mice_id.index('204FR')
# exclude_211MRR_index = mice_id.index('211MRR')
# exclude_206FRL_index = mice_id.index('206FRL')
#
# # Get the scores of NF mice, excluding 204FR
# nf_scores_NF1 = [score_values[i, nf1_index] for i in NF_indexes if i != exclude_204FR_index]
# mean_nf_score_NF1 = np.mean(nf_scores_NF1)
# nf_scores_NF3 = [score_values[i, nf3_index] for i in control_indexes if i != exclude_211MRR_index]
# mean_nf_score_NF3 = np.mean(nf_scores_NF3)
# nf_scores_NF5 = [score_values[i, nf5_index] for i in control_indexes if i != exclude_206FRL_index]
# mean_nf_score_NF5 = np.mean(nf_scores_NF5)
#
# # Set the score for 204FR during NF1 to the mean of the other NF mice
# score_values[exclude_204FR_index, nf1_index] = mean_nf_score_NF1
# score_values[exclude_211MRR_index, nf3_index] = mean_nf_score_NF3
# score_values[exclude_206FRL_index, nf5_index] = mean_nf_score_NF5






# first_column = score_values[:, 0]
# norm_score_values = score_values - first_column[:, np.newaxis]
# norm_score_values = [[element / first_column[i] for element in row] for i, row in enumerate(norm_score_values)]

norm_score_values = score_values

# second_column = score_values[:, 1]
# norm_score_values = score_values - second_column[:, np.newaxis]
# norm_score_values = [[element / second_column[i] for element in row] for i, row in enumerate(norm_score_values)]


# ############STATS WITH CONTROL GROUP##################################################################################
control_norm_score_values = [norm_score_values[i] for i in control_indexes]
NF_norm_score_values = [norm_score_values[i] for i in NF_indexes]
#
# #NF group
# df_NF = pd.DataFrame({'Mice':np.repeat([21,31,54,63,64,187,204],5),'Sessions': np.tile([0,1,2,3,4],7),'Scores':[item for sublist in NF_norm_score_values for item in sublist]})
# #res_metric = pg.rm_anova(dv = 'Scores', within = 'Sessions', subject = 'Mice', data = df_NF)
# #post_hocs_metric = pg.pairwise_tests(dv='Scores', within='Sessions',subject='Mice', data=df_NF)
# res_stats_NF = AnovaRM(data = df_NF, depvar = 'Scores', subject = 'Mice', within=['Sessions']).fit()
# posthoc_tukey_NF = pairwise_tukeyhsd(df_NF['Scores'], df_NF['Sessions'])
#
# #control group
# df_control = pd.DataFrame({'Mice':np.repeat([203,206,211,218],5),'Sessions': np.tile([0,1,2,3,4],4),'Scores':[item for sublist in control_norm_score_values for item in sublist]})
# # res_control = pg.rm_anova(dv = 'Scores', within = 'Sessions', subject = 'Mice', data = df_control)
# # post_hocs_control = pg.pairwise_tests(dv='Scores', within='Sessions',subject='Mice', data=df_control)
# res_stats_control = AnovaRM(data = df_control, depvar = 'Scores', subject = 'Mice', within=['Sessions']).fit()
# posthoc_tukey_control = pairwise_tukeyhsd(df_control['Scores'], df_control['Sessions'])
#
#
#control vs. nf
subjects = list([31,54,187,203,204,206,211,218])
#subjects = list([21,31,54,63,64,187,203,204,206,211,218])
#groups = ['NF'] * 6 + ['control'] + ['NF'] +['control'] * 3  # Group labels
groups = ['NF'] * 3 + ['control'] + ['NF'] +['control'] * 3  # Group labels
sessions = list(range(1,6)) # Session numbers (1 to 7)
#sessions = list(range(1,4)) # Session numbers (1 to 7)
#sessions = list(range(1,8)) # Session numbers (1 to 7)


# Create a long-format DataFrame
long_data = []
for subject, group, scores in zip(subjects, groups, norm_score_values):
    for session1, score1 in zip(sessions, scores):
        long_data.append([subject, group, session1, score1])

df_controlandNF = pd.DataFrame(long_data, columns=['Subject', 'Group', 'Session', 'Score'])

# Convert columns to appropriate types
df_controlandNF['Subject'] = df_controlandNF['Subject'].astype('category')
df_controlandNF['Group'] = df_controlandNF['Group'].astype('category')
df_controlandNF['Session'] = df_controlandNF['Session'].astype('category')


anova = pg.mixed_anova(dv="Score", within="Session", between="Group", subject="Subject", data=df_controlandNF)
# Fit a mixed-effects model
# 'Score' is the dependent variable
# 'Group' is a between-subject factor
# 'Session' is a within-subject factor
# model_controlandNF = mixedlm("Score ~ Group * Session", df_controlandNF, groups=df_controlandNF["Subject"], re_formula="~Session")
# result_controlandNF = model_controlandNF.fit()
#model_controlandNF = mixedlm("Score ~ Group * Session", df_controlandNF, groups="Subject")
#result_controlandNF = model_controlandNF.fit(reml=True)
##############################################################################################################################

sum = []
for a,b,c,d in zip (control_norm_score_values[0], control_norm_score_values[1],control_norm_score_values[2],control_norm_score_values[3]):
    sum.append(a+b+c+d)

mean_control = [num /len(control_indexes) for num in sum]


sum = []
for a,b,c,d in zip (NF_norm_score_values[0], NF_norm_score_values[1]
         ,NF_norm_score_values[2]
         ,NF_norm_score_values[3]
        #,NF_norm_score_values[4],NF_norm_score_values[5],NF_norm_score_values[6]
                    ):
    sum.append(a+b+c+d)

mean_NF = [num /len(NF_indexes) for num in sum]

a=5


########################## HEMISPHERES##########################
# base_path = '/data/Lena/WideFlow_prj'
# cortex_map_path_187 = f'{base_path}/187FN/functional_parcellation_cortex_map.h5'
# rois_dict_path_187 = f'{base_path}/187FN/functional_parcellation_rois_dict.h5'
# cortex_map_path_203 = f'{base_path}/203MN/functional_parcellation_cortex_map.h5'
# rois_dict_path_203 = f'{base_path}/203MN/functional_parcellation_rois_dict.h5'
# # 20221122_{mouse_id}_CRC3functional_parcellation_cortex_map.h5
# # 20221122_{mouse_id}_CRC3functional_parcellation_rois_dict.h5
# # FLfunctional_parcellation_cortex_map_CRC3.h5
# # FLfunctional_parcellation_rois_dict_CRC3.h5
# with h5py.File(cortex_map_path_187, 'r') as f:
#     cortex_mask_187 = f["mask"][()]
#     cortex_map_187 = f["map"][()]
# with h5py.File(cortex_map_path_203, 'r') as f:
#     cortex_mask_203 = f["mask"][()]
#     cortex_map_203 = f["map"][()]
# cortex_mask_187 = cortex_mask_187[:, :168]
# cortex_map_187 = cortex_map_187[:, :168]
# cortex_map_187 = skeletonize(cortex_map_187)
# cortex_mask_203 = cortex_mask_203[:, :168]
# cortex_map_203 = cortex_map_203[:, :168]
# cortex_map_203 = skeletonize(cortex_map_203)
# rois_dict_187 = load_rois_data(rois_dict_path_187)
# rois_dict_203 = load_rois_data(rois_dict_path_203)
# zvmax = 4.5#np.max(list(rois_metric_traces_dict.values()))
# zvmin = -0.8#np.min(list(rois_metric_traces_dict.values()))

#################################################################

############plotting
#f = plt.figure(figsize=(3,5)) #LK figsize is inches
# gs = f.add_gridspec(3, 5)




## Plot line per mouse:
#ax_bottom = f.add_subplot(gs[2, :])
for i, mouse_id1,color_name,line in zip(norm_score_values, mice_id,colors,line_styles):
    plt.plot(sessions_vec, i, label = f'{mouse_id1}', linewidth=1.5,color = color_name,linestyle=line)

### Plot means over all mice:
plt.plot(sessions_vec, mean_control, color='black',linewidth=4, linestyle='--', label='Mean control group')
plt.plot(sessions_vec, mean_NF, color='black',linewidth=4, label='Mean NF group')

plt.ylim(3.0, 5.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f'{title}')
plt.legend()

#
# ax_top00 = f.add_subplot(gs[0, 0])
# ax_top00.set_title('187 NF1')
# ax_top00.axis('off')
# ax_top10 = f.add_subplot(gs[1, 0])
# ax_top10.set_title('203 NF1')
# ax_top10.axis('off')
# # ax_top00.scatter(metric_outline[0], metric_outline[1], marker='.', s=0.5, c='k')
# # ax_top10.scatter(metric_outline[0], metric_outline[1], marker='.', s=10.0, c='k')
# #wf_imshow(ax_top00, frame0, mask=cortex_mask, map=cortex_map, show_cb=False, conv_ker=conv_ker, cm_name='inferno', vmin=vmin, vmax=vmax)
# _, _, Z00 = paint_roi(rois_dict_187, cortex_map_187, list(rois_dict_187.keys()), average_dict1['187FN']['NF1'])
# wf_imshow(ax_top00, Z00, mask=cortex_mask_187, map=cortex_map_187, show_cb=False, cm_name='inferno', vmin=zvmin, vmax=zvmax)#, vmin=zvmin, vmax=zvmax
# _, _, Z01 = paint_roi(rois_dict_203, cortex_map_203, list(rois_dict_203.keys()), average_dict1['203MN']['NF1'])
# wf_imshow(ax_top10, Z01, mask=cortex_mask_203, map=cortex_map_203, show_cb=False, cm_name='inferno', vmin=zvmin, vmax=zvmax)#, vmin=zvmin, vmax=zvmax
#
#
# ax_top01 = f.add_subplot(gs[0, 1])
# ax_top01.set_title('187 NF2')
# ax_top01.axis('off')
# ax_top11 = f.add_subplot(gs[1, 1])
# ax_top11.set_title('203 NF2')
# ax_top11.axis('off')
# # ax_top00.scatter(metric_outline[0], metric_outline[1], marker='.', s=0.5, c='k')
# # ax_top10.scatter(metric_outline[0], metric_outline[1], marker='.', s=10.0, c='k')
# #wf_imshow(ax_top00, frame0, mask=cortex_mask, map=cortex_map, show_cb=False, conv_ker=conv_ker, cm_name='inferno', vmin=vmin, vmax=vmax)
# _, _, Z01 = paint_roi(rois_dict_187, cortex_map_187, list(rois_dict_187.keys()), average_dict1['187FN']['NF2'])
# wf_imshow(ax_top01, Z01, mask=cortex_mask_187, map=cortex_map_187, show_cb=False, cm_name='inferno', vmin=zvmin, vmax=zvmax)#, vmin=zvmin, vmax=zvmax
# _, _, Z11 = paint_roi(rois_dict_203, cortex_map_203, list(rois_dict_203.keys()), average_dict1['203MN']['NF2'])
# wf_imshow(ax_top11, Z11, mask=cortex_mask_203, map=cortex_map_203, show_cb=False, cm_name='inferno', vmin=zvmin, vmax=zvmax)#, vmin=zvmin, vmax=zvmax
#
#
#
# ax_top02 = f.add_subplot(gs[0, 2])
# ax_top02.set_title('187 NF3')
# ax_top02.axis('off')
# ax_top12 = f.add_subplot(gs[1, 2])
# ax_top12.set_title('203 NF3')
# ax_top12.axis('off')
# # ax_top00.scatter(metric_outline[0], metric_outline[1], marker='.', s=0.5, c='k')
# # ax_top10.scatter(metric_outline[0], metric_outline[1], marker='.', s=10.0, c='k')
# #wf_imshow(ax_top00, frame0, mask=cortex_mask, map=cortex_map, show_cb=False, conv_ker=conv_ker, cm_name='inferno', vmin=vmin, vmax=vmax)
# _, _, Z02 = paint_roi(rois_dict_187, cortex_map_187, list(rois_dict_187.keys()), average_dict1['187FN']['NF3'])
# wf_imshow(ax_top02, Z02, mask=cortex_mask_187, map=cortex_map_187, show_cb=False, cm_name='inferno', vmin=zvmin, vmax=zvmax)#, vmin=zvmin, vmax=zvmax
# _, _, Z12 = paint_roi(rois_dict_203, cortex_map_203, list(rois_dict_203.keys()), average_dict1['203MN']['NF3'])
# wf_imshow(ax_top12, Z12, mask=cortex_mask_203, map=cortex_map_203, show_cb=False, cm_name='inferno', vmin=zvmin, vmax=zvmax)#, vmin=zvmin, vmax=zvmax
#
# ax_top03 = f.add_subplot(gs[0, 3])
# ax_top03.set_title('187 NF4')
# ax_top03.axis('off')
# ax_top13 = f.add_subplot(gs[1, 3])
# ax_top13.set_title('203 NF4')
# ax_top13.axis('off')
# # ax_top00.scatter(metric_outline[0], metric_outline[1], marker='.', s=0.5, c='k')
# # ax_top10.scatter(metric_outline[0], metric_outline[1], marker='.', s=10.0, c='k')
# #wf_imshow(ax_top00, frame0, mask=cortex_mask, map=cortex_map, show_cb=False, conv_ker=conv_ker, cm_name='inferno', vmin=vmin, vmax=vmax)
# _, _, Z03 = paint_roi(rois_dict_187, cortex_map_187, list(rois_dict_187.keys()), average_dict1['187FN']['NF4'])
# wf_imshow(ax_top03, Z03, mask=cortex_mask_187, map=cortex_map_187, show_cb=False, cm_name='inferno', vmin=zvmin, vmax=zvmax)#, vmin=zvmin, vmax=zvmax
# _, _, Z13 = paint_roi(rois_dict_203, cortex_map_203, list(rois_dict_203.keys()), average_dict1['203MN']['NF4'])
# wf_imshow(ax_top13, Z13, mask=cortex_mask_203, map=cortex_map_203, show_cb=False, cm_name='inferno', vmin=zvmin, vmax=zvmax)#, vmin=zvmin, vmax=zvmax
#
# ax_top04 = f.add_subplot(gs[0, 4])
# ax_top04.set_title('187 NF5')
# ax_top04.axis('off')
# ax_top14 = f.add_subplot(gs[1, 4])
# ax_top14.set_title('203 NF5')
# ax_top14.axis('off')
# # ax_top00.scatter(metric_outline[0], metric_outline[1], marker='.', s=0.5, c='k')
# # ax_top10.scatter(metric_outline[0], metric_outline[1], marker='.', s=10.0, c='k')
# #wf_imshow(ax_top00, frame0, mask=cortex_mask, map=cortex_map, show_cb=False, conv_ker=conv_ker, cm_name='inferno', vmin=vmin, vmax=vmax)
# _, _, Z04 = paint_roi(rois_dict_187, cortex_map_187, list(rois_dict_187.keys()), average_dict1['187FN']['NF5'])
# wf_imshow(ax_top04, Z04, mask=cortex_mask_187, map=cortex_map_187, show_cb=True, cm_name='inferno', vmin=zvmin, vmax=zvmax)#, vmin=zvmin, vmax=zvmax
# _, _, Z14 = paint_roi(rois_dict_203, cortex_map_203, list(rois_dict_203.keys()), average_dict1['203MN']['NF5'])
# wf_imshow(ax_top14, Z14, mask=cortex_mask_203, map=cortex_map_203, show_cb=True, cm_name='inferno', vmin=zvmin, vmax=zvmax)#, vmin=zvmin, vmax=zvmax
#
#



plt.show()


# #################to save fig####################
# # Get the current figure
# fig = plt.gcf()
#
# # Option 1: Set a specific size based on your desired width and height (in inches)
# # For example, setting to a common screen size or full-screen size (in inches)
# screen_width_inch = 14  # Adjust as necessary
# screen_height_inch = 8  # Adjust as necessary
#
# # Set the figure size in inches
# fig.set_size_inches(screen_width_inch, screen_height_inch)
# plt.rcParams['svg.fonttype'] = 'none'  # or 'path' or 'none'
# plt.savefig(f'{base_path}/Figures_exp2.1/{title}_mice_21_63_64_removed_channelswitch_used_only_frames20000-22000_multiplied_to_fill_length.svg',format='svg',
#             bbox_inches='tight', dpi=300)
# ############################################################################################

# print(f'Control group {res_stats_control} {posthoc_tukey_control} NF group {res_stats_NF} {posthoc_tukey_NF}')
#print(result_controlandNF.summary())
print(anova)

# Post-hoc: Interaction (Group × Session)
print("\nPost-hoc: Group Comparisons for Each Session")
for session in df_controlandNF["Session"].unique():
    print(f"Session {session}:")
    session_data = df_controlandNF[df_controlandNF["Session"] == session]
    tukey_interaction = pg.pairwise_tukey(dv="Score", between="Group", data=session_data)
    print(tukey_interaction)