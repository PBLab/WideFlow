from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import mixedlm
import h5py
from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict


from wideflow.config import DATA_STAGING_PATH  #added by Claude 20260906
# base_path = '/data/Lena/WideFlow_prj'
base_path = DATA_STAGING_PATH  #added by Claude 20260906
#dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
#dataset_path_MH = '/data/Lena/WideFlow_prj/Results/results_exp2.h5'
# dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/results_exp2.1.h5'
dataset_path_noMH = DATA_STAGING_PATH + '/Results/results_exp2.1.h5'  #added by Claude 20260906

#dates_vec = ['20230615', '20230618', '20230619', '20230620', '20230621', '20230622']
#dates_vec = ['20230604', '20230618', '20230619', '20230620', '20230621', '20230622']
#dates_vec = ['20241121',
#             '20241126', '20241129', '20241130','20241201','20241202','20241203']
mice_id = [
    #'21ML'
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
# NF_indexes = [0,1,2,3
#     ,4,5,7
#               ] #indexes of NF mice in mice vector
NF_indexes = [0,1,2,4] #indexes of NF mice in mice vector
#control_indexes = [6,8,9,10] #indexes of control mice in mice vector
control_indexes = [3,5,6,7] #indexes of control mice in mice vector
#line_styles = ['solid','solid','solid','solid','solid','solid','dashed','solid','dashed','dashed','dashed']
line_styles = ['solid','solid','solid','dashed','solid','dashed','dashed','dashed']
    #['21ML','31MN','54MRL', '63MR', '64ML']
colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta','blue','red','olivedrab','grey','green','yellow'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'
# colors = ['forestgreen','limegreen','darkgreen','seagreen','mediumseagreen','aquamarine',
#           'rosybrown',
#           'springgreen',
#           'lightcoral','darkred','brown']
#colors = ['cyan','orange']
mice_and_roi = 'All_mice_uptoNF5'
spont_sess_length_frames = 60000
CRC_sess_length_frames = 60000
NF_sess_length_frames = 65000
#sessions_vec = ['spont_mockNF_ROI2_excluded_closest','NF21', 'NF22', 'NF23', 'NF24', 'NF25']
# sessions_vec = ['spont_mockNF_NOTexcluded_closest',
#                 'CRC4','NF1', 'NF2', 'NF3', 'NF4', 'NF5'] #when changing sessions, note to change normalization and stats
sessions_vec = ['spont','CRC4','NF1','NF2','NF3','NF4','NF5']
#sessions_vec_MH = ['spont_mockNF_excluded_closest',
                 #  'CRC4','NF1', 'NF2', 'NF3', 'NF4', 'NF5'] #when changing sessions, note to change normalization and stats
#sessions_vec = ['spont_mockNF_ROI2_excluded_closest', 'NF1_mock_ROI2','NF2_mock_ROI2','NF3_mock_ROI2','NF4_mock_ROI2', 'NF5_mock_ROI2']
#sessions_vec = ['NF5', 'NF21_mock_ROI1','NF22_mock_ROI1','NF23_mock_ROI1','NF24_mock_ROI1', 'NF25_mock_ROI1']
#set_threshold = 1.27
set_threshold_vec = [#This is for metric ROI
    0.65, #21
    1.29,  #31
    1.6,  #54
    1.27,  #63
    0.56,  #64
    1.47,  #187
    0.82,  #203
    0.82,  #204
    0.72,  #206
    1.2,  #211
    1.14  #218
]

# set_threshold_vec = [#This is for RANDOM ROI21 for all mice
#     1.14,
#     0.53,
#     0.69,
#     1.4,
#     0.8,
#     0.73,
#     4.07,
#     1.55,
#     0.72,
#     0.82,
#     0.97
#
# ]

# set_threshold_vec = [#This is for RANDOM ROI44 for all mice
#     1.14,
#     0.71,
#     1.49,
#     0.67,
#     0.69,
#     0.73,
#     1.69,
#     0.82,
#     0.87,
#     3.43,
#     0.96
#
#
# ]
#set_threshold = 0.82
# indexes_vec = [134, 105, 85, 52, 71
#                ]#(those are the indexes of ROI1, the actual ROI numbers are this +1)
indexes_vec = [
   # 134  #21
    105  #31
    ,85  #54
    #,52  #63
    #,71  #64
    ,56  #187
    ,41  #203
    ,69  #204
    ,50  #206
    ,53  #211
    ,46  #218
             ]#(those are the indexes of ROI1, the actual ROI numbers are this +1)

#indexes_vec = [43,43,43,43,43,43,43,43,43,43,43]
num_frames_21ML = 11000

crossings_allROIs = np.zeros((len(mice_id),len(sessions_vec)))
stds = np.zeros((len(mice_id),len(sessions_vec)))
crossings = np.zeros((len(mice_id),len(sessions_vec)))
#df = pd.DataFrame({'Mice':np.repeat([mice_id],len(sessions_vec)),'Sessions':np.tile([0,1,2,3,4,5],len(mice_id)), 'Mean_crossings_all_ROIs':np.zeros((len(mice_id)*len(sessions_vec)))})
a=5

for mouse_id in mice_id:
    #set_threshold = set_threshold_vec[mice_id.index(mouse_id)]
    set_threshold = 1.5
    if mouse_id == '187FN' or mouse_id == '203MN' or mouse_id == '204FR' or mouse_id == '206FRL' or mouse_id == '211MRR' or mouse_id == '218MN':
        dates_vec = ['20241121','20241126', '20241129', '20241130', '20241201', '20241202', '20241203']
        sessions_vec = ['spont', 'CRC4', 'NF1', 'NF2', 'NF3', 'NF4', 'NF5']
        spont_sess_length_frames = 60000
        CRC_sess_length_frames = 60000
        NF_sess_length_frames = 65000
        # dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/results_exp2.1.h5'
        dataset_path_noMH = DATA_STAGING_PATH + '/Results/results_exp2.1.h5'  #added by Claude 20260906
    else:
        dates_vec = ['20230604','20230608', '20230611', '20230612', '20230613', '20230614', '20230615']
        sessions_vec = ['spont_mockNF_NOTexcluded_closest','CRC4','NF1', 'NF2', 'NF3', 'NF4', 'NF5']
        spont_sess_length_frames = 50000
        CRC_sess_length_frames = 50000
        NF_sess_length_frames = 65000
        #dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'

    for date, session_name in zip(dates_vec, sessions_vec):
        session_id = f'{date}_{mouse_id}_{session_name}'
        if session_name == 'CRC4' and mouse_id == '63MR':
            session_id = '20230607_63MR_CRC3'
        if session_name == 'CRC4' and mouse_id == '203MN':
            session_id = '20241125_203MN_CRC3'
        if session_name == 'CRC4' and mouse_id == '204FR':
            session_id = '20241125_204FR_CRC3'
        #timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')
        if (mouse_id == '21ML' or mouse_id == '31MN' or mouse_id == '54MRL' or mouse_id == '63MR' or mouse_id == '64ML') and session_name == 'CRC4':
            # dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'
            dataset_path_noMH = DATA_STAGING_PATH + '/Results/Results_exp2_CRC_sessions.h5'  #added by Claude 20260906
        if (mouse_id == '21ML' or mouse_id == '31MN' or mouse_id == '54MRL' or mouse_id == '63MR' or mouse_id == '64ML') and session_name != 'CRC4':
            # dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
            dataset_path_noMH = DATA_STAGING_PATH + '/Results/results_exp2_noMH.h5'  #added by Claude 20260906

        # if session_name == 'CRC4':
        #     dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'
        # else:
        #     dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/results_exp2.1.h5'

        data = {}
        with h5py.File(dataset_path_noMH, 'r') as f:
            decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/')

        # if mouse_id == '21ML' and session_name == 'spont_mockNF_NOTexcluded_closest':
        zscores_dict_long = data["post_session_analysis_LK2"]["zsores_MH_diff5"]
        zscores_dict = {}

        if session_id == '20230604_21ML_spont_mockNF_NOTexcluded_closest' or session_id == '20230613_21ML_NF3' or session_id == '20241126_187FN_CRC4'\
                or session_id == '20241126_218MN_CRC4' or session_id=='20241203_206FRL_NF5' or session_id=='20241201_211MRR_NF3' or session_id=='20241129_204FR_NF1':
            for a, b in zscores_dict_long.items():
                shortened_list = b[:num_frames_21ML]
                zscores_dict[a] = shortened_list
        else:
            zscores_dict = zscores_dict_long


        # crossings_sess = []
        # for key in zscores_dict.keys():
        #     count = 0
        #     zscores_roi = zscores_dict[key]
        #     for value in zscores_roi:
        #         if value > set_threshold:
        #             count += 1
        #
        #     a=5
        #     if len(zscores_dict['roi_01']) < NF_sess_length_frames/2:
        #         count = count*(NF_sess_length_frames/(2*len(zscores_dict['roi_01'])))
        #     a=5
        #     crossings_sess.append(count)
        #     a=5
        #
        # mean_crossings = np.mean(crossings_sess)
        # #std_sess = np.std(crossings_sess)
        # crossings_allROIs[mice_id.index(mouse_id),sessions_vec.index(session_name)] = mean_crossings
        # #stds[mice_id.index(mouse_id), sessions_vec.index(session_name)] = std_sess
        # a=5



        metric_result = zscores_dict[f'roi_{indexes_vec[mice_id.index(mouse_id)] + 1}']


        count_met = 0
        for value in metric_result:
            if value > set_threshold:
                count_met += 1

        a=5
        if len(metric_result) < NF_sess_length_frames / 2:
            count_met = count_met * (NF_sess_length_frames / (2 * len(metric_result)))
        crossings[mice_id.index(mouse_id), sessions_vec.index(session_name)] = count_met

        # timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(
        #     f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')
        #
        # count = 0
        # for value in metric_result:
        #     if value > set_threshold:
        #         count += 1
        #
        # crossings[mice_id.index(mouse_id), sessions_vec.index(session_name)] = count



# for mouse_id in mice_id:
#     for date, session_name in zip(dates_vec, sessions_vec_MH):
#         session_id = f'{date}_{mouse_id}_{session_name}'
#         if session_name == 'CRC4' and mouse_id == '63MR':
#             session_id = '20230607_63MR_CRC3'
#         #timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')
#
#         if session_name == 'CRC4':
#             dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'
#         else:
#             dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
#
#         data = {}
#         with h5py.File(dataset_path_noMH, 'r') as f:
#             decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/')
#
#         zscores_dict = data["post_session_analysis_LK"]["zsores_MH"]
#
#         metric_result = zscores_dict[f'roi_{indexes_vec[mice_id.index(mouse_id)]+1}']
#
#
#
#         count = 0
#         for value in metric_result:
#             if value > set_threshold:
#                 count += 1
#
#         crossings[mice_id.index(mouse_id),sessions_vec_MH.index(session_name)] = count
a=5
#crossings[3,3] = (crossings[3,2]+crossings[3,4])/2

##normalizing crossings all ROIs in spont and CRC

# first_column = (NF_sess_length_frames/spont_sess_length_frames)*first_column
# crossings_allROIs[:,0] = first_column
# second_column = crossings_allROIs[:, 1]
# second_column = (NF_sess_length_frames/CRC_sess_length_frames)*second_column
# crossings_allROIs[:,1] = second_column

# first_column = crossings_allROIs[:, 0]
# norm_crossings_allROIs = crossings_allROIs - first_column[:, np.newaxis]
# norm_crossings_allROIs = [[element / first_column[i] for element in row] for i, row in enumerate(norm_crossings_allROIs)]



second_column = crossings_allROIs[:, 1]
norm_crossings_allROIs = crossings_allROIs - second_column[:, np.newaxis]
norm_crossings_allROIs = [[element / second_column[i] for element in row] for i, row in enumerate(norm_crossings_allROIs)]



# first_column_stds = stds[:, 0]
# # first_column = (NF_sess_length_frames/spont_sess_length_frames)*first_column
# #crossings[:,0] = first_column
# norm_stds = stds - first_column_stds[:, np.newaxis]
# norm_stds = [[element / first_column_stds[i] for element in row] for i, row in enumerate(norm_stds)]
# a=5



##normalizing crossings metric ROI in spont and CRC

# first_column = (NF_sess_length_frames/spont_sess_length_frames)*first_column
# crossings[:,0] = first_column
# second_column = crossings[:, 1]
# second_column = (NF_sess_length_frames/CRC_sess_length_frames)*second_column
# crossings[:,1] = second_column

# first_column = crossings[:, 0]
# norm_crossings = crossings - first_column[:, np.newaxis]
# norm_crossings = [[element / first_column[i] for element in row] for i, row in enumerate(norm_crossings)]



second_column = crossings[:, 1]
norm_crossings = crossings - second_column[:, np.newaxis]
norm_crossings = [[element / second_column[i] for element in row] for i, row in enumerate(norm_crossings)]
norm_crossings = crossings

a=5
# #after = [np.max(crossings[i,-2:]) for i in range(len(mice_id))]
# after = [crossings[i,-3] for i in range(len(mice_id))]
# #after = [norm_crossings[i][-3] for i in range(len(mice_id))]
# #first_column1 = np.zeros(len(mice_id))
# #after = [0.2,0.1,0.5,1,0.2,0.2]
# #before = np.zeros(len(mice_id))
# t_statistic, p_value = stats.ttest_rel(first_column, after)

# Stats all ROIs
# NOTE!!!!! all stats here will be with no mexican hat (meaning, the closest neighbors of the target ROI are included in
# all calculations). To calculate stats with mexican hat - use script fix_thresh_crossings_graph_all_mice


a=5
##### STATS!!!
# df = pd.DataFrame({'Mice':np.repeat([21,31,54,63,64],7),'Sessions': np.tile([0,1,2,3,4,5,6],5),'Scores':[item for sublist in norm_crossings_allROIs for item in sublist]})
# res_allROIs = pg.rm_anova(dv = 'Scores', within = 'Sessions', subject = 'Mice', data = df)
# post_hocs_allROIs = pg.pairwise_tests(dv='Scores', within='Sessions',subject='Mice', data=df)
# res_stats_allROIs = AnovaRM(data = df, depvar = 'Scores', subject = 'Mice', within=['Sessions']).fit()
# posthoc_tukey_allROIs = pairwise_tukeyhsd(df['Scores'], df['Sessions'])

a=5

# # #Stats metric
# # NOTE!!!!! all stats here will be with no mexican hat (meaning, the closest neighbors of the target ROI are included in
# # all calculations). To calculate stats with mexican hat - use script fix_thresh_crossings_graph_all_mice
# df_metric = pd.DataFrame({'Mice':np.repeat([21,31,54,63,64],7),'Sessions': np.tile([0,1,2,3,4,5,6],5),'Scores':[item for sublist in norm_crossings for item in sublist]})
# res_metric = pg.rm_anova(dv = 'Scores', within = 'Sessions', subject = 'Mice', data = df_metric)
# post_hocs_metric = pg.pairwise_tests(dv='Scores', within='Sessions',subject='Mice', data=df_metric)
# res_stats_metric = AnovaRM(data = df_metric, depvar = 'Scores', subject = 'Mice', within=['Sessions']).fit()
# posthoc_tukey_metric = pairwise_tukeyhsd(df_metric['Scores'], df_metric['Sessions'])

############STATS WITH CONTROL GROUP
norm_crossings_noSpont = [sublist[1:] for sublist in norm_crossings]
control_norm_crossings = [norm_crossings_noSpont[i] for i in control_indexes]
NF_norm_crossings = [norm_crossings_noSpont[i] for i in NF_indexes]

#control vs. nf
subjects = list([31,54,187,203,204,206,211,218])
#subjects = list([21,31,54,63,64,187,203,204,206,211,218])
#groups = ['NF'] * 6 + ['control'] + ['NF'] +['control'] * 3  # Group labels
groups = ['NF'] * 3 + ['control'] + ['NF'] +['control'] * 3  # Group labels
#sessions = list(range(1,6)) # Session numbers (1 to 7)
#sessions = list(range(1,4)) # Session numbers (1 to 7)
sessions = list(range(1,8)) # Session numbers (1 to 7)


# Create a long-format DataFrame
long_data = []
for subject, group, scores in zip(subjects, groups, norm_crossings_noSpont):
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



#
# #NF group
# df_NF = pd.DataFrame({'Mice':np.repeat([21,31,54,63,64,187,204],6),'Sessions': np.tile([0,1,2,3,4,5],7),'Scores':[item for sublist in NF_norm_crossings for item in sublist]})
# #res_metric = pg.rm_anova(dv = 'Scores', within = 'Sessions', subject = 'Mice', data = df_NF)
# #post_hocs_metric = pg.pairwise_tests(dv='Scores', within='Sessions',subject='Mice', data=df_NF)
# res_stats_NF = AnovaRM(data = df_NF, depvar = 'Scores', subject = 'Mice', within=['Sessions']).fit()
# posthoc_tukey_NF = pairwise_tukeyhsd(df_NF['Scores'], df_NF['Sessions'])
#
# #control group
# df_control = pd.DataFrame({'Mice':np.repeat([203,206,211,218],6),'Sessions': np.tile([0,1,2,3,4,5],4),'Scores':[item for sublist in control_norm_crossings for item in sublist]})
# # res_control = pg.rm_anova(dv = 'Scores', within = 'Sessions', subject = 'Mice', data = df_control)
# # post_hocs_control = pg.pairwise_tests(dv='Scores', within='Sessions',subject='Mice', data=df_control)
# res_stats_control = AnovaRM(data = df_control, depvar = 'Scores', subject = 'Mice', within=['Sessions']).fit()
# posthoc_tukey_control = pairwise_tukeyhsd(df_control['Scores'], df_control['Sessions'])
#
#
# #control vs. nf
# #subjects = list([21,31,54,63,64,187,203,204,206,211,218])
# subjects = list([21,31,54,63,64,187,203,204,206,211,218])
# groups = ['NF'] * 6 + ['control'] + ['NF'] +['control'] * 3  # Group labels
# #groups = ['NF'] * 4  +['control'] * 4  # Group labels
# sessions = list(range(1,7)) # Session numbers (1 to 7)


# # Create a long-format DataFrame
# long_data = []
# for subject, group, scores in zip(subjects, groups, norm_crossings_noSpont):
#     for session1, score in zip(sessions, scores):
#         long_data.append([subject, group, session1, score])
#
# df_controlandNF = pd.DataFrame(long_data, columns=['Subject', 'Group', 'Session', 'Score'])
a=5
# # Convert columns to appropriate types
# df_controlandNF['Subject'] = df_controlandNF['Subject'].astype('category')
# df_controlandNF['Group'] = df_controlandNF['Group'].astype('category')
# df_controlandNF['Session'] = df_controlandNF['Session'].astype('category')
#
# # Fit a mixed-effects model
# # 'Score' is the dependent variable
# # 'Group' is a between-subject factor
# # 'Session' is a within-subject factor
# model_controlandNF = mixedlm("Score ~ Group * Session", df_controlandNF, groups=df_controlandNF["Subject"], re_formula="~Session")
# result_controlandNF = model_controlandNF.fit()



a=5
# #### Calculate means over mice:
control_norm_crossings = [norm_crossings[i] for i in control_indexes]
NF_norm_crossings = [norm_crossings[i] for i in NF_indexes]

sum = []
for a,b,c,d in zip (control_norm_crossings[0], control_norm_crossings[1],control_norm_crossings[2],control_norm_crossings[3]):
    sum.append(a+b+c+d)

mean_control = [num /len(control_indexes) for num in sum]


sum = []
for a,b,c,d in zip (NF_norm_crossings[0], NF_norm_crossings[1],NF_norm_crossings[2]
        ,NF_norm_crossings[3]
        #,NF_norm_crossings[4],NF_norm_crossings[5],NF_norm_crossings[6]
                          ):
    sum.append(a+b+c+d)

mean_NF = [num /len(NF_indexes) for num in sum]

## Plot line per mouse:
for i, mouse_id1,color_name,line in zip(norm_crossings, mice_id,colors,line_styles):
    plt.plot(sessions_vec, i, label = f'{mouse_id1} Metric ROI', linewidth=3,color = color_name,linestyle=line)

# for i, mouse_id1, color_name in zip(norm_crossings_allROIs, mice_id, colors):
#     plt.plot(sessions_vec, i, label = f'{mouse_id1} All ROIs', linewidth=0.75,linestyle='--', color = color_name)
#

### Plot means over all mice:
plt.plot(sessions_vec, mean_control, color='black',linewidth=3.5, linestyle='--', label='Mean')
plt.plot(sessions_vec, mean_NF, color='black',linewidth=3.5, label='Mean')


#plt.gcf().set_size_inches(10, 5)
plt.suptitle(f'{mice_and_roi}_fix_crossings_{set_threshold}')
#plt.grid()
plt.legend()
plt.show()
#print(f'All ROIs {res_stats_allROIs} {posthoc_tukey_allROIs} Target ROI {res_stats_metric} {posthoc_tukey_metric}')
#print(f'Control group {res_stats_control} {posthoc_tukey_control} NF group {res_stats_NF} {posthoc_tukey_NF}')
#print(result_controlandNF.summary())





# plt.rcParams['svg.fonttype'] = 'none'  # or 'path' or 'none'
# plt.savefig(f'{base_path}/Figs_for_paper/All_mice_{sessions_vec[0]}-{sessions_vec[-1]}_'
#             f'norm to CRC4_TCR_allROIs_vs_metricROI_zscore_diff5_thr={set_threshold} 1400 frames only 21ML spont+NF3.svg',format='svg',dpi=500)
#
#

#plt.savefig(f'{base_path}/Figures_exp2_all_mice_compare/{mice_and_roi}_fix_crossings_{set_threshold}.png',dpi=500)
#plt.savefig(f'{base_path}/Figures_exp2_all_mice_compare/{mice_and_roi}_fix_crossings_{set_threshold}.pdf', format="pdf",dpi=500)

print(anova)

# Post-hoc: Interaction (Group × Session)
print("\nPost-hoc: Group Comparisons for Each Session")
for session in df_controlandNF["Session"].unique():
    print(f"Session {session}:")
    session_data = df_controlandNF[df_controlandNF["Session"] == session]
    tukey_interaction = pg.pairwise_tukey(dv="Score", between="Group", data=session_data)
    print(tukey_interaction)


a=5