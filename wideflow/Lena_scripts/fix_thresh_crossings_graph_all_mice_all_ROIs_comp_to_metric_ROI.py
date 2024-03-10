from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM
import h5py
from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict


base_path = '/data/Lena/WideFlow_prj'
dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
dataset_path_MH = '/data/Lena/WideFlow_prj/Results/results_exp2.h5'

#dates_vec = ['20230615', '20230618', '20230619', '20230620', '20230621', '20230622']
#dates_vec = ['20230604', '20230618', '20230619', '20230620', '20230621', '20230622']
dates_vec = ['20230604',
             '20230608', '20230611', '20230612', '20230613', '20230614', '20230615']
mice_id = ['21ML','31MN','54MRL', '63MR', '64ML'
           ]
colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'
mice_and_roi = 'All_mice_exp_24_46_ROI1'
spont_sess_length_frames = 50000
CRC_sess_length_frames = 60000
NF_sess_length_frames = 65000
#sessions_vec = ['spont_mockNF_ROI2_excluded_closest','NF21', 'NF22', 'NF23', 'NF24', 'NF25']
sessions_vec = ['spont_mockNF_NOTexcluded_closest',
                'CRC4','NF1', 'NF2', 'NF3', 'NF4', 'NF5'] #when changing sessions, note to change normalization and stats
#sessions_vec_MH = ['spont_mockNF_excluded_closest',
                 #  'CRC4','NF1', 'NF2', 'NF3', 'NF4', 'NF5'] #when changing sessions, note to change normalization and stats
#sessions_vec = ['spont_mockNF_ROI2_excluded_closest', 'NF1_mock_ROI2','NF2_mock_ROI2','NF3_mock_ROI2','NF4_mock_ROI2', 'NF5_mock_ROI2']
#sessions_vec = ['NF5', 'NF21_mock_ROI1','NF22_mock_ROI1','NF23_mock_ROI1','NF24_mock_ROI1', 'NF25_mock_ROI1']
set_threshold = 1.5
indexes_vec = [134, 105, 85, 52, 71
               ]#(those are the indexes of ROI1, the actual ROI numbers are this +1)


crossings_allROIs = np.zeros((len(mice_id),len(sessions_vec)))
stds = np.zeros((len(mice_id),len(sessions_vec)))
crossings = np.zeros((len(mice_id),len(sessions_vec)))
#df = pd.DataFrame({'Mice':np.repeat([mice_id],len(sessions_vec)),'Sessions':np.tile([0,1,2,3,4,5],len(mice_id)), 'Mean_crossings_all_ROIs':np.zeros((len(mice_id)*len(sessions_vec)))})
a=5

for mouse_id in mice_id:
    for date, session_name in zip(dates_vec, sessions_vec):
        session_id = f'{date}_{mouse_id}_{session_name}'
        if session_name == 'CRC4' and mouse_id == '63MR':
            session_id = '20230607_63MR_CRC3'
        #timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')

        if session_name == 'CRC4':
            dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/Results_exp2_CRC_sessions.h5'
        else:
            dataset_path_noMH = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'

        data = {}
        with h5py.File(dataset_path_noMH, 'r') as f:
            decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/')

        # if mouse_id == '21ML' and session_name == 'spont_mockNF_NOTexcluded_closest':
        #     zscores_dict_long = data["post_session_analysis_LK2"]["zsores_MH_diff5"]
        #     zscores_dict = {}
        #     for a, b in zscores_dict_long.items():
        #         shortened_list = b[:14000]
        #         zscores_dict[a] = shortened_list
        # else:
        zscores_dict = data["post_session_analysis_LK2"]["zsores_MH_diff5"]


        crossings_sess = []
        for key in zscores_dict.keys():
            count = 0
            zscores_roi = zscores_dict[key]
            for value in zscores_roi:
                if value > set_threshold:
                    count += 1

            a=5
            if len(zscores_dict['roi_01']) < NF_sess_length_frames/2:
                count = count*(NF_sess_length_frames/(2*len(zscores_dict['roi_01'])))
            a=5
            crossings_sess.append(count)
            a=5

        mean_crossings = np.mean(crossings_sess)
        #std_sess = np.std(crossings_sess)
        crossings_allROIs[mice_id.index(mouse_id),sessions_vec.index(session_name)] = mean_crossings
        #stds[mice_id.index(mouse_id), sessions_vec.index(session_name)] = std_sess
        a=5



        metric_result = zscores_dict[f'roi_{indexes_vec[mice_id.index(mouse_id)] + 1}']


        count_met = 0
        for value in metric_result:
            if value > set_threshold:
                count_met += 1

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
first_column = crossings_allROIs[:, 0]
# first_column = (NF_sess_length_frames/spont_sess_length_frames)*first_column
# crossings_allROIs[:,0] = first_column
# second_column = crossings_allROIs[:, 1]
# second_column = (NF_sess_length_frames/CRC_sess_length_frames)*second_column
# crossings_allROIs[:,1] = second_column
norm_crossings_allROIs = crossings_allROIs - first_column[:, np.newaxis]
norm_crossings_allROIs = [[element / first_column[i] for element in row] for i, row in enumerate(norm_crossings_allROIs)]




# first_column_stds = stds[:, 0]
# # first_column = (NF_sess_length_frames/spont_sess_length_frames)*first_column
# #crossings[:,0] = first_column
# norm_stds = stds - first_column_stds[:, np.newaxis]
# norm_stds = [[element / first_column_stds[i] for element in row] for i, row in enumerate(norm_stds)]
# a=5



##normalizing crossings metric ROI in spont and CRC
first_column = crossings[:, 0]
# first_column = (NF_sess_length_frames/spont_sess_length_frames)*first_column
# crossings[:,0] = first_column
# second_column = crossings[:, 1]
# second_column = (NF_sess_length_frames/CRC_sess_length_frames)*second_column
# crossings[:,1] = second_column
norm_crossings = crossings - first_column[:, np.newaxis]
norm_crossings = [[element / first_column[i] for element in row] for i, row in enumerate(norm_crossings)]




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

df = pd.DataFrame({'Mice':np.repeat([21,31,54,63,64],7),'Sessions': np.tile([0,1,2,3,4,5,6],5),'Scores':[item for sublist in norm_crossings_allROIs for item in sublist]})
res_allROIs = pg.rm_anova(dv = 'Scores', within = 'Sessions', subject = 'Mice', data = df)
post_hocs_allROIs = pg.pairwise_tests(dv='Scores', within='Sessions',subject='Mice', data=df)
res_stats_allROIs = AnovaRM(data = df, depvar = 'Scores', subject = 'Mice', within=['Sessions']).fit()
posthoc_tukey_allROIs = pairwise_tukeyhsd(df['Scores'], df['Sessions'])

a=5

# #Stats metric
# NOTE!!!!! all stats here will be with no mexican hat (meaning, the closest neighbors of the target ROI are included in
# all calculations). To calculate stats with mexican hat - use script fix_thresh_crossings_graph_all_mice
df_metric = pd.DataFrame({'Mice':np.repeat([21,31,54,63,64],7),'Sessions': np.tile([0,1,2,3,4,5,6],5),'Scores':[item for sublist in norm_crossings for item in sublist]})
res_metric = pg.rm_anova(dv = 'Scores', within = 'Sessions', subject = 'Mice', data = df_metric)
post_hocs_metric = pg.pairwise_tests(dv='Scores', within='Sessions',subject='Mice', data=df_metric)
res_stats_metric = AnovaRM(data = df_metric, depvar = 'Scores', subject = 'Mice', within=['Sessions']).fit()
posthoc_tukey_metric = pairwise_tukeyhsd(df_metric['Scores'], df_metric['Sessions'])



#### Calculate means over mice:
sum = []
for a,b,c,d,e in zip (norm_crossings_allROIs[0], norm_crossings_allROIs[1],norm_crossings_allROIs[2],norm_crossings_allROIs[3],norm_crossings_allROIs[4]):
    sum.append(a+b+c+d+e)

mean_allROIs = [num /len(mice_id) for num in sum]

sum = []
for a,b,c,d,e in zip (norm_crossings[0], norm_crossings[1],norm_crossings[2],norm_crossings[3],norm_crossings[4]):
    sum.append(a+b+c+d+e)

mean = [num /len(mice_id) for num in sum]

## Plot line per mouse:
for i, mouse_id1,color_name in zip(norm_crossings, mice_id,colors):
    plt.plot(sessions_vec, i, label = f'{mouse_id1} Metric ROI', linewidth=3,color = color_name)

for i, mouse_id1, color_name in zip(norm_crossings_allROIs, mice_id, colors):
    plt.plot(sessions_vec, i, label = f'{mouse_id1} All ROIs', linewidth=0.75,linestyle='--', color = color_name)


### Plot means over all mice:
plt.plot(sessions_vec, mean_allROIs, color='black', linestyle='--', label='Mean')
plt.plot(sessions_vec, mean, color='black',linewidth=3.5, label='Mean')


#plt.gcf().set_size_inches(10, 5)
plt.suptitle(f'{mice_and_roi}_fix_crossings_{set_threshold}')
#plt.grid()
plt.legend()
plt.show()
print(f'All ROIs {res_stats_allROIs} {posthoc_tukey_allROIs} Target ROI {res_stats_metric} {posthoc_tukey_metric}')
# plt.rcParams['svg.fonttype'] = 'none'  # or 'path' or 'none'
# plt.savefig(f'{base_path}/Figs_for_paper/All_mice_CRC4-NF5_TCR_allROIs_vs_metricROI_zscore_diff5_thr=1.5.svg',format='svg',dpi=500)
#plt.savefig(f'{base_path}/Figures_exp2_all_mice_compare/{mice_and_roi}_fix_crossings_{set_threshold}.png',dpi=500)
#plt.savefig(f'{base_path}/Figures_exp2_all_mice_compare/{mice_and_roi}_fix_crossings_{set_threshold}.pdf', format="pdf",dpi=500)


a=5