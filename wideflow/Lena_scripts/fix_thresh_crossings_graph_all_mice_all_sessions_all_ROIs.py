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
dataset_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'

#dates_vec = ['20230615', '20230618', '20230619', '20230620', '20230621', '20230622']
#dates_vec = ['20230604', '20230618', '20230619', '20230620', '20230621', '20230622']
dates_vec = ['20230604', '20230611', '20230612', '20230613', '20230614', '20230615']
mice_id = ['21ML','31MN','54MRL', '63MR', '64ML']
colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'
mice_and_roi = 'All_mice_exp_24_46_ROI1'
spont_sess_length_frames = 50000
NF_sess_length_frames = 65000
#sessions_vec = ['spont_mockNF_ROI2_excluded_closest','NF21', 'NF22', 'NF23', 'NF24', 'NF25']
sessions_vec = ['spont_mockNF_NOTexcluded_closest','NF1', 'NF2', 'NF3', 'NF4', 'NF5']
#sessions_vec = ['spont_mockNF_ROI2_excluded_closest', 'NF1_mock_ROI2','NF2_mock_ROI2','NF3_mock_ROI2','NF4_mock_ROI2', 'NF5_mock_ROI2']
#sessions_vec = ['NF5', 'NF21_mock_ROI1','NF22_mock_ROI1','NF23_mock_ROI1','NF24_mock_ROI1', 'NF25_mock_ROI1']
set_threshold = 1.3


crossings = np.zeros((len(mice_id),len(sessions_vec)))
stds = np.zeros((len(mice_id),len(sessions_vec)))
#df = pd.DataFrame({'Mice':np.repeat([mice_id],len(sessions_vec)),'Sessions':np.tile([0,1,2,3,4,5],len(mice_id)), 'Mean_crossings_all_ROIs':np.zeros((len(mice_id)*len(sessions_vec)))})
a=5

for mouse_id in mice_id:
    for date, session_name in zip(dates_vec, sessions_vec):
        session_id = f'{date}_{mouse_id}_{session_name}'
        #timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')

        data = {}
        with h5py.File(dataset_path, 'r') as f:
            decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/')

        zscores_dict = data["post_session_analysis_LK"]["zsores_MH"]

        crossings_sess = []
        for key in zscores_dict.keys():
            count = 0
            zscores_roi = zscores_dict[key]
            for value in zscores_roi:
                if value > set_threshold:
                    count += 1

            crossings_sess.append(count)
            a=5

        mean_crossings = np.mean(crossings_sess)
        std_sess = np.std(crossings_sess)
        crossings[mice_id.index(mouse_id),sessions_vec.index(session_name)] = mean_crossings
        stds[mice_id.index(mouse_id), sessions_vec.index(session_name)] = std_sess
        a=5

a=5
#crossings[3,3] = (crossings[3,2]+crossings[3,4])/2
first_column = crossings[:, 0]
first_column = (NF_sess_length_frames/spont_sess_length_frames)*first_column
crossings[:,0] = first_column
norm_crossings = crossings - first_column[:, np.newaxis]
norm_crossings = [[element / first_column[i] for element in row] for i, row in enumerate(norm_crossings)]

first_column_stds = stds[:, 0]
# first_column = (NF_sess_length_frames/spont_sess_length_frames)*first_column
#crossings[:,0] = first_column
norm_stds = stds - first_column_stds[:, np.newaxis]
norm_stds = [[element / first_column_stds[i] for element in row] for i, row in enumerate(norm_stds)]
a=5

# #after = [np.max(crossings[i,-2:]) for i in range(len(mice_id))]
# after = [crossings[i,-3] for i in range(len(mice_id))]
# #after = [norm_crossings[i][-3] for i in range(len(mice_id))]
# #first_column1 = np.zeros(len(mice_id))
# #after = [0.2,0.1,0.5,1,0.2,0.2]
# #before = np.zeros(len(mice_id))
# t_statistic, p_value = stats.ttest_rel(first_column, after)

df = pd.DataFrame({'Mice':np.repeat([21,31,54,63,64],6),'Sessions': np.tile([0,1,2,3,4,5],5),'Scores':[item for sublist in norm_crossings for item in sublist]})
res = pg.rm_anova(dv = 'Scores', within = 'Sessions', subject = 'Mice', data = df)
post_hocs = pg.pairwise_tests(dv='Scores', within='Sessions',subject='Mice', data=df)
res_stats = AnovaRM(data = df, depvar = 'Scores', subject = 'Mice', within=['Sessions']).fit()
posthoc_tukey = pairwise_tukeyhsd(df['Scores'], df['Sessions'])

sum = []
for a,b,c,d,e in zip (norm_crossings[0], norm_crossings[1],norm_crossings[2],norm_crossings[3],norm_crossings[4]):
    sum.append(a+b+c+d+e)

mean = [num /len(mice_id) for num in sum]

for i, mouse_id1,color_name in zip(norm_crossings, mice_id,colors):
    plt.errorbar(sessions_vec, i, label = f'{mouse_id1}', color = color_name)

plt.plot(sessions_vec, mean, color='black', linestyle='--', label='Mean')
plt.suptitle(f'{mice_and_roi}_fix_crossings_{set_threshold}')
plt.grid()
plt.legend()
plt.show()
#plt.savefig(f'{base_path}/Figures_exp2_all_mice_compare/{mice_and_roi}_fix_crossings_{set_threshold}.png',dpi=500)
#plt.savefig(f'{base_path}/Figures_exp2_all_mice_compare/{mice_and_roi}_fix_crossings_{set_threshold}.pdf', format="pdf",dpi=500)


a=5