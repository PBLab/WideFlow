from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM

# base_path = '/data/Lena/WideFlow_prj'
base_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj'
#dates_vec = ['20250731','20250731','20250731', '20250801','20250801','20250801','20250802','20250802','20250802']
mice_id = [
    '245FRL',
    '246FN',
    '248FL',
    '252MR',
    '256FLL',
    '257FR'
]
colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta','green'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'
mice_and_roi = 'All_mice_target_ROI'
spont_sess_length_frames = 65000
NF_sess_length_frames = 65000
#sessions_vec = ['spont_mockNF_ROI2_excluded_closest','NF21', 'NF22', 'NF23', 'NF24', 'NF25']
sessions_vec = ['NF1_p1', 'NF1_p2', 'NF1_p3', 'NF2_p1', 'NF2_p2', 'NF2_p3', 'NF3_p1', 'NF3_p2', 'NF3_p3', 'NF4_p1',
                'NF4_p2', 'NF4_p3', 'NF5_p1',
                'NF5_p2', 'NF5_p3']#sessions_vec = ['spont_mockNF_ROI2_excluded_closest', 'NF1_mock_ROI2','NF2_mock_ROI2','NF3_mock_ROI2','NF4_mock_ROI2', 'NF5_mock_ROI2']
#sessions_vec = ['NF5', 'NF21_mock_ROI1','NF22_mock_ROI1','NF23_mock_ROI1','NF24_mock_ROI1', 'NF25_mock_ROI1']
set_threshold = 0.5


crossings = np.zeros((len(mice_id),len(sessions_vec)))

for mouse_id in mice_id:
    if mouse_id == '248FL' or mouse_id == '252MR' or mouse_id == '257FR':
        dates_vec = [ '20250801', '20250801', '20250801', '20250802', '20250802','20250802', '20250803', '20250803','20250803',
                      '20250804', '20250804','20250804', '20250805', '20250805','20250805']
        sessions_vec = [ 'NF2_p1', 'NF2_p2', 'NF2_p3', 'NF3_p1', 'NF3_p2', 'NF3_p3', 'NF4_p1', 'NF4_p2', 'NF4_p3', 'NF5_p1',
                         'NF5_p2', 'NF5_p3' , 'NF6_p1', 'NF6_p2', 'NF6_p3'  ]
    else:
        dates_vec = ['20250731', '20250731', '20250731', '20250801', '20250801', '20250801', '20250802', '20250802','20250802',
                     '20250803', '20250803','20250803', '20250804', '20250804','20250804']
        sessions_vec = ['NF1_p1', 'NF1_p2', 'NF1_p3', 'NF2_p1', 'NF2_p2', 'NF2_p3', 'NF3_p1', 'NF3_p2', 'NF3_p3', 'NF4_p1', 'NF4_p2', 'NF4_p3', 'NF5_p1',
                         'NF5_p2', 'NF5_p3']

    for date, session_name in zip(dates_vec, sessions_vec):
        session_id = f'{date}_{mouse_id}_{session_name}'
        timestamp, cue, metric_result, threshold, serial_readout, trial_number = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')

        count = 0
        for value in metric_result:
            if value > set_threshold:
                count += 1

        crossings[mice_id.index(mouse_id),sessions_vec.index(session_name)] = count


crossings=crossings.reshape(6, 5, 3).mean(axis=2)
#crossings[3,3] = (crossings[3,2]+crossings[3,4])/2
first_column = crossings[:, 0]
# first_column = (NF_sess_length_frames/spont_sess_length_frames)*first_column
crossings[:,0] = first_column
norm_crossings = crossings - first_column[:, np.newaxis]
norm_crossings = [[element / first_column[i] for element in row] for i, row in enumerate(norm_crossings)]
#norm_crossings = crossings

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

sum = []
for a,b,c,d,e in zip (norm_crossings[0], norm_crossings[1],norm_crossings[2],norm_crossings[3],norm_crossings[4]):
    sum.append(a+b+c+d+e)

mean = [num /len(mice_id) for num in sum]

sessions_vec1 = ['NF1','NF2','NF3','NF4','NF5']
for i, mouse_id1,color_name in zip(norm_crossings, mice_id,colors):
    plt.plot(sessions_vec1, i, label = f'{mouse_id1}', color = color_name)

#plt.plot(sessions_vec1, mean, color='black', linestyle='--', label='Mean')
plt.suptitle(f'{mice_and_roi}_fix_crossings_{set_threshold}')
plt.grid()
plt.legend()
plt.show()
print(res_stats, posthoc_tukey)
#plt.savefig(f'{base_path}/Figures_exp2_all_mice_compare/{mice_and_roi}_fix_crossings_{set_threshold}.png',dpi=500)
#plt.savefig(f'{base_path}/Figures_exp2_all_mice_compare/{mice_and_roi}_fix_crossings_{set_threshold}.pdf', format="pdf",dpi=500)


a=5