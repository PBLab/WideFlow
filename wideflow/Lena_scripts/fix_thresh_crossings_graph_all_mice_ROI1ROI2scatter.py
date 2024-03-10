from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import pingouin as pg

base_path = '/data/Lena/WideFlow_prj'
#dates_vec = ['20230615', '20230618', '20230619', '20230620', '20230621', '20230622']
dates_vec_roi2 = ['20230604', '20230618', '20230619', '20230620', '20230621', '20230622']
dates_vec_roi1 = ['20230604', '20230611', '20230612', '20230613', '20230614', '20230615']
mice_id = ['21ML','31MN','54MRL', '63MR', '64ML']
colors = ['cyan', 'orange', 'purple', 'chartreuse', 'magenta'] #21'cyan',24'blue',31'orange',46'green',54'purple', 63'chartreuse', 64'magenta'
mice_and_roi = 'All_mice_exp_24_46'
spont_sess_length_frames = 50000
NF_sess_length_frames = 65000
sessions_vec_roi2 = ['spont_mockNF_ROI2_excluded_closest','NF21', 'NF22', 'NF23', 'NF24', 'NF25']
sessions_vec_roi1 = ['spont_mockNF_excluded_closest','NF1', 'NF2', 'NF3', 'NF4', 'NF5']
#sessions_vec = ['spont_mockNF_ROI2_excluded_closest', 'NF1_mock_ROI2','NF2_mock_ROI2','NF3_mock_ROI2','NF4_mock_ROI2', 'NF5_mock_ROI2']
#sessions_vec = ['NF5', 'NF21_mock_ROI1','NF22_mock_ROI1','NF23_mock_ROI1','NF24_mock_ROI1', 'NF25_mock_ROI1']
set_threshold = 1.3

#crossings ROI1
crossings_roi1 = np.zeros((len(mice_id),len(sessions_vec_roi1)))

for mouse_id in mice_id:
    for date, session_name in zip(dates_vec_roi1, sessions_vec_roi1):
        session_id = f'{date}_{mouse_id}_{session_name}'
        timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')

        count = 0
        for value in metric_result:
            if value > set_threshold:
                count += 1

        crossings_roi1[mice_id.index(mouse_id),sessions_vec_roi1.index(session_name)] = count


#crossings[3,3] = (crossings[3,2]+crossings[3,4])/2
first_column_roi1 = crossings_roi1[:, 0]
first_column_roi1 = (NF_sess_length_frames/spont_sess_length_frames)*first_column_roi1
crossings_roi1[:,0] = first_column_roi1
norm_crossings_roi1 = crossings_roi1 - first_column_roi1[:, np.newaxis]
norm_crossings_roi1 = [[element / first_column_roi1[i] for element in row] for i, row in enumerate(norm_crossings_roi1)]


#crossings ROI2
crossings_roi2 = np.zeros((len(mice_id),len(sessions_vec_roi2)))

for mouse_id in mice_id:
    for date, session_name in zip(dates_vec_roi2, sessions_vec_roi2):
        session_id = f'{date}_{mouse_id}_{session_name}'
        timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')

        count = 0
        for value in metric_result:
            if value > set_threshold:
                count += 1

        crossings_roi2[mice_id.index(mouse_id),sessions_vec_roi2.index(session_name)] = count


#crossings[3,3] = (crossings[3,2]+crossings[3,4])/2
first_column_roi2 = crossings_roi2[:, 0]
first_column_roi2 = (NF_sess_length_frames/spont_sess_length_frames)*first_column_roi2
crossings_roi2[:,0] = first_column_roi2
norm_crossings_roi2 = crossings_roi2 - first_column_roi2[:, np.newaxis]
norm_crossings_roi2 = [[element / first_column_roi2[i] for element in row] for i, row in enumerate(norm_crossings_roi2)]

#maxes ROI1 and ROI2
maxes = np.zeros ((len(mice_id),2))
for i in range(len(mice_id)):
    maxes[i,0] = np.max(norm_crossings_roi1[i])
    maxes[i,1] = np.max(norm_crossings_roi2[i])

a=5


# sum = []
# for a,b,c,d,e in zip (norm_crossings[0], norm_crossings[1],norm_crossings[2],norm_crossings[3],norm_crossings[4]):
#     sum.append(a+b+c+d+e)
#
# mean = [num /len(mice_id) for num in sum]

for i,mouse,color_name in zip(range(len(mice_id)),mice_id,colors):
    plt.scatter(maxes[i,0],maxes[i,1],label = f'{mouse}', color = color_name)

#plt.plot(sessions_vec, mean, color='black', linestyle='--', label='Mean')
#plt.suptitle(f'{mice_and_roi}_fix_crossings_{set_threshold}')
plt.grid()
plt.ylabel ('ROI2 Best score')
plt.xlabel ('ROI1 Best score')
plt.legend()
plt.show()
#plt.savefig(f'{base_path}/Figures_exp2_all_mice_compare/{mice_and_roi}_fix_crossings_{set_threshold}.png',dpi=500)


a=5