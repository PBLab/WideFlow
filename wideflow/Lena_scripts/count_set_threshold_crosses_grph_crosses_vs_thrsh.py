from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
import numpy as np
import matplotlib.pyplot as plt
import h5py


from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict



base_path = '/data/Lena/WideFlow_prj'
dates_vec = ['20230604','20230608', '20230611', '20230612', '20230613', '20230614', '20230615']
#dates_vec = ['20230604', '20230618', '20230619', '20230620', '20230621', '20230622']
#mouse_id_vec = ['24MLL', '24MLL', '24MLL', '24MLL', '24MLL', '24MLL']
mice_id = ['31MN']
roi = 'ROI1'
sessions_vec = ['spont_mockNF_NOTexcluded_closest','CRC4','NF1', 'NF2', 'NF3', 'NF4', 'NF5']
#sessions_vec = ['spont_mockNF_ROI2_excluded_closest','NF21', 'NF22', 'NF23', 'NF24', 'NF25']
#sessions_vec = ['spont_mockNF_ROI2_excluded_closest', 'NF1_mock_ROI2','NF2_mock_ROI2','NF3_mock_ROI2','NF4_mock_ROI2', 'NF5_mock_ROI2']

# set_threshold = 2.3

indexes_vec = [105]#(those are the indexes of ROI1, the actual ROI numbers are this +1) [134, 105, 85, 52, 71]
NF_sess_length_frames = 65000
spont_sess_frames = 50000
set_threshold = (np.arange(0.5, 4.5, 0.1)).tolist()
crossings = np.zeros((len(sessions_vec),len(set_threshold)))

num_frames_21ML = 14000

# for date, session_id in zip(dates_vec, sessions_vec):
#
#     #timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(f'{base_path}/{mouse_id}/{session_id}/metadata.txt')
#     timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{date}_{mouse_id}_{session_id}/metadata.txt')
#     #count = np.zeros(len(set_threshold))
#     #rewards = np.sum(cue)
#     for i in set_threshold:
#         for value in metric_result:
#             if value > i:
#                 crossings[sessions_vec.index(session_id), set_threshold.index(i)] += 1

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
        zscores_dict_long = data["post_session_analysis_LK2"]["zsores_MH_diff5"]
        zscores_dict = {}

        if session_id == '20230604_21ML_spont_mockNF_NOTexcluded_closest' or session_id == '20230613_21ML_NF3':
            for a, b in zscores_dict_long.items():
                shortened_list = b[:num_frames_21ML]
                zscores_dict[a] = shortened_list
        else:
            zscores_dict = zscores_dict_long

        metric_result = zscores_dict[f'roi_{indexes_vec[mice_id.index(mouse_id)] + 1}']



        #     for i in set_threshold:
        #         for value in metric_result:
        #             if value > i:
        #                 crossings[sessions_vec.index(session_id), set_threshold.index(i)] += 1
        for i in set_threshold:
            count_met = 0
            for value in metric_result:
                if value > i:
                    count_met += 1

            if len(metric_result) < NF_sess_length_frames / 2:
                count_met = count_met * (NF_sess_length_frames / (2 * len(metric_result)))
            crossings[sessions_vec.index(session_name), set_threshold.index(i)] = count_met



# first_row = crossings[0,:]
# first_row = (NF_sess_frames/spont_sess_frames)*first_row
# crossings[0,:] = first_row
# norm_crossings = crossings - first_row[np.newaxis,:]
# norm_crossings = [[element / first_row[i] for element in column] for i, column in enumerate(norm_crossings)]
# #crossings[0,:] = (NF_sess_frames/spont_sess_frames)*crossings[0,:]


# first_column = crossings[:, 0]
# norm_crossings = crossings - first_column[:, np.newaxis]
# norm_crossings = [[element / first_column[i] for element in row] for i, row in enumerate(norm_crossings)]
second_row = crossings[1, :]
norm_crossings = crossings - second_row[np.newaxis,:]
norm_crossings = [[element / second_row[i] for element in column] for i, column in enumerate(norm_crossings)]

color_palette = plt.cm.get_cmap('plasma', len(sessions_vec)) #colormap options https://matplotlib.org/stable/tutorials/colors/colormaps.html
for i, session_id in zip(norm_crossings, sessions_vec):
    plt.plot(set_threshold,i, color =color_palette(sessions_vec.index(session_id)), label = f'{session_id}')


plt.grid()
plt.legend()
plt.title(f'{mouse_id}_{roi}')
plt.xlabel ('Threshold')
plt.ylabel ('Threshold crossing rate (a.u.)')
#plt.show()


plt.rcParams['svg.fonttype'] = 'none'  # or 'path' or 'none'
plt.savefig(f'{base_path}/Figs_for_paper/{mouse_id}_{roi}_rate_vs_thrsh_all_sessions norm to CRC4.svg',format='svg',dpi=500)


#plt.savefig(f'{base_path}/Figures_exp2_all_mice_compare/{mouse_id}_{roi}_rate_vs_thrsh_all_sessions.png',dpi=500)




#[item[0] for item in norm_crossings]
a=5
#print(count, rewards)
