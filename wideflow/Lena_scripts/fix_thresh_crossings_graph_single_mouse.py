from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
import numpy as np
import matplotlib.pyplot as plt

def divide_array(arr, divisor):
    return [num / divisor for num in arr]

def subtract_from_array(arr, value):
    return [num - value for num in arr]

base_path = '/data/Lena/WideFlow_prj'
dates_vec = ['20230604', '20230611', '20230612', '20230613', '20230614', '20230615']
#dates_vec = ['20230604', '20230618', '20230619', '20230620', '20230621', '20230622']
#mouse_id_vec = ['24MLL', '24MLL', '24MLL', '24MLL', '24MLL', '24MLL']
mouse_id = '63MR'
#sessions_vec = ['spont_mockNF_excluded_closest','NF1', 'NF2', 'NF3', 'NF4', 'NF5']
sessions_vec = ['spont_mockNF_ROI2_excluded_closest', 'NF1_mock_ROI2','NF2_mock_ROI2','NF3_mock_ROI2','NF4_mock_ROI2', 'NF5_mock_ROI2']
#sessions_vec = ['spont_mockNF_ROI2_excluded_closest','NF21', 'NF22', 'NF23', 'NF24', 'NF25']
set_threshold = 2.3

NF_sess_frames = 50000
spont_sess_frames = 50000

crossings = np.zeros(len(sessions_vec))

for date, session_name in zip(dates_vec, sessions_vec):
    session_id = f'{date}_{mouse_id}_{session_name}'
    timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(
        f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')

    count = 0
    for value in metric_result:
        if value > set_threshold:
            count += 1

    crossings[sessions_vec.index(session_name)] = count


crossings[0] = (NF_sess_frames/spont_sess_frames)*crossings[0]
norm_crossings = subtract_from_array(crossings, crossings[0])
norm_crossings = divide_array(norm_crossings, crossings[0])


plt.plot (sessions_vec, norm_crossings)
plt.suptitle(f'{mouse_id}_ROI2_fix_crossings_{set_threshold}')
plt.grid()
plt.show()
#plt.savefig(f'{base_path}/{mouse_id}/simple_figs/{mouse_id}_ROI2_fix_crossings_{set_threshold}.png',dpi=500)


a=5