from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
import numpy as np

base_path = '/data/Lena/WideFlow_prj'
date = '20230613'
mouse_id = '54MRL'
session_id = f'{date}_{mouse_id}_NF3'
#session_id = f'20230618_54MRL_NF21_mocknF_ROI1'

NF_sess_frames = 65000
spont_sess_frames = 50000

#timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(f'{base_path}/{mouse_id}/{session_id}/metadata.txt')
timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')

set_threshold = [2.8, 3.2, 3.6, 4.0]
count = np.zeros(len(set_threshold))
rewards = np.sum(cue)

for i in set_threshold:
    for value in metric_result:
        if value > i:
            count[set_threshold.index(i)] += 1

count[0] = (NF_sess_frames/spont_sess_frames)*count[0]
a=5
print(count, rewards)
