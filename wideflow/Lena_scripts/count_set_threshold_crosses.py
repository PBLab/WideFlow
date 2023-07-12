from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
import numpy as np

base_path = '/data/Lena/WideFlow_prj'
date = '20230622'
mouse_id = '64ML'
session_id = f'{date}_{mouse_id}_NF25'
#session_id = f'20230618_54MRL_NF21_mocknF_ROI1'

#timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(f'{base_path}/{mouse_id}/{session_id}/metadata.txt')
timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt')

set_threshold = [2.8, 3.2, 3.6, 4.0]
count = np.zeros(len(set_threshold))
rewards = np.sum(cue)

for i in set_threshold:
    for value in metric_result:
        if value > i:
            count[set_threshold.index(i)] += 1

print(count, rewards)
