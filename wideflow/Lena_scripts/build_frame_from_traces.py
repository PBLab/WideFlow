import h5py
import numpy as np
import matplotlib.pyplot as plt

from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict
from wideflow.analysis.utils.extract_from_metadata_file import extract_from_metadata_file
from scipy.ndimage.filters import maximum_filter1d

results_path = '/data/Lena/WideFlow_prj/Results/sessions_xxx.h5'
base_path = '/data/Lena/WideFlow_prj'
mouse_id = 'MNL'
#session_id = '20220324_neurofeedback'
session_id = f'20230119_{mouse_id}_NF19'
[timestamp, cues, metric_result, threshold, serial_readout] = extract_from_metadata_file(f'{base_path}/{mouse_id}/{session_id}/metadata.txt')
serial_readout = [1-x for x in serial_readout]
cues = maximum_filter1d(cues, 2)[::2]


data1 = {}

with h5py.File(results_path, 'r') as f:
    decompose_h5_groups_to_dict(f, data1, f'/{mouse_id}/{session_id}/')

metric_roi_dffs_delta5 = data1['post_session_analysis']['dff_delta5']['traces'][56]
metric_roi_dffs = data1['post_session_analysis']['dff']['traces'][56]
#removing weird outliers


metric_roi_dffs1 = [x for x in metric_roi_dffs if -0.15 < x < 0.15]
plt.plot(metric_roi_dffs1)
plt.show()
# other_dffs = data1['post_session_analysis']['dff_delta5']['traces'][0:55]
# other_dffs_mean = np.mean(other_dffs, axis=0)
# other_dffs_mean1 = [x for x in other_dffs_mean if -1 < x < 1]
# # plt.plot(other_dffs_mean1)
# # plt.show()
# a=5


#to access something in data1: data1['rois_traces']['channel_0']['roi_01'][0] - this will give the trace in frame 0 of roi_01 in channel 0.
#data1['post_session_analysis']['dff']['traces'][:,0] - this will give all dffs of all rois in the first frame
#data1['post_session_analysis']['dff']['traces'][56] - access all dffs of roi_57 (the index is 56 because indexes start at 0 and roi names start at 1)