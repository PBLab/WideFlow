from wideflow.utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict
import h5py
import numpy as np
import matplotlib.pyplot as plt
from wideflow.analysis.utils.extract_from_metadata_file import extract_from_metadata_file

base_path_qnap = '/data/Lena/WideFlow_prj'
base_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj'

# dates = ['20251123','20251123']
# mice_id = ['248FL', '248FL']
# sessions = ['try_newcode_diff10_yes15removal', 'try_newcode_diff10_yes15removal_fake']
dates = ['20250831','20250831']
mice_id = ['260FN', '260FN']
sessions = ['spont_p2', 'spont_p2']
metric_rois = ['89_87', '89_87'] #260
# metric_rois = ['35_36', '35_36'] #248
session_id_0 = f'{dates[0]}_{mice_id[0]}_{sessions[0]}'
session_id_1 = f'{dates[1]}_{mice_id[1]}_{sessions[1]}'

dataset_path_0 = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp3.5_NEW.h5'
dataset_path_1 = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp3.5_NEW.h5'
# dataset_path_0 = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_tests.h5'
# dataset_path_1 = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_tests.h5'

data_0 = {}
data_1 = {}

with h5py.File(f'{dataset_path_0}', 'r') as f:
    decompose_h5_groups_to_dict(f, data_0, f'{mice_id[0]}/{session_id_0}/')

with h5py.File(f'{dataset_path_1}', 'r') as f:
    decompose_h5_groups_to_dict(f, data_1, f'{mice_id[1]}/{session_id_1}/')

timestamp_0, cue_0, metric_result_0, threshold_0, serial_readout_0, trial_number_0 = (
    extract_from_metadata_file(f'{base_path_qnap}/{dates[0]}/{mice_id[0]}/{session_id_0}/metadata.txt'))

timestamp_1, cue_1, metric_result_1, threshold_1, serial_readout_1, trial_number_1 = (
    extract_from_metadata_file(f'{base_path_qnap}/{dates[1]}/{mice_id[1]}/{session_id_1}/metadata.txt'))

metric_LK2_0 = data_0['post_session_analysis_LK2']['zsores_MH_diff10_exc_top15'][f'roi_{metric_rois[0]}']
metric_LK2_1 = data_1['post_session_analysis_LK2']['zsores_MH_diff10_exc_top15'][f'roi_{metric_rois[1]}']

plt.plot(metric_result_1[::2], color = 'red', label = f'metadata from session {session_id_1}')
# plt.plot(metric_result_1, color = 'blue', alpha = 0.5, label = f'metadata from session {session_id_1}')
plt.plot(metric_LK2_1, color = 'blue', alpha = 0.5, label = f'LK2 from session {session_id_1}')
plt.plot(threshold_1[::2], color = 'green', label = f'threshold from session {session_id_1}')
plt.legend()

plt.show()