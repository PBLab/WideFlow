import numpy as np
import h5py
from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict
from analysis.utils.rois_proximity import calc_rois_proximity
from utils.load_rois_data import load_rois_data

base_path = '/data/Lena/WideFlow_prj'
date = '20230604'
mouse_id = '31MN'
session_id = f'{date}_{mouse_id}_spont'
metric_index = 105 #21ML - 134, 31MN - 105, 54MRL - 85, 63MR - 52, 64ML - 71 (those are the indexes, the actual ROI numbers are this +1)

results_path = '/data/Lena/WideFlow_prj/Results/results_exp2.h5'

data = {}
with h5py.File(results_path, 'r') as f:
    decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/')


traces_sess = data["post_session_analysis"]["dff_delta5"]["traces"]
traces_to_copy = traces_sess[metric_index]
new_traces = np.tile(traces_to_copy,(traces_sess.shape[0],1))



functional_rois_dict_path = f'{base_path}/{mouse_id}/functional_parcellation_rois_dict.h5'

functional_rois_dict = load_rois_data(functional_rois_dict_path)

# rois_proximity = {}
# for i, (key, val) in enumerate(functional_rois_dict.items()):
#        rois_proximity[key] = calc_rois_proximity(functional_rois_dict, key)

rois_proximity = calc_rois_proximity(functional_rois_dict,f'roi_{metric_index+1}')

for i in range(len(rois_proximity)):
    new_traces[i] *= (list(rois_proximity.values())[i]+100)


metric_corr = np.corrcoef(new_traces,new_traces[metric_index])

a=5