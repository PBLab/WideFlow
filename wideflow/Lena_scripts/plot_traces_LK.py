import matplotlib.pyplot as plt
import numpy as np
import h5py

from wideflow.analysis.plots import plot_pstr
from wideflow.analysis.plots import plot_traces
from wideflow.analysis.utils.extract_from_metadata_file import extract_from_metadata_file
from analysis.utils.peristimulus_time_response import calc_pstr
from wideflow.utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict
from scipy.ndimage.filters import maximum_filter1d
from utils.load_rois_data import load_rois_data
from utils.load_config import load_config

#[timestamp, cues, metric_result, threshold, serial_readout] = extract_from_metadata_file('/data/Rotem/WideFlow prj/2680/20220324_neurofeedback/metadata.txt')

base_path = '/data/Lena/WideFlow_prj'
mouse_id = 'MR'
#session_id = '20220324_neurofeedback'
session_id = f'20230201_{mouse_id}_NF23'
[timestamp, cues, metric_result, threshold, serial_readout] = extract_from_metadata_file(f'{base_path}/{mouse_id}/{session_id}/metadata.txt')
serial_readout_correct = [1-x for x in serial_readout]
cue = maximum_filter1d(cues, 2)[::2]
config = load_config(f'{base_path}/{mouse_id}/{session_id}/session_config.json')
metric_roi = config['analysis_pipeline_config']['args']['metric_args'][1][0]


functional_rois_dict = load_rois_data(f'{base_path}/{mouse_id}/20221122_{mouse_id}_CRC3functional_parcellation_rois_dict.h5')
#/data/Lena/WideFlow_prj/MNL/20221122_{mouse_id}_CRC3functional_parcellation_rois_dict.h5
#/data/Rotem/WideFlow prj/2680/functional_parcellation_rois_dict_left_hemi.h5
#/data/Lena/WideFlow_prj/FL/FLfunctional_parcellation_rois_dict_CRC3.h5
session_meta = {}
session_meta[f'{session_id}'] = {"timestamp": timestamp, "cue": cue, "metric_result": metric_result,
                              "threshold": threshold, "serial_readout": serial_readout_correct, "rois_dict": functional_rois_dict, "metric_roi": metric_roi}

dt = np.mean(np.diff(timestamp))

sessions_data = {}
dataset_path = '/data/Lena/WideFlow_prj/Results/sessions_xxx.h5'
#'/data/Rotem/WideFlow prj/results/sessions_20220320.h5'
#/data/Lena/WideFlow_prj/Results/sessions_xxx.h5
with h5py.File(dataset_path, 'r') as f:
    sessions_data[f'{session_id}'] = {}
    decompose_h5_groups_to_dict(f, sessions_data[f'{session_id}'], f'/{mouse_id}/{session_id}/')

fig, ax = plt.subplots()

cues1 = session_meta[f'{session_id}']['cue']
# pstr = calc_pstr(cues1,  sessions_data['20221224_MNL_NF11']['post_session_analysis']['dff_delta5']['zscore'], 10)
pstr = {}
for i, key in enumerate(session_meta[f'{session_id}']['rois_dict'].keys()):
    pstr[key] = np.mean(
        calc_pstr(cues1, sessions_data[f'{session_id}']['post_session_analysis']['dff_delta5']['zscore'][i], delta_t=10)
        , axis=0)

#pstr = np.mean(calc_pstr(cues,  sessions_data['20221224_MNL_NF11']['post_session_analysis']['dff_delta5']['zscore'], dt)
#        , axis=0)

#plot_pstr(ax, pstr, dt, bold_list=['roi_16'], proximity_dict={}, color_code='turbo', fig=fig)
#bold_list=[session_meta[f'{session_id}']['metric_roi']]
#bold_list=['roi_02']

#def plot_traces(ax, rois_traces_dict, dt, bold_list=[], proximity_dict={}, color_code='turbo', **kwargs):



fig.suptitle(f'Traces_{session_id}')
# plt.savefig(f'{base_path}/{mouse_id}/simple_figs/PSTR_{session_id}')


plt.show()
#plt.savefig(f'/data/Lena/WideFlow_prj/Figures_Rotem/PSTR_{mouse_id}_{session_id}')

# fig, ax = plt.subplots()
# plot_pstr(ax, pstr, dt, bold_list=['roi_02'], proximity_dict={}, color_code='turbo', fig=None)
# plt.show()