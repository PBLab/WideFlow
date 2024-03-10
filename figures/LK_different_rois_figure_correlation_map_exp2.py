import numpy as np
from skimage.transform import resize
from skimage.morphology import skeletonize
from scipy.ndimage.filters import maximum_filter1d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py

from utils.load_tiff import load_tiff
from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict

from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
from analysis.utils.peristimulus_time_response import calc_pstr
from utils.load_config import load_config
from utils.load_rois_data import load_rois_data
from analysis.plots import plot_traces, wf_imshow
from analysis.utils.rois_proximity import calc_rois_proximity
from utils.paint_roi import paint_roi


def plot_pstr(ax, rois_pstr_dict, dt, bold_list=[], proximity_dict={}, color_code='turbo', cbar_show=True):
    delta_t2 = len(rois_pstr_dict[list(rois_pstr_dict.keys())[0]])
    delta_t = np.floor(delta_t2/2)
    dt = dt * 1000  # convert to milliseconds
    t = np.linspace(-delta_t * dt, (delta_t + 1) * dt, delta_t2)

    cmap = plot_traces(ax, rois_pstr_dict, t, bold_list, proximity_dict, color_code)
    rois_pstr_mat = np.array(list(rois_pstr_dict.values()))
    ax.axvline(x=0, ymin=0, ymax=np.max(rois_pstr_mat), color='k')

    ax.set_title('Peristimulus Time Response')
    ax.set_ylabel("PSTR", fontsize=14)
    ax.set_xlabel("Time [ms]", fontsize=14)
    ax.set_yticks([-1, 0, 2, 4])
    ax.set_yticklabels(["-1", "0", "2", "4"], fontsize=14)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    if cbar_show:
        cbar = plt.colorbar(sm)
        # cbar.set_ticks([])
        return cbar

    return None


delta_t = 10

mouse_id = '31MN'
session_id0 = '20230604_31MN_spont'
session_id1 = '20230604_31MN_spont'
session_id2 = '20230604_31MN_spont'
session_id3 = '20230604_31MN_spont'
sessions_id = [session_id0, session_id1, session_id2, session_id3]
base_path = '/data/Lena/WideFlow_prj'

session_path0 = f'/data/Lena/WideFlow_prj/20230604/{mouse_id}/{session_id0}/'
session_path1 = f'/data/Lena/WideFlow_prj/20230604/{mouse_id}/{session_id1}/'
session_path2 = f'/data/Lena/WideFlow_prj/20230604/{mouse_id}/{session_id2}/'
session_path3 = f'/data/Lena/WideFlow_prj/20230604/{mouse_id}/{session_id3}/'

functional_cortex_map_path = f'{base_path}/{mouse_id}/functional_parcellation_cortex_map.h5'
functional_rois_dict_path = f'{base_path}/{mouse_id}/functional_parcellation_rois_dict.h5'
with h5py.File(functional_cortex_map_path, 'r') as f:
    functional_cortex_mask = f["mask"][()]
    functional_cortex_map = f["map"][()]
functional_cortex_mask = functional_cortex_mask[:, :168]
functional_cortex_map = functional_cortex_map[:, :168]
functional_cortex_map = skeletonize(functional_cortex_map)
functional_rois_dict = load_rois_data(functional_rois_dict_path)

cortex_map_path = '/data/Rotem/Wide Field/WideFlow/data/cortex_map/allen_2d_cortex.h5'
rois_dict_path = '/data/Rotem/Wide Field/WideFlow/data/cortex_map/allen_2d_cortex_rois_left_hemi.h5'
with h5py.File(cortex_map_path, 'r') as f:
    cortex_mask = np.transpose(f["mask"][()])
    cortex_map = np.transpose(f["map"][()])
cortex_mask = cortex_mask[:, :168]
cortex_map = cortex_map[:, :168]
cortex_map = skeletonize(cortex_map)
rois_dict = load_rois_data(rois_dict_path)

# dataset_path0 = '/data/Rotem/WideFlow prj/results/archive/sessions_dataset_new.h5'
dataset_path0 = '/data/Lena/WideFlow_prj/Results/results_exp2.h5'
dataset_path1 = '/data/Lena/WideFlow_prj/Results/results_exp2.h5'
dataset_path2 = '/data/Lena/WideFlow_prj/Results/results_exp2.h5'
dataset_path3 = '/data/Lena/WideFlow_prj/Results/results_exp2.h5'

###################################################################################
################################ prepare data #####################################
sessions_meta = {}

[timestamp, cue, metric_result, threshold, serial_readout] = extract_from_metadata_file(session_path0 + 'metadata.txt')
serial_readout = 1 - np.array(serial_readout)
serial_readout = maximum_filter1d(serial_readout, 2)[::2]
cue = maximum_filter1d(cue, 2)[::2]
config = load_config(session_path0 + 'session_config.json')

# closest = config["supplementary_data_config"]["closest_rois"]
# for key in closest:
#     del functional_rois_dict[key]

if 'metric_args' in config['analysis_pipeline_config']['args']:

    map_temp = functional_cortex_map
    rois_dict_temp = functional_rois_dict
# if config['supplementary_data_config']['rois_dict_path'].endswith('functional_parcellation_rois_dict_left_hemi.h5'):
#     map_temp = functional_cortex_map
#     rois_dict_temp = functional_rois_dict
else:
    map_temp = cortex_map
    rois_dict_temp = rois_dict
metric_roi = config['analysis_pipeline_config']['args']['metric_args'][1][0]
#metric_roi = 'roi_106'

dt = np.mean(np.diff(timestamp))
rois_proximity = calc_rois_proximity(rois_dict_temp, metric_roi)
sessions_meta[session_id0] = {"timestamp": timestamp, "cue": cue, "metric_result": metric_result,
                              "threshold": threshold, "serial_readout": serial_readout, "config": config,
                              "metric_roi": metric_roi, "map": map_temp, "rois_dict": rois_dict_temp, "dt": dt,
                              "rois_proximity": rois_proximity}

[timestamp, cue, metric_result, threshold, serial_readout] = extract_from_metadata_file(session_path1 + 'metadata.txt')
serial_readout = 1 - np.array(serial_readout)
serial_readout = maximum_filter1d(serial_readout, 2)[::2]
cue = maximum_filter1d(cue, 2)[::2]
config = load_config(session_path1 + 'session_config.json')
if 'metric_args' in config['analysis_pipeline_config']['args']:

    map_temp = functional_cortex_map
    rois_dict_temp = functional_rois_dict
else:
    map_temp = cortex_map
    rois_dict_temp = rois_dict
metric_roi = config['analysis_pipeline_config']['args']['metric_args'][1][0]
#metric_roi = 'roi_106'

dt = np.mean(np.diff(timestamp))
rois_proximity = calc_rois_proximity(rois_dict_temp, metric_roi)
sessions_meta[session_id1] = {"timestamp": timestamp, "cue": cue, "metric_result": metric_result,
                              "threshold": threshold, "serial_readout": serial_readout, "config": config,
                              "metric_roi": metric_roi, "map": map_temp, "rois_dict": rois_dict_temp, "dt": dt,
                              "rois_proximity": rois_proximity}

[timestamp, cue, metric_result, threshold, serial_readout] = extract_from_metadata_file(session_path2 + 'metadata.txt')
serial_readout = 1 - np.array(serial_readout)
serial_readout = maximum_filter1d(serial_readout, 2)[::2]
cue = maximum_filter1d(cue, 2)[::2]
config = load_config(session_path2 + 'session_config.json')
if 'metric_args' in config['analysis_pipeline_config']['args']:
    map_temp = functional_cortex_map
    rois_dict_temp = functional_rois_dict
else:
    map_temp = cortex_map
    rois_dict_temp = rois_dict
metric_roi = config['analysis_pipeline_config']['args']['metric_args'][1][0]
#metric_roi = 'roi_106'

dt = np.mean(np.diff(timestamp))
rois_proximity = calc_rois_proximity(rois_dict_temp, metric_roi)
sessions_meta[session_id2] = {"timestamp": timestamp, "cue": cue, "metric_result": metric_result,
                              "threshold": threshold, "serial_readout": serial_readout, "config": config,
                              "metric_roi": metric_roi, "map": map_temp, "rois_dict": rois_dict_temp, "dt": dt,
                              "rois_proximity": rois_proximity}

[timestamp, cue, metric_result, threshold, serial_readout] = extract_from_metadata_file(session_path3 + 'metadata.txt')
serial_readout = 1 - np.array(serial_readout)
serial_readout = maximum_filter1d(serial_readout, 2)[::2]
cue = maximum_filter1d(cue, 2)[::2]
config = load_config(session_path2 + 'session_config.json')
if 'metric_args' in config['analysis_pipeline_config']['args']:
    map_temp = functional_cortex_map
    rois_dict_temp = functional_rois_dict
else:
    map_temp = cortex_map
    rois_dict_temp = rois_dict
metric_roi = config['analysis_pipeline_config']['args']['metric_args'][1][0]
#metric_roi = 'roi_106'

dt = np.mean(np.diff(timestamp))
rois_proximity = calc_rois_proximity(rois_dict_temp, metric_roi)
sessions_meta[session_id3] = {"timestamp": timestamp, "cue": cue, "metric_result": metric_result,
                              "threshold": threshold, "serial_readout": serial_readout, "config": config,
                              "metric_roi": metric_roi, "map": map_temp, "rois_dict": rois_dict_temp, "dt": dt,
                              "rois_proximity": rois_proximity}

del timestamp, cue, metric_result, threshold, serial_readout, config, map_temp, rois_dict_temp

sessions_data = {}
with h5py.File(dataset_path0, 'r') as f:
    sessions_data[session_id0] = {}
    decompose_h5_groups_to_dict(f, sessions_data[session_id0], f'/{mouse_id}/{session_id0}/')

with h5py.File(dataset_path1, 'r') as f:
    sessions_data[session_id1] = {}
    decompose_h5_groups_to_dict(f, sessions_data[session_id1], f'/{mouse_id}/{session_id1}/')

with h5py.File(dataset_path2, 'r') as f:
    sessions_data[session_id2] = {}
    decompose_h5_groups_to_dict(f, sessions_data[session_id2], f'/{mouse_id}/{session_id2}/')

with h5py.File(dataset_path3, 'r') as f:
    sessions_data[session_id3] = {}
    decompose_h5_groups_to_dict(f, sessions_data[session_id3], f'/{mouse_id}/{session_id3}/')

for sess_id in sessions_id:
    pstr_cat = {}
    for i, key in enumerate(sessions_meta[sess_id]['rois_dict'].keys()):
        pstr_mat = calc_pstr(sessions_meta[sess_id]['cue'], sessions_data[sess_id]['post_session_analysis']['dff_delta5']['zscore'][i], delta_t)
        # pstr_mat = calc_pstr(sessions_meta[sess_id]['cue'], sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i], delta_t)
        pstr_mat = pstr_mat[:, :int(pstr_mat.shape[1]/2)+1]  # calculate based on pre-stim only
        pstr_cat[key] = np.reshape(pstr_mat, newshape=(np.prod(pstr_mat.shape), ))

    metric_roi = sessions_meta[sess_id]['metric_roi']
    metric_index = int(metric_roi[4:]) - 1
    metric_corr = {}
    metric_trace = sessions_data[sess_id]['post_session_analysis']['dff_delta5']['zscore'][metric_index]
    for i, (key, val) in enumerate(sessions_meta[sess_id]['rois_proximity'].items()):
        metric_corr[key] = np.corrcoef(pstr_cat[key], pstr_cat[metric_roi])[0, 1]  # correlation with metric ROI
        # metric_corr[key] = np.corrcoef(metric_trace, sessions_data[sess_id]['post_session_analysis']['dff']['traces'][i])[0, 1]
    sessions_data[sess_id]["metric_corr"] = metric_corr
    sessions_data[sess_id]["zscore_corr"] = np.corrcoef(sessions_data[sess_id]['post_session_analysis']['dff_delta5']['zscore'])
    sessions_data[sess_id]["dff_corr"] = np.corrcoef(sessions_data[sess_id]['post_session_analysis']['dff']['traces'])


###################################################################################
############################### start plotting ####################################
f = plt.figure(constrained_layout=True, figsize=(11, 6))
gs = f.add_gridspec(2, 3)

#########################################################################
sess_id = session_id0
metric_outline = np.unravel_index(sessions_meta[sess_id]['rois_dict'][sessions_meta[sess_id]['metric_roi']]['outline'],
                                    (sessions_meta[sess_id]['map'].shape[1], sessions_meta[sess_id]['map'].shape[0]))
# metric_outline = (sessions_meta[sess_id]['rois_dict'][sessions_meta[sess_id]['metric_roi']]['outline'][:, 0],
#                   sessions_meta[sess_id]['rois_dict'][sessions_meta[sess_id]['metric_roi']]['outline'][:, 1])
ax_left0 = f.add_subplot(gs[0, 0])
_, _, pmap0 = paint_roi(sessions_meta[sess_id]['rois_dict'],
                      sessions_meta[sess_id]['map'],
                      list(sessions_meta[sess_id]['rois_dict'].keys()),
                      sessions_data[sess_id]['metric_corr'])
pmap0[cortex_mask==0] = None
im, _ = wf_imshow(ax_left0, pmap0, mask=None, map=None, conv_ker=None, show_cb=False, cm_name='inferno', vmin=None, vmax=None, cb_side='right')
ax_left0.set_ylabel("PSTR", fontsize=12)
ax_left0.scatter(metric_outline[1], metric_outline[0], marker='.', s=0.5, c='k')
ax_left0.axis('off')

ax_left1 = f.add_subplot(gs[1, 0])
cues = sessions_meta[sess_id]['cue']
pstr = {}
for i, key in enumerate(sessions_meta[sess_id]['rois_dict'].keys()):
    pstr[key] = np.mean(
        calc_pstr(cues, sessions_data[sess_id]['post_session_analysis']['dff_delta5']['zscore'][i], delta_t)
        , axis=0)
plot_pstr(ax_left1, pstr, sessions_meta[sess_id]['dt'], bold_list=[sessions_meta[sess_id]['metric_roi']],
          proximity_dict=sessions_meta[sess_id]['rois_proximity'], color_code='RdBu', cbar_show=False)
ax_left1.set_title("")

mock_lines = [Line2D([0], [0], linestyle='dashed', color='black'),
              Line2D([0], [0], linestyle='dotted', color='black')]

ax_left1.legend(mock_lines, ["mean", "median"])
ax_left1.set_xticks([-400, -200, 0, 200, 400])
ax_left1.set_xticklabels(["-400", "-200", "0", "200", "400"], fontsize=12)
ax_left1.spines['top'].set_visible(False)
ax_left1.spines['right'].set_visible(False)

#########################################################################
sess_id = session_id1
metric_outline = np.unravel_index(sessions_meta[sess_id]['rois_dict'][sessions_meta[sess_id]['metric_roi']]['outline'],
                                   (sessions_meta[sess_id]['map'].shape[1], sessions_meta[sess_id]['map'].shape[0]))
ax_med0 = f.add_subplot(gs[0, 1])
_, _, pmap0 = paint_roi(sessions_meta[sess_id]['rois_dict'],
                      sessions_meta[sess_id]['map'],
                      list(sessions_meta[sess_id]['rois_dict'].keys()),
                      sessions_data[sess_id]["metric_corr"])
pmap0[cortex_mask==0] = None
im, _ = wf_imshow(ax_med0, pmap0, mask=None, map=None, conv_ker=None, show_cb=False, cm_name='inferno', vmin=None, vmax=None, cb_side='right')
ax_med0.scatter(metric_outline[0], metric_outline[1], marker='.', s=0.5, c='k')
ax_med0.axis('off')

ax_med1 = f.add_subplot(gs[1, 1])
cues = sessions_meta[sess_id]['cue']
pstr = {}
for i, key in enumerate(sessions_meta[sess_id]['rois_dict'].keys()):
    pstr[key] = np.mean(
        calc_pstr(cues, sessions_data[sess_id]['post_session_analysis']['dff_delta5']['zscore'][i], delta_t)
        , axis=0)
plot_pstr(ax_med1, pstr, sessions_meta[sess_id]['dt'], bold_list=[sessions_meta[sess_id]['metric_roi']],
          proximity_dict=sessions_meta[sess_id]['rois_proximity'], color_code='RdBu', cbar_show=False)
ax_med1.set_title("")
ax_med1.set_ylabel("")
ax_med1.set_xticks([-400, -200, 0, 200, 400])
ax_med1.set_xticklabels(["-400", "-200", "0", "200", "400"], fontsize=12)
ax_med1.spines['top'].set_visible(False)
ax_med1.spines['right'].set_visible(False)

#########################################################################
sess_id = session_id2
metric_outline = np.unravel_index(sessions_meta[sess_id]['rois_dict'][sessions_meta[sess_id]['metric_roi']]['outline'],
                                   (sessions_meta[sess_id]['map'].shape[1], sessions_meta[sess_id]['map'].shape[0]))
ax_right0 = f.add_subplot(gs[0, 2])
_, _, pmap0 = paint_roi(sessions_meta[sess_id]['rois_dict'],
                      sessions_meta[sess_id]['map'],
                      list(sessions_meta[sess_id]['rois_dict'].keys()),
                      sessions_data[sess_id]["metric_corr"])
pmap0[cortex_mask==0] = None
im, _ = wf_imshow(ax_right0, pmap0, mask=None, map=None, conv_ker=None, show_cb=False, cm_name='inferno', vmin=None, vmax=None, cb_side='right')
cbar = plt.colorbar(im, ax=ax_right0, ticks=[-0.4, 0, 0.4, 1])
cbar.ax.set_yticklabels(["-0.4", "0", "0.4", "1"], fontsize=14)
cbar.ax.set_ylabel("Correlation", fontsize=14)
ax_right0.scatter(metric_outline[0], metric_outline[1], marker='.', s=0.5, c='k')
ax_right0.axis('off')

ax_right1 = f.add_subplot(gs[1, 2])
cues = sessions_meta[sess_id]['cue']
pstr = {}
for i, key in enumerate(sessions_meta[sess_id]['rois_dict'].keys()):
    pstr[key] = np.mean(
        calc_pstr(cues, sessions_data[sess_id]['post_session_analysis']['dff_delta5']['zscore'][i], delta_t)
        , axis=0)

cbar = plot_pstr(ax_right1, pstr, sessions_meta[sess_id]['dt'], bold_list=[sessions_meta[sess_id]['metric_roi']],
          proximity_dict=sessions_meta[sess_id]['rois_proximity'], color_code='RdBu')
ax_right1.set_title("")
ax_right1.set_ylabel("")
ax_right1.set_xticks([-400, -200, 0, 200, 400])
ax_right1.set_xticklabels(["-400", "-200", "0", "200", "400"], fontsize=12)
ax_right1.spines['top'].set_visible(False)
ax_right1.spines['right'].set_visible(False)
cbar.ax.set_ylabel("Morphological Proximitry\n(to Metric ROI)", fontsize=14)
cbar.set_ticks([0, 1])
cbar.ax.set_yticklabels(["close", "far"], fontsize=14, rotation=90)

plt.show()
