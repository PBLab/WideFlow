import numpy as np
from skimage.transform import resize
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import h5py

from utils.load_tiff import load_tiff
from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict

from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
from utils.load_config import load_config
from utils.load_rois_data import load_rois_data
from analysis.plots import *
from utils.paint_roi import paint_roi


mouse_id = '2604'
session_id = '20220321_neurofeedback'

# load cortex map and mask
base_path = '/data/Rotem/WideFlow prj'
cortex_map_path = f'{base_path}/{mouse_id}/functional_parcellation_cortex_map.h5'
rois_dict_path = f'{base_path}/{mouse_id}/functional_parcellation_rois_dict_left_hemi.h5'
with h5py.File(cortex_map_path, 'r') as f:
    cortex_mask = f["mask"][()]
    cortex_map = f["map"][()]
cortex_mask = cortex_mask[:, :168]
cortex_map = cortex_map[:, :168]
cortex_map = skeletonize(cortex_map)
rois_dict = load_rois_data(rois_dict_path)

# load data
data = {}
with h5py.File('/data/Rotem/WideFlow prj/results/sessions_20220320.h5', 'r') as f:
    decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/{session_id}/')
rois_metric_traces = data['post_session_analysis']['dff_delta5']['zscore']
# convert to dict assuming rois are sorted
rois_metric_traces_dict = {}
for i, key in enumerate(rois_dict.keys()):
    rois_metric_traces_dict[key] = rois_metric_traces[i]

# load session indices
session_path = f'/data/Rotem/WideFlow prj/{mouse_id}/{session_id}/'
[timestamp, cue, metric_result, threshold, serial_readout] = extract_from_metadata_file(session_path + 'metadata.txt')
cue = np.array(cue)
serial_readout = 1 - np.array(serial_readout)
dt = np.mean(np.diff(timestamp)) / 60
cues_inds = np.where(np.array(cue) == 1)[0]
cues_inds = np.array(cues_inds/2, dtype=np.int32) # correction for one channel

# load session config
config = load_config(session_path + 'session_config.json')
metric_roi = config['analysis_pipeline_config']['args']['metric_args'][1][0]
metric_outline = np.unravel_index(rois_dict[metric_roi]['outline'], (cortex_map.shape[1], cortex_map.shape[0]))


# load analysis dff
vid = load_tiff(session_path + '/post_analysis_results/dff_blue.tif')
n_frames = vid.shape[0]

non_rewards_inds = [11400, 26115]
frame0 = resize(vid[cues_inds[0]], cortex_map.shape)
frame0_metric = {key: val[cues_inds[0]] for key, val in rois_metric_traces_dict.items()}

frame1 = resize(vid[non_rewards_inds[0]], cortex_map.shape)
frame1_metric = {key: val[non_rewards_inds[0]] for key, val in rois_metric_traces_dict.items()}

frame2 = resize(vid[cues_inds[int(len(cues_inds)/2)]], cortex_map.shape)
frame2_metric = {key: val[cues_inds[int(len(cues_inds)/2)]] for key, val in rois_metric_traces_dict.items()}

frame3 = resize(vid[non_rewards_inds[1]], cortex_map.shape)
frame3_metric = {key: val[non_rewards_inds[1]] for key, val in rois_metric_traces_dict.items()}

frame4 = resize(vid[cues_inds[-1]], cortex_map.shape)
frame4_metric = {key: val[cues_inds[-1]] for key, val in rois_metric_traces_dict.items()}

vmax = np.max(np.stack((frame0, frame1, frame2, frame3, frame4)))
vmin = np.min(np.stack((frame0, frame1, frame2, frame3, frame4)))
zvmax = 3#np.max(rois_metric_traces)
zvmin = -3#np.min(rois_metric_traces)
del vid
del rois_metric_traces

conv_ker = np.ones((5, 5)) / 25
# f = plt.figure(constrained_layout=True, figsize=(11, 6))
f = plt.figure()
gs = f.add_gridspec(3, 5)

ax_bottom = f.add_subplot(gs[2, :])
plot_session(ax_bottom, metric_result, cue, serial_readout, np.empty((len(threshold),))*np.nan, dt)
ax_bottom.legend(['metric', 'metric threshold', 'rewards timing', 'licking timing'])
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
axt = ax_bottom.twiny()
axt.set_xlim(ax_bottom.get_xlim())
t = np.arange(0, dt * len(metric_result), dt)
ticks = [t[2*cues_inds[0]], t[2*non_rewards_inds[0]], t[2*cues_inds[int(len(cues_inds)/2)]], t[2*non_rewards_inds[1]], t[2*cues_inds[-1]]]
axt.set_xticks(ticks)
axt.set_xticklabels(['1', '2', '3', '4', '5'], fontsize=14)
axtx = ax_bottom.twinx()
axtx.plot(t, threshold, 'r')
axtx.tick_params(axis='y', colors='r')
ax_bottom.set_ylabel('Z-score', fontsize=14)
ax_bottom.set_xlabel('Time [minutes]', fontsize=14)
plt.yticks(fontsize=12)


ax_top00 = f.add_subplot(gs[0, 0])
ax_top00.set_title('1           ')
ax_top00.axis('off')
ax_top10 = f.add_subplot(gs[1, 0])
ax_top10.axis('off')
ax_top00.scatter(metric_outline[0], metric_outline[1], marker='.', s=0.5, c='k')
ax_top10.scatter(metric_outline[0], metric_outline[1], marker='.', s=0.5, c='k')
wf_imshow(ax_top00, frame0, mask=cortex_mask, map=cortex_map, show_cb=False, conv_ker=conv_ker, cm_name='inferno', vmin=vmin, vmax=vmax)
_, _, Z01 = paint_roi(rois_dict, cortex_map, list(rois_dict.keys()), frame0_metric)
wf_imshow(ax_top10, Z01, mask=cortex_mask, map=cortex_map, show_cb=False, cm_name='inferno', vmin=zvmin, vmax=zvmax)

ax_top01 = f.add_subplot(gs[0, 1])
ax_top01.set_title('2           ')
ax_top01.axis('off')
ax_top11 = f.add_subplot(gs[1, 1])
ax_top11.axis('off')
ax_top01.scatter(metric_outline[0], metric_outline[1], marker='.', s=0.5, c='k')
ax_top11.scatter(metric_outline[0], metric_outline[1], marker='.', s=0.5, c='k')
wf_imshow(ax_top01, frame1, mask=cortex_mask, map=cortex_map, show_cb=False, conv_ker=conv_ker, cm_name='inferno', vmin=vmin, vmax=vmax)
_, _, Z01 = paint_roi(rois_dict, cortex_map, list(rois_dict.keys()), frame1_metric)
wf_imshow(ax_top11, Z01, mask=cortex_mask, map=cortex_map, show_cb=False, cm_name='inferno', vmin=zvmin, vmax=zvmax)

ax_top02 = f.add_subplot(gs[0, 2])
ax_top02.set_title('3           ')
ax_top02.axis('off')
ax_top12 = f.add_subplot(gs[1, 2])
ax_top12.axis('off')
ax_top02.scatter(metric_outline[0], metric_outline[1], marker='.', s=0.5, c='k')
ax_top12.scatter(metric_outline[0], metric_outline[1], marker='.', s=0.5, c='k')
wf_imshow(ax_top02, frame2, mask=cortex_mask, map=cortex_map, show_cb=False, conv_ker=conv_ker, cm_name='inferno', vmin=vmin, vmax=vmax)
_, _, Z01 = paint_roi(rois_dict, cortex_map, list(rois_dict.keys()), frame2_metric)
wf_imshow(ax_top12, Z01, mask=cortex_mask, map=cortex_map, show_cb=False, cm_name='inferno', vmin=zvmin, vmax=zvmax)

ax_top03 = f.add_subplot(gs[0, 3])
ax_top03.set_title('4           ')
ax_top03.axis('off')
ax_top13 = f.add_subplot(gs[1, 3])
ax_top13.axis('off')
ax_top03.scatter(metric_outline[0], metric_outline[1], marker='.', s=0.5, c='k')
ax_top13.scatter(metric_outline[0], metric_outline[1], marker='.', s=0.5, c='k')
wf_imshow(ax_top03, frame3, mask=cortex_mask, map=cortex_map, show_cb=False, conv_ker=conv_ker, cm_name='inferno', vmin=vmin, vmax=vmax)
_, _, Z01 = paint_roi(rois_dict, cortex_map, list(rois_dict.keys()), frame3_metric)
wf_imshow(ax_top13, Z01, mask=cortex_mask, map=cortex_map, show_cb=False, cm_name='inferno', vmin=zvmin, vmax=zvmax)

ax_top04 = f.add_subplot(gs[0, 4])
ax_top04.set_title('5           ')
ax_top04.axis('off')
ax_top14 = f.add_subplot(gs[1, 4])
ax_top14.axis('off')
_, _, cax04 = wf_imshow(ax_top04, frame2, mask=cortex_mask, map=cortex_map, show_cb=True, conv_ker=conv_ker, cm_name='inferno', vmin=vmin, vmax=vmax, cb_side='right')
ax_top04.scatter(metric_outline[0], metric_outline[1], marker='.', s=0.5, c='k')
plt.yticks(fontsize=12)

ax_top14.scatter(metric_outline[0], metric_outline[1], marker='.', s=0.5, c='k')
_, _, Z01 = paint_roi(rois_dict, cortex_map, list(rois_dict.keys()), frame4_metric)
_, _, cax14 = wf_imshow(ax_top14, Z01, mask=cortex_mask, map=cortex_map, show_cb=True, cm_name='inferno', vmin=zvmin, vmax=zvmax, cb_side='right')
plt.yticks(fontsize=12)

# plt.tight_layout()
# plt.savefig('/data/Rotem/WideFlow prj/results/figures/session_performance.png')
plt.show()