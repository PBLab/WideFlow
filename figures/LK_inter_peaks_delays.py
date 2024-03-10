import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from analysis.utils.generate_color_list import *
import seaborn as sns

import h5py
from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict
from analysis.utils.peaks_finder import find_trace_peaks

from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
from scipy.signal import savgol_filter

# feedback delay
delay_time = 32
delay_delta = 3

# mesoscale states transition typical time
# base_path = '/data/Rotem/WideFlow prj'
# dataset_path = '/data/Rotem/WideFlow prj/results/sessions_20220320.h5'
# mice_ids = ['2604',
#             '2601',
#             '2680']
#
# sessions_ids = [
#     '20220320_neurofeedback',
#     '20220321_neurofeedback',
#     '20220322_neurofeedback',
#     '20220324_neurofeedback'
#     ]

base_path = '/data/Lena/WideFlow_prj'
dataset_path = '/data/Lena/WideFlow_prj/Results/results_exp2_noMH.h5'
mice_ids = ['21ML','31MN','54MRL','63MR','64ML']

sessions_ids = [    'spont_mockNF_NOTexcluded_closest'
    ,'NF1'
    ,'NF2'
    ,'NF3'
    ,'NF4'
    , 'NF5'
                ]
dates_vec = [
    '20230604'
    ,'20230611'
    ,'20230612'
    ,'20230613'
    ,'20230614'
    ,'20230615'
             ]

# peaks finder hyperparameters
height_th, min_dst, prominence_th = 0.01, None, 0.005

widths = []  # prominent peaks width
within_inds_diff = []  # time interval between prominent peaks of the same roi
cross_rois_inds_diff = []  # time interval between prominent peaks of all rois
widths_accum = []
accum_inds_diff = []


for mouse_id in mice_ids:
    for sess_id,date in zip(sessions_ids,dates_vec):
        [timestamp, cue, metric_result, threshold, serial_readout] = \
            extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{date}_{mouse_id}_{sess_id}/metadata.txt')
        dt = np.mean(np.diff(timestamp)) * 2 * 1000  # factor by two for the two channels and convert to milli

        data = {}
        with h5py.File(dataset_path, 'a') as file:
            decompose_h5_groups_to_dict(file, data, f'/{mouse_id}/{date}_{mouse_id}_{sess_id}/')
        rois_traces = data['rois_traces']['channel_0']

        cross_rois_inds = []
        traces_accum = np.ndarray(rois_traces['roi_01'].shape)
        for roi_key, trace in rois_traces.items():
            trace_filt = savgol_filter(trace, window_length=7, polyorder=2)
            traces_accum = traces_accum + trace_filt

            peaks, peaks_inds, peaks_props = find_trace_peaks(trace_filt, height_th, min_dst, prominence_th)
            widths.extend(np.array(peaks_props['widths']) * dt)
            within_inds_diff.extend(np.diff(peaks_inds) * dt)

            cross_rois_inds.extend(peaks_inds)

        traces_accum = traces_accum / len(rois_traces)
        peaks, peaks_inds, peaks_props = find_trace_peaks(traces_accum, height_th, min_dst, prominence_th)
        widths_accum.extend(peaks_props['widths'] * dt)
        accum_inds_diff.extend(np.diff(peaks_inds) * dt)

        cross_rois_inds.sort()
        cross_rois_inds = list(dict.fromkeys(cross_rois_inds))
        cross_rois_inds_diff.extend(np.diff(cross_rois_inds) * dt)

t_cut_off = 12000

within_inds_diff.sort()
within_inds_diff = np.array(within_inds_diff)
within_inds_diff = within_inds_diff[:np.where(within_inds_diff > t_cut_off)[0][0]]

cross_rois_inds_diff.sort()
cross_rois_inds_diff = np.array(cross_rois_inds_diff)
cross_rois_inds_diff = cross_rois_inds_diff[:np.where(cross_rois_inds_diff > t_cut_off)[0][0]]

accum_inds_diff.sort()
accum_inds_diff = np.array(accum_inds_diff)
accum_inds_diff = accum_inds_diff[:np.where(accum_inds_diff > t_cut_off)[0][0]]

widths.sort()
widths = np.array(widths)
widths = widths[:np.where(widths > t_cut_off)[0][0]]

#ploting
f = plt.figure(constrained_layout=True, figsize=(16,8))
gs = f.add_gridspec(3, 3)
nbins = 100
ax1 = f.add_subplot(gs[1, 2])
ax2 = f.add_subplot(gs[2, 2])
sns.kdeplot(within_inds_diff, ax=ax2, color='blue', log_scale=True, shade=True)
sns.kdeplot(cross_rois_inds_diff, ax=ax2, color='green', log_scale=True, shade=True)
sns.kdeplot(widths, ax=ax2, color='orange', log_scale=True, shade=True)
# sns.kdeplot(accum_inds_diff, ax=ax2, color='red', log_scale=True, shade=True)

ax2.set_yticks([])
ax2.set_xticks([10, delay_time, 1e2, 1e3, 1e4])
ax2.set_xticklabels(['10', 'delay\ntime', '1e2', '1e3', '1e4'])
ax2.vlines(delay_time, 0, 2.5, color='k', linestyles='--')
# ax2.vlines(delay_time + exp_time / 2 + readout_time, 0, 2.5, color='k', linestyles='--')

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.set_ylabel("")
ax2.set_xlabel("Time [ms]", fontsize=14)
ax2.legend(['inter peaks time difference\n(within ROI)', 'inter peaks time difference\n(cross ROI)', 'peaks width'])
ax2.set_xlim(20, t_cut_off+3000)

# # plot traces
#
# lim1, lim2 = 20770, 20770+100
# mouse_id = mice_ids[0]
# sess_id = sessions_ids[-1]
# date = dates_vec[-1]
# with h5py.File(dataset_path, 'a') as file:
#     decompose_h5_groups_to_dict(file, data, f'/{mouse_id}/{date}_{mouse_id}_{sess_id}/')
# rois_traces = data['rois_traces']['channel_0']
#
# # trace1 should have two peaks
# trace1 = rois_traces['roi_90']
# trace1 = savgol_filter(trace1, window_length=7, polyorder=2)
# trace1 = trace1[lim1:lim2]
# _, peaks_inds1, peaks_props1 = find_trace_peaks(trace1, 0, min_dst, prominence_th)
#
# # trace2 should have one peak that lays between the two peaks of trace1
# trace2 = rois_traces['roi_45']
# trace2 = savgol_filter(trace2, window_length=7, polyorder=2)
# trace2 = trace2[lim1:lim2]
# _, peaks_inds2, peaks_props2 = find_trace_peaks(trace2, 0, min_dst, prominence_th)
#
# mn = np.min(trace1)
# mx = np.max(trace1)
# ax1.plot(trace1, color='k')
# ax1.vlines(peaks_inds1[0], mn, mx, color='k', linestyles='--')
# ax1.vlines(peaks_inds1[1], mn, peaks_props1['peak_heights'][1], color='k', linestyles='--')
# ax1.hlines(mn,
#           peaks_inds1[0],
#           peaks_inds1[1],
#           color='blue', linewidth=2)
#
# ax1.plot(trace2, color='gray')
# ax1.vlines(peaks_inds2[0], mn, peaks_props2['peak_heights'][0], color='gray', linestyles='--')
# ax1.hlines(peaks_props2['peak_heights'][0] - 0.008,
#           peaks_inds2[0],
#           peaks_inds1[0],
#           color='green', linewidth=2)
#
# ax1.hlines(peaks_props1['width_heights'][1],
#           peaks_inds1[1] - peaks_props2['widths'][1] / 2,
#           peaks_inds1[1] + peaks_props2['widths'][1] / 2,
#           color='orange', linewidth=2)

########################################

# ax.plot(trace_accum, color='gray')
# ax.vlines(peaks_inds_accum[0], -0.02, 0.12 + peaks_props_accum['peak_heights'][0], color='k', linestyles='--')
# ax.vlines(peaks_inds_accum[1], -0.02, 0.12 + peaks_props_accum['peak_heights'][1], color='k', linestyles='--')
# ax.hlines(peaks_props_accum['peak_heights'][0],
#           peaks_inds_accum[0],
#           peaks_inds_accum[1],
#           color='red')


#ax1.set_yticks([trace1[0], trace2[0]])
#ax1.set_yticklabels(["ROI 1", "ROI 2"], fontsize=14)
#ax1.set_xticks([])
#ax1.spines['top'].set_visible(False)
#ax1.spines['left'].set_visible(False)
#ax1.spines['right'].set_visible(False)
#ax1.spines['bottom'].set_visible(False)

#plt.show()

plt.rcParams['svg.fonttype'] = 'none'  # or 'path' or 'none'
plt.savefig( f'/data/Lena/WideFlow_prj/Figs_for_paper/inter_peaks_delays_histogram_LK_data.svg',format='svg',dpi=500)
