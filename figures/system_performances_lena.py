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


def add_line(axis, x, y, height, width_list, color_list, annotation_list=[]):
    rects = []
    if len(annotation_list) == 0:
        annotation_list = [""] * len(width_list)
    for w, c, a in zip(width_list, color_list, annotation_list):
        w = w - w_delta
        rect = Rectangle((x, y), w, height, color=c)
        rects.append(rect)
        axis.add_patch(rect)
        if len(annotation_list) > 0:
            x_ann = x + w/2
            axis.annotate(a, (x_ann, y + 0.5), color='w', ha='center', rotation=0)
        x = x + w + w_delta

    return rects

# profiling parameters
# feedback delay
delay_time = 32
delay_delta = 3
# camera
exp_time = 30
readout_time = 10
# image processing
reg_t, reg_g_t = 8.5, 6.9
mask_t, mask_g_t = 0.5, 0.3
dff_t, dff_g_t = 5.5, 1.6
hemo_t, hemo_g_t = 3, 0.6
img_proc_t = reg_t + mask_t + dff_t + hemo_t
metric_proc_t = 2
# serial communication
ser_r_t = 5
ser_w_t = 2
ser_comm_t = ser_w_t + ser_r_t

# arduino
arduino_t = 6
arduino_r_t = 4
arduino_w_t = 2

# mesoscale states transition typical time
base_path = '/data/Rotem/WideFlow prj'
dataset_path = '/data/Rotem/WideFlow prj/results/sessions_20220320.h5'
mice_ids = ['2604',
            '2601',
            '2680']
sessions_ids = [
    '20220320_neurofeedback',
    '20220321_neurofeedback',
    '20220322_neurofeedback',
    '20220324_neurofeedback'
]

# peaks finder hyperparameters
height_th, min_dst, prominence_th = 0.01, None, 0.005

widths = []  # prominent peaks width
within_inds_diff = []  # time interval between prominent peaks of the same roi
cross_rois_inds_diff = []  # time interval between prominent peaks of all rois
widths_accum = []
accum_inds_diff = []
for mouse_id in mice_ids:
    for sess_id in sessions_ids:
        [timestamp, cue, metric_result, threshold, serial_readout] = \
            extract_from_metadata_file(f'{base_path}/{mouse_id}/{sess_id}/metadata.txt')
        dt = np.mean(np.diff(timestamp)) * 2 * 1000  # factor by two for the two channels and convert to milli

        data = {}
        with h5py.File(dataset_path, 'a') as file:
            decompose_h5_groups_to_dict(file, data, f'/{mouse_id}/{sess_id}/')
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

# plotsssssssssssssssssssssssssssssssssssssssssssssss
f = plt.figure(constrained_layout=True)
gs = f.add_gridspec(5, 3)

# 11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
# Flame graph
height = 2.5
h_delta = 0.1
w_delta = 0.2

# colors = ['orangered', 'lightcoral']
process_colors = generate_gradient_color_list(3, 'red', 'orange')
colors_camera = ['steelblue', 'deepskyblue', 'magenta', 'plum']
serial_colors = ['steelblue', 'deepskyblue', 'gray']
gpu_color = ["limegreen", "lime"]
arduino_color = ["gray"]

ax = f.add_subplot(gs[3:, 1:])
add_line(ax, 0, (height + h_delta) * 0, height,
         [exp_time, readout_time],
         colors_camera[:2],
         ['exposure\ntime', 'sensor\nreadout'])
add_line(ax, 40, (height + h_delta) * 1, height,
         [img_proc_t, metric_proc_t, ser_comm_t],
         [process_colors[1], process_colors[0], "limegreen"],
         ['image\nprocessing', 'metric\neval', 'serial\ncomm'])

add_line(ax, 40 + img_proc_t + metric_proc_t + ser_comm_t, (height + h_delta) * 2, height,
         [arduino_t],
         ['steelblue'],
         ['Arduino\nIteration'])

ax.hlines(4*height, exp_time + readout_time, exp_time + readout_time + 5, colors='k', linewidth=0.5)
ax.hlines(4*height, exp_time + readout_time + delay_time - 5, exp_time + readout_time + delay_time, colors='k', linewidth=0.5)
ax.vlines(exp_time + readout_time, height * 4.2, height * 3.8, colors='k', linewidth=0.5)
ax.vlines(exp_time + readout_time + delay_time, height * 4.2, height * 3.8, colors='k', linewidth=0.5)
ax.annotate(f"system\nfeedback delay time\n {delay_time}[ms] {delay_delta}[ms]",
            (exp_time + readout_time + delay_time / 2, 4.0*height), color='k', ha='center', rotation=0, fontsize=12)

# ax.hlines(3.5*height, exp_time / 2, exp_time + readout_time - 5, colors='k', linewidth=0.5)
# ax.hlines(3.5*height, 55, exp_time + readout_time + delay_time, colors='k', linewidth=0.5)
# ax.vlines(exp_time / 2, height * 3.7, height * 3.3, colors='k', linewidth=0.5)
# ax.vlines(exp_time + readout_time + delay_time, height * 3.7, height * 3.3, colors='k', linewidth=0.5)
# ax.annotate(f"CLN\nfeedback delay time\n {delay_time + readout_time + exp_time/2}[ms] {delay_delta + exp_time/2}[ms]",
#             (exp_time + readout_time - 10 + 15, 3.0*height), color='k', ha='center', rotation=0, fontsize=12)


ax.set_ylim(0, (height + h_delta) * 7)
ax.set_xlim(0, 75)

ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_position('zero')

ax.get_yaxis().set_ticks([])
ax.set_xlabel('Time [ms]', fontsize=14)
ax.tick_params(axis='x', labelsize=12)

# 222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
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

# plot traces

lim1, lim2 = 20770, 20770+100
mouse_id = mice_ids[0]
sess_id = sessions_ids[-1]
with h5py.File(dataset_path, 'a') as file:
    decompose_h5_groups_to_dict(file, data, f'/{mouse_id}/{sess_id}/')
rois_traces = data['rois_traces']['channel_0']

# trace1 should have two peaks
trace1 = rois_traces['roi_10']
trace1 = savgol_filter(trace1, window_length=7, polyorder=2)
trace1 = trace1[lim1:lim2]
_, peaks_inds1, peaks_props1 = find_trace_peaks(trace1, 0, min_dst, prominence_th)

# trace2 should have one peak that lays between the two peaks of trace1
trace2 = rois_traces['roi_45']
trace2 = savgol_filter(trace2, window_length=7, polyorder=2)
trace2 = trace2[lim1:lim2]
_, peaks_inds2, peaks_props2 = find_trace_peaks(trace2, 0, min_dst, prominence_th)

mn = np.min(trace1)
mx = np.max(trace1)
ax1.plot(trace1, color='k')
ax1.vlines(peaks_inds1[0], mn, mx, color='k', linestyles='--')
ax1.vlines(peaks_inds1[1], mn, peaks_props1['peak_heights'][1], color='k', linestyles='--')
ax1.hlines(mn,
          peaks_inds1[0],
          peaks_inds1[1],
          color='blue', linewidth=2)

ax1.plot(trace2, color='gray')
ax1.vlines(peaks_inds2[0], mn, peaks_props2['peak_heights'][0], color='gray', linestyles='--')
ax1.hlines(peaks_props2['peak_heights'][0] - 0.008,
          peaks_inds2[0],
          peaks_inds1[0],
          color='green', linewidth=2)

ax1.hlines(peaks_props1['width_heights'][1],
          peaks_inds1[1] - peaks_props2['widths'][1] / 2,
          peaks_inds1[1] + peaks_props2['widths'][1] / 2,
          color='orange', linewidth=2)

# ax.plot(trace_accum, color='gray')
# ax.vlines(peaks_inds_accum[0], -0.02, 0.12 + peaks_props_accum['peak_heights'][0], color='k', linestyles='--')
# ax.vlines(peaks_inds_accum[1], -0.02, 0.12 + peaks_props_accum['peak_heights'][1], color='k', linestyles='--')
# ax.hlines(peaks_props_accum['peak_heights'][0],
#           peaks_inds_accum[0],
#           peaks_inds_accum[1],
#           color='red')


ax1.set_yticks([trace1[0], trace2[0]])
ax1.set_yticklabels(["ROI 1", "ROI 2"], fontsize=14)
ax1.set_xticks([])
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)


# 333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
ax3 = f.add_subplot(gs[0, 2])
feedback_threshold = 2.5
ledBaseline = 0
adjLedAnalogValMax = 15
ledAnalogValMax = 50

x = np.arange(-feedback_threshold, feedback_threshold, 0.1)
# value sent from computer to the Arduino
y = (1 + np.clip(x / feedback_threshold, -1, 1)) / 2
# Arduino calculation
y = (np.power(y, 8) + ledBaseline) * adjLedAnalogValMax
x = np.concatenate((x, np.arange(feedback_threshold, feedback_threshold + 0.5, 0.1)))
y = np.concatenate((y, np.ones((len(x) - len(y), )) * ledAnalogValMax))

x = np.concatenate((np.arange(-feedback_threshold - 0.5, -feedback_threshold, 0.1), x))
y = np.concatenate((np.ones((len(x) - len(y), ))*np.min(y), y))

ax3.plot(x, y)
ax3.set_xticks([-feedback_threshold, 0, feedback_threshold])
ax3.set_xticklabels(["-th", 0, "th"], fontsize=12)
ax3.set_xlabel("Metric", fontsize=14)

ax3.set_yticks([np.min(y), adjLedAnalogValMax-2, ledAnalogValMax])
ax3.set_yticklabels(["baseline\nintensity", "max\nadjustable\nintensity", "max\nintensity -\n(reward granted)"], fontsize=12)
ax3.set_ylim(np.min(y) - 5, ledAnalogValMax + 5)
ax3.set_ylabel("Illumination\nIntensity", fontsize=14)

ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.grid()

ax3.set_aspect(0.06)
plt.show()


from utils.load_tiff import load_tiff
tiffs_path = f'{base_path}/extras/20220217_CueRewardCoupling/'
[timestamp, cue, metric_result, threshold, serial_readout] = \
    extract_from_metadata_file(f'{tiffs_path}/metadata.txt')
dt = np.mean(np.diff(timestamp)) * 2

ch1_z_prj = np.ndarray((5000, ))
ch2_z_prj = np.ndarray((5000, ))
for i in range(5):
    vid = load_tiff(f'{tiffs_path}/wf_raw_data_{i}.tif')
    ch1 = vid[::2]
    ch2 = vid[1::2]

    ch1_z_prj[i*1000: (i+1)*1000] = np.mean(ch1, axis=(1, 2))
    ch2_z_prj[i*1000: (i+1)*1000] = np.mean(ch2, axis=(1, 2))


ch2_z_prj_pre = ch2_z_prj[98::100][:-1]
ch2_z_prj_peri = ch2_z_prj[99::100][:-1]
ch2_z_prj_post = ch2_z_prj[100::100]

intensity_diff = (ch2_z_prj_peri - ch2_z_prj_pre)/(ch2_z_prj_post - ch2_z_prj_pre)
delay_time = intensity_diff * dt
delay_time_mean = np.mean(delay_time)
delay_time_sd = np.std(delay_time)
