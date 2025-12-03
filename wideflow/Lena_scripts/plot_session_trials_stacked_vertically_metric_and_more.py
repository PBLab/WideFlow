import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib import gridspec

from wideflow.analysis.utils.extract_from_metadata_file import extract_from_metadata_file
from wideflow.analysis.plots import plot_reward_response
from wideflow.utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict

# --- Paths and parameters ---
base_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj'
dataset_path_noMH = f'{base_path}/Results/results_exp3.5.h5'
# dataset_path_noMH = f'{base_path}/Results/results_exp3.5_with_GFP_pre_hemo.h5'
# dataset_path_noMH = f'{base_path}/Results/results_exp3.5_with_GFP_pre_hemo_and_pre_dff_and_baseline.h5'
# dataset_path_noMH = f'{base_path}/Results/results_exp3.5_with_GFP_pre_hemo_and_pre_dff_and_baseline_and_hemo.h5'
# dataset_path_noMH = f'{base_path}/Results/results_exp3.5_with_GFP_pre_hemo_and_pre_dff_and_baseline7.h5'
#dataset_path_noMH = f'{base_path}/Results/results_exp3.4.h5'

date = '20250909'
mouse_id = '260FN'
session_id = f'{date}_{mouse_id}_NF5_p2'
metric_roi = '89_87'

first_frame = 2000
last_frame = 5000

# --- Load metadata ---
[timestamp, cue, metric_result, threshold, serial_readout, trial_number] = extract_from_metadata_file(
    f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt'
)

timestamp = np.array(timestamp[first_frame:last_frame])[::2]
cue = np.array(cue[first_frame:last_frame])
cue = np.maximum(cue[0::2], cue[1::2])
metric_result = np.array(metric_result[first_frame:last_frame])[::2]
threshold = np.array(threshold[first_frame:last_frame])[::2]
serial_readout = np.array(serial_readout[first_frame:last_frame])[::2]
serial_readout_correct = 1 - serial_readout
trial_number = np.array(trial_number[first_frame:last_frame])[::2]

# --- Load H5 datasets ---
d3 = {}
with h5py.File(dataset_path_noMH, 'r') as f:
    decompose_h5_groups_to_dict(f, d3, f'/{mouse_id}/{session_id}/')

# --- Select multiple datasets to compare ---
datasets_to_plot = {
    # 'GFP raw pre-dff': d3['rois_traces']['channel_3'][f'roi_{metric_roi}'][int(first_frame/2):int(last_frame/2)],
    # 'GFP baseline for dff': d3['rois_traces']['channel_4'][f'roi_{metric_roi}'][int(first_frame/2):int(last_frame/2)],
    # 'GFP-channel dff pre-HemoSubstract' : d3['rois_traces']['channel_2'][f'roi_{metric_roi}'][int(first_frame/2):int(last_frame/2)],
    # # 'Hemo raw pre-dff': d3['rois_traces']['channel_5'][f'roi_{metric_roi}'][int(first_frame / 2):int(last_frame / 2)],
    # # 'Hemo baseline for dff': d3['rois_traces']['channel_6'][f'roi_{metric_roi}'][int(first_frame / 2):int(last_frame / 2)],
    # # 'pre-Hemo-corrected violet dff' : d3['rois_traces']['channel_7'][f'roi_{metric_roi}'][int(first_frame/2):int(last_frame/2)],
    # 'Hemo-channel dff (after HemoCorrect)' : d3['rois_traces']['channel_1'][f'roi_{metric_roi}'][int(first_frame/2):int(last_frame/2)],
    'Hemo-substracted GFP dff (final)' : d3['rois_traces']['channel_0'][f'roi_{metric_roi}'][int(first_frame/2):int(last_frame/2)],
    #'Diff10' : d3['post_session_analysis_LK2']['diff10'][f'roi_{metric_roi}'][int(first_frame/2):int(last_frame/2)],
    'Metric': metric_result,
    #'Metric': d3['post_session_analysis_LK2']['zsores_MH_diff10_exc_top15'][f'roi_{metric_roi}'][int(first_frame/2):int(last_frame/2)],
}

# --- Prepare time vector ---
dt = np.mean(np.diff(timestamp))
t = np.arange(first_frame, last_frame) * dt
t = t[::2]

# --- Define grouping of datasets to overlay ---  # <<< ADDED
# Each inner list defines one plot (i.e., datasets to overlay together)
dataset_groups = [
    # ['GFP raw pre-dff', 'GFP baseline for dff'],  # plotted together
    # # ['Hemo-channel dff (after HemoCorrect)' ],
    # ['Hemo-channel dff (after HemoCorrect)','GFP-channel dff pre-HemoSubstract', 'Hemo-substracted GFP dff (final)'],
    ['Hemo-substracted GFP dff (final)'],
    ['Metric'],
    (['Hemo-substracted GFP dff (final)','Metric'],True)
    # ['pre-Hemo-corrected violet dff','Hemo-channel dff (after HemoCorrect)']
    # ['Hemo-corrected dff']# plotted separately
    # ['Hemo raw pre-dff', 'Hemo baseline for dff'],  # plotted together
    # ['Hemo-channel dff']                           # plotted separately
]

#
# # --- Create figure with extra column for histograms ---
# n_datasets = len(datasets_to_plot)
# fig = plt.figure(figsize=(14, 2.5 * n_datasets))
# gs = gridspec.GridSpec(n_datasets, 2, width_ratios=[4, 1], wspace=0.15)
#
# axes = []
# hist_axes = []
#
# # --- Plot each dataset and its histogram ---
# for i, (label, data) in enumerate(datasets_to_plot.items()):
#     ax = fig.add_subplot(gs[i, 0])
#     ax_hist = fig.add_subplot(gs[i, 1], sharey=ax)
#     axes.append(ax)
#     hist_axes.append(ax_hist)
#
#     y = np.array(data)
#
#     # Plot main trace
#     ax.plot(t, y, color='black', alpha=0.8, linewidth=0.7, label=label)
#     # if date != '20250831' and date != '20250903':
#     #     ax.plot(t, threshold, color='green', alpha=0.7, linewidth=1)
#
#     # Overlay cue/reward
#     if date != '20250831' and date != '20250727':
#         plot_reward_response(ax, cue, serial_readout_correct,
#                              ymin=np.min(y), ymax=np.max(y),
#                              c_reward='r',
#                              linewidth_response=0.2,
#                              t=t)
#
#     ax.set_ylim(np.min(y)-0.01, np.max(y)+0.01)
#     ax.set_ylabel(label)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.legend(
#         #loc='upper center',  # position
#         #bbox_to_anchor=(0.5, 1.15),  # optional: move legend above plot
#         ncol=4,  # number of columns → horizontal layout
#         frameon=True,  # remove border (optional)
#         fontsize=8
#     )
#
#     # # --- Histogram on right ---
#     # counts, bins, patches = ax_hist.hist(
#     #     y,
#     #     bins=40,
#     #     orientation='horizontal',
#     #     color='gray',
#     #     alpha=0.6,
#     #     edgecolor='black',
#     #     linewidth=0.3
#     # )
#     counts, bins = np.histogram(y, bins=40)
#
#     # Compute bin centers for plotting
#     bin_centers = 0.5 * (bins[1:] + bins[:-1])
#
#     # Plot as a horizontal line
#     ax_hist.plot(counts, bin_centers, color='gray', linewidth=1.5)
#
#     # Optionally match visual style
#     ax_hist.fill_betweenx(bin_centers, 0, counts, color='gray', alpha=0.2)
#
#     # Optional: adjust axis limits if needed
#     ax_hist.set_ylim(bins[0], bins[-1])
#
#     #ax_hist.hist(y, bins=40, orientation='horizontal', color='gray', alpha=0.6, edgecolor='black',linewidth=0.3)
#     ax_hist.set_xlabel('Count')
#     ax_hist.spines['top'].set_visible(False)
#     ax_hist.spines['right'].set_visible(False)
#     # # --- Find and mark the most common value (mode) ---
#     # bin_centers = 0.5 * (bins[:-1] + bins[1:])
#     # mode_index = np.argmax(counts)
#     # mode_value = bin_centers[mode_index]
#     #
#     # # draw horizontal line at mode value
#     # ax_hist.axhline(mode_value, color='red', linestyle='--', linewidth=0.8, alpha=0.8, zorder=5)
#     #
#     # # add text label for mode value
#     # ax_hist.text(
#     #     ax_hist.get_xlim()[1] * 0.95,  # near right edge of histogram
#     #     mode_value,
#     #     f"mode = {mode_value:.3f}",
#     #     va='bottom',
#     #     ha='right',
#     #     fontsize=14,
#     #     color='red',
#     #     alpha=0.8
#     # )
#
#     # --- Find and mark the average (mean) value ---
#     mean_value = np.mean(y)
#
#     # draw horizontal line at mean value
#     ax_hist.axhline(mean_value, color='green', linestyle='--', linewidth=0.8, alpha=0.8, zorder=5)
#
#     # add text label for mean value
#     ax_hist.text(
#         ax_hist.get_xlim()[1] * 0.95,  # near right edge
#         mean_value,
#         f"mean = {mean_value:.3f}",
#         va='bottom',
#         ha='right',
#         fontsize=14,
#         color='green',
#         alpha=0.8
#     )
#     # ax_hist.spines['left'].set_visible(False)
#     # ax_hist.tick_params(left=False, labelleft=False)
#
# axes[-1].set_xlabel("Time [sec]")
# # fig.text(0.04, 0.5, "Metric", va='center', rotation='vertical')
# plt.suptitle(f"{session_id}: Multiple metrics", y=0.98)
# plt.tight_layout(rect=[0.06, 0.05, 1, 0.95])
#
# # # --- Add vertical dotted lines at the end of each trial ---
# # trial_changes = np.where(np.diff(trial_number) != 0)[0]  # indices where trial changes
# # trial_end_times = t[trial_changes]
# #
# # for ax in axes:
# #     for te in trial_end_times:
# #         ax.axvline(te, color='gray', linestyle=':', alpha=0.6, linewidth=0.8, zorder=100)
# #
# #
#
#
# if date != '20250831' and date != '20250903':
#
#     # --- Add vertical dotted lines at the end of each trial, with labels for next trial ---
#     trial_changes = np.where(np.diff(trial_number) != 0)[0]  # indices where trial changes
#     trial_end_times = t[trial_changes]
#     trial_start_numbers = trial_number[trial_changes + 1]  # the number of the trial beginning after each change
#
#     for ax in axes:
#         ymin, ymax = ax.get_ylim()
#         #ymid = (ymin + ymax) / 2  # vertical midpoint for label placement
#
#         for te, tn in zip(trial_end_times, trial_start_numbers):
#             # draw the vertical line
#             ax.axvline(te, color='gray', linestyle=':', alpha=0.6, linewidth=0.8, zorder=100)
#
#             # add the label (rotated vertical text)
#             ax.text(
#                 te + (t[-1] - t[0]) * 0.002,  # small horizontal offset to avoid overlap
#                 ymax,
#                 #str(int(tn)),
#                 f"trial {int(tn)}",
#                 #rotation=90,
#                 va='center',
#                 ha='left',
#                 fontsize=10,
#                 color='gray',
#                 # alpha=0.3,
#                 zorder=200
#             )
#
#
# # --- Show ---
# plt.show()


# --- Create figure with extra column for histograms ---
n_datasets = len(dataset_groups)
fig = plt.figure(figsize=(14, 2.5 * n_datasets))
gs = gridspec.GridSpec(n_datasets, 2, width_ratios=[4, 1], wspace=0.15)

axes = []
hist_axes = []
#
# # --- Plot each group of datasets ---
# for i, group in enumerate(dataset_groups):  # <<< MODIFIED
#     ax = fig.add_subplot(gs[i, 0])
#     # ax_hist = fig.add_subplot(gs[i, 1], sharey=ax)
#     ax_hist = fig.add_subplot(gs[i, 1])  # no sharey, each histogram gets independent y-limits
#     axes.append(ax)
#     hist_axes.append(ax_hist)
#
#     # --- Compute combined y-range for the group (for both trace and hist) ---
#     y_all = np.concatenate([np.array(datasets_to_plot[l]) for l in group])
#     y_min, y_max = np.min(y_all), np.max(y_all)
#
#     # --- Combine all data in this group ---
#     for label in group:  # <<< MODIFIED
#         data = np.array(datasets_to_plot[label])
#         y = data
#
#         # Plot main trace
#         ax.plot(t, y, linewidth=0.7, alpha=0.8, label=label)
#
#         # Overlay cue/reward
#         if date != '20250831' and date != '20250727':
#             plot_reward_response(
#                 ax, cue, serial_readout_correct,
#                 ymin=np.min(y), ymax=np.max(y),
#                 c_reward='r',
#                 linewidth_response=0.2,
#                 t=t
#             )
#
#     # Compute y range from all datasets in this group
#     y_all = np.concatenate([np.array(datasets_to_plot[l]) for l in group])
#     ax.set_ylim(np.min(y_all)-0.01, np.max(y_all)+0.01)
#     ax.set_ylabel(", ".join(group))  # <<< MODIFIED
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.legend(ncol=4, frameon=True, fontsize=8)
#
#     # --- Histogram using combined data ---
#     # y = y_all
#     # counts, bins = np.histogram(y, bins=40)
#     # bin_centers = 0.5 * (bins[1:] + bins[:-1])
#     #
#     # ax_hist.plot(counts, bin_centers, color='gray', linewidth=1.5)
#     for label in group:
#         y = np.array(datasets_to_plot[label])
#         counts, bins = np.histogram(y, bins=40)
#         bin_centers = 0.5 * (bins[1:] + bins[:-1])
#         ax_hist.plot(counts, bin_centers, linewidth=1.2, label=f'{label} hist')
#         ax_hist.fill_betweenx(bin_centers, 0, counts, color='gray', alpha=0.2)
#         ax_hist.set_ylim(np.min(y_all) - 0.01, np.max(y_all) + 0.01)
#         # ax_hist.set_ylim(bins[0], bins[-1])
#         ax_hist.set_xlabel('Count')
#         ax_hist.spines['top'].set_visible(False)
#         ax_hist.spines['right'].set_visible(False)
#
#         # --- Mean line and label ---
#         mean_value = np.mean(y)
#         ax_hist.axhline(mean_value, color='green', linestyle='--', linewidth=0.8, alpha=0.8, zorder=5)
#         ax_hist.text(
#             ax_hist.get_xlim()[1] * 0.95,
#             mean_value,
#             f"mean = {mean_value:.3f}",
#             va='bottom',
#             ha='right',
#             fontsize=14,
#             color='green',
#             alpha=0.8
#         )
#
# axes[-1].set_xlabel("Time [sec]")
# plt.suptitle(f"{session_id}: Multiple metrics", y=0.98)
# plt.tight_layout(rect=[0.06, 0.05, 1, 0.95])
#
# # --- Add vertical trial lines as before ---
# if date != '20250831' and date != '20250903':
#     trial_changes = np.where(np.diff(trial_number) != 0)[0]
#     trial_end_times = t[trial_changes]
#     trial_start_numbers = trial_number[trial_changes + 1]
#     for ax in axes:
#         ymin, ymax = ax.get_ylim()
#         for te, tn in zip(trial_end_times, trial_start_numbers):
#             ax.axvline(te, color='gray', linestyle=':', alpha=0.6, linewidth=0.8, zorder=100)
#             ax.text(
#                 te + (t[-1] - t[0]) * 0.002,
#                 ymax,
#                 f"trial {int(tn)}",
#                 va='center',
#                 ha='left',
#                 fontsize=10,
#                 color='gray',
#                 zorder=200
#             )
#
# plt.show()


# --- Plot each group of datasets ---
for i, group_def in enumerate(dataset_groups):
    # Allow for tuple format: (group, dual_yaxis_flag)
    if isinstance(group_def, tuple):
        group, dual_yaxis = group_def
    else:
        group = group_def
        dual_yaxis = False

    ax = fig.add_subplot(gs[i, 0])
    ax_hist = fig.add_subplot(gs[i, 1])
    axes.append(ax)
    hist_axes.append(ax_hist)

    y_all = np.concatenate([np.array(datasets_to_plot[l]) for l in group])
    y_min, y_max = np.min(y_all), np.max(y_all)

    if not dual_yaxis:
        # Standard single y-axis case
        for label in group:
            y = np.array(datasets_to_plot[label])
            ax.plot(t, y, linewidth=0.7, alpha=0.8, label=label)

            if date != '20250831' and date != '20250727':
                plot_reward_response(
                    ax, cue, serial_readout_correct,
                    ymin=np.min(y), ymax=np.max(y),
                    c_reward='r',
                    linewidth_response=0.2,
                    t=t
                )

        ax.set_ylim(y_min - 0.01, y_max + 0.01)
        ax.set_ylabel(", ".join(group))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(ncol=4, frameon=True, fontsize=8)

    else:
        # Dual y-axis case (only supports 2 datasets)
        if len(group) != 2:
            raise ValueError("Dual y-axis plotting only supported for exactly two datasets.")

        left_label, right_label = group
        y_left = np.array(datasets_to_plot[left_label])
        y_right = np.array(datasets_to_plot[right_label])

        ax2 = ax.twinx()  # right axis

        # Plot left dataset
        ax.plot(t, y_left, color='blue', linewidth=0.7, alpha=0.8, label=left_label)
        # Plot right dataset
        ax2.plot(t, y_right, color='tab:orange', linewidth=0.7, alpha=0.8, label=right_label)

        if date != '20250831' and date != '20250727':
            plot_reward_response(ax, cue, serial_readout_correct,
                                 ymin=np.min(y_left), ymax=np.max(y_left),
                                 c_reward='r',
                                 linewidth_response=0.2, t=t)

        ax.set_ylabel(left_label, color='blue')
        ax2.set_ylabel(right_label, color='tab:orange')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax.legend(lines + lines2, labels + labels2, ncol=2, frameon=True, fontsize=8)

    # --- Histogram(s) ---
    for label in group:
        y = np.array(datasets_to_plot[label])
        counts, bins = np.histogram(y, bins=40)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        ax_hist.plot(counts, bin_centers, linewidth=1.2, label=f'{label} hist')
        ax_hist.fill_betweenx(bin_centers, 0, counts, color='gray', alpha=0.2)
        ax_hist.set_ylim(y_min - 0.01, y_max + 0.01)
        ax_hist.set_xlabel('Count')
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)

        mean_value = np.mean(y)
        ax_hist.axhline(mean_value, color='green', linestyle='--', linewidth=0.8, alpha=0.8)
        ax_hist.text(ax_hist.get_xlim()[1] * 0.95, mean_value,
                     f"mean = {mean_value:.3f}",
                     va='bottom', ha='right', fontsize=14, color='green', alpha=0.8)


axes[-1].set_xlabel("Time [sec]")
plt.suptitle(f"{session_id}: Multiple metrics", y=0.98)
plt.tight_layout(rect=[0.06, 0.05, 1, 0.95])

# --- Add vertical trial lines as before ---
if date != '20250831' and date != '20250903':
    trial_changes = np.where(np.diff(trial_number) != 0)[0]
    trial_end_times = t[trial_changes]
    trial_start_numbers = trial_number[trial_changes + 1]
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        for te, tn in zip(trial_end_times, trial_start_numbers):
            ax.axvline(te, color='gray', linestyle=':', alpha=0.6, linewidth=0.8, zorder=100)
            ax.text(
                te + (t[-1] - t[0]) * 0.002,
                ymax,
                f"trial {int(tn)}",
                va='center',
                ha='left',
                fontsize=10,
                color='gray',
                zorder=200
            )

plt.show()