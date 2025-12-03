import matplotlib.pyplot as plt
import numpy as np

from wideflow.analysis.utils.extract_from_metadata_file import extract_from_metadata_file
from wideflow.analysis.plots import plot_reward_response

base_path_qnap = '/data/Lena/WideFlow_prj'
base_path = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj'
date = '20251119'
mouse_id = '260FN'
session_id = f'{date}_{mouse_id}_NF5_p2_as_is'

first_frame = 0
last_frame = 19000


[timestamp, cue, metric_result, threshold, serial_readout, trial_number] = extract_from_metadata_file(
    f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt'
)

timestamp = timestamp[first_frame:last_frame]
cue = cue[first_frame:last_frame]
metric_result = metric_result[first_frame:last_frame]
threshold = threshold[first_frame:last_frame]
serial_readout = serial_readout[first_frame:last_frame]
trial_number = trial_number[first_frame:last_frame]

serial_readout_correct = [1 - x for x in serial_readout]

# --- Split indices by trial ---
unique_trials = np.unique(trial_number)
trial_indices = {t: np.where(trial_number == t)[0] for t in unique_trials}

# --- Time axis conversion to minutes ---
dt = np.mean(np.diff(timestamp))   # minutes per sample
max_len = max(len(idxs) for idxs in trial_indices.values())
t_full = np.arange(0, dt * max_len, dt)  # reference time axis in minutes

# --- Create subplots: one row per trial ---
fig, axes = plt.subplots(len(unique_trials), 1, figsize=(12, 2 * len(unique_trials)), sharex=True)

if len(unique_trials) == 1:
    axes = [axes]  # ensure iterable if only one trial

for ax, t in zip(axes, unique_trials):
    idx = trial_indices[t]

    # Extract trial data
    metric_t = np.array(metric_result)[idx]
    threshold_t = np.array(threshold)[idx]
    cue_t = np.array(cue)[idx]
    sr_t = np.array(serial_readout_correct)[idx]

    # Time vector with exact length, in seconds
    t_trial = np.arange(len(metric_t)) * dt

    # Plot metric
    ax.plot(t_trial, metric_t, color='black', alpha=0.7, linewidth=1.5)
    ax.plot(t_trial, threshold_t, color='green', alpha=0.7, linewidth=1)
    ax.set_ylim(ymin=-6, ymax=8)

    # Overlay cue/reward responses
    plot_reward_response(ax, cue_t, sr_t,
                         # ymin=np.min(metric_t),
                         # ymax=np.max(metric_t),
                         ymin = -6,
                         ymax = 8,
                         c_reward = 'r',
                         linewidth_response = 0.4,
                         t=t_trial)

    # --- Add overlays ---
    trial_end = t_trial[-1]

    # if np.any(cue_t):   # trial has cue
    #     # Grey overlay for last 5 s
    #     ax.axvspan(trial_end - 5, trial_end,
    #                color='lightgrey', alpha=0.4, zorder=2)
    #     # Green overlay for 3 s before grey
    #     ax.axvspan(trial_end - 7.61, trial_end - 5,
    #                color='lightgreen', alpha=0.4, zorder=2)
    # else:  # no cue: grey 10 s at end
    #     ax.axvspan(trial_end - 8.12, trial_end,
    #                color='lightgrey', alpha=0.4, zorder=2)

    # Force common x range (seconds)
    ax.set_xlim([0, dt * max_len])

    #ax.set_title(f"Trial {int(t)}", loc='left')
    #ax.set_ylabel("Metric")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)



axes[-1].set_xlabel("Time [sec]")
fig.text(0.04, 0.5, "Metric", va='center', rotation='vertical')

# Global title (move slightly above tight layout area)
plt.suptitle(f"{mouse_id}_{session_id}", y=0.98)

# Adjust layout to avoid overlap with title and label
plt.tight_layout(rect=[0.06, 0.05, 1, 0.95])

# Save
#plt.savefig(f'{base_path_qnap}/figs_for_paper_EXP3.5/trials_stacked_vertically_{session_id}.svg',format='svg',dpi=200)
plt.show()
