import matplotlib.pyplot as plt
import numpy as np

from wideflow.analysis.utils.extract_from_metadata_file import extract_from_metadata_file

base_path = '/data/Lena/WideFlow_prj'
date = '20250909'
mouse_id = '260FN'
session_id = f'{date}_{mouse_id}_NF5_p3'

first_frame = 0
last_frame = 7866

[timestamp, cue, metric_result, threshold, serial_readout, trial_number] = extract_from_metadata_file(
    f'{base_path}/{date}/{mouse_id}/{session_id}/metadata.txt'
)

# Slice to window of interest
timestamp = timestamp[first_frame:last_frame]
cue = cue[first_frame:last_frame]
metric_result = metric_result[first_frame:last_frame]
serial_readout = serial_readout[first_frame:last_frame]
trial_number = trial_number[first_frame:last_frame]

serial_readout_correct = [1 - x for x in serial_readout]

# --- Trial organization ---
unique_trials = np.unique(trial_number)
trial_indices = {t: np.where(trial_number == t)[0] for t in unique_trials}

# --- Time axis in seconds ---
dt = np.mean(np.diff(timestamp))   # seconds per sample
max_len = max(len(idxs) for idxs in trial_indices.values())

fig, ax = plt.subplots(figsize=(12, 6))

for trial_idx, t in enumerate(unique_trials):
    idx = trial_indices[t]
    cue_t = np.array(cue)[idx]
    sr_t = np.array(serial_readout_correct)[idx]
    metric_t = np.array(metric_result)[idx]
    t_trial = np.arange(len(idx)) * dt

    # --- Normalize metric and shift to trial row ---
    metric_t_norm = (metric_t - np.min(metric_t)) / (np.ptp(metric_t) + 1e-9)  # 0–1 scaling
    metric_t_shifted = metric_t_norm * 0.8 + trial_idx  # scale and shift to row

    ax.plot(t_trial, metric_t_shifted, color='grey', alpha=0.8, linewidth=0.8)

    # --- Cue ticks ---
    cue_times = t_trial[np.where(cue_t > 0)[0]]
    ax.vlines(cue_times, trial_idx - 0.4, trial_idx + 0.4, color='black', linewidth=1)

    # --- Reward/response ticks ---
    reward_times = t_trial[np.where(sr_t > 0)[0]]
    ax.vlines(reward_times, trial_idx - 0.4, trial_idx + 0.4, color='blue', linewidth=1)

    # --- Shaded regions at trial end ---
    trial_end = t_trial[-1]
    if np.any(cue_t):   # with cue
        ax.axvspan(trial_end - 7.62, trial_end - 5,
                   ymin=(trial_idx)/len(unique_trials),
                   ymax=(trial_idx+1)/len(unique_trials),
                   color='lightgreen', alpha=0.4, zorder=0)
        ax.axvspan(trial_end - 5, trial_end,
                   ymin=(trial_idx)/len(unique_trials),
                   ymax=(trial_idx+1)/len(unique_trials),
                   color='lightgrey', alpha=0.4, zorder=0)
    else:  # no cue
        ax.axvspan(trial_end - 10, trial_end,
                   ymin=(trial_idx)/len(unique_trials),
                   ymax=(trial_idx+1)/len(unique_trials),
                   color='lightgrey', alpha=0.4, zorder=0)

# --- Formatting ---
ax.set_ylim(-0.5, len(unique_trials) - 0.5 + 1)
ax.set_xlim(0, dt * max_len)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Trials")
ax.set_yticks(range(len(unique_trials)))
ax.set_yticklabels([f"{int(t)}" for t in unique_trials])
ax.set_title(f"Raster + metric: {mouse_id}_{session_id}")

plt.tight_layout()
plt.show()
