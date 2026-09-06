"""
Core success-rate calculation, extracted from the original plotting script.
The trial-simulation logic (threshold crossing -> reward -> timeout) is
UNCHANGED from the original -- only wrapped as a function, with an added
optional frame_range to restrict which frames of metric_result/timestamp
are used.
"""

import numpy as np


def calculate_success_rate(timestamp, metric_result, threshold,
                            max_trial_frames=750, timeout_rewarded=200,
                            timeout_no_reward=250, frame_range=None):
    """
    timestamp, metric_result: full-session arrays (same length)
    threshold: float, feedback threshold for this mouse/ROI
    frame_range: optional (start, end) tuple -- if given, only that slice of
                 timestamp/metric_result is used, exactly as if the session
                 had only ever contained those frames. None = use all frames.

    Returns success_rate_sess (float): rewards / trials, matching the
    original script's calculation exactly.
    """
    if frame_range is not None:
        start, end = frame_range
        timestamp = timestamp[start:end]
        metric_result = metric_result[start:end]

    feedback_time = 0
    cue = np.zeros(len(metric_result))
    trial_counter = 0
    frame_counter = 0
    trial_number = []

    while frame_counter < len(timestamp):
        trial_counter += 1
        reward_given = False

        for trial_frame in range(max_trial_frames):
            if frame_counter >= len(timestamp):
                break

            frame_clock_start = timestamp[frame_counter]

            # Note by Claude (2026-08-04): The 1000ms refractory term
            # (`frame_clock_start - feedback_time`) guards against the live system
            # firing twice on one real crossing, since its metric value is duplicated
            # across both raw frames of a hemo-corrected pair. In this offline replay
            # it's redundant: the `break` below already stops a trial from
            # re-checking the duplicate frame, and the timeout between trials
            # (200/250 raw frames) already exceeds 1000ms on its own. Kept as-is (not
            # removed) to stay faithful to the original live logic.
            if (metric_result[frame_counter] > threshold) and \
                    (frame_clock_start - feedback_time) * 1000 > 1000 and \
                    frame_counter > 1000:
                cue[frame_counter] = 1
                feedback_time = timestamp[frame_counter]
                reward_given = True
                timeout_duration = timeout_rewarded
                frame_counter += 1
                trial_number.append(trial_counter)
                break

            frame_counter += 1
            trial_number.append(trial_counter)

        if not reward_given:
            timeout_duration = timeout_no_reward

        for _ in range(timeout_duration):
            if frame_counter >= len(timestamp):
                break
            frame_counter += 1
            trial_number.append(trial_counter)

    rewards = np.sum(cue)
    trials = trial_number[-1] - 1
    return rewards / trials
