"""
Plots success rates across sessions for chosen mice, using metadata loaded
from mice_metadata.h5 (synced from Excel) instead of hardcoded per-mouse
if/elif blocks.

EDIT THE CONFIG BLOCK BELOW to choose mice, sessions, ROI, parts, and frames.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py

from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict
from mouse_metadata_loader import load_mouse_metadata, get_roi_info, build_results_path
from success_rate_core import calculate_success_rate
from Lena_scripts.mice_meta.parsing import build_session_id  # from the mice_h5_sync package


# ============================== CONFIG ======================================

MICE_METADATA_H5 = '/home/elenakreines/WideFlow/wideflow/Lena_scripts/mice_meta/mice_metadata.h5'   # output of sync_excel_to_h5.py
BASE_PATH = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj'
BASE_RESULTS_DIR = '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results'

# --- 1. Choose mice ---
MICE_ID = [
# '245FRL',
# '246FN',
# '248FL',
# '252MR',
# '256FLL',
# '257FR',

# '228MN', #BC-RC
# '258FL', #BC-RC
# '259FRL', #BC-RC
# '260FN', #BC-RC
# '261MR', #BC-RC
# '263MRL', #BC-RC
#
# '266FR', #BC-RC
# '276FL', #RC-BC
# '277FRL', #M1-RC
# '281MRL', #RC-M1

# '322MR', #M1-BC
# '327FL', #RC-BC
# '329FRR', #M1-BC
# '331FN', #RC-M1

]

# --- 2. Choose sessions (by session_name, e.g. 'NF2','NF3',...). ---
# None = use every session stored for that mouse in mice_metadata.h5.
# If given, only these session_names are used (mice missing a listed
# session simply skip it).
SESSIONS_TO_USE = [
    # 'NF1',
    'NF2', 'NF3', 'NF4', 'NF5','NF6',
    'NF7',
    # 'NF7', 'NF8', 'NF9', 'NF10', 'NF11'
]
# Example restricting to a subset:
# SESSIONS_TO_USE = ['NF2', 'NF3', 'NF4', 'NF5']

# --- 3. Choose which ROI (1 or 2, as configured per-mouse in the Excel/H5) ---
ROI_CHOICE = 1

# --- 4. Choose which parts to use per session ---
# 'all'   -> use every part stored for that session in mice_metadata.h5
#            (this already reflects each mouse's actual recorded parts,
#             including any part_exceptions_and_skips overrides)
# ['p1']  -> restrict to only these parts, if present for that session
PARTS_TO_USE = 'all'
# Example: PARTS_TO_USE = ['p1', 'p2', 'p3']

# --- 4b. Average multi-part sessions into a single point? ---
# False -> one point per part (e.g. 'NF2_p1','NF2_p2','NF2_p3' each plotted
#          separately -- matches the original script's default behavior).
# True  -> sessions with more than one part are averaged (simple mean of
#          each part's success rate) into a single 'NF2' point. Sessions
#          with only one part (or no-suffix sessions) are unaffected either
#          way, since there's nothing to average.
AGGREGATE_PARTS = True

# --- 5. Optional: restrict to specific frames within a session/part ---
# Keyed by exact session_id (the same id used to look up metadata.txt /
# results, e.g. '20251208_266FR_NF2_p1'). Sessions not listed here use all
# frames. Format: {session_id: (start_frame, end_frame)}
FRAME_RANGE_OVERRIDES = {
    # '20251208_266FR_NF2_p1': (500, 3000),
}

# --- Manual overrides for specific sessions where extraction is known-bad ---
# Keyed by exact session_id -- matches the original script's hardcoded patch.
MANUAL_SUCCESS_RATE_OVERRIDES = {
    '20251211_266FR_NF5_p2': 15 / 26,
}

MAX_TRIAL_FRAMES = 750
TIMEOUT_REWARDED = 200
TIMEOUT_NO_REWARD = 250

# --- 6. Normalization ---
# NORMALIZE = False -> plot raw success rates (default).
# NORMALIZE = True  -> plot (session - reference) / reference, per mouse,
#                      where "reference" is NORMALIZE_TO_SESSION (a
#                      session_name, e.g. 'NF2'). Normalization is applied
#                      AFTER part-averaging, so each session must already
#                      be a single value -- this requires AGGREGATE_PARTS
#                      to be True (enforced below; the script will raise a
#                      clear error if NORMALIZE=True and AGGREGATE_PARTS
#                      is False, rather than silently misnormalizing).
NORMALIZE = True
NORMALIZE_TO_SESSION = 'NF2'

PLOT_TITLE = 'BC ROI1'

# --- 7. Session label equivalences ---
# Some mice use a differently-named session as a stand-in for a numbered NF
# session (e.g. 'NF_control' instead of 'NF7'). To plot such mice alongside
# ones that have the real NF7, map the stand-in name to the slot it should
# share: {stand-in_name: target_session_name}. This only affects the
# DISPLAY label (x-axis position / SESSIONS_TO_USE matching) -- the real
# session_id lookup for that mouse still uses its own actual session name
# and date, exactly as stored in mice_metadata.h5.
#
# Example: mice using NF_control in place of NF7 both plot at the 'NF7' slot:
#     SESSION_LABEL_EQUIVALENCES = {'NF_control': 'NF7'}
#
# If a mouse has BOTH a real NF7 session AND the stand-in (e.g. NF_control)
# in mice_metadata.h5, that's treated as a data error and raises -- a mouse
# should only ever have one or the other filling that slot.
SESSION_LABEL_EQUIVALENCES = {
    'NF_control': 'NF7',
}

# --- 8. Color plots by ROI cortical area instead of by mouse ---
# False -> each mouse gets its own color (default), legend shows mouse IDs.
# True  -> all mice sharing the same ROI CORTICAL AREA (e.g. 'BC', 'RC') get
#          the same color. Legend shows one entry per cortical area instead
#          of per mouse. NF vs control still differ by linestyle (solid vs
#          dashed) as before -- this only changes color/legend grouping, not
#          which mice count as NF or control. The mean is now computed
#          separately per (cortical_area, group) pair -- e.g. 'BC' NF mean,
#          'BC' control mean, 'RC' NF mean, etc. -- each drawn bold in its
#          area's color with its group's linestyle.
COLOR_BY_ROI = False

# =============================================================================


def get_sessions_for_mouse(mouse_record, sessions_to_use, parts_to_use, mouse_id=None):
    """Filters the mouse's stored sessions by SESSIONS_TO_USE, and applies
    PARTS_TO_USE (either 'all' -> keep stored parts as-is, or a list ->
    intersect with what's actually stored for that session).

    Also applies SESSION_LABEL_EQUIVALENCES: a stand-in session name (e.g.
    'NF_control') is relabeled to its target slot (e.g. 'NF7') for display
    purposes -- raw_session_name is left untouched, so the real data lookup
    is unaffected. If a mouse has BOTH the stand-in and the real target
    session, that's a data error and raises."""
    sessions = mouse_record['sessions']

    if SESSION_LABEL_EQUIVALENCES:
        present_names = {s['session_name'] for s in sessions}
        remapped = []
        for s in sessions:
            target = SESSION_LABEL_EQUIVALENCES.get(s['session_name'])
            if target is not None:
                if target in present_names:
                    raise ValueError(
                        f"Mouse {mouse_id!r} has both {s['session_name']!r} "
                        f"and its equivalence target {target!r} in "
                        f"mice_metadata.h5 -- a mouse should only have one "
                        f"of these filling that slot. Fix the data or remove "
                        f"this pairing from SESSION_LABEL_EQUIVALENCES."
                    )
                s = {**s, 'session_name': target}
            remapped.append(s)
        sessions = remapped

    if sessions_to_use is not None:
        sessions = [s for s in sessions if s['session_name'] in sessions_to_use]

    filtered = []
    for s in sessions:
        if parts_to_use == 'all':
            parts = s['parts']
        else:
            parts = [p for p in s['parts'] if p in parts_to_use]
        filtered.append({**s, 'parts': parts})
    return filtered


def compute_success_rates_for_mouse(mouse_id, mouse_record, roi_choice):
    """Returns dict of {session_label: success_rate}.

    If AGGREGATE_PARTS is False: session_label is e.g. 'NF2_p1' (session_name
    + part), matching the original script's per-part granularity. Sessions
    with no parts (parts == []) use the session_name alone as the label.

    If AGGREGATE_PARTS is True: each part's rate is still computed
    individually (frame ranges / overrides are still per-part), but sessions
    with more than one part are then collapsed into a single 'NF2' point via
    simple mean of that session's part rates. Single-part and no-suffix
    sessions are unaffected."""
    roi_name, threshold, cortical_area = get_roi_info(mouse_record, roi_choice)
    results_path = build_results_path(BASE_RESULTS_DIR, mouse_record['group_number'], roi_choice)

    sessions = get_sessions_for_mouse(mouse_record, SESSIONS_TO_USE, PARTS_TO_USE, mouse_id=mouse_id)

    rates = {}
    for s in sessions:
        parts = s['parts'] if s['parts'] else [None]  # None = no-suffix session
        part_rates = {}  # part -> rate, for this session only

        for part in parts:
            # IMPORTANT: use raw_session_name (the real column/folder this
            # session's data came from), not session_name (the display name,
            # which may have been relabeled by a skip_shift exception).
            session_id = build_session_id(mouse_id, s['date'], s['raw_session_name'], part)

            if session_id in MANUAL_SUCCESS_RATE_OVERRIDES:
                part_rates[part] = MANUAL_SUCCESS_RATE_OVERRIDES[session_id]
                continue

            from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
            timestamp, cue, metric_result_meta, threshold_meta, serial_readout, trial_number = \
                extract_from_metadata_file(f'{BASE_PATH}/{s["date"]}/{mouse_id}/{session_id}/metadata.txt')

            data = {}
            with h5py.File(results_path, 'r') as f:
                decompose_h5_groups_to_dict(
                    f, data, f'/{mouse_id}/{session_id}/post_session_analysis_LK2/'
                )
            metric_result = np.repeat(data['zsores_MH_diff10_exc_top15'][f'roi_{roi_name}'], 2)

            frame_range = FRAME_RANGE_OVERRIDES.get(session_id)

            part_rates[part] = calculate_success_rate(
                timestamp, metric_result, threshold,
                max_trial_frames=MAX_TRIAL_FRAMES,
                timeout_rewarded=TIMEOUT_REWARDED,
                timeout_no_reward=TIMEOUT_NO_REWARD,
                frame_range=frame_range,
            )

        if AGGREGATE_PARTS and len(part_rates) > 1:
            rates[s['session_name']] = float(np.mean(list(part_rates.values())))
        else:
            for part, rate in part_rates.items():
                label = f"{s['session_name']}_{part}" if part else s['session_name']
                rates[label] = rate

    return rates


def main():
    if NORMALIZE and not AGGREGATE_PARTS:
        raise ValueError(
            "NORMALIZE=True requires AGGREGATE_PARTS=True, since normalization "
            "expects one success rate value per session (not per part). "
            "Set AGGREGATE_PARTS = True in the config, or set NORMALIZE = False."
        )

    all_mouse_rates = {}   # {mouse_id: {label: rate}}
    mouse_groups = {}      # {mouse_id: 'NF' | 'control'}
    mouse_cortical_areas = {}  # {mouse_id: 'BC' | 'RC' | ...}
    all_labels_ordered = []  # preserves first-seen order across mice for the x-axis

    for mouse_id in MICE_ID:
        mouse_record = load_mouse_metadata(MICE_METADATA_H5, mouse_id)
        mouse_groups[mouse_id] = mouse_record['group']
        _, _, cortical_area = get_roi_info(mouse_record, ROI_CHOICE)
        mouse_cortical_areas[mouse_id] = cortical_area
        rates = compute_success_rates_for_mouse(mouse_id, mouse_record, ROI_CHOICE)

        if NORMALIZE:
            if NORMALIZE_TO_SESSION not in rates:
                raise ValueError(
                    f"NORMALIZE_TO_SESSION={NORMALIZE_TO_SESSION!r} was not "
                    f"computed for mouse {mouse_id!r} -- check it's included "
                    f"in SESSIONS_TO_USE (or SESSIONS_TO_USE is None) and "
                    f"exists for this mouse in mice_metadata.h5."
                )
            reference = rates[NORMALIZE_TO_SESSION]
            rates = {label: (rate - reference) / reference for label, rate in rates.items()}

        all_mouse_rates[mouse_id] = rates
        for label in rates:
            if label not in all_labels_ordered:
                all_labels_ordered.append(label)

    # Build a (mice x labels) matrix, NaN where a mouse has no data for a label
    matrix = np.full((len(MICE_ID), len(all_labels_ordered)), np.nan)
    for i, mouse_id in enumerate(MICE_ID):
        for j, label in enumerate(all_labels_ordered):
            if label in all_mouse_rates[mouse_id]:
                matrix[i, j] = all_mouse_rates[mouse_id][label]

    # NF -> solid line, control -> dashed line. Any group value other than
    # these two falls back to solid, so a typo in the Excel/H5 doesn't
    # silently drop a mouse from the plot -- but check GROUP_LINESTYLES
    # if you add more groups later.
    GROUP_LINESTYLES = {'NF': 'solid', 'control': 'dashed'}

    if COLOR_BY_ROI:
        cortical_areas_sorted = sorted(set(mouse_cortical_areas.values()))
        area_colors = {a: cm.tab20(i % 20) for i, a in enumerate(cortical_areas_sorted)}

        # One line per mouse, colored by its cortical area, styled by group.
        # No legend label here -- the bold per-area mean lines (below) carry
        # the legend, so the legend shows ROIs, not individual mice.
        for i, mouse_id in enumerate(MICE_ID):
            area = mouse_cortical_areas[mouse_id]
            linestyle = GROUP_LINESTYLES.get(mouse_groups[mouse_id], 'solid')
            plt.plot(all_labels_ordered, matrix[i],
                      color=area_colors[area], linestyle=linestyle, alpha=0.5)

        # Mean per (cortical_area, group) pair -- bold, in that area's color,
        # with that group's linestyle. One legend entry per area (using the
        # NF-group mean's line to represent it, since NF is virtually always
        # present); if an area has no NF mice in this selection, its label
        # falls to whichever group does have mice, so the area still shows.
        labeled_mean_areas = set()
        for area in cortical_areas_sorted:
            for group_name, linestyle in GROUP_LINESTYLES.items():
                row_indices = [
                    i for i, mouse_id in enumerate(MICE_ID)
                    if mouse_cortical_areas[mouse_id] == area and mouse_groups[mouse_id] == group_name
                ]
                if not row_indices:
                    continue
                group_mean = np.nanmean(matrix[row_indices], axis=0)
                mean_label = f'Mean {area}' if area not in labeled_mean_areas else None
                labeled_mean_areas.add(area)
                plt.plot(all_labels_ordered, group_mean, color=area_colors[area],
                          linestyle=linestyle, linewidth=5, label=mean_label)
    else:
        mouse_colors = {m: cm.tab20(i % 20) for i, m in enumerate(sorted(MICE_ID))}

        for i, mouse_id in enumerate(MICE_ID):
            linestyle = GROUP_LINESTYLES.get(mouse_groups[mouse_id], 'solid')
            plt.plot(all_labels_ordered, matrix[i], label=mouse_id,
                      color=mouse_colors[mouse_id], linestyle=linestyle)

        # Separate mean per group, each plotted with that group's linestyle.
        for group_name, linestyle in GROUP_LINESTYLES.items():
            group_row_indices = [i for i, mouse_id in enumerate(MICE_ID) if mouse_groups[mouse_id] == group_name]
            if not group_row_indices:
                continue  # no mice from this group in the current MICE_ID selection
            group_mean = np.nanmean(matrix[group_row_indices], axis=0)
            plt.plot(all_labels_ordered, group_mean, color='black', linestyle=linestyle,
                      label=f'Mean ({group_name})', linewidth=5)

    plt.suptitle(PLOT_TITLE)
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()