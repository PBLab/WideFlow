"""
Loads per-mouse metadata (ROI choice, threshold, session dates/parts) from
mice_metadata.h5, using the lab's decompose_h5_groups_to_dict reader, and
reconstructs everything the success-rate calculation needs -- replacing the
manual per-mouse if/elif blocks from the original script.
"""

import h5py
import numpy as np

from utils.decompose_dict_and_h5_groups import decompose_h5_groups_to_dict


from wideflow.config import BASE_PATH  #added by Claude 20260906
def _to_str(val):
    """h5py datasets round-trip strings as bytes sometimes -- normalize."""
    if isinstance(val, bytes):
        return val.decode()
    return val


def load_mouse_metadata(mice_metadata_h5_path, mouse_id):
    """
    Reads one mouse's full metadata dict from mice_metadata.h5 via
    decompose_h5_groups_to_dict, and reshapes it into a clean structure:

    {
        'group': 'NF' | 'control',
        'group_number': 4.0,
        'roi1_name': '62_43', 'roi1_cortical_area': 'BC', 'roi1_threshold': 3.6,
        'roi2_name': '13_40', 'roi2_cortical_area': 'RC', 'roi2_threshold': 5.2,
        'sessions': [
            {'session_name': 'NF2', 'date': '20251208', 'parts': ['p1','p2','p3'], 'position': 4},
            ...
        ]
    }
    """
    data = {}
    with h5py.File(mice_metadata_h5_path, 'r') as f:
        if mouse_id not in f:
            raise KeyError(f"Mouse {mouse_id!r} not found in {mice_metadata_h5_path}")
        decompose_h5_groups_to_dict(f, data, f'/{mouse_id}/')

    record = {
        'group': _to_str(data['group']),
        'group_number': float(data['group_number']),
        'roi1_name': _to_str(data['roi1_name']),
        'roi1_cortical_area': _to_str(data['roi1_cortical_area']),
        'roi1_threshold': float(data['roi1_threshold']),
        'roi2_name': _to_str(data['roi2_name']),
        'roi2_cortical_area': _to_str(data['roi2_cortical_area']),
        'roi2_threshold': float(data['roi2_threshold']),
    }

    names = [_to_str(n) for n in data['sessions']['names']]
    # raw_names is new -- fall back to names for H5 files written before
    # this field existed, so old files don't break.
    if 'raw_names' in data['sessions']:
        raw_names = [_to_str(n) for n in data['sessions']['raw_names']]
    else:
        raw_names = names
    dates = [_to_str(d) for d in data['sessions']['dates']]
    parts_raw = [_to_str(p) for p in data['sessions']['parts']]
    positions = list(data['sessions']['position'])

    sessions = []
    for name, raw_name, date, parts_str, pos in zip(names, raw_names, dates, parts_raw, positions):
        parts = parts_str.split(',') if parts_str else []
        sessions.append({
            'session_name': name,
            'raw_session_name': raw_name,
            'date': date,
            'parts': parts,
            'position': int(pos),
        })
    record['sessions'] = sessions

    return record


def get_roi_info(mouse_record, roi_choice):
    """
    roi_choice: 1 or 2 -- which ROI (as configured in mice_metadata.h5) to use
    for this mouse.

    Returns (roi_name, threshold, cortical_area) e.g. ('62_43', 3.6, 'BC')
    """
    if roi_choice == 1:
        return mouse_record['roi1_name'], mouse_record['roi1_threshold'], mouse_record['roi1_cortical_area']
    elif roi_choice == 2:
        return mouse_record['roi2_name'], mouse_record['roi2_threshold'], mouse_record['roi2_cortical_area']
    else:
        raise ValueError(f"roi_choice must be 1 or 2, got {roi_choice!r}")


def build_results_path(base_results_dir, group_number, roi_choice):
    """
    Matches the pattern:
        # '/claustrum-storage/pblab_shared_data/Lena/WideFlow_prj/Results/results_exp{group_num}_parc_{ROI}.h5'
        BASE_PATH + '/Results/results_exp{group_num}_parc_{ROI}.h5'  #added by Claude 20260906

    group_number is stored as a float (e.g. 3.4, 4.0) -- group 4 is written
    as plain '4' (not '4.0') to match 'results_exp4_parc_ROI1.h5', while
    3.4/3.5/4.1 keep their decimal.
    """
    if group_number == int(group_number):
        group_str = str(int(group_number))
    else:
        group_str = str(group_number)
    return f'{base_results_dir}/results_exp{group_str}_parc_ROI{roi_choice}.h5'