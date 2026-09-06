"""
Reads and writes per-mouse metadata groups in the mice_metadata.h5 file.

IMPORTANT: everything is stored as HDF5 GROUPS and DATASETS, never as
h5py attrs (.attrs). This is deliberate -- the lab's existing reader,
decompose_h5_groups_to_dict, only walks groups/datasets and has no code
path for attrs, so anything stored as an attr would be invisible to it.

H5 layout per mouse:
    /{mouse_id}/
        group:               string dataset, scalar  (e.g. 'NF' or 'control')
        group_number:        float dataset, scalar    (e.g. 3.4)
        roi1_name:           string dataset, scalar
        roi1_cortical_area:  string dataset, scalar
        roi1_threshold:      float dataset, scalar
        roi2_name:           string dataset, scalar
        roi2_cortical_area:  string dataset, scalar
        roi2_threshold:      float dataset, scalar
        sessions/
            names:     string dataset, DISPLAY names, e.g. ['NF1','NF2',...]
                       -- after a skip_shift, this is the renamed slot
                       (e.g. 'NF7' for data that actually came from NF8)
            raw_names: string dataset, the REAL column each session's data
                       came from (e.g. 'NF8') -- use this, not names, when
                       building the actual session_id for file/results
                       lookups. Same order as names.
            dates:    string dataset, 'YYYYMMDD', same order as names
            parts:    string dataset, comma-joined parts per session
                      (e.g. 'p1,p2,p3' or '' for no-suffix sessions)
            position: int dataset, same order as names

Missing string values are stored as '' (empty string) and missing
numeric values as NaN -- HDF5 datasets can't hold Python None, so these
are the placeholders read_existing_mice() converts back to None.
"""

import os
import h5py
import numpy as np

# Fields stored per mouse, and whether each is a string or a number.
# Edit this (and nothing else) if you need to add/remove a stored field.
MOUSE_STRING_FIELDS = [
    'group',
    'roi1_name', 'roi1_cortical_area',
    'roi2_name', 'roi2_cortical_area',
]
MOUSE_NUMERIC_FIELDS = [
    'group_number',
    'roi1_threshold', 'roi2_threshold',
]
MOUSE_ATTR_FIELDS = MOUSE_STRING_FIELDS + MOUSE_NUMERIC_FIELDS  # kept for diffing.py


def read_existing_mice(h5_path):
    """Returns dict of {mouse_id: mouse_record} reconstructed from the H5,
    in the same shape produced by excel_reader.read_excel (minus the raw/
    derived-only fields), so it can be diffed directly against fresh Excel
    records. Returns {} if the file doesn't exist yet."""
    if not os.path.exists(h5_path):
        return {}

    mice = {}
    with h5py.File(h5_path, 'r') as f:
        for mouse_id in f.keys():
            g = f[mouse_id]
            record = {}

            for field in MOUSE_STRING_FIELDS:
                if field in g:
                    val = g[field][()]
                    if isinstance(val, bytes):
                        val = val.decode()
                    record[field] = val if val != '' else None
                else:
                    record[field] = None

            for field in MOUSE_NUMERIC_FIELDS:
                if field in g:
                    val = float(g[field][()])
                    record[field] = None if np.isnan(val) else val
                else:
                    record[field] = None

            sessions = []
            if 'sessions' in g:
                sg = g['sessions']
                names = [n.decode() if isinstance(n, bytes) else n for n in sg['names'][:]]
                # raw_names is new -- fall back to names for H5 files written
                # before this field existed, so old files don't break.
                if 'raw_names' in sg:
                    raw_names = [n.decode() if isinstance(n, bytes) else n for n in sg['raw_names'][:]]
                else:
                    raw_names = names
                dates = [d.decode() if isinstance(d, bytes) else d for d in sg['dates'][:]]
                parts_raw = [p.decode() if isinstance(p, bytes) else p for p in sg['parts'][:]]
                positions = list(sg['position'][:])
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
            mice[mouse_id] = record

    return mice


def write_mouse(h5_path, mouse_id, record):
    """Fully overwrite one mouse's group in the H5 with the given record
    (dict as produced by excel_reader.read_excel). Creates the file if it
    doesn't exist."""
    with h5py.File(h5_path, 'a') as f:
        if mouse_id in f:
            del f[mouse_id]  # full overwrite of this mouse's group only

        g = f.create_group(mouse_id)

        for field in MOUSE_STRING_FIELDS:
            val = record.get(field)
            g.create_dataset(field, data=(val if val is not None else ''))

        for field in MOUSE_NUMERIC_FIELDS:
            val = record.get(field)
            g.create_dataset(field, data=(float(val) if val is not None else float('nan')))

        sessions = record.get('sessions') or []
        sg = g.create_group('sessions')

        names = [s['session_name'] for s in sessions]
        raw_names = [s.get('raw_session_name', s['session_name']) for s in sessions]
        dates = [s['date'] for s in sessions]
        parts = [','.join(s['parts']) for s in sessions]
        positions = [s['position'] for s in sessions]

        dt = h5py.string_dtype(encoding='utf-8')
        sg.create_dataset('names', data=np.array(names, dtype=object), dtype=dt)
        sg.create_dataset('raw_names', data=np.array(raw_names, dtype=object), dtype=dt)
        sg.create_dataset('dates', data=np.array(dates, dtype=object), dtype=dt)
        sg.create_dataset('parts', data=np.array(parts, dtype=object), dtype=dt)
        sg.create_dataset('position', data=np.array(positions, dtype='i8'))


def delete_mouse(h5_path, mouse_id):
    with h5py.File(h5_path, 'a') as f:
        if mouse_id in f:
            del f[mouse_id]