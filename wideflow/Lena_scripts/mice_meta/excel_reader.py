"""
Reads the mice metadata Excel sheet into a list of per-mouse dicts, ready
for comparison against the H5 and/or writing.
"""

import openpyxl

from parsing import (
    SESSION_COLUMNS,
    parse_part_exceptions,
    parts_to_list,
    resolve_mouse_sessions,
)

# Non-session columns we pull directly, mapped to a clean internal key name.
# Edit this if your Excel headers change.
FIELD_MAP = {
    'Mouse': 'mouse_id',
    'exp group': 'group',
    'Group number': 'group_number',
    'ROI1': 'roi1_name',
    'ROI1 cortical area': 'roi1_cortical_area',
    'Threshold 30% from NF1 ROI1': 'roi1_threshold',
    'ROI2': 'roi2_name',
    'ROI2 cortical area': 'roi2_cortical_area',
    'Threshold 30% from NF1 ROI2': 'roi2_threshold',
    'Default_parts': 'default_parts_raw',
    'part_exceptions_and_skips': 'part_exceptions_raw',
}


def read_excel(path, sheet_name='Sheet1'):
    """
    Returns (mice, warnings)

    mice: dict of {mouse_id: mouse_record}
    mouse_record contains all FIELD_MAP values (renamed), plus:
        - 'default_parts_list': e.g. ['p1','p2','p3']
        - 'sessions': list of resolved session dicts (see resolve_mouse_sessions)
    warnings: list of (mouse_id, message) tuples for anything that needs a
              human's attention (unmatched exception keys, bad dates, etc.)
              -- collected across ALL mice, never raised mid-loop, so one
              mouse's problem doesn't hide problems in mice processed after it.
    """
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet_name]
    header = [c.value for c in ws[1]]

    mice = {}
    warnings = []

    for row in ws.iter_rows(min_row=2, values_only=True):
        row_dict = dict(zip(header, row))
        mouse_id = row_dict.get('Mouse')
        if mouse_id is None:
            continue  # blank row

        record = {}
        for excel_col, internal_key in FIELD_MAP.items():
            record[internal_key] = row_dict.get(excel_col)

        default_parts_list = parts_to_list(record['default_parts_raw'])
        record['default_parts_list'] = default_parts_list

        mouse_session_row = {col: row_dict.get(col) for col in SESSION_COLUMNS}

        exceptions, exc_warnings = parse_part_exceptions(
            record['part_exceptions_raw'], SESSION_COLUMNS
        )
        for w in exc_warnings:
            warnings.append((mouse_id, w))

        try:
            record['sessions'] = resolve_mouse_sessions(
                mouse_session_row, default_parts_list, exceptions
            )
        except ValueError as e:
            warnings.append((mouse_id, str(e)))
            record['sessions'] = None  # signal: this mouse could not be resolved

        mice[mouse_id] = record

    return mice, warnings