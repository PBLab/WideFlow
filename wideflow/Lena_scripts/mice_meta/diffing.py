"""
Compares a mouse record freshly read from Excel against what's already
stored in the H5, to classify each mouse as new / unchanged / changed,
plus flags mice present in the H5 but no longer in the Excel.
"""

# Fields compared for "did this mouse change" purposes (mirrors what's
# actually stored -- see h5_store.MOUSE_ATTR_FIELDS plus sessions).
from h5_store import MOUSE_ATTR_FIELDS


def _sessions_equal(a, b):
    a = a or []
    b = b or []
    if len(a) != len(b):
        return False
    for sa, sb in zip(a, b):
        sa_key = (sa['session_name'], sa.get('raw_session_name', sa['session_name']),
                  sa['date'], sa['parts'], sa['position'])
        sb_key = (sb['session_name'], sb.get('raw_session_name', sb['session_name']),
                  sb['date'], sb['parts'], sb['position'])
        if sa_key != sb_key:
            return False
    return True


def diff_mouse(old_record, new_record):
    """Returns a list of human-readable change descriptions (empty list =
    no differences). old_record may be None (mouse is new)."""
    if old_record is None:
        return ['NEW MOUSE']

    changes = []
    for field in MOUSE_ATTR_FIELDS:
        old_val = old_record.get(field)
        new_val = new_record.get(field)
        if old_val != new_val:
            changes.append(f"{field}: {old_val!r} -> {new_val!r}")

    if not _sessions_equal(old_record.get('sessions'), new_record.get('sessions')):
        old_n = len(old_record.get('sessions') or [])
        new_n = len(new_record.get('sessions') or [])
        changes.append(f"sessions changed ({old_n} session(s) -> {new_n} session(s))")

    return changes


def compute_diff(existing_mice, fresh_mice):
    """
    existing_mice, fresh_mice: dicts of {mouse_id: record} as produced by
    h5_store.read_existing_mice / excel_reader.read_excel

    Returns:
        new_mice: list of mouse_ids only in fresh_mice
        changed_mice: dict of {mouse_id: [change descriptions]} for mice in
                      both, where something differs
        unchanged_mice: list of mouse_ids identical in both
        removed_mice: list of mouse_ids in existing_mice but not fresh_mice
    """
    new_mice = []
    changed_mice = {}
    unchanged_mice = []

    for mouse_id, new_record in fresh_mice.items():
        old_record = existing_mice.get(mouse_id)
        changes = diff_mouse(old_record, new_record)
        if old_record is None:
            new_mice.append(mouse_id)
        elif changes:
            changed_mice[mouse_id] = changes
        else:
            unchanged_mice.append(mouse_id)

    removed_mice = [m for m in existing_mice if m not in fresh_mice]

    return new_mice, changed_mice, unchanged_mice, removed_mice