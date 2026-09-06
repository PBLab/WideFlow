"""
Parsing and session-resolution logic for the mice metadata Excel -> H5 sync.

This module has no side effects (no file I/O) so it can be unit-tested and
reasoned about independently of the sync script itself.
"""

# Session columns in the order they appear in the Excel sheet. Order matters:
# it defines the "position" sequence used for skip_shift renumbering.
SESSION_COLUMNS = [
    'spont', 'CRC',
    'NF1', 'NF2', 'NF3', 'NF4', 'NF5', 'NF6', 'NF7', 'NF_control',
    'NF8', 'NF9', 'NF10', 'NF11', 'NF12', 'NF13', 'NF14',
]

SKIP_KEYWORDS = {'skip_shift', 'skip_gap'}
NONE_KEYWORD = 'none'


class ExceptionParseWarning(Exception):
    """Raised (and caught by the caller) when part_exceptions references a
    session column that doesn't exist for that mouse / at all."""
    pass


def parse_part_exceptions(raw, valid_session_names):
    """
    Parse a part_exceptions_and_skips cell into a dict:
        {session_name: 'none' | 'skip_shift' | 'skip_gap' | ['p1','p2',...]}

    Session name keys are matched case-insensitively against
    valid_session_names, but the returned dict uses the CANONICAL
    (properly-cased) session name so downstream lookups are exact.

    Returns (parsed_dict, warnings) where warnings is a list of human-readable
    strings for any entry that didn't match a real session column -- the
    caller decides whether/how to surface these (we never fail silently).
    """
    parsed = {}
    warnings = []

    if not raw or not isinstance(raw, str):
        return parsed, warnings

    # case-insensitive lookup: lowercased column name -> canonical column name
    lower_to_canonical = {name.lower(): name for name in valid_session_names}

    for entry in raw.split(';'):
        entry = entry.strip()
        if not entry:
            continue  # tolerate trailing ';' or accidental ';;'

        if ':' not in entry:
            warnings.append(
                f"Could not parse exception entry {entry!r} (missing ':') -- ignored."
            )
            continue

        session_key, _, value = entry.partition(':')
        session_key = session_key.strip()
        value = value.strip()

        canonical = lower_to_canonical.get(session_key.lower())
        if canonical is None:
            warnings.append(
                f"Exception entry {entry!r} references session "
                f"{session_key!r}, which is not a recognized session column "
                f"for this mouse. This entry will be IGNORED."
            )
            continue

        if canonical in parsed:
            warnings.append(
                f"Duplicate exception entry for session {canonical!r} "
                f"(entry {entry!r}) -- later entry overwrites earlier one."
            )

        if ',' in value:
            parts = [v.strip() for v in value.split(',') if v.strip()]
            parsed[canonical] = parts
        elif value.lower() == NONE_KEYWORD:
            parsed[canonical] = NONE_KEYWORD
        elif value.lower() in SKIP_KEYWORDS:
            parsed[canonical] = value.lower()
        else:
            # single part like "p1" with no comma
            parsed[canonical] = [value]

    return parsed, warnings


def parts_to_list(default_parts):
    """Convert the Default_parts cell (e.g. 3, '3', or 'p1,p2,p3') into a
    canonical list like ['p1', 'p2', 'p3']."""
    if default_parts is None:
        return []
    if isinstance(default_parts, (int, float)):
        n = int(default_parts)
        return [f'p{i}' for i in range(1, n + 1)]
    s = str(default_parts).strip()
    if ',' in s:
        return [v.strip() for v in s.split(',') if v.strip()]
    if s.isdigit():
        return [f'p{i}' for i in range(1, int(s) + 1)]
    return [s]  # single value like 'p1'


def resolve_mouse_sessions(mouse_row, default_parts_list, exceptions):
    """
    Build the ordered list of usable sessions for one mouse.

    mouse_row: dict of {session_column_name: date_value_or_None_or_str}
               (only session columns, e.g. from SESSION_COLUMNS)
    default_parts_list: e.g. ['p1', 'p2', 'p3']
    exceptions: dict from parse_part_exceptions()

    Returns a list of dicts:
        {
          'session_name': str,       # DISPLAY name, e.g. 'NF5' -- this is
                                      # what shows up on plots/labels, and
                                      # what SESSIONS_TO_USE filters against.
                                      # After a skip_shift, this is the
                                      # RENAMED slot (e.g. 'NF7'), not the
                                      # column the data actually came from.
          'raw_session_name': str,   # REAL column name the date/data came
                                      # from, e.g. 'NF8' after a skip_shift
                                      # on NF7. Use this (not session_name)
                                      # when building the actual session_id
                                      # for file/results lookups.
          'date': str,               # 'YYYYMMDD'
          'parts': list[str],        # e.g. ['p1','p3'] or [] for no-suffix session
          'position': int,           # sequence index used for renumbered labels
        }
    Sessions with a bad/non-numeric date cell (e.g. "20250803 bad, don't use")
    that are NOT covered by a skip_* exception are raised as a ValueError --
    this is a real data problem the user must resolve explicitly rather than
    have the code silently guess.

    skip_shift behavior: when a session (e.g. NF7) is flagged skip_shift, its
    own date is dropped, and every session AFTER it in SESSION_COLUMNS order
    is relabeled back by one slot to fill the gap -- so NF8's date becomes
    session_name 'NF7' (raw_session_name stays 'NF8'), NF9 becomes 'NF8', and
    so on. Multiple skip_shift flags on the same mouse compound (each shifts
    everything after it back by one more slot).
    """
    # First pass: collect the raw (name, date, exception) for every session
    # that actually has a date, in column order -- this is the sequence we
    # then relabel according to skip_shift.
    raw_entries = []
    for col in SESSION_COLUMNS:
        if col not in mouse_row:
            continue
        raw_date = mouse_row[col]
        if raw_date is None:
            continue  # blank cell = session never happened, always silently skip
        raw_entries.append((col, raw_date, exceptions.get(col)))

    resolved = []
    position = 0
    shift_offset = 0  # how many slots to shift raw_session_name back by

    for raw_name, raw_date, exc in raw_entries:
        if exc == 'skip_shift':
            shift_offset += 1  # this session is dropped; everything after
            continue           # it shifts back one more slot

        if exc == 'skip_gap':
            position += 1  # slot reserved, but no session emitted
            continue

        # At this point we expect a clean date. If it's not int/clean numeric
        # string, and there's no skip_* exception marking it, that's a real
        # problem -- fail loudly instead of guessing.
        date_str = _coerce_date(raw_date, mouse_row=mouse_row, col=raw_name)

        position += 1

        if exc == NONE_KEYWORD:
            parts = []
        elif isinstance(exc, list):
            parts = exc
        elif exc is None:
            parts = default_parts_list
        else:
            # shouldn't happen given parse_part_exceptions' vocabulary
            raise ValueError(f"Unrecognized exception value {exc!r} for session {raw_name!r}")

        display_name = _shift_session_name(raw_name, shift_offset) if shift_offset else raw_name

        resolved.append({
            'session_name': display_name,
            'raw_session_name': raw_name,
            'date': date_str,
            'parts': parts,
            'position': position,
        })

    return resolved


def _shift_session_name(raw_name, shift_offset):
    """Renames e.g. 'NF8' back by shift_offset slots -> 'NF7' (shift_offset=1).
    Only NF-numbered columns are shiftable; non-numbered columns (spont, CRC,
    NF_control) are never touched by skip_shift and are returned unchanged --
    a skip_shift should only ever be set on a numbered NF session."""
    if not raw_name.startswith('NF') or raw_name == 'NF_control':
        return raw_name
    suffix = raw_name[2:]
    if not suffix.isdigit():
        return raw_name
    new_num = int(suffix) - shift_offset
    return f'NF{new_num}'


def _coerce_date(raw_date, mouse_row, col):
    """Turn a raw Excel cell value into a clean 'YYYYMMDD' string, or raise
    ValueError with a clear message if it looks like a flagged/dirty cell
    (e.g. '20250803 bad, don't use') that has no matching skip exception."""
    if isinstance(raw_date, int):
        return str(raw_date)
    if isinstance(raw_date, float):
        return str(int(raw_date))
    if isinstance(raw_date, str):
        s = raw_date.strip()
        if s.isdigit() and len(s) == 8:
            return s
        raise ValueError(
            f"Session {col!r} has a non-clean date value {raw_date!r} "
            f"(looks like a flagged/annotated cell) but no skip_shift/skip_gap "
            f"exception is set for it. Add one in part_exceptions_and_skips, "
            f"or fix the cell to a plain YYYYMMDD date."
        )
    raise ValueError(f"Unexpected type for date cell {col!r}: {raw_date!r} ({type(raw_date)})")


def build_session_id(mouse_id, date_str, session_name, part=None):
    """Match the original script's session_id convention:
        f'{date}_{mouse_id}_{session_name}'         (no-suffix sessions)
        f'{date}_{mouse_id}_{session_name}_{part}'  (parted sessions)
    """
    if part is None:
        return f'{date_str}_{mouse_id}_{session_name}'
    return f'{date_str}_{mouse_id}_{session_name}_{part}'