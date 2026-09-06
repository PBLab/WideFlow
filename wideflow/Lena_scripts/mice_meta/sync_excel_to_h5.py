#!/usr/bin/env python3
"""
Sync mouse metadata from the Excel sheet into mice_metadata.h5.

Behavior:
  - New mice (not yet in the H5): written automatically, no confirmation needed.
  - Unchanged mice: skipped, nothing touched.
  - Changed mice (any field or session differs from what's in the H5):
        the exact change is printed, and you must confirm before it's
        overwritten. This exists specifically so a typo in Excel can't
        silently clobber good data.
  - Mice that were in the H5 but are no longer in the Excel (row deleted):
        printed, and you must confirm before they're removed from the H5.
        Nothing is deleted by default.

Usage:
    # Excel file already on local disk:
    python sync_excel_to_h5.py <excel_path> <h5_path> [--sheet Sheet1] [--yes-to-all]

    # Excel file lives in OneDrive (fetched via rclone before syncing):
    python sync_excel_to_h5.py --onedrive "Documents/Lab/mice_data.xlsx" <h5_path> \\
        [--onedrive-remote onedrive] [--onedrive-local-dir ~/local_data/] \\
        [--sheet Sheet1] [--yes-to-all]

    --yes-to-all skips ALL confirmation prompts (use for scripted/CI runs
    only -- for normal interactive use, leave this off so you get to review
    every change).

    --onedrive requires rclone installed and configured with a remote
    (see onedrive_fetch.py). When --onedrive is used, the positional
    excel_path is not needed -- the downloaded local copy is used instead.
"""

import argparse
import sys

from excel_reader import read_excel
from h5_store import read_existing_mice, write_mouse, delete_mouse
from diffing import compute_diff
from onedrive_fetch import get_onedrive_file, OneDriveFetchError



def confirm(prompt):
    while True:
        resp = input(f"{prompt} [y/n]: ").strip().lower()
        if resp in ('y', 'yes'):
            return True
        if resp in ('n', 'no'):
            return False
        print("Please answer y or n.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('excel_path', nargs='?', default=None,
                         help='Local path to the Excel file. Omit this if using --onedrive.')
    parser.add_argument('h5_path')
    parser.add_argument('--sheet', default='Sheet1')
    parser.add_argument('--yes-to-all', action='store_true',
                         help='Skip all confirmation prompts (non-interactive mode).')
    parser.add_argument('--onedrive', metavar='REMOTE_PATH', default=None,
                         help="Path of the Excel file inside OneDrive (e.g. "
                              "'Documents/Lab/mice_data.xlsx'). If given, the "
                              "file is fetched via rclone before syncing, and "
                              "the positional excel_path is ignored.")
    parser.add_argument('--onedrive-remote', default='onedrive',
                         help="Name of the rclone remote to use (default: 'onedrive'). "
                              "Must match a remote you set up with `rclone config`.")
    parser.add_argument('--onedrive-local-dir', default='~/local_data/',
                         help="Local directory the OneDrive file is downloaded into.")
    args = parser.parse_args()

    if args.onedrive:
        print(f"Fetching {args.onedrive!r} from OneDrive (remote={args.onedrive_remote!r}) ...")
        try:
            excel_path = get_onedrive_file(
                args.onedrive,
                local_dir=args.onedrive_local_dir,
                remote_name=args.onedrive_remote,
            )
        except OneDriveFetchError as e:
            print(f"ERROR fetching file from OneDrive: {e}")
            return 1
        print(f"Downloaded to {excel_path}")
    elif args.excel_path:
        excel_path = args.excel_path
    else:
        print("ERROR: provide either a local excel_path or --onedrive <remote_path>.")
        return 1

    print(f"Reading {excel_path} ...")
    fresh_mice, parse_warnings = read_excel(excel_path, sheet_name=args.sheet)

    if parse_warnings:
        print("\n=== PARSE WARNINGS (these entries were ignored or flagged) ===")
        for mouse_id, msg in parse_warnings:
            print(f"  [{mouse_id}] {msg}")
        print()

    # Mice whose sessions could not be resolved at all (hard error, e.g. a
    # dirty date cell with no skip exception) are excluded from writing --
    # print them clearly so they don't just vanish silently.
    unresolved = [m for m, r in fresh_mice.items() if r.get('sessions') is None]
    if unresolved:
        print("=== MICE SKIPPED ENTIRELY (unresolved sessions, see warnings above) ===")
        for m in unresolved:
            print(f"  {m}")
        print()
        fresh_mice = {m: r for m, r in fresh_mice.items() if m not in unresolved}

    print(f"Reading existing H5 state from {args.h5_path} ...")
    existing_mice = read_existing_mice(args.h5_path)

    new_mice, changed_mice, unchanged_mice, removed_mice = compute_diff(existing_mice, fresh_mice)

    print(f"\n=== SUMMARY ===")
    print(f"  New mice:        {len(new_mice)}")
    print(f"  Changed mice:    {len(changed_mice)}")
    print(f"  Unchanged mice:  {len(unchanged_mice)}")
    print(f"  Removed mice:    {len(removed_mice)}")
    print()

    # --- New mice: written automatically, no confirmation needed ---
    for mouse_id in new_mice:
        print(f"[NEW] Writing {mouse_id} ...")
        write_mouse(args.h5_path, mouse_id, fresh_mice[mouse_id])

    # --- Changed mice: show the diff, require confirmation per mouse ---
    for mouse_id, changes in changed_mice.items():
        print(f"\n[CHANGED] {mouse_id}:")
        for c in changes:
            print(f"    - {c}")
        if args.yes_to_all or confirm(f"Overwrite {mouse_id} in the H5 with these changes?"):
            write_mouse(args.h5_path, mouse_id, fresh_mice[mouse_id])
            print(f"  -> {mouse_id} updated.")
        else:
            print(f"  -> {mouse_id} left UNCHANGED in the H5.")

    # --- Removed mice: require confirmation per mouse before deleting ---
    for mouse_id in removed_mice:
        print(f"\n[REMOVED FROM EXCEL] {mouse_id} still exists in the H5 but not in the Excel.")
        if args.yes_to_all or confirm(f"Delete {mouse_id} from the H5?"):
            delete_mouse(args.h5_path, mouse_id)
            print(f"  -> {mouse_id} deleted from H5.")
        else:
            print(f"  -> {mouse_id} left IN the H5 (not deleted).")

    print("\nDone.")


if __name__ == '__main__':
    sys.exit(main())
