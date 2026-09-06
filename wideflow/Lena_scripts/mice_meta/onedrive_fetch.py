"""
Fetches a file from OneDrive (via rclone) to a local path, so the sync
script can read an Excel file that lives in OneDrive rather than only
a plain local path.

Requires rclone to be installed and already configured with a remote
(default assumed name: 'onedrive') pointing at your OneDrive account.
Set that up once with:  rclone config

This module does not touch OneDrive credentials itself -- it only shells
out to whatever rclone remote you've already configured.
"""

import os
import shutil
import subprocess


class OneDriveFetchError(Exception):
    pass


def get_onedrive_file(remote_path, local_dir="~/local_data/", remote_name="onedrive"):
    """
    Downloads remote_path (a path inside your OneDrive, e.g.
    'Documents/Lab/mice_data.xlsx') to local_dir via rclone, and returns
    the local path to the downloaded file.

    remote_name is the rclone remote you configured with `rclone config`
    (default 'onedrive' -- change this if you named yours differently).
    """
    if shutil.which("rclone") is None:
        raise OneDriveFetchError(
            "rclone is not installed or not on PATH. Install it and run "
            "`rclone config` to set up a remote named "
            f"'{remote_name}' pointing at your OneDrive account, then retry."
        )

    local_dir = os.path.expanduser(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    try:
        subprocess.run(
            ["rclone", "copy", f"{remote_name}:{remote_path}", local_dir],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise OneDriveFetchError(
            f"rclone failed to fetch '{remote_path}' from remote "
            f"'{remote_name}'.\nstdout: {e.stdout}\nstderr: {e.stderr}\n"
            f"Check that the remote is configured (`rclone listremotes`) "
            f"and that the path exists in OneDrive."
        ) from e

    filename = os.path.basename(remote_path)
    local_path = os.path.join(local_dir, filename)

    if not os.path.exists(local_path):
        raise OneDriveFetchError(
            f"rclone reported success but expected file was not found at "
            f"{local_path!r}. Check remote_path is correct."
        )

    return local_path
