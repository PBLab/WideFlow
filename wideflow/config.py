# Machine-portable storage paths. See config_local.example.py for what to put in
# config_local.py (gitignored, one real copy per machine).
# added by Claude 20260906

try:
    from wideflow.config_local import BASE_PATH, DATA_STAGING_PATH
except ImportError as e:
    raise ImportError(
        "Missing wideflow/config_local.py. Copy wideflow/config_local.example.py to "
        "wideflow/config_local.py and set this machine's real claustrum-storage/data paths."
    ) from e
