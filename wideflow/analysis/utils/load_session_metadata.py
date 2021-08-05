import os
from wideflow.utils.imaging_utils import load_config
from wideflow.analysis.utils.extract_from_metadata_file import extract_from_metadata_file


def load_session_metadata(dir_path):
    for file in os.listdir(dir_path):
        if file.endswith(".txt"):
            metadata_path = os.path.join(dir_path, file)
        if file.endswith(".json"):
            config = load_config(os.path.join(dir_path, file))

    timestamp, cue, metric_result, serial_readout = extract_from_metadata_file(metadata_path)
    metadata = {"timestamp": timestamp, "cue": cue, "metric_result": metric_result, "serial_readout": serial_readout}

    return metadata, config

