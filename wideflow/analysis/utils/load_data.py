import os
from tifffile import TiffFile
from wideflow.utils.imaging_utils import load_config
import cv2
from .extract_from_metadata_file import extract_from_metadata_file
import numpy as np


def load_data(dir_path):
    wf_video_paths = []
    for file in os.listdir(dir_path):
        if file.endswith(".tif"):
            wf_video_paths.append(os.path.join(dir_path, file))
        if file.endswith(".txt"):
            metadata_path = os.path.join(dir_path, file)
        if file.endswith(".avi"):
            behavioral_video_path = os.path.join(dir_path, file)
        if file.endswith(".json"):
            config = load_config(file)

    print("Extracting metadata")
    timestamp, cue, metric_result, serial_readout = extract_from_metadata_file(metadata_path)
    metadata = {"timestamp": timestamp, "cue": cue, "metric_result": metric_result, "serial_readout": serial_readout}

    print("Loading behavioral camera video")
    cap = cv2.VideoCapture(behavioral_video_path)
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    behavioral_data = np.empty((n_frame, width, height), np.dtype('uint8'))
    for i in range(n_frame):
        ret, behavioral_data[i] = cap.read()

    cap.release()

    print("Loading wide field camera video")
    n_frames = 0
    wf_vids = []
    for i, vid_p in enumerate(wf_video_paths):
        print(f"     part {i+1}")
        with TiffFile(vid_p) as tif:
            wf_vids.append(tif.series[0].asarray())
            n_frames += wf_vids[-1].shape[0]

    wf_data = np.ndarray((n_frames, wf_vids[0].shape[1], wf_vids[0].shape[2]), dtype=wf_vids[0].dtype)
    offset = 0
    for wf_vid in wf_vids:
        wf_data[offset: offset + wf_vid.shape[0]] = wf_vid

    print("Done")
    return wf_data, behavioral_data, metadata, config

