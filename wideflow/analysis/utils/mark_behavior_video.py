import cv2
import numpy as np
from analysis.utils.extract_from_metadata_file import extract_from_metadata_file


def mark_video(vid_path, meta_path):
    cue_rect = ((50, 50), (150, 150))
    cue_color = (0, 255, 0)
    readout_rect = ((200, 50), (300, 150))
    readout_color = (0, 0, 255)

    [_, cue, _, serial_readout] = extract_from_metadata_file(meta_path)
    readout = 1 - np.array(serial_readout)

    cap = cv2.VideoCapture(vid_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    marked_vid_path = vid_path[:-4] + '_cueGreen&lickRed_mark.avi'
    wrt = cv2.VideoWriter(marked_vid_path, fourcc=fourcc, fps=fps, frameSize=(width, height))

    for i in range(num_of_frames):
        ret, frame = cap.read()
        if ret:
            if cue[i]:
                frame = cv2.rectangle(frame, cue_rect[0], cue_rect[1], cue_color, -1)
            if readout[i]:
                frame = cv2.rectangle(frame, readout_rect[0], readout_rect[1], readout_color, -1)

            wrt.write(frame)
            print(f'appending frame: {i}', end='\r')
        else:
            print(f"couldn't retrieve frame {i}")

    cap.release()
    wrt.release()