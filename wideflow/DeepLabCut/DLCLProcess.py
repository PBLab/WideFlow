from devices.FLIRCam import FLIRCam
import PySpin

# import sys
# sys.path.insert(1, '/home/pb/PycharmProjects/DeepLabCut-live/dlclive')
import dlclive
from dlclive import DLClive

from DeepLabCut.DLCLProcessor import WFProcessor

import numpy as np
from multiprocessing import shared_memory


class BehavioralMonitoring:
    def __init__(self, query, model_path, model_config, processor_config):
        self.query = query
        self.model_path = model_path  # path to the DLC folder containing the .pb files
        self.model_config = model_config
        self.processor_config = processor_config

        self.dlc_processor = WFProcessor(**self.processor_config)
        self.dlc_live = DLClive(self.model_path, processor=self.dlc_processor)
        self.dlc_live.init_inference()  # TODO: provide frame?

        self.result = None

    def __call__(self, frame_mem_name, pose_mem_name):
        existing_shm = shared_memory.SharedMemory(name=frame_mem_name)
        frame = np.ndarray(shape=self.processor_config["pose_shape"],
                          dtype=self.processor_config["pose_shape"], buffer=existing_shm.buf)

        existing_shm = shared_memory.SharedMemory(name=pose_mem_name)
        pose = np.ndarray(shape=self.processor_config["pose_shape"],
                           dtype=self.processor_config["pose_shape"], buffer=existing_shm.buf)
        # pose shape should be the same as DLC processor output pose

        while True:
            if not self.query.empty():
                q = self.query.get()
                if q == 'estimate':
                    pose[:] = self.dlc_live.get_pose(frame)
                    self.camera.save_to_avi(frame)
                elif q == 'finish':
                    break