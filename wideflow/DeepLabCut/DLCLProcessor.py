import numpy as np
import sys
sys.path.insert(1, '/home/pb/PycharmProjects/DeepLabCut-live/dlclive')

# from dlclive.processor import Processor


# class WFProcessor(Processor):
class WFProcessor:
    """
    deep lab cut live processor.
    must implement two methods -
        "process": take a pose and return a pose
        "save": saves internal data generated by the processor
    """
    def __init__(self, lick_th, leg_th, whiskers_th):
        self.lick_th = lick_th
        self.leg_th = leg_th
        self.whiskers_th = whiskers_th

        self.prev_pose = np.empty((4, ), dtype=np.float32)

    def process(self, pose, **kwargs):
        pose = self.estimate_motion(pose)
        return pose

    def save(self, file=""):
        pass

    def estimate_motion(self, pose):
        pass

