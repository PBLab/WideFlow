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
    def __init__(self, detection_th, leg_th, nose_th, tounge_th):
        self.detection_th = detection_th

        self.leg_th = leg_th
        self.nose_th = nose_th
        self.tounge_th = tounge_th
        self.th = np.array([leg_th, leg_th, nose_th, tounge_th])  # same order as pose

        self.prev_pose = np.empty((4, 3), dtype=np.float32)

    def process(self, pose, **kwargs):
        '''

        Args:
            pose: ndarray with shape (4, 3): row represent annotation
                first column - x coordinate
                second column - y coordinate
                third column - detection probability
            **kwargs:

        Returns: bool: True for detected movement otherwise False

        '''
        certainty = pose[:, 2] > self.detection_th

        displacement = pose[:, :2] - self.prev_pose[:, :2]
        displacement = np.sqrt(displacement[:, 0]**2 + displacement[:, 1]**2)
        displacement = displacement > self.th
        displacement = True in certainty * displacement  # True if annotation and movement detected

        self.prev_pose = pose
        return displacement

    def save(self, file=""):
        pass



