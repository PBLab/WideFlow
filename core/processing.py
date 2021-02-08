import numpy as np
import cv2
from utils.gen_utils import extract_rois_data


class Processing:
    def __init__(self):
        pass

    def process(self):
        pass


class OpticFlow(Processing):
    def __init__(self, method):
        super().__init__()
        self.method = method

    def process(self, image1, image2):
        pass


class SeqNMFPatternsDecomposition(Processing):
    def __init__(self, patterns, shape=None, rois_dict=None, extract_rois_data=False):
        super().__init__()
        self.patterns = patterns
        self.shape = shape
        self.rois_dict = rois_dict
        self.extract_rois_data = extract_rois_data

    def process(self, images, patterns):
        sequence = self.preprocess(images)
        [n, k, t] = patterns.shape
        overlap = np.zeros(k, t)
        for l in range(t):
            overlap += np.multiply(np.squeeze(patterns[:, l, :]), np.roll(sequence, l))

        return overlap

    def preprocess(self, images):
        if not self.shape:
            sequence = np.resize(images, self.shape)
        sequence.reshape(self.shape[0] * self.shape[1])
        if self.extract_rois_data:
            sequence = extract_rois_data(sequence)

        return sequence


