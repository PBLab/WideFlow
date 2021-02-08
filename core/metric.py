import numpy as np
from utils.gen_utils import mse


class Metric:
    def __init__(self):
        pass

    def metric(self, input):
        pass


class SeqNMFPatternsCorr(Metric):
    def __init__(self, threshold, pattern):
        super().__init__()
        self.threshold = threshold
        self.pattern = pattern

    def mertic(self, overlaps):
        [k, t] = overlaps.shape
        th = self.threshold * t
        weights = np.mean(overlaps, 1)
        weights = weights / sum(weights)
        score = mse(weights)
        return score
