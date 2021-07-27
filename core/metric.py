import importlib
import numpy as np
from utils.gen_utils import mse


class Metric:
    def __init__(self):
        self.module = importlib.import_module("core.processing")

    def calc_metric(self, input):
        pass

    def get_child_from_str(self, child_name, **kwargs):
        class_ = getattr(self.module, child_name)
        return class_(kwargs)


class SeqNMFPatternsCorr(Metric):
    def __init__(self, threshold, pattern):
        super().__init__()
        self.threshold = threshold
        self.pattern = pattern

    def calc_metric(self, overlaps):
        [k, t] = overlaps.shape
        th = self.threshold * t
        weights = np.mean(overlaps, 1)
        weights = weights / sum(weights)
        score = mse(weights)
        return score