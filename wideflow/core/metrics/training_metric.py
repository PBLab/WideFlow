from core.abstract_metric import AbstractMetric
import random
import numpy as np


class Training(AbstractMetric):
    def __init__(self, min_frame_count, max_frame_count):
        self.min_frame_count = min_frame_count
        self.max_frame_count = max_frame_count

        self.counter = 0
        self.cue_delay = 0
        self.cue = 0
        self.result = 0

    def initialize_buffers(self):
        pass

    def evaluate(self):
        if self.counter == 0:
            self.cue = 0
            self.cue_delay = random.choice(range(self.min_frame_count, self.max_frame_count, 1))

        self.counter += 1
        if self.counter > self.cue_delay - 15:
            self.cue = (15 - (self.cue_delay - self.counter)) / 15

        else:
            self.cue = np.clip(self.cue + random.random() / random.choice([10, 7, -10, -7]), -1, 0.8)

        if self.counter == self.cue_delay:
            self.counter = 0
            self.cue = 1

        self.result = self.cue
