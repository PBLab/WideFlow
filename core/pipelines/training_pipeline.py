from core.abstract_pipeline import AbstractPipeLine
import random
import time


class TrainingPipe(AbstractPipeLine):
    def __init__(self, time_delay, min_frame_count, max_frame_count):
        self.time_delay = time_delay
        self.min_frame_delay = min_frame_count
        self.max_frame_delay = max_frame_count

        self.counter = 0

    def fill_buffers(self):
        pass

    def get_input(self):
        pass

    def process(self):
        time.sleep(self.time_delay)

    def evaluate(self):

        if self.counter == 0:
            cue = 0
            cue_delay = random.choice(range(self.min_frame_count, self.max_frame_count, 1))

        self.counter += 1
        if self.counter == cue_delay:
            self.counter = 0
            cue = 1

        return cue
