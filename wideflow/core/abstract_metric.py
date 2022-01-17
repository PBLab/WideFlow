from abc import ABC, abstractmethod


class AbstractMetric(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def initialize_buffers(self):
        pass

    @abstractmethod
    def evaluate(self):
        return