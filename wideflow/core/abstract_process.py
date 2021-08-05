from abc import ABC, abstractmethod


class AbstractProcess(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def initialize_buffers(self):
        pass

    @abstractmethod
    def process(self):
        pass