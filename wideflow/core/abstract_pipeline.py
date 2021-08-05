from abc import ABC, abstractmethod


class AbstractPipeLine(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fill_buffers(self):
        pass

    @abstractmethod
    def clear_buffers(self):
        pass

    @abstractmethod
    def get_input(self, input):
        pass

    @abstractmethod
    def process(self):
        pass

    @abstractmethod
    def evaluate(self):
        return

