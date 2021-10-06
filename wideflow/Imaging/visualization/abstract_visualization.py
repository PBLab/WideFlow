from abc import ABC, abstractmethod


class AbstractVis(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        return

    @abstractmethod
    def terminate(self):
        pass