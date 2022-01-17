from abc import ABC, abstractmethod


class AbstractSession(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def session_preparation(self):
        pass

    @abstractmethod
    def run_session_pipeline(self):
        pass

    @abstractmethod
    def session_termination(self, input):
        pass

