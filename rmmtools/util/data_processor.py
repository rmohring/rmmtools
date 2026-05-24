import functools

from abc import ABC, abstractmethod

from .general import get_now_local_and_utc, get_logger


class DataProcessor(ABC):
    def __init__(self, shortname="Processor", description=None, log_level="INFO"):
        self._description = description

        self.args = self.setup_command_line()
        try:
            if self.args.debug:
                log_level = "DEBUG"
        except AttributeError:
            pass

        self._startup_dttm = get_now_local_and_utc()

        self.log = get_logger(shortname, level=log_level)
        self.log.info("DataProcessor Instantiated")

        self._errors = []

        return

    @abstractmethod
    def setup_command_line(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def shutdown(self):
        pass

    # TODO: make this a track_exceptions decorator over the individual methods
    # def execute_method(self, method):
    #     try:
    #         getattr(self, method)()
    #     except Exception as e:
    #         msg = f"{method} failed: {e}"
    #         self.log.error(msg)
    #         self._errors.append(msg)
