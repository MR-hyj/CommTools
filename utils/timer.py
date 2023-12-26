import time
import logging

class Timer:
    """
    用于循环计时
    """
    def __init__(self, from_file:str):
        self.__start_time = None
        self.__end_time = None
        self.__total_time_elapsed = 0
        self.__file_processing_time = 0
        self.__cnt_processed_files = 0
        self.__from_file = from_file

    def clock_start(self):
        self.__start_time = time.time()

    def clock_end(self):
        self.__end_time = time.time()
        self.__file_processing_time = self.__end_time - self.__start_time
        self.__total_time_elapsed += self.__file_processing_time
        self.__cnt_processed_files += 1

    def __format_time(self, seconds):
        """seconds -> 00 days: 00 hours: 00 mins: 00 secs

        Args:
            seconds: _description_

        Returns:
            str
        """
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)
        return f"{int(days)}d:{int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s"

    def clock_summary(self, cnt_remaining: int, logger: logging.Logger=None):
        log_func: callable = print if logger is None else logger.info
        average_time_per_file = self.__total_time_elapsed / self.__cnt_processed_files
        estimated_remaining_time = average_time_per_file * cnt_remaining

        log_func(f"Time taken for current file: {self.__file_processing_time*1000:.2f} ms.")
        log_func(f"Total time elapsed: {self.__format_time(self.__total_time_elapsed)}.")
        log_func(f"Average time per file: {average_time_per_file*1000:.2f} ms.")
        log_func(f"Estimated remaining time: {self.__format_time(estimated_remaining_time)}.")
