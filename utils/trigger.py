import logging

class Trigger:

    def __init__(self, from_file) -> None:
        """统计每个item的执行情况, 分为succeed(顺利执行), fail(文件不存在等硬错误), corrupt(标签越界等软错误).
        item根据具体情况, 可以是file, 或者是某个json

        Args:
            from_file (_type_): _description_
        """
        self.__reset()
        self.__from_file=from_file

    def __reset(self) -> None:
        self.__cnt_succeed    = 0
        self.__cnt_fail       = 0
        self.__cnt_corrupted  = 0
        self.__failed_list    = []
        self.__corrupted_set = set()

    def trigger_succeed(self):
        self.__cnt_succeed += 1

    def trigger_fail(self, failed_item, exception: Exception=None, logger: logging.Logger=None):
        self.__cnt_fail += 1
        self.__failed_list.append(failed_item)

        if exception is not None:
            log_func: callable = print if logger is None else logger.error
            log_func(f'Exception={exception} when processing item={failed_item}')


    def trigger_corrupt(self, corrupted_item):
        self.__cnt_corrupted += 1
        self.__corrupted_set.add(corrupted_item)

    @property
    def cnt_succeed(self):
        return self.__cnt_succeed
    
    @property
    def cnt_fail(self):
        return self.__cnt_fail

    @property
    def cnt_corrupt(self):
        return self.__cnt_corrupted

    def summary(self, logger: logging.Logger=None):
        if logger is None:
            log_func: callable = print
        else:
            log_func: callable = logger.info
        log_func(f'trigger from {self.__from_file}')
        if self.__cnt_fail > 0:
            log_func(f"failed items: {self.__failed_list}")
        if self.__cnt_corrupted > 0:
            log_func(f"corrupted items: {self.__corrupted_set}")
        if 0 == self.__cnt_fail and 0 == self.__cnt_corrupted:
            log_func("all items succeed.")
