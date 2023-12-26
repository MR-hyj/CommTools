import os
import logging
import colorlog
import time
import datetime

class LoggerSingleton(object):
    """单例日志类, 允许输出到指定文件

    Args:
        object (_type_): _description_

    Returns:
        _type_: _description_
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerSingleton, cls).__new__(cls)
            cls._instance._init_logger()
        return cls._instance

    def _init_logger(self):
        # log_level: color
        COLOR_CONFIG = {
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'WARN': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'purple',
        }

        self.__logger = logging.getLogger('logger')
        self.__log_root_dir = f'logs/{datetime.datetime.today().date()}'

        sh = logging.StreamHandler()

        if not os.path.exists(self.__log_root_dir):
            os.makedirs(self.__log_root_dir)

        # console log setting
        console_fmt = colorlog.ColoredFormatter(
            fmt="%(log_color)s%(asctime)s | %(filename)15s:%(lineno)4s | %(levelname)-8s | %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S', log_colors=COLOR_CONFIG
        )

        self.__logger.addHandler(sh)
        sh.setFormatter(console_fmt)

        # file log setting
        log_file = logging.FileHandler(f'{self.__log_root_dir}/{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.log', encoding='utf-8')

        file_fmt = logging.Formatter(
            fmt="%(asctime)s | %(filename)10s:%(lineno)4s | %(levelname)-8s | %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        log_file.setFormatter(file_fmt)
        self.__logger.addHandler(log_file)

    @property
    def logger(self):
        return self.__logger


if __name__ == '__main__':
    logger_1 = LoggerSingleton().logger
    logger_1.setLevel('DEBUG')

    logger_1.debug('This is a debug message')
    logger_1.info('This is an info message')
    logger_1.warning('This is a warning message')
    logger_1.error('This is an error message')
    logger_1.critical('This is a critical message')


    logger_2 = LoggerSingleton().logger
    print(logger_1 is logger_2)
