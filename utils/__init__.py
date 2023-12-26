from .loggerSingleton import LoggerSingleton
from .tools import *
from .argparser import parse_json2yolo_args, parse_augment_from_yolo_args
from .exit_code import EXIST_CODE
from .trigger import Trigger
from .timer import Timer

logger = LoggerSingleton().logger



UNIX_PLATFORMS = ['LINUX', 'DARWIN']
ALLOWED_IMAGE_FORMATS = [
    'jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP'
]

__all__ = [
    'logger', 'remove_last_sep_from_dir', 'check_path', 'EXIST_CODE', 'Timer', 'Trigger', 
    'parse_json2yolo_args', 'parse_augment_from_yolo_args', 
]
