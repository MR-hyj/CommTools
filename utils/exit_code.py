from enum import Enum

class EXIST_CODE(Enum):
    SHELL_RUNTIME_ERROR      = 101      # shell脚本运行错误
    ILLEGAL_RE_PATTERN_ERROR = 201      # 非法正则模式
    FILE_IO_ERROR            = 301      # 文件IO错误
    DIR_NOT_EMPTY_ERROR      = 401      # 目标目录非空 
