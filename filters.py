"""built-in modules"""

import os
import re
import logging
from xml.etree import ElementTree
from abc import abstractmethod, ABC
from typing import Any, Union, List, Tuple, Dict, Optional, Callable, Literal
import datetime
from xml.etree import ElementTree

"""3rd party modules"""
import numpy as np

"""customized modules"""
from .tools import (
    run_cmd,
    bbox_xnynwnhn2xyxy,
    is_unix_platform,
    calc_iou_one2N,
    calc_iou_one2one,
)


class BaseFileFilter(ABC):
    #! 一个FileFilter类必须且仅能针对一种filetype, 要么是image要么是label
    #! 如果是filepath, 则表示该filter两者都能接受
    FILEPATH_TYPE: Literal["image_filepath", "label_filepath", "filepath"] = (
        "filepath"
    )

    def __init__(
        self,
        drop_prob: float = 1.0,
        keep_prob: float = 1.0,
        reverse: bool = False,
        *args: Any,
        **kwargs: Dict[Any, Any],
    ) -> None:
        """file filter的基类, 不应该被实例化

        Args:
            drop_prob (float): 如果不符合过滤条件, 则该样本有多少概率会被丢弃
            keep_prob (float): 如果符合过滤条件, 则该样本有多少概率被保留
            reverse (bool, optional): 是否翻转过滤条件的结果. Defaults to False.
        """
        super().__init__()
        self.reverse = reverse
        assert (
            0 <= drop_prob <= 1
        ), f"$drop_prob is expected to be between [0, 1], but received ${drop_prob}."
        self.drop_prob = drop_prob

        assert (
            0 <= keep_prob <= 1
        ), f"$drop_prob is expected to be between [0, 1], but received ${keep_prob}."
        self.keep_prob = keep_prob

        self.keep_negative: bool = bool(kwargs.get("keep_negative", False))

    def __call__(self, *args: Any, **kwargs: Dict[Any, Any]) -> bool:
        filepath = kwargs.get(self.__class__.FILEPATH_TYPE, None)
        if not os.path.isfile(filepath):
            self._log(
                f"no such file as {filepath}",
                verbose=kwargs.get("verbose", False),
                logger=kwargs.get("logger", None),
                logger_func="error",
            )
            return self.keep_negative

        filter_res = self._filter(filepath, *args, **kwargs)

        # ; 是否反转过滤结果. 即该过滤器应该把符合条件的样本加进来还是去除.
        if self.reverse:
            filter_res = not (bool(filter_res))

        if not filter_res:
            # ; 此时filter_res依然为False, 表示不符合过滤条件
            # ; 随机选择一个[0, 1]之间的值, 若小于prob则返回False, 丢弃该样本
            # ; 若大于等于prob, 则返回True, 保留该样本
            rand = np.random.rand()  # rand in [0, 1)
            if_keep = rand >= self.drop_prob
            self._log(
                f"random float: {rand}, drop_prob: {self.drop_prob}, keep: {if_keep}",
                verbose=kwargs.get("verbose", False),
                logger=kwargs.get("logger", None),
            )
            return if_keep
        else:
            # ; 此时filter_res为True, 符合过滤条件
            # ; 随机选择一个[0, 1]之间的值, 若小于prob则返回True, 保留该样本
            # ; 若大于等于prob, 则返回False, 丢弃该样本
            rand = np.random.rand()  # rand in [0, 1)
            if_keep = rand < self.keep_prob
            self._log(
                f"random float: {rand}, keep_prob: {self.keep_prob}, keep: {if_keep}",
                verbose=kwargs.get("verbose", False),
                logger=kwargs.get("logger", None),
            )
            return if_keep

    def __repr__(self) -> str:
        return f"{self._summary_dict()}"

    def _summary_dict(self) -> Dict[str, Any]:
        attr_dict = {"name": self.__class__.__name__}
        attr_dict.update(vars(self))
        return attr_dict

    def __repr__(self) -> str:
        return str(self.summary_dict)

    @property
    def summary_dict(self) -> Dict[str, Any]:
        return self._summary_dict()

    def _log(
        self,
        msg: str,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
        logger_func: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not verbose:
            return

        if logger:
            if logger_func is not None:
                log_func = (
                    getattr(logger, logger_func)
                    if hasattr(logger, logger_func)
                    else logger.info
                )
            else:
                log_func = logger.info
        else:
            log_func = print
        log_func(msg)

    @abstractmethod
    def _filter(
        self, filepath: str, *args: Any, **kwargs: Dict[Any, Any]
    ) -> bool:
        pass


class LabelBaseFileFilter(BaseFileFilter, ABC):
    FILEPATH_TYPE: Literal["image_filepath", "label_filepath", "filepath"] = (
        "label_filepath"
    )

    def _get_cmd(self, filepath: str) -> str:
        ann_type = filepath.rsplit(".", 1)[1]
        if ann_type.lower() == "xml":
            cmd = self._get_cmd_xml(filepath)
        elif ann_type.lower() == "txt":
            cmd = self._get_cmd_yolo(filepath)
        else:
            raise ValueError(
                f"non support file type, expected to be one of [xml, txt], but detected {ann_type}"
            )
        return cmd

    def _parse_by_shell(
        self,
        filepath: str,
        map_func: Optional[Callable[..., Any]] = int,
        *args: Any,
        **kwargs: Dict[Any, Any],
    ):
        cmd = self._get_cmd(filepath)
        res = run_cmd(cmd)
        return map_func(res) if map_func is not None else res

    def _parse_by_python_read(
        self, filepath: str, *args: Any, **kwargs: Dict[Any, Any]
    ) -> Any:
        ann_type = filepath.rsplit(".", 1)[1].lower()
        if ann_type == "xml":
            return self._parse_by_python_read_xml(filepath, *args, **kwargs)
        elif ann_type == "txt":
            return self._parse_by_python_read_txt(filepath, *args, **kwargs)
        else:
            raise ValueError(
                f"non support file type, expected to be one of [xml, txt], but detected {ann_type}"
            )

    def _parse_by_python_read_txt(
        self, yolo_label_filepath: str, *args: Any, **kwargs: Dict[Any, Any]
    ) -> Any:
        raise NotImplementedError

    def _parse_by_python_read_xml(
        self, xml_label_filepath: str, *args: Any, **kwargs: Dict[Any, Any]
    ) -> Any:
        raise NotImplementedError

    def _get_cmd_xml(self, filepath: str) -> str:
        raise NotImplementedError

    def _get_cmd_txt(self, filepath: str) -> str:
        raise NotImplementedError


class ImageBaseFileFilter(BaseFileFilter, ABC):
    FILEPATH_TYPE: Literal["image_filepath", "label_filepath", "filepath"] = (
        "image_filepath"
    )


class DirectPassFileFilter(BaseFileFilter):
    FILEPATH_TYPE: Literal["image_filepath", "label_filepath", "filepath"] = (
        "image_filepath"
    )

    def __init__(
        self,
        drop_prob: float = 1.0,
        keep_prob: float = 1.0,
        reverse: bool = False,
        *args: Any,
        **kwargs: Dict[Any, Any],
    ) -> None:
        """
        不作任何处理, 直接返回True, 即不作任何过滤

        Args:
            drop_prob (float, optional): _description_. Defaults to 1.
            reverse (bool, optional): _description_. Defaults to False.
        """
        super().__init__(drop_prob, keep_prob, reverse, *args, **kwargs)

    def _filter(
        self, filepath: str, *args: Any, **kwargs: Dict[Any, Any]
    ) -> bool:
        self._log(
            f"called {self}",
            verbose=kwargs.get("verbose", False),
            logger=kwargs.get("logger", None),
            logger_func="debug",
        )
        return True


class PathPatternFileFilter(ImageBaseFileFilter):
    def __init__(
        self,
        path_patterns: Union[str, Tuple[str], List[str]],
        drop_prob: float = 1.0,
        keep_prob: float = 1.0,
        reverse: bool = False,
        *args: Any,
        **kwargs: Dict[Any, Any],
    ) -> None:
        """根据文件的路径中是否含有特定的pattern来匹配
        #; 比如pattern可以取 {直线杆, 耐张杆, 变压器, ...} 来过滤一些场景
        #; 或者pattern可以取线路名来过滤特定的线路

        Args:
            path_patterns (list[str]): 只要符合这个list中的一个即可
            drop_prob (float): 如果不符合过滤条件, 则该样本有prob的概率会被丢弃
            reverse (bool, optional): 是否翻转过滤条件的结果. Defaults to False.
        """
        super().__init__(drop_prob, keep_prob, reverse, *args, **kwargs)
        if isinstance(path_patterns, (str)):
            self.path_patterns = [path_patterns]
        elif isinstance(path_patterns, (List, Tuple)):
            self.path_patterns = path_patterns
        else:
            raise TypeError(
                f"wrong type of ScenarioTypeFilter.scenario_types, expected to be one of [str, List[str], Tuple[str]], but {type(self.path_patterns)} detected"
            )

    def _filter(self, filepath: str, *args: any, **kwargs: any) -> bool:
        self._log(
            f"called {self}",
            verbose=kwargs.get("verbose", False),
            logger=kwargs.get("logger", None),
            logger_func="debug",
        )
        # ; 过滤结果, 保险起见将filepath转换成绝对路径,
        # ; 因为相对路径会丢失一部分路径信息, 这部分被丢失的信息里面可能包含有pattern
        filepath_ = os.path.abspath(filepath)
        # ; 根据filepath的路径, 如果其中包含了self.path_patterns中的任何一项, 则认为符合条件, 该样本不会被过滤
        res = next(
            (
                re.search(f"{path_pattern}", filepath_).group()
                for path_pattern in self.path_patterns
                if re.search(f"{path_pattern}", filepath_)
            ),
            None,
        )

        self._log(
            f"matches pattern: {res} in {self.path_patterns}",
            verbose=kwargs.get("verbose", False),
            logger=kwargs.get("logger", None),
        )
        return res is not None


class PostfixFileFilter(BaseFileFilter):
    """
    #? 是否真的需要后缀名的filter?
    #? 因为后缀名也包含在path中, 同样可以使用PathPatternFileFilter来实现过滤

    Args:
        BaseFileFilter (_type_): _description_
    """

    pass


class ModTimeFileFilter(LabelBaseFileFilter):
    def __init__(
        self,
        mod_time_ddl: Optional[Union[int, datetime.datetime]] = None,
        drop_prob: float = 1.0,
        keep_prob: float = 1.0,
        reverse: bool = False,
        *args: Any,
        **kwargs: Dict[Any, Any],
    ) -> None:
        """根据文件最后修改时间进行filter

        Args:
            mod_time_ddl (Union[int, datetime.datetime]): 允许的最晚修改时间, 当文件的最终修改时间大于这个值时, 返回true. 可以是int或者datetime. int以hour为单位. Defaults to None
            drop_prob (float): 如果不符合过滤条件, 则该样本有prob的概率会被丢弃
            reverse (bool, optional): 是否翻转过滤条件的结果. Defaults to False.
        """
        super().__init__(drop_prob, keep_prob, reverse, *args, **kwargs)
        if mod_time_ddl is None:
            self.mod_time_ddl = datetime.datetime.now()
        else:
            self.mod_time_ddl = mod_time_ddl

    def _filter(
        self, filepath: str, *args: Any, **kwargs: Dict[Any, Any]
    ) -> bool:
        self._log(
            f"called {self}",
            verbose=kwargs.get("verbose", False),
            logger=kwargs.get("logger", None),
            logger_func="debug",
        )
        file_last_mod_time = datetime.datetime.fromtimestamp(
            os.stat(filepath).st_mtime
        )
        if isinstance(self.mod_time_ddl, int):
            mod_time_ddl = (
                datetime.timedelta(hours=self.mod_time_ddl) + file_last_mod_time
            )
        elif isinstance(self.mod_time_ddl, datetime.datetime):
            mod_time_ddl = self.mod_time_ddl
        else:
            raise TypeError(
                f"wrong type of ModTimeFilter.mod_time_ddl, expected to be one of [int, datetime.datetime], but {type(self.mod_time_ddl)} received"
            )

        self._log(
            f"file last mod time: {file_last_mod_time}",
            verbose=kwargs.get("verbose", False),
            logger=kwargs.get("logger", None),
        )
        return file_last_mod_time <= mod_time_ddl

    def _summary_dict(self) -> Dict[str, Any]:
        attr_dict = super()._summary_dict()
        attr_dict["mod_time_ddl"] = datetime.datetime.strftime(
            attr_dict["mod_time_ddl"], "%Y-%m-%d %H:%M:%S"
        )
        return attr_dict


class ByteSizeFileFilter(LabelBaseFileFilter):
    def __init__(
        self,
        min_byte_size: int = 0,
        max_byte_size: int = 3 * 1024,
        drop_prob: float = 1.0,
        keep_prob: float = 1.0,
        reverse: bool = False,
        *args: Any,
        **kwargs: Dict[Any, Any],
    ) -> None:
        """根据文件大小(Byte)为单位进行filter

        Args:
            min_byte_size (int): 允许的最小文件大小(Byte). Defaults to 0
            max_byte_size (int): 允许的最大文件大小(Byte). Defaults to 3*1024 (3KiB)
            drop_prob (float): 如果不符合过滤条件, 则该样本有prob的概率会被丢弃
            reverse (bool, optional): 是否翻转过滤条件的结果. Defaults to False.
        """
        super().__init__(drop_prob, keep_prob, reverse, *args, **kwargs)
        self.min_byte_size = min_byte_size
        self.max_byte_size = max_byte_size

    def _filter(
        self, filepath: str, *args: Any, **kwargs: Dict[Any, Any]
    ) -> bool:
        self._log(
            f"called {self}",
            verbose=kwargs.get("verbose", False),
            logger=kwargs.get("logger", None),
            logger_func="debug",
        )
        file_byte_size = os.stat(filepath).st_size
        self._log(
            f"file size: {file_byte_size}",
            verbose=kwargs.get("verbose", False),
            logger=kwargs.get("logger", None),
        )
        return self.min_byte_size <= file_byte_size <= self.max_byte_size


class InstanceNumFileFilter(LabelBaseFileFilter):

    def __init__(
        self,
        inst_min_num: int = 0,
        inst_max_num: int = 100,
        drop_prob: float = 1.0,
        keep_prob: float = 1.0,
        reverse: bool = False,
        *args: Any,
        **kwargs: Dict[Any, Any],
    ) -> None:
        """根据文件中的gt目标数量来filter

        Args:
            inst_min_num (int, optional): 允许的最少目标数量. Defaults to 0.
            inst_max_num (int, optional): 允许的最大目标数量. Defaults to 100.
            drop_prob (float): 如果不符合过滤条件, 则该样本有prob的概率会被丢弃
            reverse (bool, optional): 是否翻转过滤条件的结果. Defaults to False.
        """
        super().__init__(drop_prob, keep_prob, reverse, *args, **kwargs)
        self.inst_min_num = inst_min_num
        self.inst_max_num = inst_max_num

    def _filter(
        self, filepath: str, *args: Any, **kwargs: Dict[Any, Any]
    ) -> bool:
        exec_func: Callable[..., bool] = (
            self._parse_by_shell
            if is_unix_platform()
            else self._parse_by_python_read
        )
        self._log(
            f"called {self}.{exec_func.__name__}",
            verbose=kwargs.get("verbose", False),
            logger=kwargs.get("logger", None),
            logger_func="debug",
        )
        inst_num = exec_func(filepath, map_func=int, *args, **kwargs)
        self._log(
            f"file contains {inst_num} instances",
            verbose=kwargs.get("verbose", False),
            logger=kwargs.get("logger", None),
        )
        return self.inst_min_num <= inst_num <= self.inst_max_num

    def _get_cmd_xml(self, filepath: str) -> str:
        return f'grep -E "<name>.*</name>" -c "{filepath}"'

    def _get_cmd_txt(self, filepath: str) -> str:
        return f'grep "$" -c "{filepath}"'

    def _parse_by_python_read_xml(
        self, xml_label_filepath: str, *args: Any, **kwargs: Dict[Any, Any]
    ) -> int:
        xml_root = ElementTree.parse(xml_label_filepath).getroot()

        targets_in_xml = [
            x.find("name").text for x in xml_root.findall("object")
        ]
        self._log(
            f"loaded {len(targets_in_xml)} instances: {targets_in_xml}",
            verbose=kwargs.get("verbose", False),
            logger=kwargs.get("logger", None),
        )
        return len(targets_in_xml)

    def _parse_by_python_read_txt(
        self, yolo_label_filepath: str, *args: Any, **kwargs: Dict[Any, Any]
    ) -> int:
        yolo_labels = np.loadtxt(yolo_label_filepath, ndmin=2)
        self._log(
            f"loaded {yolo_labels.shape[0]} instances: {yolo_labels[:, 0]}",
            verbose=kwargs.get("verbose", False),
            logger=kwargs.get("logger", None),
        )
        return yolo_labels.shape[0]


class CategoryFileFilter(LabelBaseFileFilter):
    def __init__(
        self,
        categories: List[Union[str, int]],
        drop_prob: float = 1.0,
        keep_prob: float = 1.0,
        reverse: bool = False,
        *args: Any,
        **kwargs: Dict[Any, Any],
    ) -> None:
        """根据xml中是否包含特定的类别来filter, 当输入多个类别时, 只需要存在一个即可.

        Args:
            categories (List[str]): _description_
            drop_prob (float): 如果不符合过滤条件, 则该样本有prob的概率会被丢弃
            reverse (bool, optional): 是否翻转过滤条件的结果. Defaults to False.
        """
        super().__init__(drop_prob, keep_prob, reverse, *args, **kwargs)
        self.categories = categories

    def _filter(
        self, filepath: str, *args: Any, **kwargs: Dict[Any, Any]
    ) -> bool:
        exec_func: Callable[..., bool] = (
            self._parse_by_shell
            if is_unix_platform()
            else self._parse_by_python_read
        )
        self._log(
            f"called {self}.{exec_func.__name__}",
            verbose=kwargs.get("verbose", False),
            logger=kwargs.get("logger", None),
            logger_func="debug",
        )
        # ; filepath中, 类别在self.categories中的instance的数量
        inst_in_cates = exec_func(filepath, map_func=int, *args, **kwargs)
        self._log(
            f"file contains {inst_in_cates} instances in {self.categories}",
            verbose=kwargs.get("verbose", False),
            logger=kwargs.get("logger", None),
        )
        return inst_in_cates > 0

    def _parse_by_python_read_txt(
        self, yolo_label_filepath: str, *args: Any, **kwargs: Dict[Any, Any]
    ) -> int:
        yolo_labels = np.loadtxt(yolo_label_filepath, ndmin=2)

        # ; idx, xcn, ncn, wn, hn, ...
        cls = yolo_labels[:, 0].astype(np.int32)
        self._log(
            f"loaded {cls.shape[0]} instances: {cls}",
            verbose=kwargs.get("verbose", False),
            logger=kwargs.get("logger", None),
        )
        cnt = 0
        for each_cls in cls:
            if each_cls in [int(x) for x in self.categories]:
                cnt += 1
        return cnt

    def _parse_by_python_read_xml(
        self, xml_label_filepath: str, *args: Any, **kwargs: Dict[Any, Any]
    ) -> int:
        xml_root = ElementTree.parse(xml_label_filepath).getroot()

        targets_in_xml = [
            x.find("name").text
            for x in xml_root.findall("object")
            if x.find("name").text in self.categories
        ]
        self._log(
            f"loaded {len(targets_in_xml)} instances: {targets_in_xml}",
            verbose=kwargs.get("verbose", False),
            logger=kwargs.get("logger", None),
        )
        return len(targets_in_xml)

    def _get_cmd_xml(self, filepath: str) -> str:
        pattern = "|".join(
            [f"<name>{cate}</name>" for cate in self.categories]
        )  # ; <name>cate_1</name>|<name>cate_2</name> ...
        # ; 统计filepath中,在self.categories中间的目标的数量
        return f'grep -E "{pattern}" -c "{filepath}"'

    def _get_cmd_txt(self, filepath: str) -> str:
        # ; 统计filepath中,在self.categories中间的目标的数量
        # ; ^$i(\.[0]{1,}[e+[0]{1,}]{0,}\s|\s)可以匹配的模式:
        # ;      ^i(space)
        # ;      ^i.0000(若干个0)(space)
        # ;      ^i.0000(若干个0)e+00000(若干个0)(space)
        pattern = "|".join(
            [
                f"^{cate}(\.[0]{{1,}}[e+[0]{{1,}}]{{0,}}\s|\s)"
                for cate in self.categories
            ]
        )
        return f'grep -E "{pattern}" -c "{filepath}"'


class InstanceAreaFileFilter(LabelBaseFileFilter):
    def __init__(
        self,
        categories: Union[List[int], List[str]],
        min_area: Union[float, List[float]] = 0,
        max_area: Union[float, List[float]] = np.inf,
        drop_prob: float = 1,
        keep_prob: float = 1.0,
        reverse: bool = False,
        *args: Any,
        **kwargs: Dict[Any, Any],
    ) -> None:
        """
        category类别中的所有Instance的面积是否全部在[min_area, max_area]之间,
        如果存在某个inst的area不在这个区间内, 返回False

        Args:
            categories (Union[List[int], List[str]]): _description_
            min_area (Union[float, List[float]], optional): _description_. Defaults to 0.
            max_area (Union[float, List[float]], optional): _description_. Defaults to np.inf.
            drop_prob (float, optional): _description_. Defaults to 1.
            reverse (bool, optional): _description_. Defaults to False.
        """
        super().__init__(drop_prob, keep_prob, reverse, *args, **kwargs)
        self.categories = categories
        self.min_area = self.__parse_area_parameter(min_area)
        self.max_area = self.__parse_area_parameter(max_area)

        assert len(self.min_area) == len(self.max_area) == len(self.categories)

    def __parse_area_parameter(
        self, area_param: Union[float, List[float]]
    ) -> List[float]:
        """
        Parse the area parameter which can be either a single value or a list of values.

        Args:
            area_param (Union[float, List[float]]): The area parameter to parse.

        Returns:
            List[float]: A list of area values.
        """
        if isinstance(area_param, float):
            return [area_param] * len(self.categories)
        elif isinstance(area_param, List):
            return area_param
        else:
            raise TypeError(
                f"Area parameter is expected to be one of [float, List[float]], but received {type(area_param)}"
            )

    def _filter(
        self, filepath: str, *args: Any, **kwargs: Dict[Any, Any]
    ) -> bool:
        # ; [cls, area]
        instance_in_target_class: List[Tuple[Union[int, str], float]] = (
            self._parse_by_python_read(filepath, *args, **kwargs)
        )
        self._log(
            f'found instances in {self.categories}: {[{"cls": x[0], "area": x[1]} for x in instance_in_target_class]}',
            verbose=kwargs.get("verbose", False),
            logger=kwargs.get("logger", None),
        )

        # ; 遍历所有的instance, 检查其面积
        for cls, area in instance_in_target_class:
            # 这里cls一定在self.categories中, 因此index不会抛出ValueError
            index = self.categories.index(cls)
            min_area, max_area = self.min_area[index], self.max_area[index]
            # ; 只要有一个面积不在对应的阈值范围内, 返回False
            if not (min_area <= area <= max_area):
                return False

        return True

    def _parse_by_python_read_xml(
        self, xml_label_filepath: str, *args: Any, **kwargs: Dict[Any, Any]
    ) -> List[Tuple[str, float]]:
        xml_root = ElementTree.parse(xml_label_filepath).getroot()

        targets_in_xml = [
            x
            for x in xml_root.findall("object")
            if x.find("name").text in self.categories
        ]

        targets_name_area: List[Tuple[str, float]] = []
        for each_target in targets_in_xml:
            bndbox = each_target.find("bndbox")
            name = each_target.find("name")
            xmin, xmax, ymin, ymax = (
                float(bndbox.find("xmin")),
                float(bndbox.find("xmax")),
                float(bndbox.find("ymin")),
                float(bndbox.find("ymax")),
            )

            area = (xmax - xmin) * (ymax - ymin)
            targets_name_area.append((name, area))

        return targets_name_area

    def _parse_by_python_read_txt(
        self, yolo_label_filepath: str, *args: Any, **kwargs: Dict[Any, Any]
    ) -> List[Tuple[str, float]]:
        # ; cls, xcn, ycn, wn, hn
        yolo_labels = np.loadtxt(yolo_label_filepath, ndmin=2)
        image_shape = kwargs.get("image_shape", None)
        assert (
            image_shape is not None
        ), f"$image_shape is required when passing a .txt yolo label file"
        image_height, image_width = image_shape[:2]
        instance_in_target_class: List[Tuple[int, float]] = []
        for cls_id, xcn, ycn, wn, hn in yolo_labels:
            if int(cls_id) in [int(x) for x in self.categories]:
                instance_in_target_class.append(
                    (cls_id, wn * hn * image_height * image_width)
                )
        return instance_in_target_class


class LabelLossFileFilter(LabelBaseFileFilter):
    def __init__(
        self,
        bbox_loss_threshold: float,
        bg_mask_threshold: int,
        drop_prob: float = 1.0,
        keep_prob: float = 1.0,
        reverse: bool = False,
        *args: Any,
        **kwargs: Dict[Any, Any],
    ) -> None:
        """
        根据pred和gt之间的损失进行filter.
        当传递的label是voc格式时, 需要同时传递一个name到idx的映射dict[str, int];
        当label是yolo]格式时, 则需要一个image shape的tuple[height, width]

        Args:
            bbox_loss_threshold (float): bbox loss的最大阈值. 当样本的bbox loss大于这个值时, 该样本会被过滤
            bg_mask_threshold (float): background mask的最大阈值, label assign时最多允许这个数量的pred被分配到background, 否则返回Flase.
            drop_prob (float): 如果不符合过滤条件, 则该样本有prob的概率会被丢弃
            reverse (bool, optional): 是否翻转过滤条件的结果. Defaults to False.
        """
        super().__init__(drop_prob, keep_prob, reverse, *args, **kwargs)
        self.bbox_loss_threshold = bbox_loss_threshold  # bbox loss的最大阈值
        self.bg_mask_threshold = bg_mask_threshold  # cls loss的最大阈值
        self.bg_id = -1  # background class id

    def _label_assign_IoUMax(
        self,
        pred_labels: np.ndarray[float],
        gt_labels: np.ndarray[float],
        match_iou: float = 0.5,
        *args: Any,
        **kwargs: Dict[Any, Any],
    ) -> np.ndarray[int]:
        """
        将pred labels和gt labels做匹配, 返回每个pred对应的gt索引, 以及总的误差

        Args:
            pred_labels (np.ndarray[float]): 预测的标签, (cls_id, xmin, xmax, ymin, ymax, ...) 后面可以跟score, 但是不会使用
            gt_labels (np.ndarray[float]): 实际标签, (cls_id, xmin, xmax, ymin, ymax)
            match_iou (float): 将pred和eval_config匹配时的iou阈值, 同一个gt目标, 从所有大于这个iou阈值的pred中, 选择iou最大(最接近gt框)的一个pred

        Returns:
            Tuple[np.ndarray[int]: _description_
        """
        # ; 标记每个pred是否已经被分配
        # ; pred_assigned[i] = j表示pred[i]和gt[j]对应
        # ; iou_assigned[i]表示pred[i]和gt[j]之间匹配的iou的值
        pred_assigned = (
            np.ones(shape=(pred_labels.shape[0],)) * self.bg_id
        ).astype(int)
        iou_assigned = np.zeros(shape=(pred_labels.shape[0],))
        # ; 遍历每个gt目标, 从pred中选择同clsid的pred
        for gt_index, each_gt in enumerate(gt_labels):
            cls_id = each_gt[0]
            self._log(
                f"current gt: {each_gt}",
                verbose=kwargs.get("verbose", False),
                logger=kwargs.get("logger", None),
                logger_func="debug",
            )
            pred_select_index = np.where(pred_labels[:, 0] == cls_id)[0]
            # ; 同一个类别的所有pred
            pred_in_same_cls = pred_labels[pred_select_index]
            self._log(
                f"select index: {pred_select_index}, same preds:\n{pred_in_same_cls}",
                verbose=kwargs.get("verbose", False),
                logger=kwargs.get("logger", None),
                logger_func="debug",
            )
            # ; gt与每个pred之间的iou
            gt_pred_iou = calc_iou_one2N(each_gt[1:], pred_in_same_cls[:, 1:])

            self._log(
                f"iou: {gt_pred_iou}",
                verbose=kwargs.get("verbose", False),
                logger=kwargs.get("logger", None),
                logger_func="debug",
            )

            # ; 找到iou最大, 且iou大于阈值, 且还没有被分配给任何其它gt的pred
            # ; 先对pred_in_same_cls, gt_pred_iou, pred_select_index进行从大大小排序
            # ; 这样排序后, pred_in_same_cls[j], gt_pred_iou[j], pred_select_index[j]这三个依然是同一个pred
            # ; 而pred_in_same_cls[j]对应的是pred_assigned[pred_select_index[j]]
            sorted_indices = np.argsort(gt_pred_iou)[::-1]
            pred_in_same_cls = pred_in_same_cls[sorted_indices]
            gt_pred_iou = gt_pred_iou[sorted_indices]
            pred_select_index = pred_select_index[sorted_indices]
            # ; 然后从头开始查找还没有被分配给任何其它gt的pred
            best_match = self.bg_id
            best_match_iou = 0
            j = 0
            while (best_match == self.bg_id) and (j < gt_pred_iou.shape[0]):
                # ; 现在的pred_in_same_cls[j], 对应的pred_assigned索引为pred_assigned[pred_select_index[j]]
                # ; 如果该iou已经达不到阈值要求, 那么后面的iou更加达不到要求
                if gt_pred_iou[j] < match_iou:
                    self._log(
                        f"best iou {gt_pred_iou[j]} smaller than threshold {match_iou}",
                        verbose=kwargs.get("verbose", False),
                        logger=kwargs.get("logger", None),
                        logger_func="debug",
                    )
                    break

                # ; 如果该pred已经被分配(值不为self.bg_id):
                # ;      如果被分配时的best_match_iou大于该pred和当前gt的iou, 说明该pred和另外的gt更匹配.
                # ;      否则说明该pred和当前gt更匹配
                if pred_assigned[pred_select_index[j]] != self.bg_id:
                    if iou_assigned[pred_select_index[j]] > gt_pred_iou[j]:
                        self._log(
                            f"pred {pred_select_index[j]} already assigned to gt {pred_assigned[pred_select_index[j]]}, \
                                with higher iou {pred_select_index[j]}, current iou {gt_pred_iou[j]}. Skipping",
                            verbose=kwargs.get("verbose", False),
                            logger=kwargs.get("logger", None),
                            logger_func="debug",
                        )
                        j += 1
                        continue
                    else:
                        self._log(
                            f"pred {pred_select_index[j]} already assigned to gt {pred_assigned[pred_select_index[j]]}, \
                                but current iou {gt_pred_iou[j]} is higher than previous assigned iou {pred_select_index[j]}. Re-assigning",
                            verbose=kwargs.get("verbose", False),
                            logger=kwargs.get("logger", None),
                            logger_func="debug",
                        )
                        best_match = pred_select_index[j]
                        best_match_iou = gt_pred_iou[j]
                        break

                # ; 此时该pred[j]符合assign要求
                best_match = pred_select_index[j]
                best_match_iou = gt_pred_iou[j]
                j += 1
                break

            # ; 将Pred[j]分配给gt[gt_index]
            if self.bg_id != best_match:
                pred_assigned[best_match] = gt_index
                iou_assigned[best_match] = best_match_iou
                self._log(
                    f"pred {best_match} assigned to gt {gt_index}, match iou {best_match_iou}\n\n",
                    verbose=kwargs.get("verbose", False),
                    logger=kwargs.get("logger", None),
                    logger_func="debug",
                )
            else:
                self._log(
                    f"no pred assigned to gt {gt_index} \n\n",
                    verbose=kwargs.get("verbose", False),
                    logger=kwargs.get("logger", None),
                    logger_func="debug",
                )

        return pred_assigned

    def _label_assign(
        self,
        pred_labels: np.ndarray[float],
        gt_labels: np.ndarray[float],
        assign_type: Optional[str] = "IoUMax",
        *args: Any,
        **kwargs: Dict[Any, Any],
    ) -> np.ndarray[int]:
        return self._label_assign_IoUMax(
            pred_labels, gt_labels, *args, **kwargs
        )

    def _calc_bbox_loss(
        self,
        pred_bboxes: np.ndarray[float],
        gt_bboxes: np.ndarray[float],
        pred2gt_indices: np.ndarray[int],
    ) -> float:
        # ; pred2gt_indices[i] = j表示pred[i]被分配到gt[j]
        fg_mask_pred = np.array([]).astype(int)
        fg_mask_gt = np.array([]).astype(int)

        for i in range(pred2gt_indices.shape[0]):
            j = pred2gt_indices[i]
            if j == self.bg_id:
                continue
            fg_mask_pred = np.concatenate([fg_mask_pred, [i]], axis=0)
            fg_mask_gt = np.concatenate([fg_mask_gt, [j]], axis=0)

        iou = calc_iou_one2one(pred_bboxes[fg_mask_pred], gt_bboxes[fg_mask_gt])
        return (1 - iou).mean()

    def _filter(
        self, filepath: str, *args: Any, **kwargs: Dict[Any, Any]
    ) -> bool:
        self._log(
            f"called {self}",
            verbose=kwargs.get("verbose", False),
            logger=kwargs.get("logger", None),
            logger_func="debug",
        )

        pred_label_filepath: str = kwargs.get("pred_label_filepath", None)
        gt_label_filepath: str = kwargs.get("gt_label_filepath", None)
        assert os.path.isfile(pred_label_filepath) and os.path.isfile(
            gt_label_filepath
        )

        pred_labels = self._parse_label_file(
            pred_label_filepath, *args, **kwargs
        )
        gt_labels = self._parse_label_file(gt_label_filepath, *args, **kwargs)

        pred2gt_indices: np.ndarray[int] = self._label_assign(
            pred_labels, gt_labels, *args, **kwargs
        )

        # ; 统计有多少pred被分配到background
        bg_assigned = (pred2gt_indices == self.bg_id).sum()
        self._log(
            f"{bg_assigned} pred assigned to background, {len(pred2gt_indices)-bg_assigned} to foreground",
            verbose=kwargs.get("verbose", False),
            logger=kwargs.get("logger", None),
            logger_func="debug",
        )
        bbox_loss = self._calc_bbox_loss(
            pred_labels[:, 1:], gt_labels[:, 1:], pred2gt_indices
        )
        self._log(
            f"bbox loss {bbox_loss}",
            verbose=kwargs.get("verbose", False),
            logger=kwargs.get("logger", None),
            logger_func="debug",
        )
        return (bg_assigned <= self.bg_mask_threshold) and (
            bbox_loss <= self.bbox_loss_threshold
        )

    def _parse_label_file(
        self, label_filepath: str, *args: Any, **kwargs: Dict[Any, Any]
    ) -> np.ndarray[float]:
        # 根据label_filepath的后缀格式对齐标签格式
        ann_type = label_filepath.rsplit(".", 1)[1].lower()
        if ann_type == "xml":
            # ; xml(voc)标签, cls_name, xmin, ymin, xmax, ymax
            return self._parse_label_file_xml(
                label_filepath, name2idx_dict=kwargs.get("name2idx_dict", None)
            )
        elif ann_type == "txt":
            # ; txt(yolo)标签, cls_id, xn, yn, hn, wn
            return self._parse_label_file_yolo(
                label_filepath, image_shape=kwargs.get("image_shape", None)
            )
        else:
            raise ValueError(
                f"non support label format, expected to be one of [xml(voc), txt(yolo)], but detected {ann_type}"
            )

    def _parse_label_file_yolo(
        self,
        yolo_label_filepath: str,
        image_shape: Tuple[int],
        *args: Any,
        **kwargs: Dict[Any, Any],
    ) -> np.ndarray[float]:
        assert (
            image_shape is not None
        ), f"another tuple of image shape (height, width) is required when passing a .txt label file"
        # ; idx, xcn, ncn, wn, hn, ...
        # ; (后面可能带有score等信息, 先不考虑, 所以只取前5列)
        yolo_labels = np.loadtxt(yolo_label_filepath, ndmin=2).astype(
            np.float32
        )[:, :5]

        # ; 不包含任何标签
        if not yolo_labels:
            return np.array([]).reshape((-1, 5))

        # ; idx, xcn, ncn, wn, hn, ...
        cls = yolo_labels[:, 0]
        bbox_xcnycnwnhn = yolo_labels[:, 1:5]
        bbox_xyxy = bbox_xnynwnhn2xyxy(
            bboxes=bbox_xcnycnwnhn, image_shape=image_shape
        )
        return np.concatenate([cls, bbox_xyxy], axis=1).astype(np.float32)

    def _parse_label_file_xml(
        self,
        xml_label_filepath: str,
        name2idx_dict: Dict[str, int],
        *args: Any,
        **kwargs: Dict[Any, Any],
    ) -> np.ndarray[float]:
        # ; 需要一个从类名到类别id的dict来作映射
        assert (
            name2idx_dict is not None
        ), f"another dict of classname to classId is required when passing a .xml label file"
        xml_root = ElementTree.parse(xml_label_filepath).getroot()

        # xml中包含的所有target
        targets_in_xml = [x for x in xml_root.findall("object")]

        parse_labels = []
        for each_target in targets_in_xml:
            # 遍历每个target, 获取类名和box
            bndbox = each_target.find("bndbox")
            name = each_target.find("name").text
            cls_id = name2idx_dict.get(name, -1)
            if -1 == cls_id:
                # ; 该类别不在指定的dict中, 跳过
                self._log(
                    f"class name={name} not in map dict={name2idx_dict.keys()}, skipping.",
                    verbose=kwargs.get("verbose", False),
                    logger=kwargs.get("logger", None),
                    logger_func="error",
                )
                continue
            xmin, xmax, ymin, ymax = (
                float(bndbox.find("xmin").text),
                float(bndbox.find("xmax").text),
                float(bndbox.find("ymin").text),
                float(bndbox.find("ymax").text),
            )

            # 加入查询结果
            parse_labels.append([cls_id, xmin, ymin, xmax, ymax])

        return np.array(parse_labels).astype(np.float32)
