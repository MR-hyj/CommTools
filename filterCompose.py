from __future__ import annotations

import os
from abc import abstractmethod, ABC
from typing import Any, Union, List, Dict


from .filters import BaseFileFilter


class BaseFilterCompose(ABC):

    def __init__(
        self,
        filters: List[Union[BaseFileFilter, BaseFilterCompose]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """将多个file_filter按逻辑条件合并成一个, 这是FileCompose的基类, 不应该被实例化

        Args:
            filters (List[__BaseFileFilter]): 需要合并条件的FileFilters
        """
        super().__init__()
        self.file_filters = filters
        self._reset_final_res()

    def __call__(self, *args: Any, **kwargs: Any) -> bool:
        return self._compose_file_filters(*args, **kwargs)

    def _compose_file_filters(self, *args: Any, **kwargs: Any) -> bool:
        self._reset_final_res()
        for each_filter in self:
            each_res = each_filter(*args, **kwargs)
            self._update_final_res(each_res)
        return self._final_res

    @property
    def summary_dict(self) -> Dict[str, Any]:
        return self._summary_dict()

    def _summary_dict(self) -> Dict[str, Any]:
        attr_dict = {"name": self.__class__.__name__}
        attr_dict["filters"] = [
            file_filter.summary_dict for file_filter in self
        ]
        return attr_dict

    def __iter__(
        self,
    ):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.file_filters):
            result = self.file_filters[self.index]
            self.index += 1
            return result

        raise StopIteration

    def __repr__(self) -> str:
        return str(self.summary_dict)

    @abstractmethod
    def _reset_final_res(self):
        pass

    @abstractmethod
    def _update_final_res(self, res_tmp: bool):
        pass


class FilterComposeOr(BaseFilterCompose):

    def __init__(
        self,
        filters: List[Union[BaseFileFilter, BaseFilterCompose]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """将多个file_filter按Or合并, 只要有一个file filter的结果为True, 总的结果为True

        Args:
            filters (List[__BaseFileFilter]): 需要合并条件的FileFilters
        """
        super().__init__(filters, *args, **kwargs)

    def _reset_final_res(self):
        self._final_res = False

    def _update_final_res(self, res_tmp: bool):
        self._final_res = self._final_res or res_tmp


class FilterComposeAnd(BaseFilterCompose):

    def __init__(
        self,
        filters: List[Union[BaseFileFilter, BaseFilterCompose]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """将多个file_filter按And合并, 只要有一个file filter的结果为False, 总的结果为False

        Args:
            filters (List[__BaseFileFilter]): 需要合并条件的FileFilters
        """
        super().__init__(filters, *args, **kwargs)

    def _reset_final_res(self):
        self._final_res = True

    def _update_final_res(self, res_tmp: bool):
        self._final_res = self._final_res and res_tmp


if __name__ == "__main__":

    pred_dir = "D:\\project\\labels\\PreLabel\\patches_othersize\\patch_SanYuePeiDian\\xml_orig_v4.1.0"
    files = os.listdir(pred_dir)
    name2idx_dict = {
        "JYZ_ZC": 0,
        "GLKG": 1,
        "ZSKG": 2,
        "TM": 3,
        "BYQ": 4,
        "DLSRDQ": 5,
        "GANTA": 6,
        "GANTOU": 7,
        "TATOU": 8,
        "NX": 9,
        "JYZ_XS": 10,
        "GHP": 11,
        "SMCG": 12,
        "BLQ": 13,
        "DAOXIAN": 14,
    }
    from .filters import LabelLossFileFilter

    for file in files:
        pred_xml_file = os.path.join(pred_dir, file)
        gt_xml_file = os.path.join(
            "D:\\project\\labels\\PreLabel\\patches_othersize\\patch_SanYuePeiDian\\xml_orig_renamed_v4.1.0",
            file,
        )

        filefiler = LabelLossFileFilter(
            bbox_loss_threshold=1.0, bg_mask_threshold=5
        )

        res = filefiler(
            pred_label_filepath=pred_xml_file,
            gt_label_filepath=gt_xml_file,
            name2idx_dict=name2idx_dict,
            match_iou=0.5,
            verbose=True,
        )
        print(res)
