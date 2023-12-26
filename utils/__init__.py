from .loggerSingleton import LoggerSingleton
from .tools import *
from .argparser import parse_xml2json_args, parse_seg_classes_from_json_args, \
    parse_json2yolo_args, parse_delete_target_classes_from_xml_args, parse_delete_non_correspondence_args, \
    parse_override_orig_anno_args, parse_seg_all_ins_from_yolo_args, parse_merge_json_args, \
    parse_yolotxt2cocojson_args, parse_eval_yolo_pred_coco_json_args,  parse_coco2yolo_args, \
    parse_augment_from_yolo_args, parse_draw_labels_from_yolo_args, parse_json2xml_args, parse_move_target_files_args, \
    parse_yolo2xml_args
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
    'parse_seg_classes_from_json_args', 'parse_xml2json_args', 'parse_json2yolo_args', 
    'parse_delete_target_classes_from_xml_args', 'parse_delete_non_correspondence_args', 
    'parse_override_orig_anno_args', 'parse_seg_all_ins_from_yolo_args', 
    'parse_merge_json_args', 'parse_yolotxt2cocojson_args', 'parse_yolo2xml_args', 
    'parse_eval_yolo_pred_coco_json_args', 'parse_coco2yolo_args', 'parse_augment_from_yolo_args', 
    'parse_draw_labels_from_yolo_args', 'parse_json2xml_args', 'parse_move_target_files_args'
]
