"""
将目录下的所有json文件转换成yolo格式的txt数据集
可以投通过设置labels_only来仅生成标注文件, 来查看最终的类别信息

"""

import os
import shutil
import sys
import json
import yaml
from typing import Union

import numpy as np
import cv2 as cv
from typing import AnyStr, Dict

from utils import ALLOWED_IMAGE_FORMATS
from utils import logger, parse_json2yolo_args, check_path, remove_last_sep_from_dir
from utils import Trigger, Timer

args = parse_json2yolo_args()
logger.setLevel(args.log_level.upper())
logger.info(f"cmd: python {' '.join(sys.argv)}")
logger.info(f"all args: {args.__dict__}")

json_anno_dir   = check_path(remove_last_sep_from_dir(args.json_anno_dir))
orig_img_dir    = check_path(remove_last_sep_from_dir(args.orig_img_dir))
yolo_dir        = args.yolo_dir
yolo_anno_dir   = os.path.join(yolo_dir, 'labels')
yolo_img_dir    = os.path.join(yolo_dir, 'images')
img_fmt         = args.img_fmt
start_from_zero = args.start_from_zero
labels_only     = args.labels_only
val_size        = args.val_size
cls_names_file  = args.cls_names_file
copy_new_xml    = args.copy_new_xml
allow_new_cls   = args.allow_new_cls
keep_ratio      = args.keep_ratio
patch_name      = args.patch_name
if args.fixed_resolution is not None:
    yolo_img_height, yolo_img_width = args.fixed_resolution
    yolo_img_height, yolo_img_width = int(yolo_img_height), int(yolo_img_width)
else:
    yolo_img_height, yolo_img_width = None, None

logger.info(f'converted yolo annotations will be saved to {yolo_anno_dir}')

if not os.path.exists(yolo_anno_dir):
    logger.info(f'creating yolo anno directory: {yolo_anno_dir}')
    os.makedirs(yolo_anno_dir)

if yolo_img_dir is None:
    yolo_img_dir = os.path.join(os.path.dirname(yolo_anno_dir), 'images/')

logger.info(f'images whose json annotation are successfully converted to yolo format will be saved to {yolo_img_dir}')

if not os.path.exists(yolo_img_dir):
    logger.info(f'creating yolo image directory: {yolo_img_dir}')
    os.makedirs(yolo_img_dir)

if val_size != -1:
    assert 0 <= val_size < 1
    logger.info(f'train : val = [{1 - val_size} : {val_size}]')
    [os.makedirs(os.path.join(x, y), exist_ok=True) for y in ['train', 'val'] for x in [yolo_img_dir, yolo_anno_dir]]

if cls_names_file is not None:
    glb_cls_names: Dict[AnyStr, AnyStr] = yaml.safe_load(open(cls_names_file, 'r'))
    glb_cls_names = dict(map(reversed, glb_cls_names.items()))
    logger.info(f'loaded class names from {cls_names_file}: \n{glb_cls_names}')
else:
    # 记录所有的分类id   glb_cls_names[cls] = id, cls大写, id为str, 从0开始
    glb_cls_names: Dict[AnyStr, AnyStr] = {}

json_anno_files = [x for x in os.listdir(json_anno_dir) if os.path.isfile(os.path.join(json_anno_dir, x))]

# cnt

# yolo保存的文件名与json文件名之间的映射
# 只有当args.start_from_zero=true时才保存该映射关系
yolo_idx2json_file = {}

root_trigger = Trigger(__file__)
root_timer = Timer(__file__)


def resize_image(
        image_src: np.ndarray, 
        dst_size: Union[tuple, list], 
    ):
    """将图像缩放, 根据超参数, 是否等比例缩放

    Args:
        image_src (np.ndarray): 被缩放的图像
        dst_size (Union[tuple, list]): 目标尺寸

    Returns:
        np.ndarray  被缩放的图像
        tuple       如果是等比例缩放, 同时返回上下左右边界
    """
    global keep_ratio
    if not keep_ratio:
        logger.debug('not use keep ratio resizing')
        return cv.resize(image_src, dsize=dst_size), None
    
    logger.debug('use keep ratio resizing')
    src_h, src_w = image_src.shape[:2]
    dst_h, dst_w = dst_size

    # 判断应该按哪个边做等比缩放
    h = dst_w * (float(src_h) / src_w)  # 按照w做等比缩放
    w = dst_h * (float(src_w) / src_h)  # 按照h做等比缩放

    h = int(h)
    w = int(w)

    if h <= dst_h:
        image_dst = cv.resize(image_src, (dst_w, h))
    else:
        image_dst = cv.resize(image_src, (w, dst_h))

    h_, w_ = image_dst.shape[:2]

    top = int((dst_h - h_) / 2)
    bottom = int((dst_h - h_ + 1) / 2)
    left = int((dst_w - w_) / 2)
    right = int((dst_w - w_ + 1) / 2)

    value = [0, 0, 0]
    borderType = cv.BORDER_CONSTANT
    image_dst = cv.copyMakeBorder(image_dst, top, bottom, left, right, borderType, None, value)

    return image_dst, (top, bottom, left, right)


def transform_keep_ratio_labels(
        yolo_label: np.ndarray, 
        resized_img_shape: tuple, 
        padding_boundary: tuple
    ):
    """图像经过等比例缩放后, yolo格式的标签需要做相应变换

    Args:
        yolo_label (np.ndarray): 原始yolo标签
        resized_img_shape (tuple): 等比例缩放时的图像目标尺寸
        padding_boundary (tuple): 等比例缩放时的上下左右边界

    Returns:
        np.ndarray 等比例缩放的对应yolo标签
    """
    assert yolo_label.ndim == 2
    logger.debug('transform keep ratio labels')
    cls = yolo_label[:, 0]
    xcn, ycn = yolo_label[:, 1], yolo_label[:, 2]
    wn, hn   = yolo_label[:, 3], yolo_label[:, 4]
    img_h, img_w = resized_img_shape
    top, bottom, left, right = padding_boundary

    # resize后的中心点坐标
    xc_resized = (img_w - left - right) * xcn + left
    yc_resized = (img_h - top - bottom) * ycn + top
    
    # resize后box的w, h
    w_resized = (img_w - left - right) * wn
    h_resized = (img_h - top - bottom) * hn

    # 归一化
    xcn_resized = xc_resized / img_w
    ycn_resized = yc_resized / img_h
    wn_resized = w_resized / img_w
    hn_resized = h_resized / img_h

    return np.vstack([cls, xcn_resized, ycn_resized, wn_resized, hn_resized]).T


def find_corr_image_path(filename: str):
    """根据标签文件名寻找对应的图像文件名

    Args:
        filename (str): 标签文件名(不带后缀和目录路径)

    Returns:
        str 对应的图像路径
    """
    global ALLOWED_IMAGE_FORMATS, orig_img_dir
    return next(
        (
            os.path.join(orig_img_dir, f'{filename}.{fmt}') for fmt in ALLOWED_IMAGE_FORMATS 
            if os.path.isfile(os.path.join(orig_img_dir, f'{filename}.{fmt}'))
        ), None
    )

def get_yolo_image_label_save_path(
    yolo_anno_filename:str, 
    yolo_img_filename:str
    ):
    """根据一系列超参数, 生成对应的yolo数据集的保存路径

    Args:
        yolo_anno_filename (str): _description_
        yolo_img_filename (str): _description_

    Returns:
        _type_: _description_
    """
    global patch_name, yolo_anno_dir, yolo_img_dir, labels_only

    if patch_name is not None:
        yolo_anno_filename_ = f'patch_{patch_name}_{yolo_anno_filename}'
        yolo_img_filename_  = f'patch_{patch_name}_{yolo_img_filename}'
    else:
        yolo_anno_filename_ = yolo_anno_filename
        yolo_img_filename_  = yolo_img_filename

    if -1 == val_size:
        # 不划分train, val
        yolo_anno_save_path = os.path.join(yolo_anno_dir, yolo_anno_filename_)
        yolo_img_save_path  = None if labels_only else os.path.join(yolo_img_dir, yolo_img_filename_)
    elif np.random.rand() < val_size:
        # 划分给val
        yolo_anno_save_path = os.path.join(yolo_anno_dir, 'val', yolo_anno_filename_)
        yolo_img_save_path  = None if labels_only else os.path.join(yolo_img_dir, 'val', yolo_img_filename_)
    else:
        # 划分给train
        yolo_anno_save_path = os.path.join(yolo_anno_dir, 'train', yolo_anno_filename_)
        yolo_img_save_path  = None if labels_only else os.path.join(yolo_img_dir, 'train', yolo_img_filename_)

    return yolo_anno_save_path, yolo_img_save_path


def parse_instance_info_json2yolo(
        json_anno_file: str, 
        targets: list[dict], 
        img_height:int, 
        img_width: int
    ):
    # 保存该文件下所有的目标的标签, 由于yolo标签全部是数字,
    # 因此可以使用np的二维矩阵来保存
    # yolo的txt标注格式
    # id, xc, yc, w, h
    yolo_anno_content_all_targets = []
    logger.info(
        f"{len(targets)} targets found: "
        f"{[[x['name'], x['bndbox']['xmin'], x['bndbox']['ymin'], x['bndbox']['xmax'], x['bndbox']['ymax']] for x in targets]}")

    for idx_target, target in enumerate(targets):
        target_cls_name = target['name'].upper()

        # 从历史记录中获取当前分类的id
        global glb_cls_names, allow_new_cls, root_trigger
        if glb_cls_names.get(target_cls_name, None) is not None:
            # 已经记录的分类里面包括这个类
            target_yolo_idx = glb_cls_names[target_cls_name]
            logger.debug(f'found existing key={target_cls_name}, value={target_yolo_idx}')
        else:
            # 当前类不在历史记录中
            # 根据参数选择是否允许新类别加入数据集
            if cls_names_file is not None:
                # 不允许新类别加入, 直接continue
                if not allow_new_cls:
                    logger.warn(f'new class name={target_cls_name} encountered. skipping.')
                    continue
            
            # 允许新类别加入
            logger.warn(
                f'New class name={target_cls_name} encountered.'
                f'The input class name file will be updated.'
            )
            cnt_glb_cls_names = len(glb_cls_names.keys())  # 目前有多少个分类
            target_yolo_idx = str(cnt_glb_cls_names)
            glb_cls_names[target_cls_name] = target_yolo_idx
            
            logger.debug(f'update key={target_cls_name}, value={target_yolo_idx}')
        
        # xc, yc
        target_xmin, target_xmax = float(target['bndbox']['xmin']), float(target['bndbox']['xmax'])
        target_ymin, target_ymax = float(target['bndbox']['ymin']), float(target['bndbox']['ymax'])

        # 过滤不合法的标签
        if (np.any(np.array([target_xmin, target_xmax]) > img_width)) \
            or (np.any(np.array([target_ymin, target_ymax]) > img_height)):
            logger.warning(
                f"corrputed label detected in {json_anno_file}, args: {dict(image_shape=(img_height, img_width), object_bbox=target['bndbox'])}"
            )
            root_trigger.trigger_corrupt(json_anno_file)
            continue

        target_yolo_xc = (target_xmin + target_xmax) / 2 / img_width
        target_yolo_yc = (target_ymin + target_ymax) / 2 / img_height

        # width, height
        target_yolo_width = (target_xmax - target_xmin) / img_width
        target_yolo_height = (target_ymax - target_ymin) / img_height

        # yolo的txt标注格式
        # id, xc, yc, w, h
        yolo_anno_content_all_targets.append(
            [target_yolo_idx, target_yolo_xc, target_yolo_yc, target_yolo_width, target_yolo_height]
        )
    
    return np.array(yolo_anno_content_all_targets, dtype=np.float32).reshape(-1, 5)


def json2yolo_singlefile(idx_json_file:int, json_anno_file: str):
    if start_from_zero:
        yolo_idx2json_file[str(idx_json_file)] = json_anno_file

    filename = json_anno_file.rsplit('.', 1)[0]  # 去除.json后的文件名
    # 保存文件名是否从0开始，避免中文
    if start_from_zero:
        yolo_anno_filename = f'{idx_json_file}.txt'
        yolo_img_filename = f'{idx_json_file}.{img_fmt}'
    else:
        yolo_anno_filename = f'{filename}.txt'
        yolo_img_filename = f'{filename}.{img_fmt}'


    anno_data = json.load(open(os.path.join(json_anno_dir, json_anno_file), 'r'))
    if anno_data['annotation'].get('object', None) is None:
        logger.warn(f'file {json_anno_file} contains no objects')
        return True
    targets = anno_data['annotation']['object']

    img_width, img_height = float(anno_data['annotation']['size']['width']), \
        float(anno_data['annotation']['size']['height'])

    corr_img_path = find_corr_image_path(filename=filename)
    if corr_img_path is None:
        logger.error(f'found no corresponding image {corr_img_path} for annotation {json_anno_file}')
        return False

    # 如果有一个为0则这个标注失效
    # 提前读取对应的图像来获取宽高
    # 当labels_only=True时, cv.imread这一步需要耗费很多时间,
    if (0 == img_width*img_height) or (yolo_img_height is not None):
        corr_img = cv.imread(corr_img_path)
        img_height, img_width = corr_img.shape[:2]
    else:
        corr_img = None
    
    yolo_anno_content_all_targets = parse_instance_info_json2yolo(
        json_anno_file=json_anno_file, 
        targets=targets, 
        img_width=img_width, 
        img_height=img_height
    )

    if 0 == len(yolo_anno_content_all_targets):
        logger.info(f'no candidate targets in {json_anno_file}, skipping.')
        return True
    

    yolo_anno_save_path, yolo_img_save_path = get_yolo_image_label_save_path(
        yolo_anno_filename=yolo_anno_filename, yolo_img_filename=yolo_img_filename
    )

    return save_yolo_image_label(
        yolo_anno_save_path=yolo_anno_save_path, 
        yolo_img_save_path=yolo_img_save_path, 
        corr_img=corr_img, 
        yolo_anno=yolo_anno_content_all_targets
    )


def save_yolo_image_label(
        yolo_img_save_path: str, 
        yolo_anno_save_path: str, 
        corr_img: np.ndarray, 
        yolo_anno: np.ndarray
    ):
    # 将成功转换的图像保存到对应目录中
    if not labels_only:
        try:
            if yolo_img_width is not None:
                # 需要统一分辨率， 必须读取图像
                corr_img, padding_bound = resize_image(image_src=corr_img, dst_size=(yolo_img_height, yolo_img_width))
            logger.info(f'saving corresponding image  in {yolo_img_save_path}')
            cv.imwrite(yolo_img_save_path, corr_img)
        except Exception as e:
            logger.error(f'error when saving corresponding image to {yolo_img_save_path}, {e}')
            return False
    

    global keep_ratio
    if keep_ratio:
        yolo_anno = transform_keep_ratio_labels(
            yolo_anno, resized_img_shape=corr_img.shape[:2], padding_boundary=padding_bound
        )
    
    logger.info(f'saving annotation file in {yolo_anno_save_path}')
    try:
        np.savetxt(yolo_anno_save_path, yolo_anno, fmt='%.8f')
    except Exception as e:
        logger.error(f'error when saving yolo annotation file {yolo_anno_save_path}, {e}')
        return False

    return True


def json2yolo():

    global root_trigger, root_timer
    for idx_json_file, json_anno_file in enumerate(json_anno_files):

        logger.info(f'[{idx_json_file+1} / {len(json_anno_files)}] current json file {json_anno_file}, '
                    f'succeed={root_trigger.cnt_succeed}, fail={root_trigger.cnt_fail}, corrupted={root_trigger.cnt_corrupt}')

        root_timer.clock_start()
        ret = json2yolo_singlefile(idx_json_file, json_anno_file)
        root_timer.clock_end() 
        
        if ret:
            root_trigger.trigger_succeed()
        else:
            root_trigger.trigger_fail(failed_item=json_anno_file)

        remaining_files = len(json_anno_files) - (idx_json_file+1)      # 剩余文件数
        root_timer.clock_summary(
            cnt_remaining=remaining_files, 
            logger=logger
        )
        
    # 保存类别信息
    glb_cls_names_save_path = os.path.join(os.path.dirname(yolo_anno_dir), 'cls_names.yaml')
    logger.info(f'saving global cls name dict in {glb_cls_names_save_path}')
    yaml.dump(
        dict(
            map(reversed, glb_cls_names.items())
        ), 
        open(glb_cls_names_save_path, 'w'),
        indent=2, sort_keys=False, allow_unicode=True
    )

    # 如果是按id保存的, 同时保存id到原文件名的映射
    if start_from_zero:
        yolo_idx2json_file_save_path = os.path.join(os.path.dirname(yolo_anno_dir), 'yolo_idx2json_file.json')
        logger.info(f'saving yolo idx to json filename in {yolo_idx2json_file_save_path}')
        json.dump(yolo_idx2json_file, open(yolo_idx2json_file_save_path, 'w'), indent=2)

    if copy_new_xml is not None:
        copy_dir = os.path.join(os.path.dirname(yolo_anno_dir), 'copy_xml/')
        logger.info(f'coping new xml annotations into {copy_dir}')
        try:
            shutil.copytree(copy_new_xml, copy_dir)
        except Exception as e:
            logger.error(f'error when coping new xml annotations: {e}')

    root_trigger.summary(logger=logger)


if __name__ == '__main__':

    json2yolo()
