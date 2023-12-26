"""
对指定的yolo数据集做数据增强
"""

import os
import sys
import yaml
import subprocess

import cv2 as cv
import numpy as np
import albumentations as A

from utils import logger, parse_augment_from_yolo_args, check_path, EXIST_CODE, ALLOWED_IMAGE_FORMATS, UNIX_PLATFORMS


#! 当参数列表长度超过这个值, 某些命令会出错, 
#! 而且不一定能被python捕捉到这个错误
#! 这时候需要避免直接匹配目录下的所有文件
UNIX_CMD_ARGS_MAXIMUM_LENGTH = 30000

#; 已有的类别数量至少达到这个数值才可以进行增强
#; 为了确保样本分布不至于太过单一
CNT_MINIMUM_INSTANCE_TO_AUGMENT = 100


args = parse_augment_from_yolo_args()
logger.setLevel(args.log_level.upper())
logger.info(f"cmd: python {' '.join(sys.argv)}")
logger.info(f"all args: {args.__dict__}")

# 需要增强的类别id
target_classes_id = [str(x) for x in args.target_classes_id]
# 原始图像的文件夹
yolo_images_dir   = check_path(args.yolo_images_dir)
# 于原始图像对应的标签所在的文件夹, 要求必须是绝对路径
yolo_labels_dir   = check_path(args.yolo_labels_dir , require_abs=True)
# yolo数据集的类别信息
# cls_id: name
# '0': Bird
# '1': Cat
cls_id_name_dict  = yaml.safe_load(open(check_path(args.yolo_names_filepath), 'r'))
# 每一类最大样本数量, 达到这个样本数则停止增强
augment_limit     = args.augment_limit
# 增强后的数据集的保存根路径       
augment_savedir   = args.augment_root_savedir
# 增强后的数据集的版本名称
augment_version   = args.augment_version
# 文件名中的不同含义部分之间的分隔符
filename_splitor  = args.splitor
# 是否在非unix平台上强制执行
force_on_non_unix = args.force_on_non_unix

# 每一类的最大样本数量
if len(augment_limit) == 1:
    augment_limit = [augment_limit[0]] * len(target_classes_id)


if augment_savedir is None:
    augment_savedir = os.path.join(os.path.dirname(yolo_images_dir), 'augment')

if augment_version is not None:
    augment_savedir = os.path.join(augment_savedir, f'augment_{augment_version}')

augment_images_savedir  = os.path.join(augment_savedir, 'images')  # 增强后的图像被保存的目录
augment_labels_savedir  = os.path.join(augment_savedir, 'labels')  # 增强后的标签被保存的目录


os.makedirs(augment_images_savedir, exist_ok=True)
os.makedirs(augment_labels_savedir, exist_ok=True)


# 增强使用的变换
transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.5), 
        A.Rotate(p=0.5), 
        # A.GaussNoise(p=0.5), 
        A.CLAHE(p=0.5), 
        A.FancyPCA(p=0.5), 
    ], 
    bbox_params=A.BboxParams(
        format='pascal_voc', label_fields=['bbox_classes']
    )
)

logger.info(f'transforms: {[str(x) for x in transforms]}')


def run_cmd_on_UNIX_like(cmd: str):
    """在unix平台上执行cmd命令

    Args:
        cmd (str): _description_

    Returns:
        str: 执行命令的返回结果, 去除最后一个换行符
    """
    global logger
    # 所有命令输出在最后都会有一个\n换行符, rsplit('\n', 1)[0]把这个换行符去掉
    try:
        return subprocess.check_output(cmd, shell=True).decode('utf-8').rsplit(os.linesep, 1)[0]
    except Exception as e:
        logger.critical(f'failed to excute cmd={cmd}, exception={e}')
        sys.exit(EXIST_CODE.SHELL_RUNTIME_ERROR)


def run_cmd_on_nonUNIX_like(cmd: str):
    """在非unix平台上执行cmd命令

    Args:
        cmd (str): _description_

    Returns:
        str: 执行命令的返回结果, 去除最后一个换行符
    """
    # 非Unix-like系统需要指定--force参数
    global force_on_non_unix, logger
    assert force_on_non_unix, logger.error(
        'This script ONLY support Unix-like platforms, cause commands like `grep`, `cat` are used here.\n'
        '**But if you want to run anywise, try with --force flag**'
    )

    # 非Unix-like系统需要在环境变量中存在一个$bash变量,
    # 指向一个可运行bash, 比如git-bash.exe
    bash = os.environ.get('bash')
    assert bash is not None, logger.critical(
        'a $bash variable is need in $PATH for this script.'
    )

    # 非Unix-like平台运行可能出错, 比如不存在cmd中包括的命令
    try:
        # 所有命令输出在最后都会有一个\n换行符, rsplit('\n', 1)[0]把这个换行符去掉
        return subprocess.check_output(['bash', '-c', cmd]).decode('utf-8').rsplit(os.linesep, 1)[0]
    except Exception as e:
        logger.critical(f'failed to excute cmd={cmd} with bash={bash}, exception={e}')
        sys.exit(EXIST_CODE.SHELL_RUNTIME_ERROR)


def run_cmd(
        cmd: str, 
        alter_cmd: str = None, 
        arg_len_check_cmd: str = None
    )-> str:
    """
    使用Unix-like的bash运行cmd命令. 
    如果当前系统非Unix-like, 
    可以通过指定--force参数来跳过平台检查, 
    但是需要在环境变量PATH中添加一个名为bash的变量, 
    并指向一个bash.exe, 比如git-bash.exe

    Args:
        cmd (str): 需要运行的cmd命令
        alter_cmd (str): 当cmd的参数列表过长时, 调用这个命令
        arg_len_check_cmd: 获取cmd命令中参数长度

    Returns:
        str: 运行的结果字符串
    """
    import platform
    global UNIX_PLATFORMS, UNIX_CMD_ARGS_MAXIMUM_LENGTH, logger
    
    plat = platform.system().upper() 
    logger.info(f'running cmd: {cmd}')
    # Unix-like系统可以直接运行cmd命令
    if plat in UNIX_PLATFORMS:
        cmd_call_func = run_cmd_on_UNIX_like
    else:
        cmd_call_func = run_cmd_on_nonUNIX_like
    
    
    # 不进行参数长度检查
    if arg_len_check_cmd is  None:
        return cmd_call_func(cmd=cmd)
    
    # 检查参数长度
    arg_len = int(cmd_call_func(cmd=arg_len_check_cmd))
    logger.debug(f'arg len={arg_len} for cmd={cmd}')
    if arg_len <= UNIX_CMD_ARGS_MAXIMUM_LENGTH:
        return cmd_call_func(cmd=cmd)
    
    assert alter_cmd is not None, logger.critical(f'args too long for cmd={cmd}, alternative cmd expected but got {alter_cmd}')
    logger.info(f'args too long for cmd={cmd}, trying with alternative cmd={alter_cmd}, this may take quite a while, please wait...')
    return cmd_call_func(cmd=alter_cmd)


def augment_single_class(
        cls_id: str, 
        cls_augment_limit: int
    ):
    """
    给指定的类别做增强, 直到数量达到增强上限

    Args:
        cls_id (str): 类别id(字符串)
        cls_augment_limit (int): 该类别的增强上限
    """
    global yolo_images_dir, yolo_labels_dir, cls_id_name_dict, CNT_MINIMUM_INSTANCE_TO_AUGMENT

    #; 先查询当前数据集中, 包含了多少个cls_id的gt实例
    #; 使用Python一个个读取txt文件太慢, 因此使用shell查询

    # 包含cls_id示例的标注文件
    # yolo标注格式, 每一行都是cls_id开头
    # 因此用^匹配每一行开头, 获取包含cls_id开头的注释的文件的绝对路径
    cmd = f'grep -l -E "^{cls_id}(\.[0]{{1,}}[e+[0]{{1,}}]{{0,}}\s|\s)" {os.path.join(yolo_labels_dir, "*.txt")}'

    #! 上面的命令当txt文件过多时会超出grep可接受的最大参数列表长度, 这个时候启用下面的命令
    #! 原理是find将逐个处理文件, 类似一个for循环, 而上面的命令会把所有的文件路径一次性传递给grep命令 
    #! 但是由于find是逐个文件处理的, 因此会非常费时, 所以将-exec \; 处替换为 -exec + 一次性传递多个文件, 提高find的效率

    #; ^$i(\.[0]{1,}[e+[0]{1,}]{0,}\s|\s)可以匹配的模式:
    #;  i 
    #;  i.0000(若干个0)
    #;  i.0000(若干个0)e+00000(若干个0)
    alter_cmd = f'find {yolo_labels_dir} -maxdepth 1 -type f -name "*.txt" -exec grep -l -E "^{cls_id}(\.[0]{{1,}}[e+[0]{{1,}}]{{0,}}\s|\s)" {{}} +'

    # ;这里每个file已经是绝对路径
    target_files = run_cmd(
        cmd=cmd, alter_cmd=alter_cmd, 
        arg_len_check_cmd=f'ls {yolo_labels_dir} | grep ".txt" -c'
    ).split('\n')

    # 统计当前cls_id的实例数量
    
    #! 下面这个命令虽然快速, 但是cat遇到参数列表过长的情况不会抛出error
    #! 也就不能被python捕捉到exception, 这种情况会默认返回0
    cmd = f'cat {os.path.join(yolo_labels_dir, "*.txt")} | grep -E "^{cls_id}(\.[0]{{1,}}[e+[0]{{1,}}]{{0,}}\s|\s)" | wc -l'
    
    #! 所里这里先查询有多少个文件数量, 如果超过最大长度, 则采用find来查询
    #! 虽然会更慢, 但是不会碰到超出参数列表长度的问题.
    alter_cmd = f'find {yolo_labels_dir} -maxdepth 1 -type f -name "*.txt" -exec cat {{}} + | grep -E "^{cls_id}(\.[0]{{1,}}[e+[0]{{1,}}]{{0,}}\s|\s)" | wc -l'

    current_cls_instances_cnt = int(
        run_cmd(
            cmd=cmd, alter_cmd=alter_cmd, 
            arg_len_check_cmd=f'ls {yolo_labels_dir} | grep ".txt" -c'
        )
    )
    
    # 多少个txt文件中包含目标类别的总数
    logger.info(f'{len(target_files)} txt containing {cls_id_name_dict[cls_id]}, total instances={current_cls_instances_cnt}')

    #; 为了避免重复, 要求查询到的gt数量不少于一定的阈值才可以开始增强.
    assert current_cls_instances_cnt > CNT_MINIMUM_INSTANCE_TO_AUGMENT, \
        logger.critical(f'Current instance count is too low. Expect {CNT_MINIMUM_INSTANCE_TO_AUGMENT} at least, but only {current_cls_instances_cnt} instances found.')


    # 当前实例数量已经达到最大值, 不需要增强
    if cls_augment_limit <= current_cls_instances_cnt:
        logger.info('instances alread reach augment limit, skipping augment')
        return

    # 否则至少进行一次增强
    augment_rounds = cls_augment_limit//current_cls_instances_cnt + 1
    logger.info(f'{augment_rounds} rounds of augment to go.')
    for idx in range(augment_rounds):
        augment_single_round(
            txt_files=target_files, 
            round_idx=idx, 
            max_rounds=augment_rounds
        )



def augment_single_round(
        txt_files: list[str], 
        round_idx: int, 
        max_rounds: int,
    ):
    global yolo_images_dir, augment_version
    for file_idx, labelfile_filepath in enumerate(txt_files):
        logger.info(f'[{round_idx+1} / {max_rounds}], [{file_idx+1} / {len(txt_files)}], current_file: {labelfile_filepath}')
        try:
            yolo_label = np.loadtxt(labelfile_filepath)
            # 确保标签是一个二维数组
            # 当一个标签文件中只有一行标签时, 会被默认读取为一个一维向量
            if 1 == yolo_label.ndim:
                yolo_label = yolo_label.reshape(1, -1).astype(np.float32)
            
            logger.info(f'{yolo_label.shape[0]} instances detected')
        except Exception as e:
            logger.error(f'failed to load label={labelfile_filepath}, exception={e}')
            continue
        
        # 标签文件的filename
        #   given a path path/to/00000.txt, 
        #   this is a filepath: path/to/00000.txt
        #   this is a basename: 00000.txt
        #   this is a filename: 00000
        #   path/to/00000.txt => 00000
        labelfile_filename = os.path.basename(labelfile_filepath).rsplit('.', 1)[0]


        try:
            # 对应图像文件的路径
            imagefile_filepath = next(
                (
                    os.path.join(yolo_images_dir, f'{labelfile_filename}.{fmt}') for fmt in ALLOWED_IMAGE_FORMATS 
                    if os.path.isfile(os.path.join(yolo_images_dir, f'{labelfile_filename}.{fmt}'))
                ), None
            )
            assert imagefile_filepath is not None, \
                logger.error(f"find no such file as {os.path.join(yolo_images_dir, f'{labelfile_filename}.{ALLOWED_IMAGE_FORMATS}')}")
            yolo_image = cv.imread(imagefile_filepath)
            logger.info(f'succeed to load image, shape={yolo_image.shape}')
        except Exception as e:
            logger.error(f'failed to load image={imagefile_filepath} due to {e}. skipping')
            continue

        augment_image, augment_label = augment_single_image(
            image=yolo_image, 
            label=yolo_label
        )

        if augment_image is None:
            continue
        
        save_augment_image_label(
            augment_image=augment_image, 
            augment_label=augment_label, 
            save_filename=f'{labelfile_filename}-augment_{augment_version}-round_{round_idx}'
        )


def save_augment_image_label(
        augment_image: np.ndarray, 
        augment_label: np.ndarray, 
        save_filename: str
    ):
    global augment_images_savedir, augment_labels_savedir
    img_savepath = os.path.join(augment_images_savedir, f'{save_filename}.jpg')
    label_savepath = os.path.join(augment_labels_savedir, f'{save_filename}.txt')
    try:
        logger.info(f'saving augmented image to {img_savepath}')
        cv.imwrite(
            filename=img_savepath, 
            img=augment_image
        )
        logger.info(f'saving augmented label to {label_savepath}')
        np.savetxt(
            fname=label_savepath, 
            X=augment_label
        )
    except Exception as e:
        logger.error(f'failed to save image and label, error: {e}')
        return False


def augment_single_image(
        image: np.ndarray, 
        label: np.ndarray, 
    ):
    global cls_id_name_dict, transforms
    cls_ids = label[:, 0]

    bboxes_xnynwnhn = label[:, 1:5]
    # yolo bbox => x1 y1 x2 y2
    bboxes_xyxy = bbox_xnynwnhn2xyxy(
        bboxes=bboxes_xnynwnhn, 
        image_shape=image.shape[:2], 
        return_int=True
    )

    try:
        transformed = transforms(
            image=image, 
            bboxes=bboxes_xyxy, 
            bbox_classes=[str(int(x)) for x in cls_ids]
        )
    except Exception as e:
        logger.error(f'failed to transform current image, skipping. Error: {e}')
        return None, None
    
    if not transformed['bboxes']:
        logger.warning('empty transform, skipping.')
        return None, None

    logger.info(f"{len(transformed['bboxes'])} transformed bboxes")
    logger.debug(f"transformed bboxes:\n{transformed['bboxes']}")
    # 更新label中的bndbox
    # 丢弃多余的标签, transformed不一定能够保留所有的标签
    cnt_objects_removed = label.shape[0] - len(transformed['bboxes'])

    transformed_bboxes_xnynwnhn = bbox_xyxy2xnynwnhn(
        bboxes=transformed['bboxes'], 
        image_shape=transformed['image'].shape
    )
    transformed_cls_labels = np.array([int(x) for x in transformed['bbox_classes']]).reshape(1, -1).T

    # 检查是否存在越界的bbox
    if (transformed_bboxes_xnynwnhn > 1).any():
        logger.warning(
            f"corrupted instance detected. image shape: {transformed['image'].shape}, bbox:\n"
            f"{transformed_bboxes_xnynwnhn}"
        )
    
    # 将transformed的cls_id和bbox合并
    transformed_label = np.hstack([transformed_cls_labels, transformed_bboxes_xnynwnhn])
    logger.info(
        f"{cnt_objects_removed} objects filtered, {transformed_bboxes_xnynwnhn.shape[0]} remaining"
    )
    logger.debug(
        f"remaining bboxes:\n"
        f"{[dict(name=cls_id_name_dict[str(int(x[0]))], bbox=x[1:]) for x in transformed_label]}"
    )

    return transformed['image'], transformed_label


def bbox_xnynwnhn2xyxy(
        bboxes:np.ndarray, 
        image_shape: tuple, 
        return_int: False
    ):
    if not isinstance(bboxes, np.ndarray):
        bboxes = np.array(bboxes)
    if bboxes.ndim == 1:
        bboxes = bboxes.reshape(1, -1)
    
    image_h, image_w = image_shape
    xn, yn = bboxes[:, 0], bboxes[:, 1]
    wn, hn = bboxes[:, 2], bboxes[:, 3]
    xc, yc = xn*image_w, yn*image_h
    bbox_w, bbox_h = wn*image_w, hn*image_h

    x1 = (xc - bbox_w/2).reshape(1, -1)
    x2 = (xc + bbox_w/2).reshape(1, -1)
    y1 = (yc - bbox_h/2).reshape(1, -1)
    y2 = (yc + bbox_h/2).reshape(1, -1)

    bboxes_xyxy = np.concatenate([x1, y1, x2, y2], axis=0).T
    return bboxes_xyxy.astype(np.int32) if return_int else bboxes_xyxy


def bbox_xyxy2xnynwnhn(
        bboxes: np.ndarray, 
        image_shape: tuple, 
    ):
    if not isinstance(bboxes, np.ndarray):
        bboxes = np.array(bboxes)
    if bboxes.ndim == 1:
        bboxes = bboxes.reshape(1, -1)

    image_h, image_w = image_shape[:2]
    x1, y1 = bboxes[:, 0], bboxes[:, 1]
    x2, y2 = bboxes[:, 2], bboxes[:, 3]

    w, h = x2-x1, y2-y1
    xc, yc = (x1+x2)/2, (y1+y2)/2
    
    wn, hn = w/image_w, h/image_h
    xn, yn = xc/image_w, yc/image_h

    wn = wn.reshape(1, -1)
    hn = hn.reshape(1, -1)
    xn = xn.reshape(1, -1)
    yn = yn.reshape(1, -1)

    return np.concatenate([xn, yn, wn, hn], axis=0).T


def summary():
    """
    统计增强的图像中, 各个类别的实例数量

    Args:

    """
    global cls_id_name_dict, augment_labels_savedir, augment_savedir

    # 实际没有进行增强操作
    # 则这个文件夹下不会有内容
    # 跳过
    if 0 == len(os.listdir(augment_labels_savedir)):
        logger.info(f'no augmentation found.')
        return
    
    cmd = f'for i in $(seq 0 {len(cls_id_name_dict)-1}); do cat {os.path.join(augment_labels_savedir, "*.txt")} | grep -E "^$i(\.[0]{{1,}}[e+[0]{{1,}}]{{0,}}\s|\s)" | wc -l; done'
    
    #! 同理, 防止参数列表过长
    alter_cmd = f'for i in $(seq 0 {len(cls_id_name_dict)-1}); do find {augment_labels_savedir} -maxdepth 1 -type f -name "*.txt" -exec cat {{}} + | grep -E "^$i(\.[0]{{1,}}[e+[0]{{1,}}]{{0,}}\s|\s)" | wc -l; done'
    
    #; 分类别统计txt文件中的类别实例数量
    #; cls_id,cls_cnt
    splits = run_cmd(
        cmd, alter_cmd=alter_cmd, 
        arg_len_check_cmd=f'ls {augment_labels_savedir} | grep ".txt" -c'
    ).split('\n')
    # 打包成dict并保存
    summary_dict = dict(zip(cls_id_name_dict.values(), splits))
    logger.info(f'instances summary in augment patch: {summary_dict}')
    savename = os.path.join(augment_savedir, 'instances_summary.json')
    import json
    json.dump(
        summary_dict, open(savename, 'w'), indent=2
    )
    logger.info(f'summary saved in {savename}')

def main():
    global cls_id_name_dict, target_classes_id, augment_limit
    logger.info(
        f'augment classes_id: {target_classes_id}, '
        f'augment classes: {[cls_id_name_dict[x] for x in target_classes_id]}, '
        f'augment limits: {augment_limit}'
    )

    for idx, current_cls_id in enumerate(target_classes_id):
        current_cls_augment_limit = augment_limit[idx]
        logger.info(f'begin augment class={cls_id_name_dict[current_cls_id]}, augment limit: {current_cls_augment_limit}')
        augment_single_class(
            cls_id=current_cls_id, 
            cls_augment_limit=current_cls_augment_limit
        )


if __name__ == '__main__':
    main()
    summary()

    # python augment_from_yolo.py --target_classes_id 4 --yolo_images_dir /data/jianghengyu/datasets/TaiZhou_230831/v1.0.0/images/train --yolo_labels_dir /data/jianghengyu/datasets/TaiZhou_230831/v1.0.0/labels/train --augment_root_save_dir /data/jianghengyu/datasets/TaiZhou_230831/v1.0.0/augments --augment_limit 24000 --augment_version BLQ_alpha --yolo_names_filepath /data/jianghengyu/datasets/TaiZhou_230831/v1.0.0/cls_names.yaml
    
