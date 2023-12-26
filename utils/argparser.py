import argparse


def _default_args():
    """
    全局通用的默认参数
    :return:
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--log_level', '-ll', type=str, default='INFO', help='log level, [DEFAULT: INFO]', 
        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'], 

    )
    parser.add_argument(
        '--splitor', '-s', type=str, default='-', help='splitor of different components in a filename. [DEFAULT: -]'
    )
    return parser



def parse_json2yolo_args():
    """
    json2yolo.py使用的参数
    :return:
    """
    parser = argparse.ArgumentParser('json2yolo args', parents=[_default_args()])
    parser.add_argument(
        '--json_anno_dir', '-jad', type=str, required=True,
        help='directory which contains all json annotations. [REQUIRED]'
    )
    parser.add_argument(
        '--yolo_dir', '-yd', type=str, required=True,
        help='directory to save all the converted yolo dataset. [REQUIRED]'
    )
    parser.add_argument(
        '--orig_img_dir', '-id', type=str, default=None,
        help='directory contains all the original images. [DEFAULT: None]'
    )
    parser.add_argument(
        '--img_fmt', '-if', type=str, default='JPG',
        choices=['JPG', 'PNG', 'jpg', 'png'],
        help='image format for saved images in yolo dataset. [DEFAULT: jpg]'
    )
    parser.add_argument(
        '--start_from_zero', '-sfz', action='store_true', default=False,
        help="if true, files will be renamed starting from 0, in case of names that can't decode. [DEFAULT: False]"
    )
    parser.add_argument(
        '--fixed_resolution', '-fr', nargs='+', type=int, default=None,
        help='fixed resolution of converted yolo dataset, [height, width]. [DEFAULT: None]'
    )
    parser.add_argument(
        '--labels_only', '-lo', action='store_true', default=False,
        help='if true, only label files will be created. [DEFAULT: False]'
    )
    parser.add_argument(
        '--val_size', '-vs', type=float, default=-1,
        help='size of validation set, (0, 1), [DEFAULT: -1]'
    )
    parser.add_argument(
        '--cls_names_file', '-cnf', type=str, default=None,
        help='global class names of yolo dataset. if not specified, the id may be random. [DEFAULT: None]'
    )
    parser.add_argument(
        '--copy_new_xml', '-cnx', type=str, default=None,
        help='if specified, the new xml files of new class names will be copied. [DEFAULT: None]'
    )
    parser.add_argument(
        '--allow_new_cls', '-anc', action='store_true', default=False, 
        help='whether to continue when encountering new class label. [DEFAULT: False]'
    )
    parser.add_argument(
        '--keep_ratio', '-nkr', action='store_true', default=False, 
        help='whether to keep image ratio when resizing. [DEFAULT: False]'
    )
    parser.add_argument(
        '--patch_name', '-pn', type=str, default=None, 
        help='filename prefix. [DEFAULT: None]'
    )
    return parser.parse_args()


def parse_augment_from_yolo_args():
    parser = argparse.ArgumentParser('augment from yolo args', parents=[_default_args()])
    parser.add_argument(
        '--target_classes_id', '-tci', nargs='+', type=str, required=True, 
        help='which classes to be augmented. [REQUIRED]'
    )
    parser.add_argument(
        '--yolo_images_dir', '-yid', type=str, required=True, 
        help='directory containing yolo images. [REQUIRED]'
    )
    parser.add_argument(
        '--yolo_labels_dir', '-yld', type=str, required=True, 
        help='direcotry contains all yolo labels. An absolute path is requried here. [REQUIRED]'
    )
    parser.add_argument(
        '--augment_root_savedir', '-ars', type=str, default=None, 
        help='where to save the augment images and labels, [DEFAULT: dirname($yolo_images_dir)/augment_$augment_version]'
    )
    parser.add_argument(
        '--augment_limit', '-al', type=int, nargs='+', default=[10000], 
        help='maximun augmented images number of each target class. [DEFAULT: (10000)]'
    )
    parser.add_argument(
        '--augment_version', '-av', type=str, default=None, 
        help='augment version name. [DEFAULT: None]'
    )
    parser.add_argument(
        '--yolo_names_filepath', '-ynf', type=str, required=True, 
        help='class names configure yaml file, [class id: class name]. [REQUIRED]'
    )
    parser.add_argument(
        '--force_on_non_unix', '-fonu', action='store_true', default=False, 
        help='whether to force running on non-Unix systems. [DEFAULT: False]'
    )

    return parser.parse_args()
