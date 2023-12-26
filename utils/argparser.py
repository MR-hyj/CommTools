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


def parse_xml2json_args():
    """
    xml2json.py使用的参数
    :return:
    """
    parser = argparse.ArgumentParser(prog='xml2json args', parents=[_default_args()])
    parser.add_argument(
        '--xml_path', '-xp', type=str, required=True,
        help='the xml file path or directory to be json-lize. [REQUIRED]'
    )
    parser.add_argument(
        '--indent', '-i', type=int, default=2,
        help='indent of json file, [DEFAULT: 2]'
    )
    parser.add_argument(
        '--json_directory', '-jd', type=str, default=None,
        help='save directory of json files, =$xml_path by default. '
            'needs override only when $xml_path is a directory. [DEFAULT: dirname($xml_path)]'
    )
    return parser.parse_args()


def parse_seg_classes_from_json_args():
    """
    seg_cls_from_img.py使用的参数
    :return:
    """
    parser = argparse.ArgumentParser(prog='segment classes args', parents=[_default_args()])
    parser.add_argument(
        '--target_classes', '-tc', nargs='+', type=str, required=True,
        help='target class names to be segment. set --use_re if this regular expression is used here. [REQUIRED]'
    )
    parser.add_argument(
        '--img_root_dir', '-ird', type=str, required=True,
        help='directory which contains all the images. [REQUIRED]'
    )
    parser.add_argument(
        "--seg_save_dir", '-ssd', type=str, required=True,
        help='directory in which the segmented targets will be saved. [REQUIRED]'
    )
    parser.add_argument(
        '--json_anno_dir', '-jad', type=str, required=True,
        help='directory which contains all json annotation files. [REQUIRED]'
    )
    parser.add_argument(
        "--seg_save_fmt", '-tsf', type=str, default='jpg', choices=['jpg', 'JPG', 'png', 'PNG'],
        help='in which format the segmented targets will be saved. [DEFAULT: jpg]'
    )
    parser.add_argument(
        "--separate_save_seg", '-sss', action='store_true', default=False,
        help='if true, segments of same class will be saved together in the same directory. [DEFAULT: False]'
    )
    parser.add_argument(
        "--cp_xml_file", '-cjf', action='store_true', default=False,
        help='if true, save xml annotation file along with segments. '
            'require $separate_save_seg=true and $orig_xml_dir to be specified. [DEFAULT: False]'
    )
    parser.add_argument(
        "--orig_xml_dir", '-oxd', type=str, default=None,
        help='directory which contains the original xml annotations. '
            'this argument is used only to make a copy of xml annotaionts. [DEFAULT: None]'
    )
    parser.add_argument(
        '--use_re', '-ur', action='store_true', default=False, 
        help='whether to use regular expression when matching $target_classes. [DEFAULT: False]'
    )

    return parser.parse_args()


def parse_delete_target_classes_from_xml_args():
    parser = argparse.ArgumentParser(prog='delete target classes from xml args', parents=[_default_args()])
    parser.add_argument(
        '--xml_dir', '-xd', type=str, required=True,
        help='directory of xml label files. [REQUIRED]'
    )
    parser.add_argument(
        '--target_classes', '-tc', nargs='+', type=str,  default=None,
        help='target classes to delete. [DEFAULT: None]'
    )
    parser.add_argument(
        '--filter_size', '-fs', nargs='+', type=int, default=None,
        help='if specified, smaller targets will be deleted. [height, width]. [DEFAULT: None]'
    )
    parser.add_argument(
        '--use_re', '-ur', action='store_true', default=False, 
        help='whether to use regular expression when matching $target_classes. [DEFAULT: False]'
    )
    return parser.parse_args()


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


def parse_delete_non_correspondence_args():
    parser = argparse.ArgumentParser(prog='delete non correspondence args', parents=[_default_args()])
    parser.add_argument(
        '--first_dir', '-fd', type=str, required=True,
        help='first directory. [REQUIRED]'
    )
    parser.add_argument(
        '--second_dir', '-sd', type=str, required=True,
        help='second directory. [REQUIRED]'
    )
    parser.add_argument(
        '--first_dir_fmt', '-fdf', nargs='+', type=str, required=True,
        help='candidate file format in first dir. [REQUIRED]'
    )
    parser.add_argument(
        '--second_dir_fmt', '-sdf', nargs='+', type=str, required=True,
        help='candidate file format in second dir. [REQUIRED]'
    )
    parser.add_argument(
        '--check_both', '-cb', action='store_true', default=False,
        help='check both directory. [DEFAULT: False]'
    )
    parser.add_argument(
        '--delete', '-d', action='store_true', default=False,
        help='whether to delete non-correspondence files. [DEFAULT: False]'
    )

    return parser.parse_args()


def parse_override_orig_anno_args():
    parser = argparse.ArgumentParser(prog='override orig anno args', parents=[_default_args()])
    parser.add_argument(
        '--segments_dir', '-sd', type=str, required=True,
        help='directory which contains instance segments grouped by new class names. [REQUIRED]'
    )
    parser.add_argument(
        '--orig_xml_anno_dir', '-oxad', type=str, required=True,
        help='directory which contains all original xml annotations. [REQUIRED]'
    )
    parser.add_argument(
        '--new_xml_anno_dir', '-nxad', type=str, default=None,
        help='directory to save all the new xml annotation files. [Defulat: dirname($orig_xml_anno_dir)]'
    )

    return parser.parse_args()


def parse_seg_all_ins_from_yolo_args():
    parser = argparse.ArgumentParser(prog='seg all ins from yolo', parents=[_default_args()])
    parser.add_argument(
        '--yolo_cls_names', '-ycn', type=str, required=True,
        help='yolo class names yaml. [REQUIRED]'
    )
    parser.add_argument(
        '--yolo_dataset_dir', '-ydd', type=str, default=None,
        help='yolo dataset path. [DEFAULT: None]'
    )
    parser.add_argument(
        '--yolo_img_dir', '-yid', type=str, default=None,
        help='yolo image directory. [DEFAULT: None]'
    )
    parser.add_argument(
        '--yolo_label_dir', '-yld', type=str, default=None,
        help='yolo label directory. [DEFAULT: None]'
    )
    parser.add_argument(
        '--instance_segment_dir', '-isd', type=str, default=None,
        help='where to save the segments. [DEFAULT: None]'
    )
    parser.add_argument(
        '--split_train_val', '-stv', action='store_true', default=False, 
        help='whether to split train and val when augmenting. [DEFAULT: False]'
    )
    parser.add_argument(
        '--ordered_by', '-ob', type=str, choices=['CLASS', 'FILE', 'class', 'file'], default='CLASS', 
        help='in which order will the augmented samples be saved. [DEFAULT: CLASS]'
    )
    parser.add_argument(
        '--target_classes', '-tc', nargs='+', type=str, default=None,
        help='target classes to delete. [DEFAULT: None]'
    )
    parser.add_argument(
        '--use_re', '-ur', action='store_true', default=False, 
        help='whether to use regular expression when matching $target_classes. [DEFAULT: False]'
    )
    parser.add_argument(
        '--size_filter', '-sf', nargs="+", type=int, default=None, 
        help='instances of smaller size will be ignored. [height, width], [DEFAULT: None]'
    )

    return parser.parse_args()


def parse_merge_json_args():
    parser = argparse.ArgumentParser('merge json args', parents=[_default_args()])
    parser.add_argument(
        '--json_directory', '-jd', type=str, default=None,
        help='directory contains json files to be merged. [DEFAULT: None]'
    )
    parser.add_argument(
        '--indent', '-i', type=int, default=2,
        help='indent of json file. [DEFAULT: 2]'
    )
    return parser.parse_args()


def parse_yolotxt2cocojson_args():

    """

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser('yolotxt2cocojson args', parents=[_default_args()])
    parser.add_argument(
        '--yolo_images_dir', '-yid', type=str, required=True, 
        help='direcotry containing yolo images. [REQUIRED]'
    )
    parser.add_argument(
        '--yolo_labels_dir', '-yld', type=str, required=True, 
        help='directory containing yolo labels. [REQUIRED]'
    )
    parser.add_argument(
        '--yolo_names_filepath', '-ynf', type=str, required=True, 
        help='class names configure yaml file, [class id: class name]. [REQUIRED]'
    )
    parser.add_argument(
        '--start_from_one', '-sfo', action='store_true', default=False, 
        help='coco class id starts from 0(false) or 1(true). [DEFAULT: False]'
    )
    parser.add_argument(
        '--size_filter', '-sf', nargs="+", type=int, default=None, 
        help='instances of smaller size will be ignored. [height, width], [DEFAULT: None]'
    )

    return parser.parse_args()
    

def parse_eval_yolo_pred_coco_json_args():

    """

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser('eval_yolo_pred_coco_json args', parents=[_default_args()])
    parser.add_argument(
        '--coco_anno_json', '-caj', type=str, required=True, 
        help='coco annotation json filepath. [REQUIRED]'
    )
    parser.add_argument(
        '--yolo_pred_json', '-ypj', type=str, required=True, 
        help='yolo prediction json filepath. [REQUIRED]'
    )
    parser.add_argument(
        '--eval_save_dir', '-esd', type=str, default=None, 
        help='which directory to save eval results. [DEFAULT: None]'
    )

    return parser.parse_args()



def parse_coco2yolo_args():
    parser = argparse.ArgumentParser('coco2yolo args', parents=[_default_args()])
    parser.add_argument(
        '--coco_anno_path', '-cap', type=str, required=True, 
        help='coco annotation file. [REQUIRED]'
    )
    parser.add_argument(
        '--image_dir', '-id', type=str, default=None, 
        help='directory containing images. if this is specified, $image_size is not required. [DEFAULT: None]'
    )
    parser.add_argument(
        '--image_size', '-is', nargs='+',  type=int, default=None, 
        help='image size, (H, W), which means all the images are in the same resolution of (H, W). if this is specified, $image_dir is not required. [DEFAULT: None]'
    )
    parser.add_argument(
        '--yolo_save_dir', '-ysd', type=str, default=None, 
        help='directory containing yolo labels. [DEFAULT: None]'
    )
    parser.add_argument(
        '--score_thr', '-st', type=float, default=0.001, 
        help='only predictions whose socre is higher than this value will be converted. [DEFAULT: 0.001]'
    )
    parser.add_argument(
        '--with_score', '-ws', action='store_true', default=False, 
        help='whether to keep score when converting. [DEFAULT: False]'
    )
    parser.add_argument(
        '--start_from_one', '-sfo', action='store_true', default=False, 
        help='coco class id starts from 0(false) or 1(true). [DEFAULT: False]'
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


def parse_draw_labels_from_yolo_args():
    parser = argparse.ArgumentParser(prog='draw labels from yolo', parents=[_default_args()])
    parser.add_argument(
        '--yolo_cls_names', '-ycn', type=str, required=True,
        help='yolo class names yaml. [REQUIRED]'
    )
    parser.add_argument(
        '--yolo_image_dir', '-yid', type=str, required=True,
        help='yolo image directory. [REQUIRED]'
    )
    parser.add_argument(
        '--yolo_label_dir', '-yld', type=str, required=True,
        help='yolo label directory. [REQUIRED]'
    )
    parser.add_argument(
        '--target_classes', '-tc', nargs='+', type=str, default=None,
        help='target classes to visualize. [DEFAULT: None]'
    )
    parser.add_argument(
        '--use_re', '-ur', action='store_true', default=False, 
        help='whether to use regular expression when matching $target_classes. [DEFAULT: False]'
    )
    parser.add_argument(
        '--size_filter', '-sf', nargs="+",type=int, default=None, 
        help='instances of smaller size will be ignored. [height, width], [DEFAULT: None]'
    )
    parser.add_argument(
        '--labeled_image_dir', '-lid', type=str, default=None, 
        help='where to save the labeled images. [DEFAULT: os.path.dirname($yolo_img_dir)_visualized]'
    )
    parser.add_argument(
        '--line_thickness', '-lt', type=int, default=1, 
        help='line thickness of label and score. [DEFAULT: 1]'
    )
    parser.add_argument(
        '--with_score', '-ws', action='store_true', default=False, 
        help='whether show score along with labels. [DEFAULT: False]'
    )

    return parser.parse_args()


def parse_json2xml_args():
    parser = argparse.ArgumentParser(prog='xml2json args', parents=[_default_args()])
    parser.add_argument(
        '--xml_directory', '-xd', type=str, required=True,
        help='where to save the xml files. [REQUIRED]'
    )
    parser.add_argument(
        '--json_directory', '-jd', type=str, required=True,
        help='directory of json annotations to be converted into xml. [REQUIRED]'
    )
    parser.add_argument(
        '--image_directory', '-id', type=str, required=True, 
        help='directory containing all images. [REQUIRED]'
    )
    return parser.parse_args()



def parse_move_target_files_args():
    parser = argparse.ArgumentParser(prog='move target files args', parents=[_default_args()])
    parser.add_argument(
        '--target_postfix', '-tp', type=str, required=True,
        help='files of this postfix will be moved. [REQUIRED]'
    )
    parser.add_argument(
        '--file_root_dir', '-frd', type=str, required=True, 
        help='root directory of all files. [REQUIRED]'
    )
    parser.add_argument(
        '--save_dir', '-sd', type=str, default=None, 
        help='where to move the target files. [DEFAULT: dirname($file_root_dir)/$target_postfix]'
    )
    return parser.parse_args()    


def parse_yolo2xml_args():
    parser = argparse.ArgumentParser(prog='yolo2json args', parents=[_default_args()])
    parser.add_argument(
        '--xml_directory', '-xd', type=str, required=True,
        help='where to save the xml files. [REQUIRED]'
    )
    parser.add_argument(
        '--yolo_directory', '-yd', type=str, required=True,
        help='directory of yolo annotations to be converted into xml. [REQUIRED]'
    )
    parser.add_argument(
        '--image_directory', '-id', type=str, required=True, 
        help='directory containing all images. [REQUIRED]'
    )
    parser.add_argument(
        '--cls_names_filepath', '-cnf', type=str, required=True, 
        help='yolo classes names filepath, a yaml file is expected. [REQUIRED]'
    )
    return parser.parse_args()





if __name__ == '__main__':
    pass