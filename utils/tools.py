import os

def check_path(
        path: str, 
        require_abs: bool = False 
    ):
    if not require_abs:
        assert os.path.exists(path), f'no such path as {path}'
    else:
        assert os.path.exists(path) and os.path.isabs(path), f'no such path as {path} or {path} is not absolute'
    return path

def remove_last_sep_from_dir(dirpath: str):
    return dirpath.rsplit(os.sep, 1)[0] if dirpath.endswith(os.sep) else dirpath
