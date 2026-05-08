from pathlib import Path
def ensure_file_directory(file_path):
    """
    确保文件所在的目录存在，如果不存在则创建。
    """
    dir_path = Path(file_path).parent
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
