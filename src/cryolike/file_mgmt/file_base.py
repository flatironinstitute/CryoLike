from pathlib import Path
from typing import Literal, Sequence

FILE_TYPES = Literal['fourier'] | Literal['params'] | Literal['phys']


def make_dir(base: Path, branch: str):
    full_path = base / branch
    if not full_path.exists():
        full_path.mkdir(exist_ok=True, parents=True)
        return
    if not full_path.is_dir():
        raise FileExistsError(f"Requested path {full_path.absolute()} already exists and is not a directory.")


def check_files_exist(files: Sequence[str | Path]):
    missing_files: list[str] = []
    for f in files:
        f = Path(f)
        if not f.is_file():
            missing_files.append(str(f.absolute()))
    return (len(missing_files) == 0, missing_files)


def get_input_filename(base: Path, i_stack: int, type: FILE_TYPES) -> Path:
    if type == 'fourier':
        desc = "fourier"
        suffix = "pt"
    elif type == 'params':
        desc = "fourier"
        suffix = "npz"
    elif type == 'phys':
        desc = "phys"
        suffix = "pt"
    else:
        raise NotImplementedError("Impermissible file type requested.")

    fn = f"particles_{desc}_stack_{i_stack:06}.{suffix}"
    return base / fn
