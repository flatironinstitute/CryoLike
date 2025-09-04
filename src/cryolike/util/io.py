from typing import Any
from pathlib import Path
import numpy as np
from functools import reduce

def load_file(file: str | Path) -> dict:
    if file == '':
        raise FileNotFoundError("Loading from file requires a file name.")
    filepath = Path(file)
    ext = filepath.suffix
    if ext != '.npz':
        raise ValueError("Can only load from .npz files.")
    
    # TODO: figure out a way to avoid allowing pickling
    data: dict[str, Any] = np.load(file, allow_pickle=True)

    out_dict = {}
    for param in data.keys():
        y = data[param]
        if not isinstance(y, np.ndarray): # pragma: no cover
            raise NotImplementedError("Shouldn't happen: will be represented as a shape-() array instead.")
            out_dict[param] = y
            continue
        else:
            if y.size == 0:
                out_dict[param] = None
            elif y.shape == ():
                out_dict[param] = y.item()
            elif y.size == 1 and y.item() is None:
                # Must include the `y.item()` criterion, or this fails on strings
                out_dict[param] = None
            else:
                out_dict[param] = y

    return out_dict


def save_descriptors(filename: str, *args, **kwargs): # type: ignore
    """Create NPZ file named `filename` from the to_dict() representations of
    the objects passed as args.

    Args:
        filename (str): NPZ file to use as output. Operation will be canceled if
            this named file already exists unless *overwrite* is **True**

    Kwargs:
        overwrite (bool): whether to allow overwriting existing files. Default False

    Raises:
        ValueError: If the requested output filename already exists unless *overwrite* is **True**.
    """
    if len(args) == 0:
        return
    filepath = Path(filename)
    if filepath.exists() and not kwargs.get('overwrite', False):
        raise ValueError(f"Requested filename {filename} already exists. Aborting to avoid overwrite.")
    all_keys_count = sum([len(x) for x in args])
    def merge(x, y):
        x.update(y)
        return x
    data = reduce(merge, args, {})
    if len(data) != all_keys_count:
        fields = []
        [fields.append(list(x.keys())) for x in args]
        raise ValueError(f"Duplicate keys passed in save_descriptors list:\n{fields}")
    np.savez_compressed(
        filename,
        **data
    )
