from math import isfinite
from torch import Tensor, is_floating_point, is_complex, any
import numpy as np

from cryolike.util.types import FloatArrayType, IntArrayType


def ensure_positive(x: float | int |  FloatArrayType | IntArrayType | Tensor, desc: str):
    if type(x) in [float, int]:
        if (x <= 0.0):
            raise ValueError(f'Invalid value for {desc} (positive value required, received {x})')
    else:
        failed = False
        if isinstance(x, Tensor):
            failed = any(x <= 0.0).item()
        else:
            failed = np.any(x <= 0.0)
        if failed:
            raise ValueError(f'Invalid value for {desc} (positive values required, received {x})')


def ensure_positive_finite(x: float | int, desc: str):
    if not isfinite(x):
        raise ValueError(f'Invalid value for {desc} (finite value required, received {x})')
    ensure_positive(x, desc)


def ensure_integer(x: int | float, desc: str):
    if (x == int(x)):
        return
    raise ValueError(f'{desc} must be an integer')


def is_integral_torch_tensor(x: Tensor):
    return not (is_floating_point(x) or is_complex(x))
