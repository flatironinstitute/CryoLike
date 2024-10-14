from math import isfinite
from cryolike.util.enums import Precision
from cryolike.util.types import FloatArrayType, IntArrayType
from torch import Tensor, dtype, int32, int64, float32, float64, complex64, complex128, is_floating_point, is_complex
import numpy as np


def ensure_positive(x: float | int |  FloatArrayType | IntArrayType | Tensor, desc: str):
    if type(x) in ['float', 'int']:
        if (x <= 0.0):
            raise ValueError(f'Invalid value for {desc} (positive value required, received {x})')
    if np.any(x <= 0.0):
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


def set_precision(precision: Precision, default: Precision) -> tuple[dtype, dtype, dtype]:
    """Interprets a Precision enum to return the desired dtypes.

    Args:
        precision (Precision): Precision to interpret.
        default (Precision): Precision level to use if DEFAULT is requested.

    Returns:
        tuple[dtype, dtype, dtype]: Torch float-type, complex-type, and int-type
    for the requested precision.
    """
    if precision == Precision.DEFAULT:
        precision = default
    if precision == Precision.SINGLE:
        return (float32, complex64, int32)
    elif precision == Precision.DOUBLE:
        return (float64, complex128, int64)
    else:
        raise ValueError("Unreachable: unrecognized precision type.")


def set_epsilon(precision: Precision, requested: float, default: Precision=Precision.SINGLE) -> float:
    if precision == Precision.DEFAULT:
        precision = default
    if precision == Precision.SINGLE:
        target_epsilon = 1.0e-6
    if precision == Precision.DOUBLE:
        target_epsilon = 1.0e-12
    return max(requested, target_epsilon)
