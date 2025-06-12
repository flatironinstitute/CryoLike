import torch
import numpy as np
from typing import TypeVar, cast

from .device_handling import get_device
from .enums import Precision
from .typechecks import is_integral_torch_tensor
from .types import FloatArrayType, ComplexArrayType


def to_torch(array: torch.Tensor | np.ndarray, precision: Precision = Precision.DEFAULT, device: str | torch.device | None = None):
    device = get_device(device)
    if isinstance(array, torch.Tensor):
        current_precision = Precision.SINGLE if array.dtype in [torch.float32, torch.complex64, torch.int32] else Precision.DOUBLE
    else:
        current_precision = Precision.DOUBLE
    (torch_float_type, torch_complex_type, torch_int_type) = precision.get_dtypes(current_precision)

    if torch.is_tensor(array):
        assert isinstance(array, torch.Tensor)
        if torch.is_complex(array):
            result = array.to(torch_complex_type)
        elif torch.is_floating_point(array):
            result = array.to(torch_float_type)
        elif is_integral_torch_tensor(array):
            result = array.to(torch_int_type)
        return result.to(device)
    else:
        assert isinstance(array, np.ndarray)
        assert isinstance(array.dtype, np.dtype)
        try:
            if np.issubdtype(array.dtype, np.complexfloating):
                result = torch.tensor(array, dtype=torch_complex_type, device=device)
            elif np.issubdtype(array.dtype, np.floating):
                result = torch.tensor(array, dtype=torch_float_type, device=device)
            elif np.issubdtype(array.dtype, np.integer):
                result = torch.tensor(array, dtype=torch_int_type, device=device)
        except:
            raise ValueError("Cannot convert array to torch tensor")
    
    return result


T2 = TypeVar("T2", bound=float | FloatArrayType | np.ndarray | torch.Tensor | None)
def batchify(d: T2, start: int, end: int) -> T2:
    """Safely take a range from a value that might be scalar.

    Args:
        d (T2): Source (list or scalar) to take a range from
        start (int): Range start (inclusive)
        end (int): Range end (exclusive)

    Returns:
        T2: The range, if input was a vector; or the original scalar value,
            if the input was a scalar 
    """
    if isinstance(d, np.ndarray) or isinstance(d, torch.Tensor):
        return cast(T2, d[start:end])
    return d


def ensure_np(d: float | FloatArrayType | torch.Tensor) -> FloatArrayType:
    if isinstance(d, np.ndarray):
        return d
    if isinstance(d, torch.Tensor):
        return d.numpy()
    return np.array(d)


def to_float_flatten_np_array(x: float | FloatArrayType):
    if isinstance(x, float):
        return np.array([x], dtype = np.float64)
    if isinstance(x, list):
        return np.array(x, dtype = np.float64)
    if not isinstance(x, np.ndarray):
        raise ValueError(f'Invalid type for x ({type(x)})')
    if len(x.shape) > 1:
        x = x.flatten()
    return x


def absq(
    array : torch.Tensor
):
    return array.real ** 2 + array.imag ** 2


def complex_mul_real(
    array1 : torch.Tensor,
    array2 : torch.Tensor
):
    ## array1 is complex and array2 is real
    if torch.is_complex(array1) and torch.is_complex(array2):
        return (array1.real * array2.real - array1.imag * array2.imag)
    elif torch.is_complex(array1) and torch.is_floating_point(array2):
        return (array1.real * array2)
    elif torch.is_floating_point(array1) and torch.is_complex(array2):
        return (array1 * array2.real)
    elif torch.is_floating_point(array1) and torch.is_floating_point(array2):
        return (array1 * array2)


def fourier_bessel_transform(image_fourier_: torch.Tensor, axis = -1, norm = "ortho") -> torch.Tensor:
    ## perform 1D Fourier transform on the specfied axis
    if axis > image_fourier_.ndim - 1:
        raise ValueError("axis out of range")
    image_fourier_bessel_ = torch.fft.fft(image_fourier_, dim = axis, norm = norm)
    # image_fourier_bessel_ = torch.fft.ihfft(image_fourier_, dim = axis, norm = norm)
    return image_fourier_bessel_


T = TypeVar("T", bound=FloatArrayType | ComplexArrayType | torch.Tensor | None)
def pop_batch(u: T, batch_size: int) -> tuple[T, T]:
    if u is None:
        head = cast(T, None)
        tail = cast(T, None)
    else:
        head = cast(T, u[:batch_size])
        tail = cast(T, u[batch_size:])
    return (head, tail)
