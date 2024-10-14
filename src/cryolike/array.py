from cryolike.util.device_handling import get_device
import torch
import numpy as np

from cryolike.util.enums import Precision
from cryolike.util.typechecks import is_integral_torch_tensor, set_precision

def to_torch(array: torch.Tensor | np.ndarray, precision: Precision = Precision.DEFAULT, device = None):
    # ## test if device is available
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # elif not torch.cuda.is_available():
    #     device = torch.device("cpu")
    device = get_device(device)
    if isinstance(array, torch.Tensor):
        current_precision = Precision.SINGLE if array.dtype in [torch.float32, torch.complex64, torch.int32] else Precision.DOUBLE
    else:
        current_precision = Precision.DOUBLE
    (torch_float_type, torch_complex_type, torch_int_type) = set_precision(precision, current_precision)

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