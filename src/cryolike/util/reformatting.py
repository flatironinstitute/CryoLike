from typing import Literal, overload
import numpy as np
from cryolike.util.types import FloatArrayType, IntArrayType
from enum import Enum

class TargetType(Enum):
    """Enum defining int and float type labels for parameters.
    """
    INT = 1
    FLOAT = 2


def project_scalar(scalar: int | float, dims: int) -> IntArrayType | FloatArrayType:
    """Returns a numpy array of appropriate dtype and dimension from a scalar input.

    Args:
        scalar (int | float): The value to project to an array
        dims (int): Dimension (1d) of the desired array

    Returns:
        np.ndarray: A 1-d numpy array with the scalar value repeated.
    """
    # this will return float if scalar is float or preserve int if scalar is int
    return np.ones((dims,), dtype=int) * scalar


def project_vector(vector: list | np.ndarray, dims: int, label: str) -> IntArrayType | FloatArrayType:
    """Returns an appropriately trimmed ndarray with the input values, warning
    appropriately if some values are unused.

    Args:
        vector (list | np.ndarray): Input data values to reshape.
        dims (int): Desired length of the 1-d ndarray returned.
        label (str): Description for the vector being assembled.

    Raises:
        ValueError: If input data is multi-dimensional
        ValueError: If requested dimension cannot be obviously satisfied by the input
            data.

    Returns:
        np.ndarray: A 1-d numpy array of desired length with data from the input.
    """
    if (isinstance(vector, np.ndarray)):
        squeezed = np.squeeze(vector)
        if len(squeezed.shape) > 1:
            raise ValueError(f"{label} contains multi-dimensional data and is ambiguous.")
        v = np.atleast_1d(squeezed)
    else:
        assert isinstance(vector, list)
        v = vector
    if len(v) == 1:
        result = project_scalar(v[0], dims)
    else:
        if len(v) < dims:
            raise ValueError(f"{label} must be at least {dims}-dimensional.")
        if len(v) > dims:
            # QUERY: should we actually allow this? It probably represents a mistake on the caller's part.
            print(f"Warning: {label} is more than {dims}-dimensional. Ignoring higher dimensions...")
        result = np.array(v[:dims])
    return result


int_descriptors = int | list[int] | IntArrayType
float_descriptors = float | list[float] | FloatArrayType
descriptor_types = int | float | list[int] | list[float] | IntArrayType | FloatArrayType

@overload
def project_descriptor(descriptor: descriptor_types, label: str, dims: int, target_type: Literal[TargetType.INT]) -> IntArrayType:
    ...
@overload
def project_descriptor(descriptor: descriptor_types, label: str, dims: int, target_type: Literal[TargetType.FLOAT]) -> FloatArrayType:
    ...
@overload
def project_descriptor(descriptor: int_descriptors, label: str, dims: int, target_type: None) -> IntArrayType:
    ...
@overload
def project_descriptor(descriptor: float_descriptors, label: str, dims: int, target_type: None) -> FloatArrayType:
    ...
def project_descriptor(descriptor: descriptor_types, label: str, dims: int, target_type: TargetType | None) -> IntArrayType | FloatArrayType:
    """Normalizes grid inputs to desired-length 1D vectors, with descriptive error checking.

    Args:
        descriptor (int | float | list[int] | list[float] | IntArrayType | FloatArrayType): The descriptive data point
            being projected (e.g. pixel size)
        label (str): Description of the descriptor (used for error message reporting)
        dims (int): Length of the desired output vector
        target_type (TargetType): Whether the desired output should be integer or float (determining dtype of output
            array). If omitted, will be inferred from the type of the descriptor.

    Raises:
        ValueError: Raised if non-integer input and integer output desired.
        ValueError: Raised if the input descriptor contains non-positive values.

    Returns:
        IntArrayType | FloatArrayType: Appropriately shaped numpy array with input vector's data.
    """
    if target_type is None:
        if isinstance(descriptor, list) or isinstance(descriptor, np.ndarray):
            testval = descriptor[0]
        else:
            testval = descriptor
        if isinstance(testval, float):
            target_type = TargetType.FLOAT
        elif isinstance(testval, int):
            target_type = TargetType.INT
        else:
            raise ValueError("Unreachable: test val was neither float nor int")

    # TODO: Typing contravariant stuff.
    if np.isscalar(descriptor):
        assert isinstance(descriptor, (int, float))
        result = project_scalar(descriptor, dims)
    else:
        assert isinstance(descriptor, (list, np.ndarray))
        result = project_vector(descriptor, dims, label)

    # Check result for positivity and correct type
    if (target_type == TargetType.INT and not issubclass(result.dtype.type, np.integer)):
        if np.all(np.mod(result, 1) == 0):
            result = result.astype(int)
        else:
            raise ValueError(f"{label} must have integer values.")
    if (target_type == TargetType.FLOAT and not issubclass(result.dtype.type, np.floating)):
        result = result.astype(float)
    if np.any(result <= 0.0):
        raise ValueError(f"{label} must have positive values.")

    return result
