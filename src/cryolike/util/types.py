import numpy as np
import numpy.typing as npt

ComplexArrayType = npt.NDArray[np.complexfloating]
FloatArrayType = npt.NDArray[np.floating]
IntArrayType = npt.NDArray[np.integer]

Pixels_count_type = int | list[int] | IntArrayType
Pixel_size_type = float | list[float] | FloatArrayType
