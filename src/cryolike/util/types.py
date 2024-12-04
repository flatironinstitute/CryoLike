import numpy as np
import numpy.typing as npt

ComplexArrayType = npt.NDArray[np.complexfloating]
FloatArrayType = npt.NDArray[np.floating]
IntArrayType = npt.NDArray[np.integer]

# TODO: These actually probably belong in the grids directory

Pixels_count_type = int | list[int] | IntArrayType
Pixel_size_type = float | list[float] | FloatArrayType

Voxels_count_type = int | list[int] | IntArrayType
Voxel_size_type = float | list[float] | FloatArrayType

Cartesian_grid_2d_descriptor = tuple[Pixels_count_type, Pixel_size_type]
