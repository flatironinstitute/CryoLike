import numpy as np
from typing import Literal, Tuple, cast, Union

from cryolike.util import (
    FloatArrayType,
    IntArrayType,
    Cartesian_grid_2d_descriptor,
    project_descriptor,
    TargetType,
)


class CartesianGrid2D:
    """Class implementing 2D Cartesian grid.

    Attributes:
        n_pixels (IntArrayType): Number of pixels on each dimension of the grid.
        pixel_size (FloatArrayType): Size of each pixel as [x, y].
        box_size (FloatArrayType): Overall area of the grid (pixel size times number of pixels) in each dimension.
        radius (FloatArrayType): Half the box size, i.e. multidimensional radius centered at the grid center.
        x_axis (FloatArrayType): 1-D array represetting X-coordinates of the grid
        y_axis (FloatArrayType): 1-D array represetting Y-coordinates of the grid
        x_pixels (FloatArrayType): 2-D array representing X-coordinates of each pixel on the grid
        y_pixels (FloatArrayType): 2-D array representing Y-coordinates of each pixel on the grid
        n_pixels_total (int): Total pixel count of the grid.
    """
    n_pixels: IntArrayType
    pixel_size: FloatArrayType
    box_size: FloatArrayType
    radius: FloatArrayType
    x_axis: FloatArrayType
    y_axis: FloatArrayType
    x_pixels: FloatArrayType
    y_pixels: FloatArrayType
    n_pixels_total: int

    def __init__(
        self,
        n_pixels: int | list[int] | IntArrayType,
        pixel_size: float | list[float] | FloatArrayType,
        endpoint : bool = False
    ):
        """Class implementing 2D Cartesian grid.

        Args:
            n_pixels (int | list[int] | IntArrayType): Number of pixels in the grid. If scalar, will be applied
                to both dimensions. If a vector, the first 2 dimensions will be used as the X and Y dimensions
                (respectively).
            pixel_size (float | list[float] | FloatArrayType): Size of each pixel in the grid. If scalar, each
            pixel will be treated as square; if a vector, the first 2 dimensions will be used as the X and Y
                sizes (width and height), respectively.
            endpoint (bool, optional): If true, the computed grid will be inclusive of the final pixel location;
                otherwise the grid will stop short of the endpoint. Defaults to False.
        """
        self.endpoint = endpoint
        (self.n_pixels, self.pixel_size, self.box_size, self.radius) = _compute_grid_dims(2, n_pixels, pixel_size)
        (axes, xels) = _setup_grid(self.radius, self.n_pixels, self.endpoint)
        (self.x_axis, self.y_axis) = axes
        (self.x_pixels, self.y_pixels) = (xels)
        self.n_pixels_total = self.x_pixels.size # Is this right? What about non-square grids?


    @classmethod
    def from_descriptor(cls, grid_desc: Union[Cartesian_grid_2d_descriptor, 'CartesianGrid2D']):
        if isinstance(grid_desc, CartesianGrid2D):
            return grid_desc
        (n_pixels, pixel_size) = grid_desc
        return cls(n_pixels=n_pixels, pixel_size=pixel_size)


class CartesianGrid3D:
    """Class implementing 3D Cartesian grid.

    Attributes:
        n_voxels (IntArrayType): Number of voxels in each dimension of the grid.
        voxel_size (FloatArrayType): Size of each voxel as [x, y, z].
        box_size (FloatArrayType): Overall area of the grid (voxel size times number of voxels) in each dimension.
        radius (FloatArrayType): Half the box size, i.e. multidimensional radius centered at the grid center.
        x_axis (FloatArrayType): X-axis positions of the grid, assuming 0 for the grid center
        y_axis (FloatArrayType): Y-axis positions of the grid, assuming 0 for the grid center
        z_axis (FloatArrayType): Z-axis positions of the grid, assuming 0 for the grid center
        x_voxels (FloatArrayType): X-positions of each voxel
        y_voxels (FloatArrayType): Y-positions of each voxel
        z_voxels (FloatArrayType): Z-positions of each voxel
        n_pixels_total (int): Total pixel count of the grid.
    """
    n_voxels: IntArrayType
    voxel_size: FloatArrayType
    box_size: FloatArrayType
    radius: FloatArrayType
    x_axis: FloatArrayType
    y_axis: FloatArrayType
    z_axis: FloatArrayType
    x_voxels: FloatArrayType
    y_voxels: FloatArrayType
    z_voxels: FloatArrayType
    n_voxels_total: int

    def __init__(
        self,
        n_voxels: int | list[int] | IntArrayType,
        voxel_size: float | list[float] | FloatArrayType,
        endpoint : bool = False
    ):
        """Constructor for class implementing a 3D Cartesian grid.

        Args:
            n_voxels (int | list[int] | IntArrayType): Number of voxels in the grid. If scalar, will be applied
                to all three dimensions. If a vector, the first 3 dimensions will be used as the X, Y, and Z
                dimensions (respectively).
            voxel_size (float | list[float] | FloatArrayType): Size of each voxel in the grid. If scalar,
                the voxels will be treated as square; if a vector, the first 3 dimensions will be used as the
                X, Y, and Z sizes (width, height, and depth), respectively.
            endpoint (bool, optional): If True, the computed grid will be inclusive of the final pixel location;
                otherwise the grid will stop short of the stated endpoint in each dimension. Defaults to False.
        """
        self.endpoint = endpoint
        (self.n_voxels, self.voxel_size, self.box_size, self.radius) = _compute_grid_dims(3, n_voxels, voxel_size)
        (axes, xels) = _setup_grid(self.radius, self.n_voxels, self.endpoint)
        (self.x_axis, self.y_axis, self.z_axis) = axes
        (self.x_voxels, self.y_voxels, self.z_voxels) = xels
        self.n_voxels_total = self.x_voxels.size


def _compute_grid_dims(
        dims: Literal[2] | Literal[3],
        n_xels: int | list[int] | IntArrayType,
        xel_size: float | list[float] | FloatArrayType
    ) -> Tuple[IntArrayType, FloatArrayType, FloatArrayType, FloatArrayType]:
    if dims == 2:
        desc = "pixel"
    elif dims == 3:
        desc = "voxel"
    else: # pragma: no cover
        raise ValueError('Unreachable: Unsupported dimension size')
    eff_n_xels = cast(IntArrayType, project_descriptor(n_xels, f"n_{desc}s", dims, TargetType.INT))
    eff_xel_size = cast(FloatArrayType, project_descriptor(xel_size, f"{desc}_size", dims, TargetType.FLOAT))

    box_size = cast(FloatArrayType, eff_xel_size * eff_n_xels)
    radius = cast(FloatArrayType, box_size * 0.5)

    return (eff_n_xels, eff_xel_size, box_size, radius)


def _setup_grid(radius: FloatArrayType, n_xels: IntArrayType, endpoint: bool):
    axes: list[FloatArrayType] = []
    for i in range(len(radius)):
        axes.append(np.linspace(-radius[i], radius[i], n_xels[i], endpoint = endpoint))
    computed_xels: list[FloatArrayType] = np.meshgrid(*axes, indexing='ij')
    return (axes, computed_xels)
