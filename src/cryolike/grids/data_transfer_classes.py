from dataclasses import dataclass
from torch import Tensor
from numpy import array

from cryolike.util import Pixel_size_type, Voxel_size_type
from .cartesian_grid import CartesianGrid2D, CartesianGrid3D
from .polar_grid import PolarGrid
from .sphere_grid import SphereGrid


# TODO: Review the use of these--they may not actually be needed

@dataclass
class FourierImages:
    """Data class representing a set of Fourier images and the corresponding polar grid.

    Attributes:
        images_fourier (Tensor): Stack of images in Fourier space as (complex-valued) Tensor
        polar_grid (PolarGrid): A grid describing the polar space in which the Fourier images reside
    """
    images_fourier: Tensor
    polar_grid: PolarGrid

    def __post_init__(self):
        if not len(self.images_fourier.shape) in [1, 2, 3]:
            raise ValueError("image array is not 1D (single non-uniform images, (n_points)), 2D (single image (n_shells, n_inplanes) or multiple non-uniform images, (n_images, n_points)) or 3D (multiple uniform images, (n_images, n_shells, n_inplanes)).")
        if len(self.images_fourier.shape) == 1 and not self.polar_grid.uniform:
            self.images_fourier = self.images_fourier.unsqueeze(0)
        if len(self.images_fourier.shape) == 2 and self.polar_grid.uniform:
            self.images_fourier = self.images_fourier.unsqueeze(0)


class PhysicalImages:
    """Data class representing a set of Cartesian images and the 2D grid on which they reside.

    Attributes:
        images_phys (Tensor): Stack of images in Cartesian space as (real-valued) Tensor
        phys_grid (CartesianGrid2D): A grid describing the Cartesian space in which the images reside
    """
    images_phys: Tensor
    phys_grid: CartesianGrid2D

    def __init__(self, images_phys: Tensor, pixel_size: Pixel_size_type):
        if not len(images_phys.shape) in [2, 3]:
            raise ValueError("image array is not 2D (single image (n_pixels_x, n_pixels_y)) or 3D (multiple images, (n_images, n_pixels_x, n_pixels_y)).")
        if len(images_phys.shape) == 2:
            images_phys = images_phys[None, :, :]
        if len(images_phys.shape) != 3:
            raise ValueError("Invalid shape for images.")

        n_pixels = array(images_phys.shape[1:], dtype=int)
        self.phys_grid = CartesianGrid2D(n_pixels, pixel_size)
        self.images_phys = images_phys


@dataclass
class FourierVolume:
    density_fourier: Tensor
    sphere_grid: SphereGrid
    
    def __post_init__(self):
        if not len(self.density_fourier.shape) == 1:
            raise ValueError(" %% Error: Fourier volume must be a 1D array. ")


class PhysicalVolume:
    density_physical: Tensor
    voxel_size: Voxel_size_type # consider removing this member
    voxel_grid: CartesianGrid3D

    def __init__(self, density_physical: Tensor, voxel_size: Voxel_size_type):
        if not len(density_physical.shape) == 3:
            raise ValueError(" %% Error: Physical volume must be a 3D array. ")
        n_pixels = array(density_physical.shape, dtype = int)
        self.density_physical = density_physical
        self.voxel_grid = CartesianGrid3D(n_voxels = n_pixels, voxel_size = voxel_size)
