from dataclasses import dataclass
from torch import Tensor
from numpy import array

from cryolike.util.types import Pixel_size_type
from cryolike.cartesian_grid import CartesianGrid2D
from cryolike.polar_grid import PolarGrid

@dataclass
class FourierImages:
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
    images_phys: Tensor
    phys_grid: CartesianGrid2D | None

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
