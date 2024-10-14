from cryolike.util.enums import Precision
import torch
import numpy as np

from cryolike.polar_grid import PolarGrid
from cryolike.cartesian_grid import CartesianGrid2D
from cryolike.nufft import cartesian_phys_to_fourier_polar

def variance_scaling(
    phys_grid : CartesianGrid2D, # physical grid
    polar_grid : PolarGrid, # polar grid
    n_samples: int = -1, # number of samples
    precision: Precision = Precision.SINGLE,
    use_cuda = True # use CUDA
):
    if n_samples == -1:
        n_samples = polar_grid.n_shells
    noises_phys = np.random.randn(n_samples, phys_grid.n_pixels[0], phys_grid.n_pixels[1])
    noises_fourier = cartesian_phys_to_fourier_polar(
        grid_cartesian_phys = phys_grid,
        grid_fourier_polar = polar_grid,
        images_phys = noises_phys,
        eps = 1e-12,
        precision = precision,
        use_cuda = use_cuda
    )
    noises_fourier = noises_fourier.reshape((n_samples, polar_grid.n_shells, polar_grid.n_inplanes))
    variance = np.var(noises_fourier, axis = (0, 2))
    return variance