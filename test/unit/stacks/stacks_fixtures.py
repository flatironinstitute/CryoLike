import torch
import numpy as np
from unittest.mock import Mock

from cryolike.grids import CartesianGrid2D, PolarGrid
from cryolike.metadata import ViewingAngles

from cryolike.stacks import Templates
from cryolike.util import QuadratureType, to_torch, Precision

def make_image_tensor(n_im: int, d1: int, d2: int, target_fourier: bool = False):
    t = torch.arange(n_im * d1 * d2, dtype=torch.float64)
    if target_fourier:
        t = t.to(dtype=torch.complex128)
    t = t.reshape([n_im, d1, d2])
    return t


def make_mock_data_obj(img: torch.Tensor, grid: CartesianGrid2D | PolarGrid):
    ret = Mock()
    if img.dtype == torch.complex128 or img.dtype == torch.complex64:
        assert isinstance(grid, PolarGrid)
        ret.images_fourier = img
        ret.polar_grid = grid
    else:
        assert isinstance(grid, CartesianGrid2D)
        ret.images_phys = img
        ret.phys_grid = grid
    return ret


def make_mock_phys_grid(x_dim: int, y_dim: int, box_size: float):
    # ret = Mock()
    ret = CartesianGrid2D(n_pixels=[x_dim, y_dim], pixel_size=[10., 10.])
    return ret


def make_mock_polar_grid(n_shells: int = 2, n_inplanes: int = 3, uniform: bool = True):
    # ret = Mock()
    if n_shells < 2:
        raise ValueError("n_shells must be >= 2 for make_mock_polar_grid")
    if n_inplanes < 3:
        raise ValueError("n_inplanes must be >= 3 for make_mock_polar_grid")
    if not uniform:
        _n_inplanes = np.arange(3, 3 + n_shells)
    else:
        _n_inplanes = n_inplanes
    _n_shells_factor = n_shells - 1
    while True:
        ret = PolarGrid(
            radius_max = np.pi * _n_shells_factor / (2.0 * np.pi) / 2.0,
            dist_radii = np.pi / (2.0 * np.pi) / 2.0,
            uniform = uniform,
            quadrature = QuadratureType.GAUSS_JACOBI_BETA_1,
            n_inplanes = _n_inplanes
        )
        if ret.n_shells == n_shells:
            break
        if ret.n_shells < n_shells:
            _n_shells_factor += 1
        else:
            _n_shells_factor -= 1
    return ret


def make_mock_viewing_angles(n_im: int):
    azimus = torch.linspace(0, 2 * np.pi, n_im, dtype=torch.float32)
    polars = torch.linspace(0, np.pi, n_im, dtype=torch.float32)
    gammas = torch.linspace(0, 2 * np.pi, n_im, dtype=torch.float32)
    return ViewingAngles(azimus, polars, gammas)


def make_basic_Templates(
    n_im: int,
    with_cimgs: bool = False,
    with_fimgs: bool = False,
    with_physgrid: bool = False,
    with_polargrid: bool = False
) -> Templates:
    n_shells = 15
    n_inplanes = 15
    n_pixels = 15
    cgrid = None
    fgrid = None
    if with_cimgs or with_physgrid:
        cgrid = make_mock_phys_grid(n_pixels, n_pixels, 1.)
    if with_fimgs or with_polargrid:
        fgrid = make_mock_polar_grid(n_shells, n_inplanes, True)

    if with_cimgs:
        assert cgrid is not None
        phys_data = make_mock_data_obj(make_image_tensor(n_im, n_pixels, n_pixels), cgrid)
    else:
        phys_data = cgrid
    if with_fimgs:
        assert fgrid is not None
        four_data = make_mock_data_obj(
            make_image_tensor(n_im, n_shells, n_inplanes, target_fourier=True),
            fgrid
        )
    else:
        four_data = fgrid
    views = make_mock_viewing_angles(n_im)

    return Templates(phys_data=phys_data, fourier_data=four_data, viewing_angles=views)


def mock_get_fourier_slices(polar_grid: PolarGrid, viewing_angles: ViewingAngles, precision: Precision, device: torch.device):
    x = to_torch(polar_grid.x_points, precision, device)
    y = to_torch(polar_grid.y_points, precision, device)
    n_imgs = viewing_angles.n_angles
    return torch.stack((x, y, torch.zeros_like(x)), dim=1).unsqueeze(0).repeat(n_imgs, 1, 1).reshape(n_imgs, polar_grid.n_shells, polar_grid.n_inplanes, 3)
