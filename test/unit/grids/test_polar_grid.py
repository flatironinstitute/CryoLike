import numpy as np
from torch import tensor, complex128, complex64, Size, ones, exp
from torch.testing import assert_close
from matplotlib import pyplot as plt
import pytest
from pytest import mark

from cryolike import Precision
from cryolike.grids.polar_grid import ArbitraryPolarQuadrature, PolarGrid
from test_polar_grid_fixtures import make_grids, integrate_functions, parameter_descriptions

parameters = make_grids()

@pytest.mark.parametrize(parameter_descriptions, parameters)
def test_polar_quadrature(n, radius_max, polar_grid, description, legend):
    integrate_functions(radius_max, polar_grid)


def test_arbitrary_quadrature_consistency():
    radius_shells = np.array([1.0, 2.0, 3.0])
    weight_shells = np.array([1.0]) # shape mismatch
    assert(radius_shells.shape != weight_shells.shape)
    with pytest.raises(ValueError, match="must match"):
        x = ArbitraryPolarQuadrature(radius_shells, weight_shells)

## TODO: add test for negative weights in _weights_arbitrary_points

## TODO: Test for manual quadrature points

## TODO: Test for more cases of integrate() fn

## TODO: Test resolution fn


# Presumably one might want to test other properties, like that it
# actually accomplishes a requested displacement, but this will do
# for now
@mark.parametrize("precision", [Precision.DOUBLE, Precision.DEFAULT])
def test_get_fourier_translation_kernel(precision):
    grid = PolarGrid(radius_max=10., dist_radii=1/4.)
    # define a 3x3 displacement grid. With default box sizes,
    # these should remain untouched after ratio conversion.
    x_disp = tensor([-.5, 0., .5, -.5, 0., .5, -.5, 0., .5])
    y_disp = tensor([-.5, -.5, -.5, 0., 0., 0., .5, .5, .5])

    result = grid.get_fourier_translation_kernel(
        x_displacements_angstrom=x_disp,
        y_displacements_angstrom=y_disp,
        precision=precision
    )
    assert result.shape == Size([9, grid.n_shells, grid.n_inplanes])
    if precision == Precision.DOUBLE:
        assert result.dtype == complex128
    else:
        assert result.dtype == complex64
    assert_close(result[4], ones([grid.n_shells, grid.n_inplanes], dtype=result.dtype, device=result.device))
    first_entry = exp(
        -2. * np.pi * 1j * \
        (tensor(grid.x_points) * -.5 + tensor(grid.y_points) * -.5)
    ).reshape(grid.n_shells, grid.n_inplanes)
    assert_close(result[0], first_entry.to(result.device).to(result.dtype))


# "debug" mark no longer necessary--prefixing it with debug_ means it won't
# be discovered as a test.
@pytest.mark.debug_test
def debug_test_polar_quadrature():
    grids = make_grids()
    last_n = None
    fig = None
    def finalize_figure(n):
        plt.legend()
        plt.title(f'n {n:d}')
        plt.savefig(f'plots/weights_test_polar_grid_n_{n:d}.png')

    for grid in grids:
        (n, radius_max, polar_grid, description, legend) = grid
        report = integrate_functions(radius_max, polar_grid)
        print(description)
        print("\n".join(report))
        if (last_n != n):
            if fig is not None:
                finalize_figure(last_n)
            last_n = n
            fig = plt.figure(figsize = (8, 8))
        assert polar_grid.weight_shells is not None
        plt.plot(polar_grid.radius_shells, polar_grid.weight_shells, 'x-', label = legend)
    finalize_figure(last_n)


if __name__ == '__main__':
    debug_test_polar_quadrature()
    print(' %% 2D polar grid quadrature test passed')
