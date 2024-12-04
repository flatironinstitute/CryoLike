import numpy as np
from matplotlib import pyplot as plt
import pytest

from cryolike.grids.polar_grid import ArbitraryPolarQuadrature
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
