from typing import Callable, List, Optional, Tuple
from cryolike.util.types import FloatArrayType
import numpy as np
from cryolike.polar_grid import PolarGrid
from cryolike.util.enums import QuadratureType

verbose = True
tol = 1e-8
ns = [32, 64, 128, 256, 512]
dist_radii = np.pi / 2.0 / (2.0 * np.pi)
dist_inplane_default = 1.0 / (2.0 * np.pi)

parameter_descriptions = "n,radius_max,polar_grid,description,legend"

# Makes a table of algorithmically-computed quadrature point models, with their descriptions.
def make_grids() -> List[Tuple[int, float, PolarGrid, str, str]]:
    grids = []
    for n in ns:
        n_inplanes = n #* 2
        radius_max = n * np.pi / 2 / (2.0 * np.pi)
        dist_inplane = dist_inplane_default
        for (n_inplanes, uniform) in [
            (n_inplanes, True),
            (None, True),
            (None, False)
        ]:
            ## TODO: fix the negative weight issue for the shells for the beta 2 quadrature with arbitrary weights assignment
            for quadrature in [QuadratureType.GAUSS_LEGENDRE, QuadratureType.GAUSS_JACOBI_BETA_1]:#, QuadratureType.GAUSS_JACOBI_BETA_2]:
                # if (n == 320 and quadrature == QuadratureType.GAUSS_JACOBI_BETA_2):
                #     continue # this particular family of test cases produces a negative weight for the shells.
                for half_space in [True, False]:
                    (description, legend) = describe_polar_grid(n, quadrature, n_inplanes, dist_inplane=dist_inplane, half_space=half_space)
                    polar_grid = PolarGrid(radius_max, dist_radii, n_inplanes = n_inplanes, dist_inplane=dist_inplane, uniform = uniform, quadrature=quadrature, half_space=half_space)
                    grids.append((n, radius_max, polar_grid, description, legend))
    return grids

def describe_polar_grid(n: int, quadrature: QuadratureType, n_inplanes: Optional[int], dist_inplane: Optional[float], half_space: bool = False):
    is_set = lambda x: len(x) > 0

    n_txt = f" % n {n:d}"
    n_inplanes_txt = f"n_inplanes {n_inplanes:d}" if n_inplanes is not None else ''
    dist_inplane_txt = f"dist_inplane {dist_inplane:.16f}" if dist_inplane is not None else ''
    half_space_txt = "half_space" if half_space else ''
    quadrature_txt = f"quadrature {quadrature.value}" if quadrature is not None else ''
    beta_txt = "beta 1" if quadrature == QuadratureType.GAUSS_JACOBI_BETA_1 else "beta 2" if quadrature == QuadratureType.GAUSS_JACOBI_BETA_2 else ''

    desc = ", ".join(filter(is_set, [n_txt, n_inplanes_txt, dist_inplane_txt, half_space_txt, quadrature_txt, beta_txt]))
    legend = " ".join(filter(is_set, [n_inplanes_txt, quadrature_txt,  beta_txt]))
    return (desc, legend)


def integrate_functions(radius_max, polar_grid: PolarGrid):
    sigma = radius_max / 2
    fs: list[Callable[[FloatArrayType, FloatArrayType], FloatArrayType]] = [
        # The first 2 don't actually use the theta_points, but this keeps the call signature consistent.
        lambda k, w: np.ones_like(k),
        lambda k, w: -1 * (1 / sigma ** 2) * np.exp(-k ** 2 / (2 * sigma ** 2)),
        lambda k, w: -(np.sin(w) * np.cos(w) + 1) * (1 / sigma ** 2) * np.exp(-k ** 2 / (2 * sigma ** 2))
    ]
    F_trus = [
        np.pi * radius_max ** 2,
        2 * np.pi * (np.exp(-radius_max ** 2 / (2 * sigma ** 2)) - 1),
        2 * np.pi * (np.exp(-radius_max ** 2 / (2 * sigma ** 2)) - 1)
    ]
    
    report = []

    for f, F_tru in zip(fs, F_trus):
        F_est = polar_grid.integrate(f(polar_grid.radius_points, polar_grid.theta_points))
        assert len(F_est.shape) == 1
        assert F_est.shape == (1,)
        error = np.abs(F_tru - F_est) / np.abs(F_tru)
        error = error.item()
        assert error < tol, 'Error in F_tru vs F_est: %0.16f, F_tru %0.16f, F_est %0.16f' % (error, F_tru, F_est)
        report.append(f' % F_tru vs F_est: {error:.16f}')
    return report
