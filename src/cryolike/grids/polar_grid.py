import numpy as np
import torch
from scipy.special import roots_jacobi
from typing import TypeVar, Union, Optional, cast
from dataclasses import dataclass

from cryolike.util import (
    ComplexArrayType,
    ensure_positive,
    FloatArrayType,
    IntArrayType,
    QuadratureType,
)


# Pydantic might be better for this
@dataclass
class ArbitraryPolarQuadrature:
    """Data class to facilitate preselected quadrature points for polar grids.

    Args:
        radius_shells (NDArray[np.floating]): Radius of each radial shell.
        weight_shells (NDArray[np.floating], optional): Weight of each radial shell.
            Will be computed if unset.

    Raises:
        ValueError: On inconsistency between radius_shells and weight_shells, which
            are required to have the same shape if both are set.
    """
    radius_shells: FloatArrayType
    weight_shells: Optional[FloatArrayType]

    def __post_init__(self):
        if self.weight_shells is not None:
            if not (self.radius_shells.shape == self.weight_shells.shape):
                # Other checks here: presumably they're both supposed to be 1-d vectors, etc.
                raise ValueError("Quadrature radii and weights must match in shape.")


QuadratureAlias = Union[QuadratureType, ArbitraryPolarQuadrature]
T = TypeVar("T", bound = FloatArrayType | ComplexArrayType | torch.Tensor)
EmptyArray = np.array([])


class PolarGrid:
    """Class implementing polar-coordinate grid.

    Attributes:
        uniform (bool): Whether the class uses uniform quadrature points.
        radius_max (float): Maximum radius.
        dist_radii (float): Distance between two radial shells.
        radius_shells (FloatArrayType): Radius of each radial shell.
        weight_shells (FloatArrayType): Weight of each radial shell.
        n_shells (int): Total number of shells.
        n_inplanes (int): Number of points in each radial shell. Only applicable to uniform grids;
            if the shells have different point counts, see n_inplane_shells.
        n_points (int): Total number of points in the quadrature grid.
        n_inplane_shells (IntArrayType): Number of points per each radial shell.
        radius_points (FloatArrayType): Radius coordinate for each point in the list of points.
        theta_shell (FloatArrayType): For uniform grids, all shells have the same sets of angles. These
            are those angles.
        theta_shells (list[FloatArrayType]): For non-uniform grids, the angles for each shell.
        theta_points (FloatArrayType): Theta coordinate for each point in the list of points.
        weight_points (FloatArrayType): Weight for each point in the list of points.
        shell_indices (IntArrayType): Index for the point lists, identifying which shell each point
            belongs to.
            
    """
    uniform: bool
    radius_max: float
    dist_radii: float
    radius_shells: FloatArrayType
    weight_shells: FloatArrayType
    n_shells: int
    n_inplanes: int
    n_points: int
    n_inplane_shells: IntArrayType
    radius_points: FloatArrayType
    theta_shell: FloatArrayType
    theta_shells: list[FloatArrayType]
    theta_points: FloatArrayType
    weight_points: FloatArrayType
    shell_indices: IntArrayType
    x_points: Optional[FloatArrayType]
    y_points: Optional[FloatArrayType]


    def __init__(
        self,
        radius_max: float = -1.,
        dist_radii: float = -1.,
        uniform: bool = True,
        quadrature: Union[QuadratureType, ArbitraryPolarQuadrature] = QuadratureType.GAUSS_JACOBI_BETA_1,
        n_inplanes: Optional[Union[int, IntArrayType]] = None,
        dist_inplane: float = 1.0 / (2.0 * np.pi),
        return_cartesian: bool = True,
        half_space: bool = False
    ) -> None:
        """Class representing a shell-based polar grid for quadrature.

        Args:
            radius_max (float): Maximum radius.
            dist_radii (float): Distance between two radial shells.
            uniform (bool): Whether to use uniform quadrature points. Defaults to True.
            quadrature (QuadratureType | ArbitraryPolarQuadrature): One of:
                - an instance of a data class, ArbitraryPolarQuadrature, containing pre-computed quadrature points; OR
                - one of the defined quadrature-point selection algorithms (QuadratureType enum)
                Defaults to GAUSS_JACOBI_BETA_1, which provides best-guess for 2D coordinates.
                GAUSS_JACOBI_BETA_2 provides best-guess for 3D coordinates, but loses an order of accuracy.
            n_inplanes (Union[int, IntArrayType], optional): Number of points in each radial shell. Will be set
                algorithmically if undefined. If set, must be an integer (for uniform points) or an integer-dtype Numpy
                array (for non-uniform points). Defaults to None.
            dist_inplane (float, optional): Distance between two points in the same radial shell. Used for nonuniform
                grid, or for uniform grid when n_inplanes is not set. Defaults to 1/(2pi).
            return_cartesian (bool): Whether to return points in Cartesian coordinates. Defaults to True.

        Raises:
            ValueError: For invalid combinations of values (uniformity vs. n_inplanes type).
        """
        ensure_positive(radius_max, "radius_max")
        ensure_positive(dist_radii, "dist_radii")
        ensure_positive(dist_inplane, "dist_inplane")
        self.uniform = uniform
        self.radius_max = radius_max
        self.dist_radii = dist_radii
        self.half_space = half_space
        self._set_radius_and_weight_shells(quadrature)

        if uniform:
            if (isinstance(n_inplanes, np.ndarray)):
                if n_inplanes.size == 1:
                    n_inplanes = int(n_inplanes)
                else:
                    raise ValueError('Array-valued n_inplanes is not allowed for uniform quadrature')
            self._do_uniform_initialization( n_inplanes=n_inplanes, dist_inplane=dist_inplane)
        else:
            if (n_inplanes is not None and not isinstance(n_inplanes, np.ndarray)):
                raise ValueError('n_inplanes must be array-valued for non-uniform quadrature')
            self._do_nonuniform_initialization(n_inplanes=n_inplanes, dist_inplane=dist_inplane)
        if return_cartesian:
            self.x_points = self.radius_points * np.cos(self.theta_points)
            self.y_points = self.radius_points * np.sin(self.theta_points)


    def _set_radius_and_weight_shells(self, quadrature: QuadratureAlias):
        if (isinstance(quadrature, ArbitraryPolarQuadrature)):
            self.radius_shells = quadrature.radius_shells
            if quadrature.weight_shells is not None:
                self.weight_shells = quadrature.weight_shells
            self.n_shells = len(quadrature.radius_shells)
        else:
            if quadrature == QuadratureType.GAUSS_LEGENDRE:
                self._gauss_legendre()
            elif quadrature == QuadratureType.GAUSS_JACOBI_BETA_1:
                self._gauss_jacobi(1)
            elif quadrature == QuadratureType.GAUSS_JACOBI_BETA_2:
                self._gauss_jacobi(2)
            else:
                raise ValueError("Unreachable: Unrecognized quadrature point selection algorithm")


    def _gauss_jacobi(self, beta: int = 1):
        self.n_shells = 1 + int(np.ceil(self.radius_max / self.dist_radii))
        jac_points, jac_weights = roots_jacobi(n = self.n_shells, alpha = 0, beta = beta)  # roots of Jacobi polynomial and weights
        self.radius_shells = (jac_points + 1.0) * self.radius_max / 2  # radius of each radial shell
        if beta == 1:
            self.weight_shells = jac_weights * (2.0 * np.pi) * (self.radius_max / 2.0) ** 2 / (2.0 * np.pi) ** 2
        elif beta == 2:
            # self.weight_shells = jac_weights * (2.0 * np.pi / self.radius_shells) * (self.radius_max / 2.0) ** 3 / (2.0 * np.pi) ** 2
            self.weight_shells = EmptyArray
            pass
        else:
            raise ValueError('Invalid beta value, should be 1 or 2')


    def _gauss_legendre(self):
        self.n_shells = 1 + int(np.ceil(self.radius_max / self.dist_radii))
        legg_points, legg_weights = np.polynomial.legendre.leggauss(self.n_shells)
        self.radius_shells = (legg_points + 1.0) * self.radius_max / 2.0
        self.weight_shells = legg_weights * (2.0 * np.pi * self.radius_shells) * self.radius_max / 2.0 / (2.0 * np.pi) ** 2


    def _do_uniform_initialization(self, *, n_inplanes: Optional[Union[float, int]], dist_inplane: float):
        self._set_n_inplanes_uniform(dist_inplane=dist_inplane, n_inplanes=n_inplanes)
        self.n_points = self.n_shells * self.n_inplanes
        self.n_inplane_shells = np.ones(self.n_shells, dtype = int) * self.n_inplanes
        self.radius_points = np.repeat(self.radius_shells, self.n_inplanes)
        if self.half_space:
            self.theta_shell = np.linspace(0, np.pi, self.n_inplanes, endpoint = False)
        else:
            self.theta_shell = np.linspace(0, 2 * np.pi, self.n_inplanes, endpoint = False)
        self.theta_points = np.tile(self.theta_shell, int(self.n_shells))
        # if (self.radius_shells is not None and weight_shells is None) or quadrature is None:
        if len(self.weight_shells) == 0:
            self._weights_arbitrary_points()
        self.weight_points = np.repeat(self.weight_shells, self.n_inplanes) / self.n_inplanes


    def _set_n_inplanes_uniform(self, *, n_inplanes: Optional[Union[float, int]], dist_inplane: float):
        if (n_inplanes is None):
            n_equator: int = 3 + np.round(2 * np.pi * self.radius_max / dist_inplane).astype(int)
            n_polar_a: int = 3 + np.round(n_equator / 2).astype(int)
            self.n_inplanes = 2 * n_polar_a
        else:
            _n_inplanes = int(n_inplanes)
            ensure_positive(_n_inplanes, "n_inplanes")
            self.n_inplanes = _n_inplanes


    def _do_nonuniform_initialization(self, *, n_inplanes: Optional[IntArrayType], dist_inplane: float):
        if n_inplanes is not None:
            self.n_inplane_shells = cast(IntArrayType, n_inplanes.astype(int))
        self._set_n_inplanes_nonuniform(n_inplanes=n_inplanes, dist_inplane=dist_inplane)
        # TODO: nonuniform initialization is not fully implemented yet

        # TODO QUERY: where does self.n_inplane_shells come from if n_inplanes is None?
        # int() is pleonastic but keeps pylance happy
        self.n_points = int(np.sum(self.n_inplane_shells))
        if len(self.weight_shells) == 0:
            self._weights_arbitrary_points()
        self.radius_points = np.zeros(self.n_points, dtype = float)
        self.theta_points = np.zeros(self.n_points, dtype = float)
        self.weight_points = np.zeros(self.n_points, dtype = float)
        self.theta_shells = []
        self.shell_indices = np.zeros(self.n_points, dtype = int)
        i_point = 0
        for i_s in range(self.n_shells):
            n_inplanes = self.n_inplane_shells[i_s]
            assert(n_inplanes is not None)
            radius_shell = self.radius_shells[i_s]
            # TODO: Reinstate this; we need a definite value for n_inplanes
            if self.half_space:
                theta_shell = np.linspace(0, np.pi, n_inplanes, endpoint = False)
            else:
                theta_shell = np.linspace(0, 2 * np.pi, n_inplanes, endpoint = False)
            self.theta_shells.append(theta_shell)
            self.radius_points[i_point : i_point + n_inplanes] = radius_shell
            self.theta_points[i_point : i_point + n_inplanes] = theta_shell
            self.weight_points[i_point : i_point + n_inplanes] = self.weight_shells[i_s] / n_inplanes
            self.shell_indices[i_point : i_point + n_inplanes] = i_s
            i_point += n_inplanes
    

    def _set_n_inplanes_nonuniform(self, *, n_inplanes: Optional[IntArrayType], dist_inplane: float):
        if n_inplanes is not None:
            if not np.issubdtype(n_inplanes.dtype, np.integer):
                raise ValueError('n_inplanes must be integer array for nonuniform grid')  
            self.n_inplane_shells = cast(IntArrayType, n_inplanes.astype(int))
        self._nonuniform_inplane(dist_inplane)


    def _nonuniform_inplane(self, dist_inplane: float = 1.0 / (2.0 * np.pi)):
        ## QUERY: This next line resets any setting of self.n_inplane_shells that happened previously
        ## (in the branch where n_inplanes is not None in set_n_inplanes_nonuniform.)
        self.n_inplane_shells = cast(IntArrayType, np.zeros(self.n_shells, dtype = int))
        for i_r in range(self.n_shells):
            n_equator = 3 + np.round(2 * np.pi * self.radius_shells[i_r] / dist_inplane).astype(int)
            n_polar_a = 3 + np.round(n_equator / 2).astype(int)
            self.n_inplane_shells[i_r] = 2 * n_polar_a


    ### Instability for certain radius_max
    def _weights_arbitrary_points(self): 
        tmp_P_ = np.zeros((self.n_shells, self.n_shells), dtype=float) #<-- polynomials of order 0:n_r-1 evaluated on r_shell_/radius_max. ;
        tmp_I_ = 2.0 * np.pi / np.arange(2, self.n_shells + 2)
        tmp_P_[0, :] = 1.0
        tmp_r_ = self.radius_shells / self.radius_max
        for i_r in range(1, self.n_shells):
            tmp_P_[i_r, :] = tmp_P_[i_r - 1, :] * tmp_r_
        tmp_W_, residual, rank, s = np.linalg.lstsq(tmp_P_, tmp_I_, rcond=None)
        self.weight_shells = tmp_W_ * self.radius_max ** 2 / (2 * np.pi) ** 2
        if np.any(self.weight_shells < 0.0):
            raise ValueError('Negative weights')
        self.weight_points = cast(FloatArrayType, np.zeros(self.n_points, dtype=float))
        i_point = 0
        for i_r in range(self.n_shells):
            n_inplane = self.n_inplane_shells[i_r]
            self.weight_points[i_point : i_point + n_inplane] = self.weight_shells[i_r] / max(1, n_inplane)


    def integrate(self, f: T) -> T:
        if not hasattr(f, '__len__'):
            raise ValueError('Invalid type for f', type(f))
        if len(f.shape) > 1:
            if f.shape[-2] == self.n_shells and f.shape[-1] == self.n_inplanes:
                f = f.reshape(-1, self.n_points) # type: ignore
            if f.shape[-1] != self.n_points:
                raise ValueError('Invalid shape for f', f.shape)
        else:
            if f.shape[0] != self.n_points: # type: ignore
                raise ValueError('Invalid shape for f', f.shape)
        if isinstance(f, np.ndarray):
            weight_points_tmp = np.expand_dims(self.weight_points, axis = tuple(range(f.ndim - 1)))
            I = cast(T, np.sum(f * weight_points_tmp, axis = -1) * (2 * np.pi) ** 2)
            if I.size == 1:
                return cast(T, np.atleast_1d(I))
            else:
                return cast(T, I)
        elif isinstance(f, torch.Tensor):
            weight_points_torch = torch.tensor(self.weight_points, dtype = f.dtype, device = f.device)
            for i in range(f.dim() - 1):
                weight_points_torch = weight_points_torch.unsqueeze(0)
            I = torch.sum(f * weight_points_torch, dim = -1) * (2 * np.pi) ** 2
            if I.size == 1:
                return torch.atleast_1d(I)
            else:
                return cast(T, I)
    

    def resolution(self, box_size):
        return box_size / (self.radius_shells * 2.0)
