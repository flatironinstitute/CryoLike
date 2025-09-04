from typing import Optional, NamedTuple
import numpy as np
from scipy.special import roots_legendre, roots_jacobi
from copy import copy
from torch import tensor, Tensor, cuda
from math import ceil

from cryolike.util import SamplingStrategy, ensure_positive_finite, FloatArrayType, IntArrayType


class CartesianShell(NamedTuple):
    """Collection of Cartesian points for a sphere or spherical shell.

    Attributes:
        x_points (FloatArrayType): x-positions of the points.
        y_points (FloatArrayType): y-positions of the points.
        z_points (FloatArrayType): z-positions of the points.
        xyz_points (FloatArrayType): arrays of x, y, z positions for each point. TODO: CONFIRM
    """
    x_points: FloatArrayType
    y_points: FloatArrayType
    z_points: FloatArrayType
    xyz_points: FloatArrayType


class SphereShell:
    """Class to sample points on a sphere shell of known radius r.

    Attributes:
        radius (float): Radius of the sphere in TODO:UNITS. Positive finite.
        dist_eq (float): Distance between points at the equator. Positive finite.
        azimuthal_sampling (SamplingStrategy): Whether the grid is uniform or adaptive.
        n_points (int): number of points
        n_polar_circles (int): number of polar circles
        polar_circles (FloatArrayType): polar angles at each polar circle
        weights_polar (FloatArrayType): weight at each point
        n_azimus_each_circle (IntArrayType): number of azimuthal points at each polar circle
        azimu_points (FloatArrayType): azimuthal angles at each point
        polar_points (FloatArrayType): polar angles at each point
        weight_points (FloatArrayType): weight at each point
        cartesian_points (Optional[CartesianShell]): If set, a NamedTuple of four float arrays
            (x_points, y_points, z_points, xyz_points) defining cartesian coordinates of each
            point.
    """
    radius: float
    dist_eq: float
    azimuthal_sampling: SamplingStrategy # consider removing from class
    n_points: int
    n_polar_circles: int
    polar_circles: FloatArrayType
    weights_polar: FloatArrayType
    n_azimus_each_circle: IntArrayType
    azimu_points: FloatArrayType
    polar_points: FloatArrayType
    weight_points: FloatArrayType
    cartesian_points: Optional[CartesianShell]

    def __init__(self,
        radius: float = 1.0,
        dist_eq: float = 1.0 / (2.0 * np.pi),
        azimuthal_sampling: SamplingStrategy = SamplingStrategy.UNIFORM,
        compute_cartesian: bool = True,
    ) -> None:
        """Class to sample points on a sphere shell of known radius r.

        Args:
            radius (float, optional): Radius of the sphere in TODO:UNITS. Must be positive finite. Defaults to 1.0.
            dist_eq (float, optional): Distance between points at equator. Must be positive finite.
                Defaults to 1.0/(2.0 * np.pi).
            azimuthal_sampling (SamplingStrategy, optional): Enum indicating whether to use uniform sampling
                (the default) or adaptive sampling (TODO: ADAPTIVE HOW?).
            compute_cartesian (bool, optional): Whether to populate a cartesian grid for the sphere. If True,
                the resulting shell will have a cartesian_points member set. Defaults to True.
        """
        ensure_positive_finite(radius, "spherical shell radius")
        ensure_positive_finite(dist_eq, "spherical shell equatorial point distance dist_eq")
        self.radius = radius
        self.dist_eq = dist_eq
        self.azimuthal_sampling = azimuthal_sampling
        self._build_shell(azimuthal_sampling, compute_cartesian)


    def _build_shell(self, azimuthal_sampling: SamplingStrategy, compute_cartesian: bool = True):
        self._build_circles(azimuthal_sampling)

        self.n_points = np.sum(self.n_azimus_each_circle).item()
        self.azimu_points = np.zeros(self.n_points, dtype = np.float64)
        self.polar_points = np.zeros(self.n_points, dtype = np.float64)
        self.weight_points = np.zeros(self.n_points, dtype = np.float64)
        radius_sq = self.radius ** 2
        i_point = 0
        # TODO: Is there a more elegant way to do this without explicit loop
        for i, n_azimus in enumerate(self.n_azimus_each_circle):
            self._compute_circle_azimus(i, n_azimus, i_point, radius_sq)
            i_point += n_azimus
        if compute_cartesian:
            self._calc_cartesian()
    

    def _build_circles(self, azimuthal_sampling: SamplingStrategy):
        angular_frequency = 2. * np.pi * self.radius / self.dist_eq
        n_point_eq = 3 + int(round(angular_frequency))  # number of points on equator
        self.n_polar_circles = 3 + ceil(n_point_eq / 2)
        lgnd_nodes, lgnd_weights = roots_legendre(self.n_polar_circles)
        self.polar_circles = np.arccos(lgnd_nodes)   # polar angle of each polar circle
        self.weights_polar = lgnd_weights
        
        n_azimu_max = 3 + int(round(angular_frequency))
        if azimuthal_sampling == SamplingStrategy.UNIFORM:
            self.n_azimus_each_circle = np.ones(self.n_polar_circles, dtype=int) * n_azimu_max
        elif azimuthal_sampling == SamplingStrategy.ADAPTIVE:
            sin_polar_circles = np.sin(self.polar_circles)
            self.n_azimus_each_circle = 3 + np.round(angular_frequency * sin_polar_circles).astype(int)
        else:
            raise ValueError('Unreachable: Unsupported value for azimuthal_sampling')


    def _compute_circle_azimus(self, i: int, n_azimus: int, i_point: int, radius_sq: float):
        azimu_per_polar = np.linspace(0, 2 * np.pi, n_azimus, endpoint = False)
        d_azimu = azimu_per_polar[1] - azimu_per_polar[0]
        self.azimu_points[i_point : i_point + n_azimus] = azimu_per_polar
        self.polar_points[i_point : i_point + n_azimus] = self.polar_circles[i]
        self.weight_points[i_point : i_point + n_azimus] = self.weights_polar[i] * radius_sq * d_azimu  ## integrate to 4pir^2


    def _calc_cartesian(self):
        cos_azimu_ = np.cos(self.azimu_points)
        sin_azimu_ = np.sin(self.azimu_points)
        cos_polar_ = np.cos(self.polar_points)
        sin_polar_ = np.sin(self.polar_points)

        x_points = self.radius * cos_azimu_ * sin_polar_
        y_points = self.radius * sin_azimu_ * sin_polar_
        z_points = self.radius * cos_polar_
        xyz_points = np.array([x_points, y_points, z_points]).T
        self.cartesian_points = CartesianShell(x_points, y_points, z_points, xyz_points)
    

    def integrate(self,
        f: FloatArrayType | Tensor, # functional values to integrate
        use_torch : bool = False,
        device = 'cpu'
    ):            
        # integrate function f over the sphere shell
        if f.shape[-1] != self.n_points:
            raise ValueError(' %% error: f.shape[0] != self.n_points')
        if use_torch:
            assert type(f) == Tensor
            if not cuda.is_available() and device == 'cuda':
                print(' %% warning: device cuda not available, using cpu instead')
                device = 'cpu'
            # if not hasattr(self, "weight_torch_"):
            #     self.weight_torch_ = torch.tensor(self.weight_points, dtype = torch.float64, device = device)
            return f.to(device) @ tensor(self.weight_points, dtype = f.dtype, device = device)
        else:
            return np.sum(f * self.weight_points, axis = -1)


    def copy(self):
        shell = copy(self)
        shell.polar_circles = self.polar_circles.copy()
        shell.weight_points = self.weight_points.copy()
        shell.n_azimus_each_circle = self.n_azimus_each_circle.copy()
        shell.azimu_points = self.azimu_points.copy()
        shell.polar_points = self.polar_points.copy()
        if self.cartesian_points is not None:
            (x_pts, y_pts, z_pts, xyz_pts) = self.cartesian_points
            shell.cartesian_points = CartesianShell(
                x_pts.copy(),
                y_pts.copy(),
                z_pts.copy(),
                xyz_pts.copy()
            )

        return shell


class SphereGrid:
    """Class to sample points on a spherical grid comprised of shells.

    Attributes
        radius_max (float): Maximum radius of the sphere
        dist_eq (float): minimum distance between two points at the equator
        equal_shell (bool): if True, all shells have the same angular points
        dist_type (SamplingStrategy): 'Uniform' or 'Adaptive' distance between points for various shells
        azimuthal_sampling (SamplingStrategy): 'Uniform' or 'adaptive' aziumthal sampling for polar circles
        n_shells (int): number of radial shells
        radius_shells (FloatArrayType): radius of each radial shell
        weights_radius_shells (FloatArrayType): weight of each radial shell
        n_points (int): total number of points
        radius_points (FloatArrayType): radius of each point
        polar_points (FloatArrayType): polar angle of each point
        azimu_points: (FloatArrayType): azimuthal angle of each point
        weight_points (FloatArrayType): weight of each point
        shells list[SphereShell(]): list of radial shells
        point_shell_start_indices (IntArrayType): start index of each shell within the point list
        cartesian_points (Optional[CartesianShell]): If set, a NamedTuple of four float arrays
            (x_points, y_points, z_points, xyz_points) defining cartesian coordinates of each
            point
    """
    radius_max: float
    dist_eq: float
    equal_shell: bool
    dist_type: SamplingStrategy
    azimuthal_sampling: SamplingStrategy
    n_shells: int
    radius_shells: FloatArrayType
    weights_radius_shells: FloatArrayType
    n_points: int
    radius_points: FloatArrayType
    polar_points: FloatArrayType
    azimu_points: FloatArrayType
    weight_points: FloatArrayType
    shells: list[SphereShell]
    point_shell_start_indices: IntArrayType
    cartesian_points: Optional[CartesianShell]

    def __init__(self,
        radius_max: float = 1.0,
        dist_eq: float = 1.0 / (2.0 * np.pi),
        azimuthal_sampling: SamplingStrategy = SamplingStrategy.UNIFORM,
        equal_shell: bool = False,
        dist_type: SamplingStrategy = SamplingStrategy.UNIFORM,
    ) -> None:
        """Class to sample points on a spherical grid comprised of shells.

        Args:
            radius_max (float, optional): maximum radius of the sphere. Defaults to 1.0.
            dist_eq (float, optional): minimum distance between two points at the equator. Defaults to 1.0/(2.0 * np.pi).
            azimuthal_sampling (SamplingStrategy, optional): 'uniform' (default) or 'adaptive' azimuthal sampling
                for various polar circles
            equal_shell (bool, optional): if True, all shells have the same angular points. Defaults to False.
            dist_type (SamplingStrategy, optional): 'uniform' (default) or 'adaptive' distance between
                 points for various shells.
        """
        ensure_positive_finite(radius_max, "maximum spherical radius")
        ensure_positive_finite(dist_eq, "minimum equatorial distance")
        self.radius_max = radius_max
        self.dist_eq = dist_eq
        self.dist_type = dist_type
        self.azimuthal_sampling = azimuthal_sampling
        self._sample_radii()
        self._init_grid()    # TODO: Can this just be skipped entirely?
        if equal_shell:
            self._build_grid_equal_shells()
        else:
            self._build_grid_varied_shells()
        self._calc_cartesian()
        

    def _sample_radii(self) -> None :
        self.n_shells = 1 + int(np.ceil(self.radius_max / self.dist_eq))          # number of radial shells
        points_jacobi, weights_jacobi = roots_jacobi(self.n_shells, 0, 2)         # Gauss-Jacobi quadrature.
        self.radius_shells = (points_jacobi + 1.0) * self.radius_max / 2          # radius of each radial shell
        self.weights_radius_shells = weights_jacobi * (self.radius_max / 2) ** 3  # weight of each radial shell


    def _init_grid(self):
        self.radius_points = np.array([])
        self.polar_points = np.array([])
        self.azimu_points = np.array([])
        self.weight_points = np.array([])
        self.shells = []
        self.point_shell_start_indices = np.zeros(self.n_shells + 1, dtype=int)
        self.cartesian_points = None


    # TODO: This could be combined fairly straightforwardly with the _varied_shells
    # version, by setting the appropriate values of radius_shells, dist_eq_shells.
    # Maybe a couple other variables. That seems worth doing to simplify the code...
    def _build_grid_equal_shells(self):
        shell: SphereShell = SphereShell(radius = 1.0, dist_eq = self.dist_eq / self.radius_max,
            azimuthal_sampling = self.azimuthal_sampling)

        for i_s in range(self.n_shells):
            tmp_shell = shell.copy()
            tmp_shell.radius = self.radius_shells[i_s]
            tmp_shell.weight_points *= self.radius_shells[i_s] ** 2
            if self.cartesian_points is not None and tmp_shell.cartesian_points is not None:
                (x_pts, y_pts, z_pts, xyz_pts) = tmp_shell.cartesian_points
                x_pts *= self.radius_shells[i_s]
                y_pts *= self.radius_shells[i_s]
                z_pts *= self.radius_shells[i_s]
                xyz_pts *= self.radius_shells[i_s]
                tmp_shell.cartesian_points = CartesianShell(x_pts, y_pts, z_pts, xyz_pts)
            self.weight_points = np.append(self.weight_points, tmp_shell.weight_points * self.weights_radius_shells[i_s] / self.radius_shells[i_s] ** 2)
            self.shells.append(tmp_shell)
        self.n_points = self.n_shells * shell.n_points
        self.radius_points = np.repeat(self.radius_shells, shell.n_points)
        self.polar_points = np.tile(shell.polar_points, self.n_shells)
        self.azimu_points = np.tile(shell.azimu_points, self.n_shells)
        self.point_shell_start_indices = np.arange(self.n_shells + 1) * shell.n_points


    def _build_grid_varied_shells(self):
        self.n_points = 0
        dist_eq_shells = np.zeros(self.n_shells, dtype=float)
        if self.dist_type == SamplingStrategy.UNIFORM:
            dist_eq_shells[:] = self.dist_eq
        elif self.dist_type == SamplingStrategy.ADAPTIVE:
            dist_eq_shells[:] = self.dist_eq * self.radius_shells / self.radius_max
        
        for i_s in range(self.n_shells):
            shell = SphereShell(radius = self.radius_shells[i_s], dist_eq = dist_eq_shells[i_s],
                azimuthal_sampling = self.azimuthal_sampling)
            self.shells.append(shell)
            
            self.point_shell_start_indices[i_s] = self.n_points
            self.n_points += shell.n_points
            self.radius_points = np.append(self.radius_points, shell.radius * np.ones(shell.n_points))
            self.polar_points = np.append(self.polar_points, shell.polar_points)
            self.azimu_points = np.append(self.azimu_points, shell.azimu_points)
            self.weight_points = np.append(self.weight_points, shell.weight_points * self.weights_radius_shells[i_s] / self.radius_shells[i_s] ** 2)

        self.point_shell_start_indices[-1] = self.n_points


    def _calc_cartesian(self):
        x_points = np.array([])
        y_points = np.array([])
        z_points = np.array([])
        xyz_points = np.array([])
        for shell in self.shells:
            if shell.cartesian_points is None:
                shell._calc_cartesian()
            assert shell.cartesian_points is not None
            (x_pts, y_pts, z_pts, xyz_pts) = shell.cartesian_points
            x_points = np.append(x_points, x_pts)
            y_points = np.append(y_points, y_pts)
            z_points = np.append(z_points, z_pts)
            if xyz_points.size == 0:
                xyz_points = xyz_pts
            else:
                xyz_points = np.vstack((xyz_points, xyz_pts))
        self.cartesian_points = CartesianShell(x_points, y_points, z_points, xyz_points)


    def integrate(self,
        f: FloatArrayType | Tensor, # functional values to integrate
        type_integration = "standard", # "standard" or "riesz"
        use_torch : bool = False,
        device = 'cpu'
    ):
        # integrate function f over the sphere shell
        if f.shape[-1] != self.n_points:
            raise ValueError(' %% error: f.shape[0] != self.n_points')
        if use_torch:
            if not cuda.is_available() and device == 'cuda':
                print(' %% warning: device cuda not available, using cpu instead')
                device = 'cpu'
            # if type(f) != torch.Tensor:
            #     f = torch.tensor(f, dtype = torch.float64, device = device)
            assert type(f) == Tensor
            if type_integration == "standard":
                return f.to(device) @ tensor(self.weight_points, dtype = f.dtype, device = device)
            elif type_integration == "riesz":
                return f.to(device) @ tensor(self.weight_points / self.radius_points, dtype = f.dtype, device = device)
        else:
            if type_integration == "standard":
                return np.sum(f * self.weight_points, axis = -1)
            elif type_integration == "riesz":
                if not hasattr(self, "weight_riesz_points"):
                    self.weight_riesz_points = self.weight_points / self.radius_points
                return np.sum(f * self.weight_riesz_points, axis = -1)
