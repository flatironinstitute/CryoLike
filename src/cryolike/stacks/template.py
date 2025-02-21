from typing import Callable, NamedTuple, Optional, cast
import numpy as np
import torch
from math import ceil

from cryolike.grids import (
    CartesianGrid2D,
    FourierImages,
    PhysicalImages,
    PolarGrid,
    Volume
)

from cryolike.microscopy import (
    CTF,
    volume_phys_to_fourier_points,
)

from .image import Images

from cryolike.metadata import (
    ViewingAngles
)

from cryolike.util import (
    AtomShape,
    AtomicModel,
    check_cuda,
    FloatArrayType,
    Precision,
    project_descriptor,
    TargetType,
)


# Nobody is asking for this (yet)
# def read_template_pt(filename: str) -> torch.Tensor:
#     if not filename.endswith('.pt'):
#         raise ValueError(f'Template file name {filename} must end with .pt extension.')
#     return torch.load(filename)

            
def _fourier_circles(
    thetas: torch.Tensor,  # inplane angle in radians
    polars: torch.Tensor,  # polars angle in radians
    azimus: torch.Tensor,  # azimusthal angle in radians
    gammas: torch.Tensor,  # inplane rotation in radians
) -> torch.Tensor:
    
    thetas = thetas[None,:] + gammas[:,None]
    
    cos_thetas = torch.cos(thetas)
    sin_thetas = torch.sin(thetas)

    cos_polars = torch.cos(polars)
    sin_polars = torch.sin(polars)
    cos_azimus = torch.cos(azimus)
    sin_azimus = torch.sin(azimus)

    cos_thetas_cos_polars = cos_thetas * cos_polars[:,None]
    sin_thetas_sin_azimus = sin_thetas * sin_azimus[:,None]
    sin_thetas_cos_azimus = sin_thetas * cos_azimus[:,None]

    x_template_points = cos_azimus[:,None] * cos_thetas_cos_polars - sin_thetas_sin_azimus
    y_template_points = sin_azimus[:,None] * cos_thetas_cos_polars + sin_thetas_cos_azimus
    z_template_points = - sin_polars[:,None] * cos_thetas
    xyz_template_points = torch.stack((x_template_points, y_template_points, z_template_points), dim = 2)
    
    return xyz_template_points


def _get_circles(viewing_angles: ViewingAngles, polar_grid: PolarGrid, float_type: torch.dtype, device: torch.device):
    azimus = viewing_angles.azimus.to(device)
    polars = viewing_angles.polars.to(device)
    gammas = viewing_angles.gammas.to(device)
    thetas = polar_grid.theta_shell if polar_grid.uniform else polar_grid.theta_points
    thetas = torch.tensor(thetas, dtype = float_type, device = device)
    circles = _fourier_circles(thetas, polars, azimus, gammas)
    return circles


def _get_offset(polar_grid: PolarGrid, float_type: torch.dtype, device: torch.device) -> torch.Tensor:
    x_points = torch.tensor(polar_grid.x_points, dtype = float_type, device = device)
    y_points = torch.tensor(polar_grid.y_points, dtype = float_type, device = device)
    offset = torch.sinc(2.0 * x_points) * torch.sinc(2.0 * y_points)
    return offset


class _ParsedAtomicModel(NamedTuple):
    atomic_radius_scaled: torch.Tensor
    radius_shells: torch.Tensor
    radius_shells_sq: torch.Tensor
    pi_atomic_radius_sq_times_two: torch.Tensor
    atomic_coordinates_scaled: torch.Tensor


def _parse_atomic_model(atomic_model: AtomicModel, polar_grid: PolarGrid, box_size: FloatArrayType, float_type: torch.dtype, device: torch.device) -> _ParsedAtomicModel:
    atomic_coordinates = torch.tensor(atomic_model.atomic_coordinates, dtype=float_type, device=device)
    box_max = np.amax(box_size)
    ## TODO: handle anisotropic box sizes
    atomic_coordinates_scaled = atomic_coordinates.T / box_max * 2.0 * (- 2.0 * np.pi)

    atomic_radii = torch.tensor(atomic_model.atom_radii, dtype=float_type, device=device)
    atomic_radius_scaled = atomic_radii / box_max * 2.0
    pi_atomic_radius_sq_times_two = 2.0 * (np.pi * atomic_radius_scaled) ** 2
    radius_shells = torch.tensor(polar_grid.radius_shells, dtype = float_type, device = device)
    radius_shells_sq = radius_shells ** 2
    return _ParsedAtomicModel(atomic_radius_scaled, radius_shells, radius_shells_sq, pi_atomic_radius_sq_times_two, atomic_coordinates_scaled)


# TODO: Consider whether DataClass is better for this application
class _CommonKernelParams(NamedTuple):
    xyz_template_points: torch.Tensor
    parsed_model: _ParsedAtomicModel
    templates_fourier: torch.Tensor  # TODO: Ensure this is a pointer
    polar_grid: PolarGrid
    n_atoms: int
    n_templates: int
    device: torch.device
    torch_float_type: torch.dtype
    torch_complex_type: torch.dtype


def _get_shared_kernel_params(
    atomic_model : AtomicModel,
    viewing_angles: ViewingAngles,
    polar_grid: PolarGrid,
    box_size: FloatArrayType,
    precision: Precision = Precision.DEFAULT,
    use_cuda: bool = True,
) -> _CommonKernelParams:
    (torch_float_type, torch_complex_type, _) = precision.get_dtypes(default=Precision.SINGLE)
    device = check_cuda(use_cuda)
    n_templates = viewing_angles.n_angles
    parsed_model = _parse_atomic_model(
        atomic_model,
        polar_grid,
        box_size,
        float_type=torch_float_type,
        device=device
    )
    xyz_template_points = _get_circles(viewing_angles, polar_grid, torch_float_type, device)
    if polar_grid.uniform:
        templates_fourier = torch.zeros((n_templates, polar_grid.n_shells, polar_grid.n_inplanes), dtype=torch_complex_type, device="cpu")
    else:
        templates_fourier = torch.zeros((n_templates, polar_grid.n_points), dtype = torch_complex_type, device = "cpu")

    return _CommonKernelParams(
        xyz_template_points,
        parsed_model,
        templates_fourier,
        polar_grid,
        atomic_model.n_atoms,
        viewing_angles.n_angles,
        device,
        torch_float_type,
        torch_complex_type
    )


def _make_uniform_hard_sphere_kernel(
    params: _CommonKernelParams,
):
    kR = 2 * np.pi * params.parsed_model.radius_shells[:,None] * params.parsed_model.atomic_radius_scaled[None,:] ## (n_shells, n_atoms)
    kernelAtoms = (torch.sin(kR) - kR * torch.cos(kR)) * params.parsed_model.radius_shells.pow(-3)[:,None] / (8 * np.pi ** 2) * 3 / params.n_atoms
    # offset = np.pi * parsed_model.atomic_radius_scaled.pow(3).sum() / atomic_model.n_atoms
    
    # offset = offset.reshape(polar_grid.n_shells, polar_grid.n_inplanes).unsqueeze(0)
    def _uniform_kernel(start: int, end: int):
        kdotr_batch = torch.matmul(params.xyz_template_points[start:end,:,:], params.parsed_model.atomic_coordinates_scaled) ## uniform (n_templates, n_inplanes, n_atoms) or nonuniform (n_templates, n_points, n_atoms)
        kdotr_batch = kdotr_batch[:,None,:,:] * params.parsed_model.radius_shells[None,:,None,None] ## (n_templates, n_shells, n_inplanes, n_atoms)
        exponent = torch.exp(1j * kdotr_batch) * kernelAtoms[None,:,None,:] ## (n_templates, n_shells, n_inplanes, n_atoms)
        # exponent.exp_()
        # exponent = kernelAtoms[None,:,None,:]
        templates_fourier_batch = torch.sum(exponent, dim = 3)
        params.templates_fourier[start:end,:,:] = templates_fourier_batch.cpu()
    return _uniform_kernel


def _make_uniform_gaussian_kernel(
    params: _CommonKernelParams,
):
    log_norm = - 1.5 * np.log(2 * np.pi) - 3 * torch.log(params.parsed_model.atomic_radius_scaled) - np.log(params.n_atoms)
    offset = _get_offset(params.polar_grid, float_type=params.torch_float_type, device=params.device)
    offset *= torch.sum(torch.exp(log_norm))
    offset = offset.reshape(params.polar_grid.n_shells, params.polar_grid.n_inplanes).unsqueeze(0)
    ## Gaussian kernel
    gaussKernelAtoms = - params.parsed_model.radius_shells_sq[:,None] * params.parsed_model.pi_atomic_radius_sq_times_two[None,:] + log_norm[None,:]

    def _uniform_kernel(start: int, end: int):
        kdotr_batch = torch.matmul(params.xyz_template_points[start:end,:,:], params.parsed_model.atomic_coordinates_scaled) ## uniform (n_templates, n_inplanes, n_atoms) or nonuniform (n_templates, n_points, n_atoms)
        kdotr_batch = kdotr_batch[:,None,:,:] * params.parsed_model.radius_shells[None,:,None,None] ## (n_templates, n_inplanes, n_shells, n_atoms)
        exponent = torch.complex(gaussKernelAtoms[None,:,None,:], kdotr_batch) ## (n_templates, n_inplanes, n_shells, n_atoms)
        exponent.exp_()
        templates_fourier_batch = torch.sum(exponent, dim = 3) - offset
        params.templates_fourier[start:end,:,:] = templates_fourier_batch.cpu()
    return _uniform_kernel


# TODO: On the pattern of the others, this should just return the kernel
def _generate_templates_from_nonuniform_positions_gaussian(
    kernel_params: _CommonKernelParams,
    verbose : bool = False
) -> torch.Tensor:
    raise NotImplementedError("kdtor is not set properly.")
    parsed_model = kernel_params.parsed_model
    polar_grid = kernel_params.polar_grid
    log_norm = - 1.5 * np.log(2 * np.pi) - 3 * torch.log(parsed_model.atomic_radius_scaled) - np.log(kernel_params.n_atoms)
    offset = _get_offset(polar_grid, float_type=kernel_params.torch_float_type, device=kernel_params.device)
    offset *=  torch.sum(torch.exp(log_norm))

    # TODO: Check this signature, is it right?
    templates_fourier = torch.zeros((kernel_params.n_templates, polar_grid.n_points), dtype = kernel_params.torch_complex_type, device = "cpu")
    gaussKernelAtoms = parsed_model.radius_shells_sq[:,None] * parsed_model.pi_atomic_radius_sq_times_two[None,:] + log_norm[None,:]
    radius_points = torch.from_numpy(polar_grid.radius_points).to(kernel_params.device)
    kdotr = kdotr * radius_points[None,:,None]
    offset = offset.unsqueeze(0)
    def _nonuniform_kernel(start: int, end: int):
        exponent = torch.complex(gaussKernelAtoms[None,:,:], kdotr[start:end,:,:]) ## (n_templates, n_points, n_atoms)
        exponent.exp_()
        templates_fourier_batch = torch.sum(exponent, dim = 2) - offset
        templates_fourier[start:end,:] = templates_fourier_batch.cpu() ## (n_templates, n_points)
    _iterate_kernel_with_memory_constraints(kernel_params.n_templates, _nonuniform_kernel, verbose=verbose)
    return templates_fourier


def _iterate_kernel_with_memory_constraints(n_templates: int, kernel: Callable[[int, int], None], verbose: bool = False):
    batch_size = 1#n_templates ## batch size = 1 is fine for now
    success = False
    while batch_size > 0:
        n_batches = ceil(n_templates / batch_size)
        try:
            # TODO: Check if performance issue from exponent being in the kernel function scope now
            if verbose:
                print(f"Batch size: {batch_size}")
                from tqdm import trange
                tmp = trange(n_batches)
            else:
                tmp = range(n_batches)
            start = 0
            end = batch_size
            for _ in tmp:
                kernel(start, end)
                start += batch_size
                end += batch_size
            success = True
            break
        except torch.cuda.OutOfMemoryError:
            batch_size //= 2
            continue
    if not success:
        raise MemoryError("Insufficient memory to compute templates from atomic model.")


def _get_fourier_slices(polar_grid: PolarGrid, viewing_angles: ViewingAngles, float_type: torch.dtype, device: torch.device) -> torch.Tensor:
    """Returns a tensor representing the Cartesian values, in space, of the points on the polar grid.

    Outer index is the image, middle index is linearized point index, inner index is 3 (for x, y, z).
    Linearized point index is (# of points per shell) x (# of shells)

    Args:
        polar_grid (PolarGrid): Grid to compute Fourier slices for
        viewing_angles (ViewingAngles): Viewing angles used to compute slices
        float_type (torch.dtype): dtype identifying level of precision to use for computation
        device (torch.device): Device on which to carry out computatino

    Returns:
        torch.Tensor: Tensor of reals, of [img x point x [x/y/z]] such that result[0,5,:] is a
            3-vector of the x,y,z coordinates of the 6th grid point of the 1st image in the stack.
    """
    radius_shells = torch.tensor(polar_grid.radius_shells, dtype = float_type, device = device)
    circles = _get_circles(viewing_angles, polar_grid, float_type=float_type, device=device)
    if not polar_grid.uniform:
        return circles * radius_shells[None,:,None]
    fourier_slices = circles.unsqueeze(1) * radius_shells[None,:,None,None]
    fourier_slices = fourier_slices.flatten(1, 2)
    return fourier_slices


def _test_callable(fn: Callable[[torch.Tensor], torch.Tensor], precision: Precision, device: torch.device | None):
    if not callable(fn):
        raise ValueError("Function must be callable.")
    (float_type, complex_type, _) = precision.get_dtypes(default=Precision.DOUBLE)
    test_input = torch.randn(1, 2, 3, dtype = float_type, device = device)
    test_output: torch.Tensor = fn(test_input)
    if test_output.shape != test_input.shape[:-1]:
        raise ValueError("Function must be a callable that takes a tensor of shape (n_templates, n_pixels, 3) and returns a tensor of shape (n_templates, n_pixels).")
    if precision != Precision.DEFAULT:
        if test_output.dtype != complex_type:
            raise ValueError(f"You have requested to work in {precision.value} precision, but the supplied function " +
                             f"returns {test_output.dtype}. Please ensure your function returns the appropriate type.")


class Templates(Images):
    """Class representing a collection of (Cartesian-space and/or Fourier-space) templates, with methods for manipulating them.
    
    Attributes:
        box_size (FloatArrayType): Size of the (Cartesian-space) viewing port
        phys_grid (CartesianGrid2D): A grid describing the physical space in which the (physical-representation) templates reside
        polar_grid (PolarGrid): A grid describing the polar space in which the Fourier templates reside
        templates_phys (torch.Tensor | None): Cartesian-space template images as pixel-value array of [template x X-index x Y-index]
        templates_fourier (torch.Tensor | None): Fourier-space template images as complex-valued array of [template x radius x angle]
        n_templates (int): Count of templates in the collection
        ctf (CTF | None): Contrast transfer function to be applied to templates, if any
        viewing_angles (ViewingAngles): Optimal viewing angles, if set
        filename (str | None): If set, the name of the file from which the templates were loaded
    """
    viewing_angles: ViewingAngles


    def __init__(self, *,
        phys_data: Optional[PhysicalImages | CartesianGrid2D] = None,
        fourier_data: Optional[FourierImages | PolarGrid] = None,
        box_size: Optional[float | FloatArrayType] = None,
        viewing_angles: ViewingAngles,
        ctf : Optional[CTF] = None,
    ):
        super().__init__(phys_data, fourier_data, box_size, viewing_angles, ctf)
        if (self.viewing_angles.n_angles != self.n_images):
            raise ValueError(f"Number of viewing angles ({self.viewing_angles.n_angles}) must match number of templates ({self.n_images}).")
    
    
    @classmethod
    def generate_from_positions(
        cls,
        atomic_model: AtomicModel,
        viewing_angles: ViewingAngles,
        polar_grid: PolarGrid,
        box_size: float | FloatArrayType,
        atom_shape: AtomShape = AtomShape.DEFAULT,
        precision: Precision = Precision.DEFAULT,
        use_cuda: bool = True,
        verbose : bool = False
    ):
        _box_size = cast(FloatArrayType, project_descriptor(box_size, "box size", 2, TargetType.FLOAT))
        if atom_shape == AtomShape.DEFAULT:
            atom_shape = AtomShape.HARD_SPHERE
            print("Atom shape not specified, using hard sphere.")

        kernel_params = _get_shared_kernel_params(
            atomic_model=atomic_model,
            viewing_angles=viewing_angles,
            polar_grid=polar_grid,
            box_size=_box_size,
            precision=precision,
            use_cuda=use_cuda
        )
        if not polar_grid.uniform:
            raise NotImplementedError("Non-uniform Fourier templates not implemented yet.")
        else:
            if atom_shape == AtomShape.GAUSSIAN:
                kernel = _make_uniform_gaussian_kernel(kernel_params)
            elif atom_shape == AtomShape.HARD_SPHERE:
                kernel = _make_uniform_hard_sphere_kernel(kernel_params)
            else:
                raise ValueError(f"Unknown atom shape {atom_shape.value}")
        
        _iterate_kernel_with_memory_constraints(kernel_params.n_templates, kernel=kernel, verbose=verbose)
        data = FourierImages(images_fourier=kernel_params.templates_fourier, polar_grid=polar_grid)
        return cls(fourier_data=data, viewing_angles=viewing_angles, box_size=box_size)


    @classmethod
    def generate_from_physical_volume(
        cls,
        volume: Volume,
        polar_grid: PolarGrid,
        viewing_angles: ViewingAngles,
        precision: Precision = Precision.DEFAULT,
        use_cuda: bool = True,
        nufft_eps: float = 1.0e-12,
        verbose: bool = False
    ):
        if volume.density_physical is None:
            raise ValueError("No physical volume found")
        device = check_cuda(use_cuda)
        (torch_float_type, _, _) = precision.get_dtypes(default=Precision.SINGLE)
        n_templates = viewing_angles.n_angles

        volume.density_physical = volume.density_physical.to(device)
        fourier_slices = _get_fourier_slices(polar_grid, viewing_angles, float_type=torch_float_type, device=device)
        templates_fourier = volume_phys_to_fourier_points(
            volume = volume,
            fourier_slices = fourier_slices,
            eps = nufft_eps,
            precision = Precision.SINGLE if precision == Precision.DEFAULT else precision,
            use_cuda = True,
            output_device = device,
            verbose = verbose
        )

        origin = torch.tensor([0.0, 0.0, 0.0], dtype = torch_float_type, device = device).unsqueeze(0)
        centers = volume_phys_to_fourier_points(
            volume = volume,
            fourier_slices = origin,
            eps = nufft_eps,
            precision = Precision.SINGLE if precision == Precision.DEFAULT else precision,
            use_cuda = True,
            output_device = device,
            verbose = verbose
        )
        offset = _get_offset(polar_grid=polar_grid, float_type=torch_float_type, device=device)
        offset = offset[None,:] * centers
        templates_fourier -= offset

        if polar_grid.uniform:
            templates_fourier = templates_fourier.reshape(n_templates, polar_grid.n_shells, polar_grid.n_inplanes)
        data = FourierImages(images_fourier=templates_fourier, polar_grid=polar_grid)

        return cls(fourier_data=data, viewing_angles=viewing_angles)


    @classmethod
    def generate_from_function(
        cls,
        function: Callable[[torch.Tensor], torch.Tensor],
        viewing_angles: ViewingAngles,
        polar_grid: PolarGrid,
        precision: Precision = Precision.DEFAULT,
        use_cuda: bool = True
    ):
        device = check_cuda(use_cuda)
        _test_callable(fn=function, precision=precision, device=device)
        (torch_float_type, torch_complex_type, _) = precision.get_dtypes(default=Precision.SINGLE)
        # n_templates = viewing_angles.n_angles

        fourier_slices = _get_fourier_slices(polar_grid, viewing_angles, float_type=torch_float_type, device=device)
        templates_fourier = function(fourier_slices)
        templates_fourier = templates_fourier.reshape(viewing_angles.n_angles, polar_grid.n_shells, polar_grid.n_inplanes)

        # Empirically, it's already this shape
        # if polar_grid.uniform:
        #     templates_fourier = templates_fourier.reshape(n_templates, polar_grid.n_shells, polar_grid.n_inplanes)
        data = FourierImages(images_fourier=templates_fourier, polar_grid=polar_grid)
        return cls(fourier_data=data, viewing_angles=viewing_angles)


    def to_images(self) -> Images:
        if getattr(self, "phys_grid", None) is not None and self.has_physical_images():
            phys_data = PhysicalImages(self.images_phys.clone(), pixel_size=self.phys_grid.pixel_size)
        elif getattr(self, "phys_grid", None) is not None:
            phys_data = self.phys_grid
        else:
            phys_data = None

        if getattr(self, "polar_grid", None) is not None and self.has_fourier_images():
            fourier_data = FourierImages(self.images_fourier.clone(), self.polar_grid)
        elif getattr(self, "polar_grid", None) is not None:
            fourier_data = self.polar_grid
        else:
            fourier_data = None

        return Images(phys_data, fourier_data, self.box_size, self.viewing_angles)
