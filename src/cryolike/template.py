from typing import Callable, NamedTuple, Optional, cast
from cryolike.util.typechecks import set_precision
import numpy as np
import torch
from math import ceil

from cryolike.polar_grid import PolarGrid
from cryolike.atomic_model import AtomicModel
from cryolike.nufft import fourier_polar_to_cartesian_phys, volume_phys_to_fourier_points
from cryolike.cartesian_grid import CartesianGrid2D, from_descriptor, Cartesian_grid_descriptor
from cryolike.volume import Volume
from cryolike.ctf import CTF
from cryolike.viewing_angles import ViewingAngles

from cryolike.util.device_handling import check_cuda
from cryolike.util.image_manipulation import get_imgs_max
from cryolike.util.data_transfer_classes import FourierImages, PhysicalImages
from cryolike.util.enums import Precision, AtomShape
from cryolike.util.types import FloatArrayType
from cryolike.util.reformatting import TargetType, project_descriptor


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


class ParsedAtomicModel(NamedTuple):
    atomic_radius_scaled: torch.Tensor
    radius_shells: torch.Tensor
    radius_shells_sq: torch.Tensor
    pi_atomic_radius_sq_times_two: torch.Tensor
    atomic_coordinates_scaled: torch.Tensor


def _parse_atomic_model(atomic_model: AtomicModel, polar_grid: PolarGrid, box_size: FloatArrayType, float_type: torch.dtype, device: torch.device) -> ParsedAtomicModel:
    atomic_coordinates = torch.tensor(atomic_model.atomic_coordinates, dtype=float_type, device=device)
    atomic_radii = torch.tensor(atomic_model.atom_radii, dtype=float_type, device=device)
    box_max = np.amax(box_size)
    ## TODO: haddle anisotropic box sizes
    atomic_coordinates_scaled = atomic_coordinates.T / box_max * 2.0 * (- 2.0 * np.pi)

    atomic_radius_scaled = atomic_radii / box_max * 2.0
    pi_atomic_radius_sq_times_two = 2.0 * (np.pi * atomic_radius_scaled) ** 2
    radius_shells = torch.tensor(polar_grid.radius_shells, dtype = float_type, device = device)
    radius_shells_sq = radius_shells ** 2
    return ParsedAtomicModel(atomic_radius_scaled, radius_shells, radius_shells_sq, pi_atomic_radius_sq_times_two, atomic_coordinates_scaled)


def _generate_templates_from_uniform_positions_hard_sphere(
    atomic_model : AtomicModel,
    viewing_angles: ViewingAngles,
    polar_grid: PolarGrid,
    box_size: FloatArrayType,
    precision: Precision = Precision.DEFAULT,
    use_cuda: bool = True,
    verbose : bool = False
) -> torch.Tensor:
    (torch_float_type, torch_complex_type, _) = set_precision(precision, default=Precision.SINGLE)
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
    kR = 2 * np.pi * parsed_model.radius_shells[:,None] * parsed_model.atomic_radius_scaled[None,:] ## (n_shells, n_atoms)
    kernelAtoms = (torch.sin(kR) - kR * torch.cos(kR)) * parsed_model.radius_shells.pow(-3)[:,None] / (8 * np.pi ** 2) * 3 / atomic_model.n_atoms
    # offset = np.pi * parsed_model.atomic_radius_scaled.pow(3).sum() / atomic_model.n_atoms
    
    templates_fourier = torch.zeros((n_templates, polar_grid.n_shells, polar_grid.n_inplanes), dtype=torch_complex_type, device="cpu")
    # offset = offset.reshape(polar_grid.n_shells, polar_grid.n_inplanes).unsqueeze(0)
    def _uniform_kernel(start: int, end: int):
        kdotr_batch = torch.matmul(xyz_template_points[start:end,:,:], parsed_model.atomic_coordinates_scaled) ## uniform (n_templates, n_inplanes, n_atoms) or nonuniform (n_templates, n_points, n_atoms)
        kdotr_batch = kdotr_batch[:,None,:,:] * parsed_model.radius_shells[None,:,None,None] ## (n_templates, n_shells, n_inplanes, n_atoms)
        exponent = torch.exp(1j * kdotr_batch) * kernelAtoms[None,:,None,:] ## (n_templates, n_shells, n_inplanes, n_atoms)
        # exponent.exp_()
        # exponent = kernelAtoms[None,:,None,:]
        templates_fourier_batch = torch.sum(exponent, dim = 3)
        templates_fourier[start:end,:,:] = templates_fourier_batch.cpu()
    _iterate_kernel_with_memory_constraints(n_templates, kernel=_uniform_kernel, verbose=verbose)
    return templates_fourier


def _generate_templates_from_uniform_positions_gaussian(
    atomic_model : AtomicModel,
    viewing_angles: ViewingAngles,
    polar_grid: PolarGrid,
    box_size: FloatArrayType,
    precision: Precision = Precision.DEFAULT,
    use_cuda: bool = True,
    verbose : bool = False
) -> torch.Tensor:
    (torch_float_type, torch_complex_type, _) = set_precision(precision, default=Precision.SINGLE)
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
    log_norm = - 1.5 * np.log(2 * np.pi) - 3 * torch.log(parsed_model.atomic_radius_scaled) - np.log(atomic_model.n_atoms)
    offset = _get_offset(polar_grid, float_type=torch_float_type, device=device)
    offset *= torch.sum(torch.exp(log_norm))
    offset = offset.reshape(polar_grid.n_shells, polar_grid.n_inplanes).unsqueeze(0)
    ## Gaussian kernel
    gaussKernelAtoms = - parsed_model.radius_shells_sq[:,None] * parsed_model.pi_atomic_radius_sq_times_two[None,:] + log_norm[None,:]
    templates_fourier = torch.zeros((n_templates, polar_grid.n_shells, polar_grid.n_inplanes), dtype=torch_complex_type, device="cpu")
    def _uniform_kernel(start: int, end: int):
        kdotr_batch = torch.matmul(xyz_template_points[start:end,:,:], parsed_model.atomic_coordinates_scaled) ## uniform (n_templates, n_inplanes, n_atoms) or nonuniform (n_templates, n_points, n_atoms)
        kdotr_batch = kdotr_batch[:,None,:,:] * parsed_model.radius_shells[None,:,None,None] ## (n_templates, n_inplanes, n_shells, n_atoms)
        exponent = torch.complex(gaussKernelAtoms[None,:,None,:], kdotr_batch) ## (n_templates, n_inplanes, n_shells, n_atoms)
        exponent.exp_()
        templates_fourier_batch = torch.sum(exponent, dim = 3) - offset
        templates_fourier[start:end,:,:] = templates_fourier_batch.cpu()
    _iterate_kernel_with_memory_constraints(n_templates, kernel=_uniform_kernel, verbose=verbose)
    return templates_fourier


def _generate_templates_from_nonuniform_positions_gaussian(
    atomic_model : AtomicModel,
    viewing_angles: ViewingAngles,
    polar_grid: PolarGrid,
    box_size: FloatArrayType,
    precision: Precision = Precision.DEFAULT,
    use_cuda: bool = True,
    verbose : bool = False
) -> torch.Tensor:
    raise NotImplementedError("kdtor is not set properly.")
    device = check_cuda(use_cuda)
    (torch_float_type, torch_complex_type, _) = set_precision(precision, default=Precision.SINGLE)
    n_templates = viewing_angles.n_angles
    parsed_model = _parse_atomic_model(
        atomic_model,
        polar_grid,
        box_size,
        float_type=torch_float_type,
        device=device
    )
    log_norm = - 1.5 * np.log(2 * np.pi) - 3 * torch.log(parsed_model.atomic_radius_scaled) - np.log(atomic_model.n_atoms)
    offset = _get_offset(polar_grid, float_type=torch_float_type, device=device)    
    offset *=  torch.sum(torch.exp(log_norm))

    # TODO: Check this signature, is it right?
    templates_fourier = torch.zeros((n_templates, polar_grid.n_points), dtype = torch_complex_type, device = "cpu")
    gaussKernelAtoms = parsed_model.radius_shells_sq[:,None] * parsed_model.pi_atomic_radius_sq_times_two[None,:] + log_norm[None,:]
    radius_points = torch.from_numpy(polar_grid.radius_points).to(device)
    kdotr = kdotr * radius_points[None,:,None]
    offset = offset.unsqueeze(0)
    def _nonuniform_kernel(start: int, end: int):
        exponent = torch.complex(gaussKernelAtoms[None,:,:], kdotr[start:end,:,:]) ## (n_templates, n_points, n_atoms)
        exponent.exp_()
        templates_fourier_batch = torch.sum(exponent, dim = 2) - offset
        templates_fourier[start:end,:] = templates_fourier_batch.cpu() ## (n_templates, n_points)
    _iterate_kernel_with_memory_constraints(n_templates, _nonuniform_kernel, verbose=verbose)
    return templates_fourier


def _iterate_kernel_with_memory_constraints(n_templates: int, kernel: Callable[[int, int], None], verbose: bool = False):
    batch_size = 1#n_templates ## batch size = 1 is fine for now
    exponent = None
    success = False
    while batch_size > 0:
        n_batches = ceil(n_templates / batch_size)
        try:
            # TODO: Check if performance issue from exponent being in the kernel function scope now
            exponent = None
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
            del exponent
            batch_size //= 2
            continue
    if not success:
        raise MemoryError("Insufficient memory to compute templates from atomic model.")


def _get_fourier_slices(polar_grid: PolarGrid, viewing_angles: ViewingAngles, float_type: torch.dtype, device: torch.device) -> torch.Tensor:
    radius_shells = torch.tensor(polar_grid.radius_shells, dtype = float_type, device = device)
    circles = _get_circles(viewing_angles, polar_grid, float_type=float_type, device=device)
    if not polar_grid.uniform:
        return circles * radius_shells[None,:,None]
    fourier_slices = circles.unsqueeze(1) * radius_shells[None,:,None,None]
    fourier_slices = fourier_slices.flatten(1, 2)
    return fourier_slices


def _test_callable(fn: Callable, precision: Precision, device: torch.device | None):
    if not callable(fn):
        raise ValueError("Function must be callable.")
    (float_type, complex_type, _) = set_precision(precision, Precision.DOUBLE)
    test_input = torch.randn(1, 2, 3, dtype = float_type, device = device)
    test_output: torch.Tensor = fn(test_input)
    if test_output.shape != test_input.shape[:-1]:
        raise ValueError("Function must be a callable that takes a tensor of shape (n_templates, n_pixels, 3) and returns a tensor of shape (n_templates, n_pixels).")
    if precision != Precision.DEFAULT:
        if test_output.dtype != complex_type:
            raise ValueError(f"You have requested to work in {precision.value} precision, but the supplied function " +
                             f"returns {test_output.dtype}. Please ensure your function returns the appropriate type.")


class Templates:
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
    box_size: FloatArrayType
    phys_grid: CartesianGrid2D
    polar_grid: PolarGrid
    templates_phys: torch.Tensor | None
    templates_fourier: torch.Tensor | None
    n_templates: int
    ctf: CTF | None
    viewing_angles: ViewingAngles
    filename: str | None


    def __init__(self, *,
        phys_templates_data: Optional[PhysicalImages] = None,
        fourier_templates_data: Optional[FourierImages] = None,
        box_size: Optional[float | FloatArrayType] = None,
        polar_grid: Optional[PolarGrid] = None,
        phys_grid: Optional[CartesianGrid2D] = None,
        ctf : Optional[CTF] = None,
        viewing_angles: ViewingAngles
    ):
        if (phys_templates_data is None and fourier_templates_data is None):
            raise ValueError("Must pass at least one of Fourier and cartesian templates.")
        
        phys_grid_set = False
        if phys_templates_data is not None:
            if phys_templates_data.phys_grid is None:
                raise ValueError("Can't happen: this should be fixed in the constructor.")
            self.phys_grid = phys_templates_data.phys_grid
            phys_grid_set = True
            self.templates_phys = phys_templates_data.images_phys
            self.n_templates = self.templates_phys.shape[0]
        else:
            self.templates_phys = None
        if phys_grid is not None:
            self.phys_grid = phys_grid
            phys_grid_set = True
            
        if fourier_templates_data is not None:
            if polar_grid is not None:
                raise ValueError("Can't pass both polar_grid and fourier_templates_data.")
            self.polar_grid = fourier_templates_data.polar_grid
            self.templates_fourier = fourier_templates_data.images_fourier
            self.n_templates = self.templates_fourier.shape[0]
        else:
            self.templates_fourier = None
            if polar_grid is not None:
                self.polar_grid = polar_grid
            
        if box_size is not None:
            _box_size = cast(FloatArrayType, project_descriptor(box_size, "box_size", 2, TargetType.FLOAT))
            self.box_size = _box_size
        elif phys_grid_set == True:
            self.box_size = self.phys_grid.box_size
        else:
            self.box_size = np.array([2., 2.])
            
        self._check_template_array()
        self.viewing_angles = viewing_angles
        if (self.viewing_angles.n_angles != self.n_templates):
            raise ValueError(f"Number of viewing angles ({self.viewing_angles.n_angles}) must match number of templates ({self.n_templates}).")
        self.ctf = ctf
    
    
    def _ensure_phys_templates(self):
        if self.templates_phys is None:
            raise ValueError("Physical templates not found.")
        if self.phys_grid is None:
            raise ValueError("No physical grid found.")
        

    def _ensure_fourier_templates(self):
        if self.templates_fourier is None:
            raise ValueError("Fourier templates not found.")
        if self.polar_grid is None:
            raise ValueError("No polar grid found.")

    
    def _check_template_array(self):
        if self.templates_phys is not None:
            if self.phys_grid is None:
                # can't happen
                raise ValueError("Physical grid is not defined for physical Templates")
            if len(self.templates_phys.shape) == 2:
                self.templates_phys = self.templates_phys[None,:,:]
            if len(self.templates_phys.shape) != 3:
                raise ValueError("Invalid shape for Templates.")
            if (self.n_templates != self.templates_phys.shape[0]):
                raise ValueError(f"Templates object lists templates count of {self.n_templates} but the physical Templates array has {self.templates_phys.shape[0]} entries.")
            # TODO: There's probably another consistency check required with the phys grid.
            if (self.phys_grid.n_pixels[0] != self.templates_phys.shape[1] or self.phys_grid.n_pixels[1] != self.templates_phys.shape[2]):
                raise ValueError('Dimension mismatch: n_pixels {self.phys_grid.n_pixels[0]} x {self.phys_grid.n_pixels[1]} but shape is {self.templates_phys.shape}')
            if not np.allclose(self.box_size, self.phys_grid.box_size):
                print(f"WARNING: Templates box size {self.box_size} is outside tolerance of physical grid box size {self.phys_grid.box_size}")
        if self.templates_fourier is not None:
            if self.polar_grid is None:
                # Can't happen
                raise ValueError("Polar grid is not defined for Fourier Templates")
            if self.n_templates != self.templates_fourier.shape[0]:
                raise ValueError(f"Templates object lists templates count of {self.n_templates} but the fourier Templates array has {self.templates_fourier.shape[0]} entries.")


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
        if isinstance(box_size, float):
            box_size = cast(FloatArrayType, project_descriptor(box_size, "box size", 2, TargetType.FLOAT))
        assert isinstance(box_size, np.ndarray)
        if atom_shape == AtomShape.DEFAULT:
            atom_shape = AtomShape.HARD_SPHERE
            print("Atom shape not specified, using hard sphere.")
        if atom_shape == AtomShape.GAUSSIAN:
            if polar_grid.uniform:
                templates_fourier = _generate_templates_from_uniform_positions_gaussian(
                    atomic_model=atomic_model, viewing_angles=viewing_angles, polar_grid=polar_grid,
                    box_size=box_size, precision=precision, use_cuda=use_cuda, verbose=verbose
                )
            else:
                raise NotImplementedError("Non-uniform Fourier templates not implemented yet.")
                # templates_fourier = _generate_templates_from_nonuniform_positions_gaussian(
                #     atomic_model=atomic_model, viewing_angles=viewing_angles, polar_grid=polar_grid,
                #     box_size=box_size, precision=precision, use_cuda=use_cuda, verbose=verbose
                # )
        elif atom_shape == AtomShape.HARD_SPHERE:
            if polar_grid.uniform:
                templates_fourier = _generate_templates_from_uniform_positions_hard_sphere(
                    atomic_model=atomic_model, viewing_angles=viewing_angles, polar_grid=polar_grid,
                    box_size=box_size, precision=precision, use_cuda=use_cuda, verbose=verbose
                )
            else:
                raise NotImplementedError("Non-uniform Fourier templates not implemented yet.")
        else:
            raise ValueError(f"Unknown atom shape {atom_shape.value}")
        data = FourierImages(images_fourier=templates_fourier, polar_grid=polar_grid)
        return cls(fourier_templates_data=data, viewing_angles=viewing_angles, box_size=box_size)


    @classmethod
    def generate_from_physical_volume(
        cls,
        volume : Volume,
        polar_grid: PolarGrid,
        viewing_angles: ViewingAngles,
        precision: Precision = Precision.DEFAULT,
        use_cuda: bool = True,
        nufft_eps : float = 1.0e-12,
        verbose : bool = False
    ):
        if volume.density_physical is None:
            raise ValueError("No physical volume found")
        device = check_cuda(use_cuda)
        (torch_float_type, _, _) = set_precision(precision, default=Precision.SINGLE)
        n_templates = viewing_angles.n_angles

        volume.density_physical = volume.density_physical.to(device)
        origin = torch.tensor([0.0, 0.0, 0.0], dtype = torch_float_type, device = device).unsqueeze(0)
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

        offset = _get_offset(polar_grid=polar_grid, float_type=torch_float_type, device=device)
        centers = volume_phys_to_fourier_points(
            volume = volume,
            fourier_slices = origin,
            eps = nufft_eps,
            precision = Precision.SINGLE if precision == Precision.DEFAULT else precision,
            use_cuda = True,
            output_device = device,
            verbose = verbose
        )
        offset = offset[None,:] * centers

        templates_fourier -= offset
        if polar_grid.uniform:
            templates_fourier = templates_fourier.reshape(n_templates, polar_grid.n_shells, polar_grid.n_inplanes)
        data = FourierImages(images_fourier=templates_fourier, polar_grid=polar_grid)
        return cls(fourier_templates_data=data, viewing_angles=viewing_angles)


    @classmethod
    def generate_from_function(
        cls,
        function: Callable,
        viewing_angles: ViewingAngles,
        polar_grid: PolarGrid,
        precision: Precision = Precision.DEFAULT,
        use_cuda: bool = True
    ):
        device = check_cuda(use_cuda)
        _test_callable(fn=function, precision=precision, device=device)
        (torch_float_type, _, _) = set_precision(precision, default=Precision.SINGLE)
        n_templates = viewing_angles.n_angles

        fourier_slices = _get_fourier_slices(polar_grid, viewing_angles, float_type=torch_float_type, device=device)
        templates_fourier = function(fourier_slices)
        assert isinstance(templates_fourier, torch.Tensor)

        if polar_grid.uniform:
            templates_fourier = templates_fourier.reshape(n_templates, polar_grid.n_shells, polar_grid.n_inplanes)
        data =FourierImages(images_fourier=templates_fourier, polar_grid=polar_grid)
        return cls(fourier_templates_data=data, viewing_angles=viewing_angles)


    def transform_to_spatial(
        self,
        phys_grid: Optional[Cartesian_grid_descriptor] = None,
        nufft_eps: float = 1e-12,
        n_templates_stop: int = -1,
        precision: Precision = Precision.DEFAULT,
        use_cuda: bool = True,
        save_to_class: bool = True
    ):
        if self.templates_fourier is None:
            raise ValueError("Fourier templates not generated.")
        if self.polar_grid is None:
            raise ValueError("No polar grid found")
        if phys_grid is not None:
            self.phys_grid = from_descriptor(phys_grid)
        if self.phys_grid is None:
            raise ValueError("No physical grid found")

        if n_templates_stop == -1 or n_templates_stop == self.n_templates:
            print("Transforming all Templates.")
            n_templates_stop = self.n_templates
        else:
            print(f"Transforming only the first {n_templates_stop} Templates, probably for testing or plotting.")

        templates_fourier = self.templates_fourier[:n_templates_stop]
        templates_fourier = templates_fourier.reshape(templates_fourier.shape[0], -1)
        templates_phys = fourier_polar_to_cartesian_phys(
            grid_fourier_polar = self.polar_grid,
            grid_cartesian_phys = self.phys_grid,
            image_polar = templates_fourier,
            eps = nufft_eps,
            precision = Precision.SINGLE if precision == Precision.DEFAULT else precision,
            use_cuda = use_cuda
        )
        if save_to_class:
            if n_templates_stop != self.n_templates:
                print("Warning: Only the first {} templates are saved to the class. This can create issues.".format(n_templates_stop))
            self.templates_phys = templates_phys
        return templates_phys


    def apply_ctf(self, ctf: CTF):
        """Applies a contrast transfer function to the Fourier-space templates.

        Args:
            ctf (CTF): CTF to apply.

        Raises:
            NotImplementedError: If the Fourier-space templates are using a non-uniform
                polar grid.

        Returns:
            torch.Tensor: The updated templates (which will also be persisted to the collection).
        """
        self._ensure_fourier_templates()
        if not self.polar_grid.uniform:
            raise NotImplementedError("Non-uniform Fourier templates not implemented yet.")
        self.ctf = ctf
        assert self.templates_fourier is not None  # TODO: Improve this assertion
        self.templates_fourier = ctf.apply(self.templates_fourier)
        return self.templates_fourier
    
    
    def normalize_templates_phys(
        self,
        ord: int = 1,
        use_max: bool = False
    ):
        """Normalize the Cartesian-space templates in the collection.

        Args:
            ord (int, optional): Degree of norm to apply. Defaults to 1.
            use_max (bool, optional): Whether to use the max in place of an LP norm. Defaults to False.

        Returns:
            torch.Tensor: The norm applied to the templates (which are modified in-place).
        """
        self._ensure_phys_templates()
        assert self.templates_phys is not None
        if use_max:
            maxval = get_imgs_max(self.templates_phys)
            self.templates_phys /= maxval
            return maxval
        else:
            lpnorm = torch.norm(self.templates_phys, dim = (1,2), p = ord, keepdim = False)
            self.templates_phys /= lpnorm[:,None,None]
            return lpnorm
    

    def normalize_templates_fourier(
        self,
        ord: int = 1,
        use_max: bool = False,
    ):
        """Normalize the Fourier-space templates in the collection.

        Args:
            ord (int, optional): Degree of norm to apply. Defaults to 1.
            use_max (bool, optional): Whether to use the max in place of an LP norm. Defaults to False.

        Returns:
            torch.Tensor: The norm applied to the templates (which are modified in-place).
        """
        self._ensure_fourier_templates()
        assert self.templates_fourier is not None
        if use_max:
            maxval = get_imgs_max(self.templates_fourier)
            self.templates_fourier /= maxval
            return maxval
        else:
            lpnorm = self.polar_grid.integrate(self.templates_fourier.abs().pow(ord)).pow(1.0 / ord)
            for _ in range(len(self.templates_fourier.shape) - 1):
                lpnorm = torch.unsqueeze(lpnorm, dim = -1)
            self.templates_fourier /= lpnorm
            return lpnorm
        
    
    def get_power_spectrum(self):
        """Gets the power spectrum of the (Fourier-space) images.

        Raises:
            ValueError: If no Fourier-space images exist in the collection.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of power-spectrum values and the resolutions.
        """
        if self.templates_fourier is None:
            raise ValueError("Fourier images not found. Please transform the images to Fourier domain before calculating the power spectrum.")
        resolutions = np.amax(self.box_size) / (2.0 * self.polar_grid.radius_shells)
        power_spectrum = torch.mean(torch.abs(self.templates_fourier) ** 2, dim = (0, 2))
        return power_spectrum, resolutions
