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
    get_device,
    FloatArrayType,
    Precision,
    project_descriptor,
    TargetType,
    to_torch,
)


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


def _get_circles(viewing_angles: ViewingAngles, polar_grid: PolarGrid, precision: Precision, device: torch.device):
    azimus = to_torch(viewing_angles.azimus, precision, device)
    polars = to_torch(viewing_angles.polars, precision, device)
    gammas = to_torch(viewing_angles.gammas, precision, device)
    thetas = to_torch(polar_grid.theta_shell if polar_grid.uniform else polar_grid.theta_points, precision, device)
    circles = _fourier_circles(thetas, polars, azimus, gammas)
    return circles


def _get_offset(polar_grid: PolarGrid, precision: Precision, device: torch.device) -> torch.Tensor:
    assert polar_grid.x_points is not None and polar_grid.y_points is not None, \
        "Polar grid must have x_points and y_points defined for offset computation."
    x_points = to_torch(polar_grid.x_points, precision, device)
    y_points = to_torch(polar_grid.y_points, precision, device)
    offset = torch.sinc(2.0 * x_points) * torch.sinc(2.0 * y_points)
    return offset


class TemplateGenerator:

    """Class for generating templates from an atomic model, viewing angles, and polar grid.
    also allow gradient computation for the templates with respect to the atomic coordinates and viewing angles."""

    atomic_model: AtomicModel
    viewing_angles: ViewingAngles
    polar_grid: PolarGrid
    box_size: float ## TODO: handle anisotropic box sizes

    n_templates: int
    n_frames: int
    n_images_per_frame: int

    atomic_coordinates: torch.Tensor
    radius_shells: torch.Tensor
    radius_shells_sq: torch.Tensor
    atomic_radius_scaled: torch.Tensor
    pi_atomic_radius_sq_times_two: torch.Tensor
    atom_shape: AtomShape
    precision: Precision
    float_type: torch.dtype
    complex_type: torch.dtype
    storage_device: torch.device
    compute_device: torch.device

    fourier_circles: torch.Tensor
    generator: Callable[[], torch.Tensor]

    use_all_angles_for_each_frame: bool

    # fix_atomic_coordinates: bool
    # fix_viewing_angles: bool

    def __init__(
        self,
        atomic_model: AtomicModel,
        viewing_angles: ViewingAngles,
        polar_grid: PolarGrid,
        box_size: float | FloatArrayType,
        atom_shape: AtomShape,
        # fix_atomic_coordinates: bool = True,
        # fix_viewing_angles: bool = True,
        use_all_angles_for_each_frame: bool = False,
        storage_device: str | torch.device = torch.device("cpu"),
        compute_device: str | torch.device = torch.device("cuda"),
        precision: Precision = Precision.DEFAULT,
    ):

        self.atomic_model = atomic_model
        self.viewing_angles = viewing_angles
        self.polar_grid = polar_grid
        if not self.polar_grid.uniform:
            raise NotImplementedError("Non-uniform polar grids not implemented yet.")
        if isinstance(box_size, (float, int)):
            self.box_size = float(box_size)
        elif isinstance(box_size, (np.ndarray, list, tuple, torch.Tensor)):
            assert box_size.ndim == 1, "Box size must be a 1D array-like structure."
            self.box_size = float(box_size[0])
        else:
            raise TypeError("Box size must be a float, int, or a 1D array-like structure.")
        assert self.box_size > 0, "Box size must be greater than 0."
        self.storage_device = get_device(storage_device)
        self.compute_device = get_device(compute_device)
        print("Using device:", self.compute_device, "for templates generation, and ", self.storage_device, "for storage.")
        self.atom_shape = atom_shape
        self.precision = precision
        self.float_type, self.complex_type, _ = precision.get_dtypes(default=Precision.SINGLE)

        self.radius_shells = to_torch(polar_grid.radius_shells, precision, self.compute_device)
        self.radius_shells_sq = self.radius_shells ** 2
        if torch.is_tensor(self.atomic_model.atomic_coordinates):
            self.atomic_coordinates = self.atomic_model.atomic_coordinates
        else:
            self.atomic_coordinates = to_torch(atomic_model.atomic_coordinates, precision, self.storage_device)
        assert self.atomic_coordinates is not None, "Atomic coordinates must be provided."
        assert self.atomic_coordinates.ndim == 3, "Atomic coordinates must be a 3D tensor of shape (n_frames, n_atoms, 3)."
        # assert self.atomic_coordinates.shape[0] == 1 or self.atomic_coordinates.shape[0] == self.n_templates, \
        #     "Invalid atomic coordinates shape, expected (n_viewing_angles, n_atoms, 3) or (1, n_atoms, 3)."
        self.use_all_angles_for_each_frame = use_all_angles_for_each_frame
        if self.atomic_coordinates.shape[0] == 1 and self.viewing_angles.n_angles > 1:
            self.use_all_angles_for_each_frame = True
            self.n_frames = 1
            self.n_images_per_frame = self.viewing_angles.n_angles
            self.n_templates = self.viewing_angles.n_angles
        if self.use_all_angles_for_each_frame:
            self.n_frames = self.atomic_coordinates.shape[0]
            self.n_images_per_frame = self.viewing_angles.n_angles
            self.n_templates = self.atomic_coordinates.shape[0] * self.viewing_angles.n_angles
        elif self.atomic_coordinates.shape[0] != self.n_templates:
            raise ValueError(f"Atomic coordinates shape {self.atomic_coordinates.shape} does not match number of templates {self.n_templates}. " +
                             "If you want to generate templates using all angles on each frames, set use_all_angles_for_each_frame=True.")
        else:
            self.n_frames = self.atomic_coordinates.shape[0]
            self.n_images_per_frame = 1
            self.n_templates = self.atomic_coordinates.shape[0]
        print(f"Using {self.n_frames} frames of atomic coordinates for {self.n_templates} templates.")
        
        assert self.n_frames > 0, "Number of frames must be greater than 0."
        assert self.n_images_per_frame > 0, "Number of images per frame must be greater than 0."
        assert self.n_templates > 0, "Number of templates must be greater than 0."

        atomic_radii = to_torch(atomic_model.atom_radii, precision, self.compute_device)
        self.atomic_radius_scaled = atomic_radii / self.box_size * 2.0
        self.pi_atomic_radius_sq_times_two = 2.0 * (np.pi * self.atomic_radius_scaled) ** 2

        # self.fix_atomic_coordinates = fix_atomic_coordinates
        # self.fix_viewing_angles = fix_viewing_angles

        if self.atom_shape == AtomShape.GAUSSIAN:
            self.generator = self._gaussian_atom_kernel
        elif self.atom_shape == AtomShape.HARD_SPHERE:
            self.generator = self._hard_sphere_atom_kernel
        else:
            raise ValueError(f"Unsupported atom shape: {self.atom_shape}. Supported shapes are: {AtomShape.GAUSSIAN}, {AtomShape.HARD_SPHERE}.")


    def _get_scaled_atomic_radius(self) -> torch.Tensor:
        """Returns the atomic radius scaled to the unit box size."""
        return self.atomic_radius_scaled * self.box_size / 2.0


    def _get_scaled_atomic_coordinates(self) -> torch.Tensor:
        """Returns the atomic coordinates scaled to the unit box size and scaled for the translation kernel."""
        return self.atomic_coordinates * 2.0 / self.box_size * (- 2.0 * np.pi)
    
    def _get_template_points(self) -> torch.Tensor:
        return _get_circles(
            self.viewing_angles, self.polar_grid, self.precision, self.compute_device
        ).to(self.storage_device)


    def _gaussian_atom_kernel(self) -> torch.Tensor:
        _atomic_coordinates_scaled = self._get_scaled_atomic_coordinates()
        _xyz_template_points = self._get_template_points()
        if self.use_all_angles_for_each_frame:
            _atomic_coordinates_scaled = _atomic_coordinates_scaled.unsqueeze(1).expand(-1, self.n_images_per_frame, -1, -1)
            _xyz_template_points = _xyz_template_points.unsqueeze(0).expand(self.n_frames, -1, -1, -1)
        log_norm = - 1.5 * np.log(2 * np.pi) - 3 * torch.log(self.atomic_radius_scaled) - np.log(self.atomic_model.n_atoms)
        offset = _get_offset(self.polar_grid, precision=self.precision, device=self.compute_device)
        offset *= torch.sum(torch.exp(log_norm))
        offset = offset.reshape(self.polar_grid.n_shells, self.polar_grid.n_inplanes).unsqueeze(0)
        gaussKernelAtoms = - self.radius_shells_sq[:,None] * self.pi_atomic_radius_sq_times_two[None,:] + log_norm[None,:]
        _templates_fourier = torch.zeros((self.n_templates, self.polar_grid.n_shells, self.polar_grid.n_inplanes),
            dtype=self.complex_type, device=self.storage_device)
        if self.use_all_angles_for_each_frame:
            def _batch_kernel(start_frame: int, end_frame: int, start_angle: int = 0, end_angle: int = self.n_images_per_frame):
                _xyz_template_points_batch = _xyz_template_points[:,start_angle:end_angle,:,:].to(self.compute_device)
                _atomic_coordinates_batch = _atomic_coordinates_scaled[start_frame:end_frame,:,:,:].to(self.compute_device)
                kdotr_batch = torch.einsum("mnkx,mnax->mnka", _xyz_template_points_batch, _atomic_coordinates_batch).flatten(0, 1)
                kdotr_batch = kdotr_batch[:,None,:,:] * self.radius_shells[None,:,None,None] ## (n_templates, n_shells, n_inplanes, n_atoms)
                exponent = torch.complex(gaussKernelAtoms[None,:,None,:], kdotr_batch) ## (n_templates, n_shells, n_inplanes, n_atoms)
                exponent.exp_()
                templates_fourier_batch = torch.sum(exponent, dim = 3) - offset
                _slice_template = slice(start_frame * self.n_images_per_frame + start_angle, end_frame * self.n_images_per_frame + end_angle)
                _templates_fourier[_slice_template,:,:] = templates_fourier_batch.to(self.storage_device) ## (n_templates, n_shells, n_inplanes)
        else:
            def _batch_kernel(start_frame: int, end_frame: int, start_angle: int = 0, end_angle: int = self.n_images_per_frame):
                _xyz_template_points_batch = _xyz_template_points[start_angle:end_angle,:,:].to(self.compute_device)
                _atomic_coordinates_batch = _atomic_coordinates_scaled[start_frame:end_frame,:,:].to(self.compute_device)
                kdotr_batch = torch.einsum("ikx,iax->ika", _xyz_template_points_batch, _atomic_coordinates_batch)
                kdotr_batch = kdotr_batch[:,None,:,:] * self.radius_shells[None,:,None,None] ## (n_templates, n_shells, n_inplanes, n_atoms)
                exponent = torch.complex(gaussKernelAtoms[None,:,None,:], kdotr_batch) ## (n_templates, n_shells, n_inplanes, n_atoms)
                exponent.exp_()
                templates_fourier_batch = torch.sum(exponent, dim = 3) - offset
                _templates_fourier[start_frame:end_frame,:,:] = templates_fourier_batch.to(self.storage_device) ## (n_templates, n_shells, n_inplanes)
        _iterate_kernel_with_memory_constraints(self.n_frames, self.n_images_per_frame, _batch_kernel)
        return _templates_fourier


    def _hard_sphere_atom_kernel(self) -> torch.Tensor:
        _atomic_coordinates_scaled = self._get_scaled_atomic_coordinates()
        _xyz_template_points = self._get_template_points()
        if self.use_all_angles_for_each_frame:
            _atomic_coordinates_scaled = _atomic_coordinates_scaled.unsqueeze(1).expand(-1, self.n_images_per_frame, -1, -1)
            _xyz_template_points = _xyz_template_points.unsqueeze(0).expand(self.n_frames, -1, -1, -1)
        kR = 2 * np.pi * self.radius_shells[:,None] * self.atomic_radius_scaled[None,:]
        kernelAtoms = (torch.sin(kR) - kR * torch.cos(kR))
        kernelAtoms *= self.radius_shells.pow(-3)[:,None] / (8 * np.pi ** 2) * 3 / self.atomic_model.n_atoms
        _templates_fourier = torch.zeros((self.n_templates, self.polar_grid.n_shells, self.polar_grid.n_inplanes),
            dtype=self.complex_type, device=self.storage_device)
        if self.use_all_angles_for_each_frame:
            def _batch_kernel(start_frame: int, end_frame: int, start_image: int = 0, end_image: int = self.n_images_per_frame):
                _xyz_template_points_batch = _xyz_template_points[:,start_image:end_image,:,:].to(self.compute_device)
                _atomic_coordinates_batch = _atomic_coordinates_scaled[start_frame:end_frame,:,:,:].to(self.compute_device)
                kdotr_batch_twa = torch.einsum("mnkx,mnax->mnka", _xyz_template_points_batch, _atomic_coordinates_batch).flatten(0, 1)
                kdotr_batch_trwa = kdotr_batch_twa[:,None,:,:] * self.radius_shells[None,:,None,None]
                print("kdotr_batch shape:", kdotr_batch_trwa.shape)
                exponent = torch.exp(1j * kdotr_batch_trwa) * kernelAtoms[None,:,None,:] ## (n_templates, n_shells, n_inplanes, n_atoms)
                print("exponent shape:", exponent.shape)
                templates_fourier_batch = torch.sum(exponent, dim = 3) ## (n_templates, n_shells, n_inplanes)
                _slice_template = slice(start_frame * self.n_images_per_frame + start_image, start_frame * self.n_images_per_frame + end_image)
                print("shape of templates_fourier_batch:", templates_fourier_batch.shape)
                _templates_fourier[_slice_template,:,:] = templates_fourier_batch.to(self.storage_device)
        else:
            def _batch_kernel(start_frame: int, end_frame: int, start_image: int = 0, end_image: int = self.n_images_per_frame):
                _xyz_template_points_batch = _xyz_template_points[start_frame:end_frame,:,:].to(self.compute_device)
                _atomic_coordinates_batch = _atomic_coordinates_scaled[start_frame:end_frame,:,:].to(self.compute_device)
                kdotr_batch = torch.einsum("ikx,iax->ika", _xyz_template_points_batch, _atomic_coordinates_batch)
                kdotr_batch = kdotr_batch[:,None,:,:] * self.radius_shells[None,:,None,None]
                exponent = torch.exp(1j * kdotr_batch) * kernelAtoms[None,:,None,:] ## (n_templates, n_shells, n_inplanes, n_atoms)
                templates_fourier_batch = torch.sum(exponent, dim = 3) ## (n_templates, n_shells, n_inplanes)
                _templates_fourier[start_frame:end_frame,:,:] = templates_fourier_batch.to(self.storage_device)
        _iterate_kernel_with_memory_constraints(self.n_frames, self.n_images_per_frame, _batch_kernel)
        return _templates_fourier


def _iterate_kernel_with_memory_constraints(n_frames: int, n_images_per_frame: int, kernel: Callable[[int, int, int, int], None], verbose: bool = False):
    batch_size_frame = n_frames
    success = False
    while not success and batch_size_frame > 1:
        n_batches = ceil(n_frames / batch_size_frame)
        try:
            # TODO: Check if performance issue from exponent being in the kernel function scope now
            if verbose:
                print(f"Batch size frames: {batch_size_frame}")
                from tqdm import trange
                tmp = trange(n_batches)
            else:
                tmp = range(n_batches)
            start_frame = 0
            end_frame = batch_size_frame
            for _ in tmp:
                end_frame = min(end_frame, n_frames)
                kernel(start_frame, end_frame, 0, n_images_per_frame)
                start_frame += batch_size_frame
                end_frame += batch_size_frame
            success = True
            break
        except torch.cuda.OutOfMemoryError:
            batch_size_frame //= 2
            continue
    batch_size_image = n_images_per_frame
    while not success and batch_size_image > 0:
        n_batches = ceil(n_images_per_frame / batch_size_image)
        try:
            if verbose:
                print(f"Batch size images: {batch_size_image}")
                from tqdm import trange
                tmp = trange(n_batches)
            else:
                tmp = range(n_batches)
            for frame in range(n_frames):
                print(f"Processing frame {frame + 1}/{n_frames}.")
                start = 0
                end = batch_size_image
                for _ in tmp:
                    print(f"Processing frame {frame + 1}/{n_frames}, images {start + 1} to {end}.")
                    end = min(end, n_images_per_frame)
                    kernel(frame, frame + 1, start, end)
                    start += batch_size_image
                    end += batch_size_image
            success = True
            break
        except torch.cuda.OutOfMemoryError:
            batch_size_image //= 2
            continue
    
    if not success:
        raise MemoryError("Insufficient memory to compute templates from atomic model.")


def _get_fourier_slices(polar_grid: PolarGrid, viewing_angles: ViewingAngles, precision: Precision, device: torch.device) -> torch.Tensor:
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
    _radius_shells = to_torch(polar_grid.radius_shells, precision, device)
    circles = _get_circles(viewing_angles, polar_grid, precision, device)
    if not polar_grid.uniform:
        return circles * _radius_shells[None,:,None]
    fourier_slices = circles.unsqueeze(1) * _radius_shells[None,:,None,None]
    fourier_slices = fourier_slices.flatten(1, 2)
    return fourier_slices


def _test_callable(fn: Callable[[torch.Tensor], torch.Tensor], precision: Precision, device: torch.device):
    if not callable(fn):
        raise ValueError("Function must be callable.")
    (float_type, complex_type, _) = precision.get_dtypes(default=Precision.SINGLE)
    test_input = torch.randn(1, 2, 3, dtype=float_type, device=device)
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


    def __init__(self, *,
        phys_data: Optional[PhysicalImages | CartesianGrid2D] = None,
        fourier_data: Optional[FourierImages | PolarGrid] = None,
        box_size: Optional[float | FloatArrayType] = None,
        viewing_angles: ViewingAngles,
        ctf : Optional[CTF] = None,
    ):
        super().__init__(phys_data, fourier_data, box_size, viewing_angles, ctf)
        assert self.viewing_angles is not None, "Viewing angles must be provided."
        if (self.viewing_angles.n_angles != self.n_images):
            raise ValueError(f"Number of viewing angles ({self.viewing_angles.n_angles}) must match number of templates ({self.n_images}).")
    
    
    @classmethod
    def generate_from_positions(
        cls,
        atomic_model: AtomicModel,
        viewing_angles: ViewingAngles,
        polar_grid: PolarGrid,
        box_size: float | FloatArrayType,
        atom_shape: AtomShape,
        compute_device: str | torch.device = "cpu",
        output_device: str | torch.device = "cpu",
        precision: Precision = Precision.DEFAULT,
        verbose : bool = False
    ):
        generator = TemplateGenerator(
            atomic_model=atomic_model,
            viewing_angles=viewing_angles,
            polar_grid=polar_grid,
            box_size=box_size,
            storage_device=output_device,
            compute_device=compute_device,
            atom_shape=atom_shape,
            precision=precision
        )
        templates_fourier = generator.generator()

        data = FourierImages(templates_fourier, polar_grid=polar_grid)
        return cls(fourier_data=data, viewing_angles=viewing_angles, box_size=box_size)


    ## TODO: merge into TemplateGenerator
    @classmethod
    def generate_from_physical_volume(
        cls,
        volume: Volume,
        polar_grid: PolarGrid,
        viewing_angles: ViewingAngles,
        precision: Precision = Precision.DEFAULT,
        compute_device: torch.device | None = None,
        output_device: str | torch.device | None = "cpu",
        nufft_eps: float = 1.0e-12,
        verbose: bool = False
    ):
        if volume.density_physical is None:
            raise ValueError("No physical volume found")
        _device = get_device(compute_device)
        _output_device = get_device(output_device)
        (torch_float_type, _, _) = precision.get_dtypes(default=Precision.SINGLE)
        n_templates = viewing_angles.n_angles

        volume.density_physical = volume.density_physical.to(_device)
        fourier_slices = _get_fourier_slices(polar_grid, viewing_angles, precision, _device)
        templates_fourier = volume_phys_to_fourier_points(
            volume = volume,
            fourier_slices = fourier_slices,
            eps = nufft_eps,
            precision = Precision.SINGLE if precision == Precision.DEFAULT else precision,
            input_device = _device,
            output_device = _output_device,
            verbose = verbose
        )

        origin = torch.tensor([0.0, 0.0, 0.0], dtype = torch_float_type, device=_device).unsqueeze(0)
        centers = volume_phys_to_fourier_points(
            volume = volume,
            fourier_slices = origin,
            eps = nufft_eps,
            precision = Precision.SINGLE if precision == Precision.DEFAULT else precision,
            input_device = _device,
            output_device= _output_device,
            verbose = verbose
        )
        offset = _get_offset(polar_grid, precision, _output_device)
        offset = offset[None,:] * centers
        templates_fourier -= offset

        if polar_grid.uniform:
            templates_fourier = templates_fourier.reshape(n_templates, polar_grid.n_shells, polar_grid.n_inplanes)
        data = FourierImages(images_fourier=templates_fourier, polar_grid=polar_grid)

        return cls(fourier_data=data, viewing_angles=viewing_angles)


    ## TODO: merge into TemplateGenerator
    @classmethod
    def generate_from_function(
        cls,
        function: Callable[[torch.Tensor], torch.Tensor],
        viewing_angles: ViewingAngles,
        polar_grid: PolarGrid,
        compute_device: str | torch.device | None = None,
        output_device: str | torch.device | None = "cpu",
        precision: Precision = Precision.DEFAULT
    ):
        _device = get_device(compute_device)
        _output_device = get_device(output_device)
        _test_callable(fn=function, precision=precision, device=_device)
        (torch_float_type, torch_complex_type, _) = precision.get_dtypes(default=Precision.SINGLE)
        fourier_slices = _get_fourier_slices(polar_grid, viewing_angles, precision, _device)
        ## TODO: batch template generation for memory management
        templates_fourier = function(fourier_slices).reshape(viewing_angles.n_angles, polar_grid.n_shells, polar_grid.n_inplanes).to(_output_device)
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
    

    def clone(self) -> "Templates":
        """Returns a copy of the current templates object."""
        phys_data = PhysicalImages(self.images_phys.clone(), pixel_size=self.phys_grid.pixel_size) if self.has_physical_images() else None
        fourier_data = FourierImages(self.images_fourier.clone(), self.polar_grid) if self.has_fourier_images() else None
        return Templates(
            phys_data=phys_data,
            fourier_data=fourier_data,
            box_size=self.box_size,
            viewing_angles=self.viewing_angles,
            ctf=self.ctf
        )