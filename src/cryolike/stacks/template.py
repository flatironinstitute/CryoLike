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
    azimus: torch.Tensor,  # azimuthal angle in radians
    gammas: torch.Tensor | None,  # inplane rotation in radians
) -> torch.Tensor:
    
    if gammas is None:
        thetas = thetas[None,:]
    else:
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
    if viewing_angles.gammas is not None:
        gammas = to_torch(viewing_angles.gammas, precision, device)
    else:
        gammas = None
    thetas = to_torch(polar_grid.theta_shell if polar_grid.uniform else polar_grid.theta_points, precision, device)
    circles = _fourier_circles(thetas, polars, azimus, gammas)
    return circles


def _get_offset(polar_grid: PolarGrid, precision: Precision, device: torch.device) -> torch.Tensor:
    assert polar_grid.x_points is not None and polar_grid.y_points is not None, \
        "Polar grid must have x_points and y_points defined for offset computation."
    x_points = to_torch(polar_grid.x_points, precision, device)
    y_points = to_torch(polar_grid.y_points, precision, device)
    offset = torch.sinc(2.0 * x_points) * torch.sinc(2.0 * y_points)
    return offset.reshape(1, polar_grid.n_shells, polar_grid.n_inplanes)


def _scale_atom_radius(atomic_radii: torch.Tensor, box_size: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the atomic radius scaled to the unit box size."""
    atomic_radii_scaled = atomic_radii / box_size * 2.0
    pi_atomic_radii_sq_times_two = 2.0 * (np.pi * atomic_radii_scaled) ** 2
    return atomic_radii_scaled, pi_atomic_radii_sq_times_two


def _scale_atomic_coordinates(atomic_coordinates: torch.Tensor, box_size: float) -> torch.Tensor:
    """Returns the atomic coordinates scaled to the unit box size and scaled for the translation kernel."""
    return atomic_coordinates * 2.0 / box_size * (- 2.0 * np.pi)


class AtomicTemplateGenerator:

    """Class for generating templates from an atomic model, viewing angles, and polar grid.
    also allow gradient computation for the templates with respect to the atomic coordinates and viewing angles."""

    polar_grid: PolarGrid
    box_size: float
    atom_shape: AtomShape

    radius_shells: torch.Tensor
    radius_shells_sq: torch.Tensor
    precision: Precision
    float_type: torch.dtype
    complex_type: torch.dtype
    device: torch.device

    generator: Callable[[torch.Tensor, ViewingAngles], torch.Tensor]

    def __init__(
        self,
        polar_grid: PolarGrid,
        box_size: float | FloatArrayType,
        atom_radii: torch.Tensor,
        atom_shape: AtomShape,
        device: str | torch.device = torch.device("cuda"),
        precision: Precision = Precision.DEFAULT,
    ):
        
        self.float_type, self.complex_type, _ = precision.get_dtypes(default=Precision.SINGLE)

        self.polar_grid = polar_grid
        if not self.polar_grid.uniform:
            raise NotImplementedError("Non-uniform polar grids not implemented yet.")
        self.device = get_device(device)
        print("Using device:", self.device, "for templates generation.")

        if isinstance(box_size, float):
            self.box_size = box_size
        elif isinstance(box_size, (list, tuple, np.ndarray)):
            self.box_size = box_size[0]
        else:
            raise ValueError(f"Unsupported box_size type: {type(box_size)}. Expected float, list, tuple, or np.ndarray.")

        self.atom_shape = atom_shape
        self.precision = precision
        
        self.radius_shells = to_torch(polar_grid.radius_shells, precision, self.device)
        self.radius_shells_sq = self.radius_shells ** 2

        atom_radii = to_torch(atom_radii, precision, self.device)
        self.atom_radii_scaled, self.pi_atomic_radii_sq_times_two = _scale_atom_radius(atom_radii, self.box_size)

        self.offset = _get_offset(self.polar_grid, precision=self.precision, device=self.device)

        if self.atom_shape == AtomShape.GAUSSIAN:
            self.generator = self._gaussian_atom_kernel
        elif self.atom_shape == AtomShape.HARD_SPHERE:
            self.generator = self._hard_sphere_atom_kernel
        else:
            raise ValueError(f"Unsupported atom shape: {self.atom_shape}. Supported shapes are: {AtomShape.GAUSSIAN}, {AtomShape.HARD_SPHERE}.")


    def _calc_kdotr(self, _xyz_template_points: torch.Tensor, _atomic_coordinates_scaled: torch.Tensor) -> torch.Tensor:
        """Calculates the dot product of the template points and atomic coordinates."""
        kdotr = torch.einsum("mnkx,mnax->mnka", _xyz_template_points, _atomic_coordinates_scaled)
        return kdotr.unsqueeze(2) * self.radius_shells[None,None,:,None,None]  # (n_templates, n_shells, n_inplanes, n_atoms)


    def _common_kernel(
        self,
        atomic_coordinates: torch.Tensor,
        viewing_angles: ViewingAngles,
    ):
        """Generates the Fourier templates for each atom using the common kernel."""
        atomic_coordinates.to(dtype=self.float_type, device=self.device)
        viewing_angles.to(precision=self.precision, device=self.device)

        n_frames = atomic_coordinates.shape[0]
        n_atoms = atomic_coordinates.shape[1]
        n_angles = viewing_angles.n_angles

        _atomic_coordinates_scaled = _scale_atomic_coordinates(atomic_coordinates, self.box_size)
        _xyz_template_points = _get_circles(viewing_angles, self.polar_grid, self.precision, self.device)

        _atomic_coordinates_scaled = _atomic_coordinates_scaled.unsqueeze(1).expand(-1, n_angles, -1, -1) # (n_frames, n_angles, n_atoms, 3)
        _xyz_template_points = _xyz_template_points.unsqueeze(0).expand(n_frames, -1, -1, -1)  # (n_frames, n_angles, n_inplanes, 3)

        return _atomic_coordinates_scaled, _xyz_template_points, n_frames, n_atoms, n_angles


    def _gaussian_atom_kernel(
        self,
        atomic_coordinates: torch.Tensor,
        viewing_angles: ViewingAngles,
    ) -> torch.Tensor:
        """Generates the Fourier templates using a Gaussian kernel for each atom."""

        _atomic_coordinates_scaled, _xyz_template_points, n_frames, n_atoms, n_angles = \
            self._common_kernel(atomic_coordinates, viewing_angles)

        _log_norm = - 1.5 * np.log(2 * np.pi) - 3 * torch.log(self.atom_radii_scaled) - np.log(n_atoms)
        _offset = self.offset * torch.sum(torch.exp(_log_norm))
        _gaussKernelAtoms = - self.radius_shells_sq[:,None] * self.pi_atomic_radii_sq_times_two[None,:] + _log_norm[None,:]

        kdotr = self._calc_kdotr(_xyz_template_points, _atomic_coordinates_scaled)
        exponent = torch.complex(_gaussKernelAtoms[None, None,:,None,:], kdotr)
        exponent.exp_()
        templates_fourier = torch.sum(exponent, dim = 4) - _offset.unsqueeze(0)

        return templates_fourier


    def _hard_sphere_atom_kernel(
        self,
        atomic_coordinates: torch.Tensor,
        viewing_angles: ViewingAngles,
    ) -> torch.Tensor:
        """Generates the Fourier templates using a hard sphere density for each atom."""
        _atomic_coordinates_scaled, _xyz_template_points, n_frames, n_atoms, n_angles = \
            self._common_kernel(atomic_coordinates, viewing_angles)

        kR = 2 * np.pi * self.radius_shells[:,None] * self.atom_radii_scaled[None,:]
        kernelAtoms = (torch.sin(kR) - kR * torch.cos(kR))
        kernelAtoms *= self.radius_shells.pow(-3)[:,None] / (8 * np.pi ** 2) * 3 / n_atoms
        
        kdotr = self._calc_kdotr(_xyz_template_points, _atomic_coordinates_scaled)
        exponent = torch.exp(1j * kdotr) * kernelAtoms[None,None,:,None,:] ## (n_templates, n_shells, n_inplanes, n_atoms)
        templates_fourier = torch.sum(exponent, dim = 4) ## (n_templates, n_shells, n_inplanes)

        return templates_fourier


def _iterate_kernel_with_memory_constraints(n_frames: int, n_images_per_frame: int, kernel: Callable[[int, int, int, int], None]):
    batch_size_frame = n_frames
    success = False
    while not success and batch_size_frame > 1:
        n_batches = ceil(n_frames / batch_size_frame)
        try:
            start_frame = 0
            end_frame = batch_size_frame
            while start_frame < end_frame and end_frame <= n_frames:
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
            for frame in range(n_frames):
                start = 0
                end = batch_size_image
                while start < end and end <= n_images_per_frame:
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
    viewing_angles: ViewingAngles

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
        """Generates templates from atomic coordinates and viewing angles."""

        n_frames = atomic_model.n_frames
        n_angles = viewing_angles.n_angles
        generator = AtomicTemplateGenerator(
            polar_grid=polar_grid,
            box_size=box_size,
            atom_radii=atomic_model.atom_radii,
            atom_shape=atom_shape,
            device=compute_device,
            precision=precision
        )
        try:
            templates_fourier = torch.zeros(
                (n_frames, n_angles, polar_grid.n_shells, polar_grid.n_inplanes),
                dtype=generator.complex_type,
                device=get_device(output_device)
            )
        except torch.OutOfMemoryError:
            if output_device == "cpu":
                raise MemoryError("Templates cannot even fit in CPU memory! Consider using a smaller polar grid or fewer frames.")
            else:
                print("Out of memory on output device, trying to allocate on CPU memory instead.")
                ## try again with CPU memory
                output_device = torch.device("cpu")
                templates_fourier = torch.zeros(
                    (n_frames, n_angles, polar_grid.n_shells, polar_grid.n_inplanes),
                    dtype=generator.complex_type,
                    device=output_device
                )
        def _kernel(start_frame: int, end_frame: int, start_image: int, end_image: int):
            _atomic_coordinates = atomic_model.atomic_coordinates[start_frame:end_frame, :, :].to(compute_device)
            _viewing_angles = viewing_angles.get_slice(start_image, end_image).to(device=compute_device)
            _templates_fourier = generator.generator(_atomic_coordinates, _viewing_angles)
            templates_fourier[start_frame:end_frame, start_image:end_image, :, :] = _templates_fourier
            return
        _iterate_kernel_with_memory_constraints(n_frames, n_angles, _kernel)
        templates_fourier = templates_fourier.view(n_frames * n_angles, polar_grid.n_shells, polar_grid.n_inplanes)
        data = FourierImages(templates_fourier, polar_grid=polar_grid)
        viewing_angles = viewing_angles.repeat(n_frames)
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
        if not polar_grid.uniform:
            raise NotImplementedError("Non-uniform polar grids not implemented yet.")
        
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
        ).reshape(n_templates, polar_grid.n_shells, polar_grid.n_inplanes)

        origin = torch.tensor([0.0, 0.0, 0.0], dtype = torch_float_type, device=_device).unsqueeze(0).unsqueeze(0)
        print(f"Origin shape: {origin.shape}, volume.shape: {volume.density_physical.shape}")
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
        print(f"Offset: {offset.shape}, centers: {centers.shape}, templates_fourier: {templates_fourier.shape}")
        templates_fourier -= offset * centers
        # templates_fourier = templates_fourier.reshape(n_templates, polar_grid.n_shells, polar_grid.n_inplanes)
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