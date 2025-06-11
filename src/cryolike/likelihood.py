import numpy as np
import torch
from scipy.special import gammaln as lgamma
from typing import Literal, overload, Callable

from cryolike.microscopy import CTF, translation_kernel_fourier, fourier_polar_to_cartesian_phys
from cryolike.stacks import Images, Templates
from cryolike.grids import PolarGrid
from cryolike.util import Precision, to_torch, absq, complex_mul_real

from cryolike.grids import Volume
from cryolike.util import AtomicModel, FloatArrayType, AtomShape
from cryolike.metadata import ViewingAngles


class LikelihoodFourier:
    """
    Class to compute the likelihood of images given templates in Fourier space
    """

    polar_grid: PolarGrid
    n_pixels: int
    weights: torch.Tensor
    s_points: torch.Tensor
    float_type: torch.dtype
    complex_type: torch.dtype
    device: torch.device
    constant: float
    Iss: float
    p: float

    def __init__(self, 
        polar_grid: PolarGrid, 
        n_pixels: int, 
        precision: Precision = Precision.SINGLE, 
        device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        identity_kernel: Callable[[PolarGrid, Precision], torch.Tensor] | None = None
    ):
        self.float_type, self.complex_type, _ = precision.get_dtypes(default=Precision.SINGLE)
        self.device = device
        self.polar_grid = polar_grid
        self.n_pixels = n_pixels
        self.weights = torch.tensor(polar_grid.weight_points, dtype = self.float_type, device = device) * (2.0 * np.pi) ** 2
        self.s_points = _ensure_identity_kernel(
            polar_grid = polar_grid,
            identity_kernel = identity_kernel,
            precision = precision,
            device = device
        ).flatten()
        self.Iss = torch.sum(self.s_points.abs() ** 2 * self.weights).cpu().item()
        self.p = self.n_pixels / 2.0 - 2.0
        lg = lgamma(self.n_pixels / 2.0 - 2.0).item()
        self.constant = (3.0 - self.n_pixels) / 2.0 * np.log(2 * np.pi) \
            - np.log(2) - 0.5 * np.log(self.Iss) \
            + lg \
            + self.p * np.log(2 * self.Iss)


    # @overload
    # def __call__(self, *args, return_cross_correlation: bool = False, **kwargs) -> torch.Tensor: ...
    # @overload
    # def __call__(self, *args, return_cross_correlation: bool = True, **kwargs) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self,
        templates_fourier: torch.Tensor,
        images_fourier: torch.Tensor,
        # return_cross_correlation: bool = False
    ):
        
        images_fourier = images_fourier.view(images_fourier.shape[0], -1).to(self.complex_type).to(self.device)
        templates_fourier = templates_fourier.view(templates_fourier.shape[0], -1).to(self.complex_type).to(self.device)

        ## BioEM likelihood
        Isy = torch.sum((self.s_points * images_fourier) * self.weights, dim = 1)
        Isx = torch.sum((self.s_points * templates_fourier) * self.weights, dim = 1)
        Iyy = torch.sum(absq(images_fourier) * self.weights, dim = 1)
        Ixx = torch.sum(absq(templates_fourier) * self.weights, dim = 1)
        Ixy = torch.sum(complex_mul_real(images_fourier, templates_fourier.conj()) * self.weights, dim = 1)

        A = - absq(Isx) + Ixx * self.Iss
        B = - Isx.real * Isy.real - Isx.imag * Isy.imag + Ixy * self.Iss
        C = absq(Isy) - Iyy * self.Iss

        D = - (B ** 2 / A + C)
        log_likelihood = -self.p * torch.log(D) - 0.5 * torch.log(A) + self.constant
        return log_likelihood

        # log_likelihood = log_likelihood.cpu()
        # _return = log_likelihood
        # if return_cross_correlation:
        #     cross_correlation = Ixy / torch.sqrt(Ixx * Iyy)
        #     cross_correlation = cross_correlation.cpu()
        #     _return = (log_likelihood, cross_correlation)
        # return _return
    

class LikelihoodFourierModel:

    """
    Class to compute the likelihood of images given a model in Fourier space
    """

    polar_grid: PolarGrid
    n_pixels: int
    box_size: float
    model: Volume | AtomicModel | Callable[[torch.Tensor], torch.Tensor] | Templates
    viewing_angles: ViewingAngles
    atom_shape: AtomShape
    precision: Precision
    device: torch.device
    float_type: torch.dtype
    complex_type: torch.dtype
    identity_kernel: Callable[[PolarGrid, Precision], torch.Tensor]
    likelihood: LikelihoodFourier

    def __init__(
        self,
        model: Volume | AtomicModel | Callable[[torch.Tensor], torch.Tensor] | Templates,
        polar_grid: PolarGrid,
        box_size: float,
        n_pixels: int,
        viewing_angles: ViewingAngles | None = None,
        atom_shape: AtomShape = AtomShape.GAUSSIAN,
        precision: Precision = Precision.SINGLE,
        device: torch.device = torch.device('cpu'),
        identity_kernel: Callable[[PolarGrid, Precision], torch.Tensor] | None = None,
        verbose: bool = False
    ):
        self.device = device
        self.model = model
        self.polar_grid = polar_grid
        self.n_pixels = n_pixels
        self.box_size = box_size
        self.precision = precision
        self.float_type, self.complex_type, _ = precision.get_dtypes(default=Precision.SINGLE)
        self.viewing_angles = _ensure_viewing_angles(
            va=viewing_angles,
            torch_float_type=self.float_type
        )
        self.atom_shape = atom_shape
        self.likelihood = LikelihoodFourier(
            polar_grid=self.polar_grid,
            n_pixels=self.n_pixels,
            precision=self.precision,
            device=self.device,
            identity_kernel=identity_kernel
        )


    def __call__(
        self,
        images: Images,
        template_indices: torch.Tensor | None = None,
        x_displacements: torch.Tensor | FloatArrayType | None = None,
        y_displacements: torch.Tensor | FloatArrayType | None = None,
        gammas: torch.Tensor | FloatArrayType | None = None,
        ctf: CTF | None = None,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Calculate the likelihood of the images given the model.
        
        Parameters:
            images (Images): The images to calculate the likelihood for.
            x_displacements (torch.Tensor | FloatArrayType | None): The x displacements to apply to the templates.
            y_displacements (torch.Tensor | FloatArrayType | None): The y displacements to apply to the templates.
            ctf (CTF | None): The CTF to apply to the templates.
        Returns:
            torch.Tensor: The likelihood of the images given the model.
        """

        if isinstance(self.model, Templates):
            templates = self.model.clone()
        else:
            templates = _make_templates_from_model(
                model=self.model,
                polar_grid=self.polar_grid,
                box_size=self.box_size,
                viewing_angles=self.viewing_angles,
                precision=self.precision,
                atom_shape=self.atom_shape,
                device=self.device,
                output_device=self.device,
                verbose=verbose
            )
        if template_indices is not None:
            templates.select_images(template_indices)
        x_disps, y_disps = _validate_displacements(x_displacements, y_displacements, search_displacments=False)
        templates.displace_images_fourier(
            x_displacements=x_disps,
            y_displacements=y_disps
        )
        if gammas is not None:
            templates.rotate_images_fourier_discrete(gammas)
        if ctf is not None:
            templates.apply_ctf(ctf)
        # templates.normalize_images_fourier(ord=2, use_max=False)
        images_fourier = images.images_fourier.to(self.complex_type).to(self.device)
        log_likelihood = self.likelihood(
            templates_fourier=templates.images_fourier,
            images_fourier=images_fourier
        )
        return log_likelihood


def _ensure_identity_kernel(
    polar_grid: PolarGrid,
    identity_kernel: Callable[[PolarGrid, Precision], torch.Tensor] | None = None,
    precision: Precision = Precision.SINGLE,
    device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
) -> torch.Tensor:
    float_type, _, _ = precision.get_dtypes(default=Precision.SINGLE)
    if identity_kernel is not None:
        if not callable(identity_kernel):
            raise ValueError(f"identity_kernel must be a callable function that takes PolarGrid and Precision as arguments, but got {type(identity_kernel)}.")
        s_points = identity_kernel(polar_grid, precision)
    else:
        _x_points = torch.tensor(polar_grid.x_points, dtype=float_type, device=device)
        _y_points = torch.tensor(polar_grid.y_points, dtype=float_type, device=device)
        s_points = torch.sinc(2.0 * _x_points) * torch.sinc(2.0 * _y_points) * 4.0
        s_points = s_points.reshape(polar_grid.n_shells, polar_grid.n_inplanes)
    if not isinstance(s_points, torch.Tensor):
        raise ValueError("identity_kernel must return a torch.Tensor.")
    if s_points.shape != (polar_grid.n_shells, polar_grid.n_inplanes):
        raise ValueError(f"identity_kernel must return a tensor of shape {(polar_grid.n_shells, polar_grid.n_inplanes)}.")
    return s_points.to(device)


def _ensure_viewing_angles(va: ViewingAngles | None, torch_float_type: torch.dtype) -> ViewingAngles:
    if va is not None:
        return va
    print("viewing_angles is None. Not rotating the model.")
    azimus = torch.tensor([0.0], dtype = torch_float_type)
    polars = torch.tensor([0.0], dtype = torch_float_type)
    gammas = torch.tensor([0.0], dtype = torch_float_type)
    return ViewingAngles(azimus = azimus, polars = polars, gammas = gammas)
   

def _make_templates_from_model(
    model: Volume | AtomicModel | Callable[[torch.Tensor], torch.Tensor],
    polar_grid: PolarGrid,
    box_size: float,
    viewing_angles: ViewingAngles,
    precision: Precision,
    atom_shape: AtomShape = AtomShape.GAUSSIAN,
    device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    output_device: torch.device = torch.device('cpu'),
    verbose: bool = False
) -> Templates:
    if isinstance(model, Volume):
        return Templates.generate_from_physical_volume(
            volume=model,
            polar_grid=polar_grid,
            viewing_angles=viewing_angles,
            precision=precision,
            device=device,
            output_device=output_device,
            verbose=verbose
        )
    elif isinstance(model, AtomicModel):
        return Templates.generate_from_positions(
            atomic_model=model,
            viewing_angles=viewing_angles,
            polar_grid=polar_grid,
            box_size=box_size,
            atom_shape=atom_shape,
            device=device,
            output_device=output_device,
            precision=precision,
            verbose=verbose
        )
    elif callable(model):
        return Templates.generate_from_function(
            function=model,
            viewing_angles=viewing_angles,
            polar_grid=polar_grid,
            precision=precision,
            device=device,
            output_device=output_device
        )
    raise ValueError("Model must be an instance of Volume of AtomicModel")


def _validate_displacements(
    x_displacements : torch.Tensor | FloatArrayType | None,
    y_displacements : torch.Tensor | FloatArrayType | None,
    search_displacments: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    x_displacements = torch.tensor(x_displacements) if isinstance(x_displacements, np.ndarray) else x_displacements
    y_displacements = torch.tensor(y_displacements) if isinstance(y_displacements, np.ndarray) else y_displacements
    if search_displacments:
        if (x_displacements is None or y_displacements is None):
            raise ValueError("x_displacements and y_displacements must be provided if search_displacements is True.")
        if x_displacements.dim() != 1 or y_displacements.dim() != 1:
            raise ValueError("x_displacements and y_displacements must be 1D tensors.")
        if x_displacements.shape[0] != y_displacements.shape[0]:
            raise ValueError("x_displacements and y_displacements must have the same number of displacements.")

    x_displacements = torch.tensor([0.]) if x_displacements is None else x_displacements
    y_displacements = torch.tensor([0.]) if y_displacements is None else y_displacements

    return (x_displacements, y_displacements)
