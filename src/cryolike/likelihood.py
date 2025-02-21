import numpy as np
import torch
from scipy.special import gammaln as lgamma
from typing import Optional, Literal

from cryolike.microscopy import CTF, translation_kernel_fourier, fourier_polar_to_cartesian_phys
from cryolike.stacks import Images, Templates
from cryolike.grids import PolarGrid
from cryolike.util import Precision, to_torch, absq, complex_mul_real

from cryolike.grids import Volume
from cryolike.util import AtomicModel, FloatArrayType, AtomShape
from cryolike.metadata import ViewingAngles
from cryolike.stacks.image import _verify_displacements


def likelihood_physical(
    templates_phys : torch.Tensor,
    images_phys : torch.Tensor,
    return_cross_correlation : bool = False,
):
    
    raise NotImplementedError("Physical likelihood is still under development and not yet available. Please use Fourier likelihood instead.")
    
    n_pixels = images_phys.shape[1] * images_phys.shape[2]

    Co = torch.sum(images_phys, dim = (1,2))
    Cc = torch.sum(templates_phys, dim = (1,2))
    Coo = torch.sum(images_phys ** 2, dim = (1,2))
    Ccc = torch.sum(templates_phys ** 2, dim = (1,2))
    Coc = torch.sum(images_phys * templates_phys, dim = (1,2))
    
    if return_cross_correlation:
        cross_correlation = Coc / torch.sqrt(Ccc * Coo)
        cross_correlation = cross_correlation.cpu()
    
    ## Version with saddle-point-approximated lambda
    # first_term = n_pixels * (Ccc * Coo - Coc ** 2) + 2 * Co * Cc * Coc - Co ** 2 * Ccc - Coo * Cc ** 2
    # second_term = (n_pixels * Ccc - Cc ** 2) * (n_pixels - 2)
    
    # first_term = (1.5 - n_pixels * 0.5) * torch.log(first_term)
    # second_term = (n_pixels * 0.5 - 2) * torch.log(second_term)
    # third_term = 0.5 * np.log(np.pi) + (1 - n_pixels * 0.5) * (np.log(2 * np.pi) + 1)
    
    # log_likelihood = first_term + second_term + third_term
    # log_likelihood = log_likelihood.cpu()
    
    ## Version with integrating over lambda
    a = (n_pixels * (Ccc * Coo - Coc ** 2) + 2 * Co * Cc * Coc - Co ** 2 * Ccc - Coo * Cc ** 2) / 2 / (n_pixels * Ccc - Cc ** 2)
    p = (3.0 - n_pixels) * 0.5
    log_likelihood = (1.0 - n_pixels * 0.5) * np.log(2 * np.pi) - 0.5 * torch.log(n_pixels * Ccc - Cc ** 2) - np.log(2) + lgamma(-p) + p * torch.log(a)
    log_likelihood = log_likelihood.cpu()
    
    if return_cross_correlation:
        return log_likelihood, cross_correlation
    else:
        return log_likelihood
    
def likelihood_fourier(
    templates_fourier: torch.Tensor,
    images_fourier: torch.Tensor,
    polar_grid: PolarGrid,
    n_pixels: int,
    return_cross_correlation: bool = False
):
    
    # n_shells = polar_grid.n_shells
    # n_inplanes = polar_grid.n_inplanes
    # n_pixels = n_shells * n_inplanes
    float_type = torch.float32 if templates_fourier.dtype == torch.complex64 else torch.float64
    device = templates_fourier.device
    weights = torch.tensor(polar_grid.weight_points, dtype = float_type, device = device) * (2.0 * np.pi) ** 2
    
    images_fourier = images_fourier.view(images_fourier.shape[0], -1)
    templates_fourier = templates_fourier.view(templates_fourier.shape[0], -1)
    
    x_points = torch.tensor(polar_grid.x_points, dtype = float_type, device = device)
    y_points = torch.tensor(polar_grid.y_points, dtype = float_type, device = device)
    s_points = torch.sinc(2.0 * x_points) * torch.sinc(2.0 * y_points) * 4.0
    
    ## BioEM likelihood
    Iss = torch.sum(s_points.abs() ** 2 * weights).cpu().item()
    Isy = torch.sum((s_points * images_fourier) * weights, dim = 1)
    Isx = torch.sum((s_points * templates_fourier) * weights, dim = 1)
    Iyy = torch.sum(absq(images_fourier) * weights, dim = 1)
    Ixx = torch.sum(absq(templates_fourier) * weights, dim = 1)
    Ixy = torch.sum(complex_mul_real(images_fourier, templates_fourier.conj()) * weights, dim = 1)
    if return_cross_correlation:
        cross_correlation = Ixy / torch.sqrt(Ixx * Iyy)
        cross_correlation = cross_correlation.cpu()

    A = - absq(Isx) + Ixx * Iss
    B = - Isx.real * Isy.real - Isx.imag * Isy.imag + Ixy * Iss
    C = absq(Isy) - Iyy * Iss
    
    D = - (B ** 2 / A + C)
    p = n_pixels / 2.0 - 2.0
    constant = (3.0 - n_pixels) / 2.0 * np.log(2 * np.pi) - np.log(2) - 0.5 * np.log(Iss) + lgamma(n_pixels / 2.0 - 2.0) + p * np.log(2 * Iss)
    log_likelihood = -p * torch.log(D) - 0.5 * torch.log(A) + constant

    log_likelihood = log_likelihood.cpu()
    if return_cross_correlation:
        return log_likelihood, cross_correlation
    else:
        return log_likelihood    


def calc_likelihood_optimal_pose(
    template : Templates,
    image : Images,
    template_indices : torch.Tensor,
    displacements_x : torch.Tensor | None = None,
    displacements_y : torch.Tensor | None = None,
    inplane_rotations : torch.Tensor | None = None,
    ctf : CTF | None = None,
    mode : str = "phys",
    return_distance : bool = True,
    return_likelihood : bool = False,
    return_cross_correlation : bool = False,
    precision : Precision = Precision.SINGLE,
    use_cuda : bool = True
):
    """
    Calculate the distance between the optimal templates and the true templates.
    """
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    n_images = image.n_images
    ## calculate the translation kernels
    translation_kernel___ = None
    if displacements_x is not None and displacements_y is not None:
        displacements_x *= 2.0 / template.box_size[0]
        displacements_y *= 2.0 / template.box_size[1]
        assert displacements_x is not None
        assert displacements_y is not None
        translation_kernel___ = translation_kernel_fourier(template.polar_grid, displacements_x, displacements_y, precision, device)   
    inplane_rotations_discrete = None
    if inplane_rotations is not None:
        inplane_rotations_step = 2 * np.pi / template.polar_grid.n_inplanes
        inplane_rotations_discrete = - torch.round(inplane_rotations / inplane_rotations_step).to(torch.int64)
    templates_optimal = to_torch(template.images_fourier, precision, device)[template_indices]
    if translation_kernel___ is not None:
        templates_optimal = templates_optimal * translation_kernel___
    if inplane_rotations is not None:
        assert inplane_rotations_discrete is not None
        # templates_optimal = templates_optimal.roll(inplane_rotations_discrete, dims = 2)
        for i in range(templates_optimal.shape[0]):
            inplane_rotation_discrete = int(inplane_rotations_discrete[i].item())
            templates_optimal[i] = torch.roll(templates_optimal[i], shifts = inplane_rotation_discrete, dims = 1)
    if ctf is not None:
        templates_optimal = ctf.apply(templates_optimal)
    log_likelihood = None
    distances = None
    cross_correlation = None
    if mode == "phys":
        templates_optimal = templates_optimal.cpu().numpy()
        templates_optimal_physical = fourier_polar_to_cartesian_phys(
            grid_fourier_polar = template.polar_grid,
            grid_cartesian_phys = image.phys_grid,
            image_polar = templates_optimal.reshape(n_images, -1),
            eps = 1e-12,
            precision = precision,
            use_cuda = use_cuda
        ).real
        images_phys = image.images_phys.real
        templates_optimal_physical = to_torch(templates_optimal_physical, precision, device)
        images_phys = to_torch(images_phys, precision, device)
        if return_distance:
            distances = torch.norm(templates_optimal_physical - images_phys, dim = (1, 2), p = 2).cpu()
        if return_likelihood:
            output_likelihood = likelihood_physical(
                templates_phys = templates_optimal_physical,
                images_phys = images_phys,
                return_cross_correlation = return_cross_correlation
            )
            if return_cross_correlation:
                log_likelihood, cross_correlation = output_likelihood
            else:
                log_likelihood = output_likelihood
    elif mode == "fourier":
        if not image.has_fourier_images():
            raise ValueError("Fourier images not found. Transform or Create Fourier images first before calculating the distance.")
        images_fourier = image.images_fourier
        images_fourier = to_torch(images_fourier, precision, device)
        polar_grid = template.polar_grid
        if return_distance:
            weight = to_torch(polar_grid.weight_points, precision, device)
            distances = torch.sum((templates_optimal - images_fourier).abs() ** 2 * weight, dim = 1).cpu()
        if return_likelihood:
            output_likelihood = likelihood_fourier(
                templates_fourier = templates_optimal,
                images_fourier = images_fourier,
                polar_grid = polar_grid,
                n_pixels = image.phys_grid.n_pixels[0] * image.phys_grid.n_pixels[1],
                return_cross_correlation = return_cross_correlation
            )
            if return_cross_correlation:
                log_likelihood, cross_correlation = output_likelihood
            else:
                log_likelihood = output_likelihood
    if return_likelihood and not return_distance and not return_cross_correlation:
        return log_likelihood
    if  not return_likelihood and return_distance and not return_cross_correlation:
        return distances
    if not return_distance and not return_likelihood and return_cross_correlation:
        return cross_correlation
    if return_likelihood and return_distance and not return_cross_correlation:
        return log_likelihood, distances
    if return_likelihood and not return_distance and return_cross_correlation:
        return log_likelihood, cross_correlation
    if not return_likelihood and return_distance and return_cross_correlation:
        return distances, cross_correlation
    if return_likelihood and return_distance and return_cross_correlation:
        return log_likelihood, distances, cross_correlation


## TODO: optimize this function for computational efficiency
## the convention here for 2D rotation and translation is that the image is translated first and then rotated
## which sounds counter-intuitive but is consistent with the convention in cross_correlation_likelihood.py
## CTF also have to be applied after translation and rotation as there may be anisotropy in the CTF
def calc_physical_likelihood_images_given_optimal_pose(
    images : Images,
    model : Volume | AtomicModel,
    atom_shape: Literal['hard-sphere'] | Literal['gaussian'] | AtomShape = AtomShape.GAUSSIAN,
    viewing_angles : ViewingAngles = None,
    search_displacements : bool = False,
    x_displacements : torch.Tensor | FloatArrayType | None = None,
    y_displacements : torch.Tensor | FloatArrayType | None = None,
    ctf : CTF | None = None,
    device : str | torch.device = 'cpu',
    precision : Precision = Precision.SINGLE,
    verbose : bool = False
):
    use_cuda = not device == 'cpu' and torch.cuda.is_available()
    device = torch.device(device) if use_cuda else torch.device('cpu')
    (torch_float_type, torch_complex_type, _) = precision.get_dtypes(default=Precision.SINGLE)
    if viewing_angles is None:
        print("viewing_angles is None. Not rotating the model.")
        azimus = torch.tensor([0.0], dtype = torch_float_type)
        polars = torch.tensor([0.0], dtype = torch_float_type)
        gammas = torch.tensor([0.0], dtype = torch_float_type)
        viewing_angles = ViewingAngles(azimus = azimus, polars = polars, gammas = gammas)
    templates = None
    if isinstance(model, Volume):
        templates = Templates.generate_from_physical_volume(
            volume=model,
            polar_grid=images.polar_grid,
            viewing_angles=viewing_angles,
            precision=precision,
            verbose=verbose
        )
    elif isinstance(model, AtomicModel):
        templates = Templates.generate_from_positions(
            atomic_model=model,
            viewing_angles=viewing_angles,
            polar_grid=images.polar_grid,
            box_size=images.box_size,
            atom_shape=atom_shape,
            precision=precision,
            verbose = verbose
        )
    else:
        raise ValueError("model must be an instance of Volume or AtomicModel")
    if templates is None:
        raise ValueError("Templates could not be generated.")
    templates.normalize_images_fourier(ord=2, use_max=False)
    images.images_phys = images.images_phys.real.to(torch_float_type).to(device)
    images.center_physical_image_signal()
    if search_displacements:
        if x_displacements is None or y_displacements is None:
            raise ValueError("x_displacements and y_displacements must be provided if search_displacements is True.")
        if x_displacements.dim() != 1 or y_displacements.dim() != 1:
            raise ValueError("x_displacements and y_displacements must be 1D tensors.")
        if x_displacements.shape[0] != y_displacements.shape[0]:
            raise ValueError("x_displacements and y_displacements must have the same number of displacements.")
        n_displacements = x_displacements.shape[0]
        n_imgs = images.n_images
        templates_original = templates.images_fourier.clone().to(device)
        likelihood = torch.zeros((n_imgs, n_displacements), dtype = torch_float_type, device = 'cpu')
        cross_correlation = torch.zeros((n_imgs, n_displacements), dtype = torch_float_type, device = 'cpu')
        for i in range(n_displacements):
            templates.images_fourier = templates_original.clone()
            x_disp = x_displacements[i] * torch.ones(n_imgs, dtype = torch_float_type, device = device)
            y_disp = y_displacements[i] * torch.ones(n_imgs, dtype = torch_float_type, device = device)
            templates.displace_images_fourier(x_disp, y_disp, precision)
            templates.rotate_images_fourier_discrete(viewing_angles.gammas)
            if ctf is not None:
                templates.apply_ctf(ctf)
            templates.images_phys = templates.transform_to_spatial(
                grid=images.phys_grid,
                precision=precision,
                use_cuda=use_cuda
            ).real.to(torch_float_type).to(device)
            templates.center_physical_image_signal()
            likelihood[:, i], cross_correlation[:, i] = likelihood_physical(
                templates_phys = templates.images_phys,
                images_phys = images.images_phys,
                return_cross_correlation = True
            )
    else:
        if x_displacements is not None or y_displacements is not None:
            templates.displace_images_fourier(x_displacements, y_displacements, precision)
        templates.rotate_images_fourier_discrete(viewing_angles.gammas)
        if ctf is not None:
            templates.apply_ctf(ctf)
        templates.images_phys = templates.transform_to_spatial(
            grid=images.phys_grid,
            precision=precision,
            use_cuda=use_cuda
        ).real.to(torch_float_type).to(device)
        templates.center_physical_image_signal()
        likelihood, cross_correlation = likelihood_physical(
            templates_phys = templates.images_phys,
            images_phys = images.images_phys,
            return_cross_correlation = True
        )
    return likelihood, cross_correlation


def calc_fourier_likelihood_images_given_optimal_pose(
    images : Images,
    model : Volume | AtomicModel,
    atom_shape: Literal['hard-sphere'] | Literal['gaussian'] | AtomShape = AtomShape.GAUSSIAN,
    viewing_angles : ViewingAngles = None,
    search_displacements : bool = False,
    x_displacements : torch.Tensor | FloatArrayType | None = None,
    y_displacements : torch.Tensor | FloatArrayType | None = None,
    ctf : CTF | None = None,
    device : str | torch.device = 'cpu',
    precision : Precision = Precision.SINGLE,
    verbose : bool = False
):
    use_cuda = not device == 'cpu' and torch.cuda.is_available()
    device = torch.device(device) if use_cuda else torch.device('cpu')
    (torch_float_type, torch_complex_type, _) = precision.get_dtypes(default=Precision.SINGLE)
    if viewing_angles is None:
        print("viewing_angles is None. Not rotating the model.")
        azimus = torch.tensor([0.0], dtype = torch_float_type)
        polars = torch.tensor([0.0], dtype = torch_float_type)
        gammas = torch.tensor([0.0], dtype = torch_float_type)
        viewing_angles = ViewingAngles(azimus = azimus, polars = polars, gammas = gammas)
    templates = None
    if isinstance(model, Volume):
        templates = Templates.generate_from_physical_volume(
            volume=model,
            polar_grid=images.polar_grid,
            viewing_angles=viewing_angles,
            precision=precision,
            verbose=verbose
        )
    elif isinstance(model, AtomicModel):
        templates = Templates.generate_from_positions(
            atomic_model=model,
            viewing_angles=viewing_angles,
            polar_grid=images.polar_grid,
            box_size=images.box_size,
            atom_shape=atom_shape,
            precision=precision,
            verbose = verbose
        )
    else:
        raise ValueError("model must be an instance of Volume or AtomicModel")
    if templates is None:
        raise ValueError("Templates could not be generated.")
    templates.normalize_images_fourier(ord=2, use_max=False)
    if search_displacements:
        if x_displacements is None or y_displacements is None:
            raise ValueError("x_displacements and y_displacements must be provided if search_displacements is True.")
        if x_displacements.dim() != 1 or y_displacements.dim() != 1:
            raise ValueError("x_displacements and y_displacements must be 1D tensors.")
        if x_displacements.shape[0] != y_displacements.shape[0]:
            raise ValueError("x_displacements and y_displacements must have the same number of displacements.")
        n_displacements = x_displacements.shape[0]
        n_imgs = images.n_images
        templates_original = templates.images_fourier.clone().cuda()
        images_fourier = images.images_fourier.to(torch_complex_type).to(device)
        likelihood = torch.zeros((n_imgs, n_displacements), dtype = torch_float_type, device = 'cpu')
        cross_correlation = torch.zeros((n_imgs, n_displacements), dtype = torch_float_type, device = 'cpu')
        for i in range(n_displacements):
            templates.images_fourier = templates_original.clone()
            x_disp = x_displacements[i] * torch.ones(n_imgs, dtype = torch_float_type, device = device)
            y_disp = y_displacements[i] * torch.ones(n_imgs, dtype = torch_float_type, device = device)
            templates.displace_images_fourier(x_disp, y_disp, precision)
            templates.rotate_images_fourier_discrete(viewing_angles.gammas)
            if ctf is not None:
                templates.apply_ctf(ctf)
            templates_fourier = templates.images_fourier.to(torch_complex_type).to(device)
            likelihood[:, i], cross_correlation[:, i] = likelihood_fourier(
                templates_fourier = templates_fourier,
                images_fourier = images_fourier,
                polar_grid = images.polar_grid,
                n_pixels = images.phys_grid.n_pixels[0] * images.phys_grid.n_pixels[1],
                return_cross_correlation = True
            )
    else:
        if x_displacements is not None or y_displacements is not None:
            templates.displace_images_fourier(x_displacements, y_displacements, precision)
        templates.rotate_images_fourier_discrete(viewing_angles.gammas)
        if ctf is not None:
            templates.apply_ctf(ctf)
        templates_fourier = templates.images_fourier.to(torch_complex_type).to(device)
        images_fourier = images.images_fourier.to(torch_complex_type).to(device)
        likelihood, cross_correlation = likelihood_fourier(
            templates_fourier = templates_fourier,
            images_fourier = images_fourier,
            polar_grid = images.polar_grid,
            n_pixels = images.phys_grid.n_pixels[0] * images.phys_grid.n_pixels[1],
            return_cross_correlation = True
        )
    return likelihood, cross_correlation
