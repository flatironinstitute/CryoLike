import numpy as np
import torch
from scipy.special import gammaln as lgamma

from cryolike.microscopy import CTF, translation_kernel_fourier, fourier_polar_to_cartesian_phys
from cryolike.stacks import Images, Templates
from cryolike.grids import PolarGrid
from cryolike.util import Precision, to_torch, absq, complex_mul_real

def integrated_likelihood_BioEM(
    templates_phys : torch.Tensor,
    images_phys : torch.Tensor,
    return_cross_correlation : bool = False,
):
    
    n_pixels = images_phys.shape[1] * images_phys.shape[2]
    print("n_pixels", n_pixels)

    Co = torch.sum(images_phys, dim = (1,2))
    Cc = torch.sum(templates_phys, dim = (1,2))
    Coo = torch.sum(images_phys ** 2, dim = (1,2))
    Ccc = torch.sum(templates_phys ** 2, dim = (1,2))
    Coc = torch.sum(images_phys * templates_phys, dim = (1,2))
    
    if return_cross_correlation:
        cross_correlation = Coc / torch.sqrt(Ccc * Coo)
        cross_correlation = cross_correlation.cpu()
        print("cross_correlation", cross_correlation.shape, cross_correlation.dtype, cross_correlation[:10])
    
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
    
def integrated_likelihood_fourier_dcoffset(
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
    Isy = torch.sum((s_points * images_fourier.real) * weights, dim = 1)
    Isx = torch.sum((s_points * templates_fourier.real) * weights, dim = 1)
    Iyy = torch.sum(absq(images_fourier) * weights, dim = 1)
    Ixx = torch.sum(absq(templates_fourier) * weights, dim = 1)
    Ixy = torch.sum(complex_mul_real(images_fourier, templates_fourier.conj()) * weights, dim = 1)
    if return_cross_correlation:
        cross_correlation = Ixy / torch.sqrt(Ixx * Iyy)
        cross_correlation = cross_correlation.cpu()
        
    # Iss = torch.sum(sinc_points ** 2).cpu().item() / pixel_size
    # Isy = torch.sum(complex_mul_real(sinc_points, images_fourier.conj()), dim = 1) / pixel_size
    # Isx = torch.sum(complex_mul_real(sinc_points, templates_fourier.conj()), dim = 1) / pixel_size
    # Iyy = torch.sum(absq(images_fourier), dim = 1) / pixel_size
    # Ixx = torch.sum(absq(templates_fourier), dim = 1) / pixel_size
    # Ixy = torch.sum(complex_mul_real(images_fourier, templates_fourier.conj()), dim = 1) / pixel_size
    
    A = - Isx ** 2 + Ixx * Iss
    B = - Isx * Isy + Ixy * Iss
    C = Isy ** 2 - Iyy * Iss
    
    D = - (B ** 2 / A + C)
    p = (n_pixels - 3.0) / 2.0
    constant = (1.0 - n_pixels * 0.5) * np.log(2 * np.pi) - np.log(2) + p * np.log(2 * Iss) + lgamma(p) #- np.log(n_displacements) - np.log(n_inplanes)
    # constant -= (n_pixels * 0.5) * np.log(pixel_size) + np.sum(np.log(polar_grid.weight_points))
    
    log_likelihood = -p * torch.log(D) - 0.5 * torch.log(A) + constant
    log_likelihood = log_likelihood.cpu()
    if return_cross_correlation:
        return log_likelihood, cross_correlation
    else:
        return log_likelihood    


def calc_distance_optimal_templates_vs_physical_images(
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
    assert template.templates_fourier is not None
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
    templates_optimal = to_torch(template.templates_fourier, precision, device)[template_indices]
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
        print("templates_optimal_physical", templates_optimal_physical.shape, templates_optimal_physical.dtype)
        if image.images_phys is None:
            raise ValueError("Physical images not found. Transform or Create physical images first before calculating the distance.")
        images_phys = image.images_phys.real
        # if np.issubdtype(images_phys.dtype, np.complexfloating):
        #     images_phys = images_phys.real
        print("images_phys", images_phys.shape, images_phys.dtype)
        templates_optimal_physical = to_torch(templates_optimal_physical, precision, device)
        images_phys = to_torch(images_phys, precision, device)
        if return_distance:
            distances = torch.norm(templates_optimal_physical - images_phys, dim = (1, 2), p = 2).cpu()
        if return_likelihood:
            output_likelihood = integrated_likelihood_BioEM(
                templates_phys = templates_optimal_physical,
                images_phys = images_phys,
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
    elif mode == "fourier":
        if image.images_fourier is None:
            raise ValueError("Fourier images not found. Transform or Create Fourier images first before calculating the distance.")
        images_fourier = image.images_fourier
        images_fourier = to_torch(images_fourier, precision, device)
        polar_grid = template.polar_grid
        if return_distance:
            weight = to_torch(polar_grid.weight_points, precision, device)
            distances = torch.sum((templates_optimal - images_fourier).abs() ** 2 * weight, dim = 1).cpu()
        if return_likelihood:
            output_likelihood = integrated_likelihood_fourier_dcoffset(
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