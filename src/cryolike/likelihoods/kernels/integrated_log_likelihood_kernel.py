from collections.abc import Callable
from math import lgamma
import numpy as np
import torch

from cryolike.util import (
    absq,
    complex_mul_real,
)

### NOTE: CF likelihood.py likelihood_fourier() l. 91-145
def ill_kernel(
    Iss: float,
    n_pixels_phys: int,
    sqrt_weighted_mask_points: torch.Tensor,
    CTF_sqrtweighted_fourier_templates_msnw: torch.Tensor,
    images_fourier_sqrtweighted_mnw: torch.Tensor,
    Ixx: torch.Tensor,
    Iyy: torch.Tensor,
    Ixy: torch.Tensor,
):
    """Compute the per-image integrated log likelihood.

    Args:
        Iss (float): Scalar value resulting from integrating the mask weights
        n_pixels_phys (int): Total number of pixels in the real-space representation
            of the images
        sqrt_weighted_mask_points (torch.Tensor): Mask points from the quadrature
            grid, pre-multiplied by the square root of the quadrature weights
        CTF_sqrtweighted_fourier_templates_msnw (torch.Tensor): Template stack
            in Fourier-space representation, pre-multiplied by the per-image CTF
            and the square root of the quadrature weights, and indexed as
            [image/ctf, template, radius, inplane-angle]
        images_fourier_sqrtweighted_mnw (torch.Tensor): Image stack in Fourier-space
            representation, pre-multiplied by the square root of the quadrature
            weights, and indexed as [image, radius, inplane-angle]
        Ixx (torch.Tensor): Template norms, as a 4-tensor [image, template, displacement, inplane]
        Iyy (torch.Tensor): Image norms, as a 4-tensor [image, template, displacement, inplane]
        Ixy (torch.Tensor): Unnormalized cross-correlation dot product between templates and images,
            indexed as [image, template, displacement, inplane-rotation]

    Returns:
        torch.Tensor: The integrated log likelihood of each image-tensor pair, indexed
            as [image, tensor]
    """
    Isx_ms = torch.sum(sqrt_weighted_mask_points * CTF_sqrtweighted_fourier_templates_msnw, dim = (2,3))
    Isy_m = torch.sum(sqrt_weighted_mask_points * images_fourier_sqrtweighted_mnw, dim=(1,2))

    Isx = Isx_ms.unsqueeze(2).unsqueeze(3)
    Isy = Isy_m.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    A = - absq(Isx) + Ixx * Iss
    B = - complex_mul_real(Isx, Isy) + Ixy * Iss
    C =   absq(Isy) - Iyy * Iss
    D = - (B ** 2 / A + C)
    p = n_pixels_phys / 2.0 - 2.0
    constant = (3.0 - n_pixels_phys) / 2.0 * np.log(2 * np.pi) \
                - np.log(2) - 0.5 * np.log(Iss) \
                + lgamma(n_pixels_phys / 2.0 - 2.0) \
                + p * np.log(2 * Iss)
    log_likelihood_msdw = -p * torch.log(D) - 0.5 * torch.log(A) + constant
    log_likelihood_ms = torch.logsumexp(log_likelihood_msdw, dim = (2, 3))
    return log_likelihood_ms
