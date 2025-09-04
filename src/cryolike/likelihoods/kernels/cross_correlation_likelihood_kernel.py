import torch

from cryolike.util import (
    absq,
    fourier_bessel_transform,
    inverse_fourier_bessel_transform,
)

## Note the general naming scheme for likelihood tensors indicates the values which can be
## used to index into them. Tensors named with _smdw are indexed left to right by image,
## template, displacement, and rotational angle; _SM indicates a tensor indexed only by
## image and template. Capital SM indicate that the tensor should contain information about
## the entire image/template stack, while lowercase sm indicates that the numbering corresponds
## to only the current subset/batch.


def compute_cross_correlation(
    n_inplanes: int,
    sqrtweighted_premultiplied_CTF_image_fourier_bessel_conj_mnq: torch.Tensor,
    sqrtweighted_displaced_fourier_bessel_templates_sdnq: torch.Tensor
) -> torch.Tensor:
    """Computes the raw cross-correlation between image and template, by
    doing a dot product on the fourier-bessel-space representations of the
    image and template stacks. We expect that the image stack will be a 3-tensor
    indexed as [image, grid-point, frequency-dimension] and the template stack
    will be a 4-tensor indexed as [template, displacement, grid-point, frequency].

    Note that both image and template stacks are expected to have been multiplied
    by the square root of the integration weights for the polar grid, and that
    the images should have been further multiplied by the CTF.

    Args:
        n_inplanes (int): Number of inplane rotations in the grid (needed to
            invert the bessel-space transform)
        sqrtweighted_premultiplied_CTF_image_fourier_bessel_conj_mnq (torch.Tensor): A
            fourier-bessel representation of the image stack, with its points pre-multiplied
            by the square root of the quadrature weights and by the CTF
        sqrtweighted_displaced_fourier_bessel_templates_sdnq (torch.Tensor): A
            fourier-bessel representation of the templates, premultiplied by the
            square root of the quadrature weights, and displaced across the searched
            displacement grid (as a separate dimension)

    Returns:
        torch.Tensor: The unnormalized cross-correlation between images and templates,
            indexed as [image, template, displacement, inplane-rotation]
    """
    ## Compute cross-correlation between image and template
    cross_correlation_msdq = torch.einsum(
        "mnq,sdnq->msdq",
        sqrtweighted_premultiplied_CTF_image_fourier_bessel_conj_mnq,
        sqrtweighted_displaced_fourier_bessel_templates_sdnq
    )
    cross_correlation_msdw = inverse_fourier_bessel_transform(
        cross_correlation_msdq,
        n_inplanes=n_inplanes
    )
    return cross_correlation_msdw


def compute_template_norms(
    n_inplanes: int, # for the inverse fft
    sqrtweighted_fourier_templates_snw: torch.Tensor,
    ctf_mnw: torch.Tensor,
):
    """Computes the (CTF-adjusted) norms for each template at every
    inplane rotation (which matters for anisotropic CTFs). This is done in
    Fourier-Bessel representation, which avoids having to materialize all the
    different template rotations.

    Args:
        n_inplanes (int): Number of inplane rotations in the polar
            representation (needed to invert the fourier-bessel transform)
        sqrtweighted_fourier_templates_snw (torch.Tensor): A tensor representing
            the template stack, indexed as [template, grid-point, inplane-rotation].
            Note that this does not need to be displaced as template norms should
            not be affected by displacement. The point values should have been
            premultiplied by the square root of the quadrature point weights.
        ctf_mnw (torch.Tensor): A tensor of the CTF values for the images,
            indexed as [image/ctf id, linear grid point, inplane-rotation]

    Returns:
        torch.Tensor: A tensor of the template norms, indexed as
            [image, template, inplane-rotation] (where the image
            index is 1:1 with the CTF index)
    """
    # ctf index corresponds to image index, so we use m here
    templates_sq_fb_msnq = fourier_bessel_transform(
        sqrtweighted_fourier_templates_snw.conj() * sqrtweighted_fourier_templates_snw,
        axis=-1
    ).unsqueeze(0)
    ctf_sq_fb_msnq = fourier_bessel_transform(
        ctf_mnw.conj() * ctf_mnw,
        axis=-1
    ).conj().unsqueeze(1)

    ctf_templates_msq = (templates_sq_fb_msnq * ctf_sq_fb_msnq).sum(2)
    ctf_template_norms_msw = inverse_fourier_bessel_transform(
        ctf_templates_msq,
        n_inplanes=n_inplanes
    )
    return ctf_template_norms_msw


def compute_image_norms(
    images_fourier: torch.Tensor
):
    """Computes the norms of each image. This is currently fairly straightforward
    since they do not require displacement, rotation, or CTF-application.

    Args:
        images_fourier (torch.Tensor): Tensor of the image stack (indexed as
            [image, radius, inplane-rotation] i.e. on the polar quadrature grid).
            The image values should have been premultiplied by the square root of
            the quadrature grid weights.

    Returns:
        torch.Tensor: The per-image norm, with the tensor expanded to include
            size-1 indices for the template, displacement, and rotation dimensions
    """
    Iyy_msdw = torch.sum(absq(images_fourier), dim = (-1,-2))[:,None,None,None]
    return Iyy_msdw

