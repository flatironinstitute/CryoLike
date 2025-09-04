import torch
from math import ceil

from .cross_correlation_iterator_types import GeneratorType, CrossCorrelationYieldType
from cryolike.stacks import Images, Templates
from cryolike.microscopy import CTF
from cryolike.likelihoods.validation import validate_operation
from cryolike.likelihoods.kernels.cross_correlation_likelihood_kernel import compute_image_norms, compute_cross_correlation, compute_template_norms
from cryolike.likelihoods.kernels.integrated_log_likelihood_kernel import ill_kernel
from cryolike.util import Precision, to_torch, fourier_bessel_transform


def template_first_comparator(
    device: torch.device,
    images: Images,
    templates: Templates,
    ctf: CTF,
    n_images_per_batch: int,
    n_templates_per_batch: int,
    return_integrated_likelihood: bool,
    precision: Precision = Precision.DEFAULT
) -> GeneratorType:
    """Creates a generator that carries out the pairwise image-template comparison
    to compute cross-correlation likelihood (and optional integrated log likelihood).
    This version favors templates over images, i.e. it expects the number of templates
    to be much smaller than the number of images and keeps the templates in memory
    as much as possible.

    Args:
        device (torch.device): Device to use to carry out the calculations (usually
            cpu or cuda)
        images (Images): Stack of images to iterate over
        templates (Templates): Stack of templates to iterate over. We expect that any
            searched displacements will already have been configured on the Templates
            object.
        ctf (CTF): Contrast transfer function object to use in adjusting templates
        n_images_per_batch (int): Number of images to use in a single iteration
        n_templates_per_batch (int): Number of templates to use in a single iteration
        return_integrated_likelihood (bool): Whether to include integrated log likelihood
        precision (Precision, optional): Precision to use for the cross-correlation
            calculation. Defaults to Precision.DEFAULT, in which case we will follow
            the precision used on the templates.

    Yields:
        CrossCorrelationYieldType: For every iteration, we will return (in order) the
            start index and end index of the templates used, the start and end indices
            of the images used, the full cross-correlation likelihood over that range,
            and the integrated log likelihood of the images (if requested, else None).
            These are strongly typed in a NamedTuple for ease of use.
    """
    validate_operation(templates, images)
    
    if precision == Precision.DEFAULT:
        precision = Precision.DOUBLE if templates.images_fourier.dtype == torch.complex128 else Precision.SINGLE

    ### NOTE: WE ASSUME that the displacements are already configured
    ctf_tensor = ctf.ctf
    ctf_is_singleton = ctf_tensor.shape[0] == 1
    
    integration_weights_sqrt = torch.sqrt(
        to_torch(
            templates.polar_grid.integration_weight_points,
            precision,
            device
        )
    ).unsqueeze(0)  # natively nw, this makes it snw

    n_pixels_phys = images.phys_grid.n_pixels_total

    Iss = None
    sqrt_mask_points = None
    if return_integrated_likelihood:
        Iss = templates.polar_grid.mask_integral
        sqrt_mask_points = to_torch(
            templates.polar_grid.mask_points,
            precision,
            device
        ) * integration_weights_sqrt

    t_start = 0

    n_batches_t = ceil(templates.n_images / n_templates_per_batch)
    n_batches_i = ceil(images.n_images / n_images_per_batch)

    for _ in range(n_batches_t):
        t_snw = to_torch(
            templates.images_fourier[t_start:t_start + n_templates_per_batch],
            precision,
            device
        ) * integration_weights_sqrt
        t_end = t_start + t_snw.shape[0]

        t_bessel_sdnq = fourier_bessel_transform(
            integration_weights_sqrt.unsqueeze(0) * # was snw, now sdnw
            templates.project_images_over_displacements(t_start, t_end, device)
        )

        i_start = 0
        for i in range(n_batches_i):
            ctf_batch = to_torch(
                ctf_tensor if ctf_is_singleton else ctf_tensor[i_start:i_start + n_images_per_batch],
                precision,
                device
            )
            i_mnw = to_torch(
                images.images_fourier[i_start:i_start + n_images_per_batch],
                precision,
                device
            ) * integration_weights_sqrt
            i_end = i_start + i_mnw.shape[0]
            i_bessel_mnq = fourier_bessel_transform(i_mnw * ctf_batch).conj()

            Ixx_msdw = compute_template_norms(
                templates.polar_grid.n_inplanes,
                t_snw,
                ctf_batch
            ).unsqueeze(2)
            Iyy_msdw = compute_image_norms(i_mnw)

            # the actual cross-correlation
            Ixy_msdw = compute_cross_correlation(
                templates.polar_grid.n_inplanes,
                i_bessel_mnq,
                t_bessel_sdnq
            )

            cross_correlation_msdw = Ixy_msdw / torch.sqrt(Ixx_msdw * Iyy_msdw)
            if torch.isnan(cross_correlation_msdw).any():
                raise ValueError("NaN detected in cross-correlation.")

            log_likelihood_ms = None
            if return_integrated_likelihood:
                assert Iss is not None
                assert sqrt_mask_points is not None
                log_likelihood_ms = ill_kernel(
                    Iss,
                    n_pixels_phys,
                    sqrt_mask_points,
                    t_snw * ctf_batch.unsqueeze(1),
                    i_mnw,
                    Ixx_msdw,
                    Iyy_msdw,
                    Ixy_msdw
                )

            res = CrossCorrelationYieldType(
                t_start, t_end, i_start, i_end, cross_correlation_msdw, log_likelihood_ms
            )
            yield res
            i_start = i_end
        t_start = t_end
