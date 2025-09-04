from __future__ import annotations

from typing import Literal, NamedTuple, overload, TYPE_CHECKING
import torch

if TYPE_CHECKING: # pragma: no cover
    from cryolike.likelihoods import GeneratorType

from cryolike.likelihoods.iteration.integrated_log_likelihood_aggregation import aggregate_ill
from cryolike.stacks import Templates, Images
from cryolike.util import (
    Precision,
    to_torch,
)

class OptimalRotationReturn(NamedTuple):
    """Class representing the cross-correlation value
    and optimal rotation angle for each image.

    Attributes:
        cross_correlation_MSd (torch.Tensor): Cross-correlation for each
            image-template pair at each displacement. Indexed by [image, template, displacement].
        optimal_inplane_rotation_SMd(torch.Tensor): The optimal inplane
            rotation for each image-template pair at each displacement,
            indexed by [image, displacement]
    """
    cross_correlation_MSd: torch.Tensor
    optimal_inplane_rotation_MSd: torch.Tensor

@overload
def compute_optimal_rotation(comparator: GeneratorType, templates: Templates, images: Images, precision: Precision,
    include_integrated_log_likelihood: Literal[False]
) -> OptimalRotationReturn:
    ... # pragma: no cover
@overload
def compute_optimal_rotation(comparator: GeneratorType, templates: Templates, images: Images, precision: Precision,
    include_integrated_log_likelihood: Literal[False]
) -> tuple[OptimalRotationReturn, torch.Tensor]:
    ... # pragma: no cover
def compute_optimal_rotation(
    comparator: GeneratorType,
    templates: Templates,
    images: Images,
    precision: Precision,
    include_integrated_log_likelihood: bool,
):
    """Compute cross-correlation between templates and images, returning score
    and optimal rotation achieving that score for each image-template pair,
    and (optionally) the integrated log likelihood for each image.

    Args:
        device (torch.device): Device to use for computation
        images_fourier (torch.Tensor): Fourier-space images
        ctf (torch.Tensor): Contrast transfer function, either singleton (applied to
            every image) or a vector (one per image).
        n_pixels_phys (int): Number of pixels in the physical image, i.e. the
            physical image source's Cartesian grid
        n_images_per_batch (int): Number of images to use per batch
        n_templates_per_batch (int): Number of templates to use per batch
        return_integrated_likelihood (bool): Whether to include integrated log likelihood
            in the return

    Returns:
        OptimalRotationReturn | tuple(OptimalRotationReturn, torch.Tensor): The
            optimal rotation for each image (if no ILL requested) or a tuple of (same, ILL). ILL is
            represented as a matrix of scores indexed as [image, template].
    """    
    (torch_float_type, _, _) = Precision.get_dtypes(precision, default=Precision.DOUBLE)
    n_disp = templates.n_displacements
    _gamma = to_torch(templates.polar_grid.theta_shell, precision, "cpu")

    if include_integrated_log_likelihood:
        _log_likelihood_MS = torch.zeros((images.n_images, templates.n_images), dtype=torch_float_type, device="cpu")

    cross_correlation_MSd = torch.zeros(
        (images.n_images, templates.n_images, n_disp),
        dtype=torch_float_type,
        device="cpu"
    )
    optimal_inplane_rotation_MSd = torch.zeros(
        (images.n_images, templates.n_images, n_disp),
        dtype=torch_float_type,
        device="cpu"
    )
    
    for batch in comparator:
        (t_start, t_end, i_start, i_end, cross_corr_msdw, ill) = batch
        if ill is not None and include_integrated_log_likelihood:
            _log_likelihood_MS[i_start:i_end, t_start:t_end] = ill.cpu()

        values, nw = torch.max(cross_corr_msdw, dim=3)
        nw = nw.cpu()
        cross_correlation_MSd[i_start:i_end, t_start:t_end] = values.cpu()
        optimal_inplane_rotation_MSd[i_start:i_end, t_start:t_end] = _gamma[nw].cpu()

    ret = OptimalRotationReturn(
        cross_correlation_MSd,
        optimal_inplane_rotation_MSd
    )

    if include_integrated_log_likelihood:
        log_likelihood_M = aggregate_ill(templates, _log_likelihood_MS, precision)
        return (ret, log_likelihood_M)

    return ret
