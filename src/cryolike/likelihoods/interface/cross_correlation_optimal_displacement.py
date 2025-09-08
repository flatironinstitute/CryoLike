from __future__ import annotations

import numpy as np
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


class OptimalDisplacementReturn(NamedTuple):
    """Class representing the cross-correlation value
    and optimal displacement for each image.

    Attributes:
        cross_correlation_MSw (torch.Tensor): Cross-correlation for each
            image-template pair at each angle. Indexed by [image, template, angle].
        optimal_displacement_x_MSw (torch.Tensor): The optimal x-displacement
            for each image-template pair at each angle, indexed by [image, template, angle]
        optimal_displacement_y_MSw (torch.Tensor): The optimal y-displacement
            for each image-template pair at each angle, indexed by [image, template, angle]
    """
    cross_correlation_MSw: torch.Tensor
    optimal_displacement_x_MSw: torch.Tensor
    optimal_displacement_y_MSw: torch.Tensor


@overload
def compute_optimal_displacement(comparator: GeneratorType, templates: Templates, images: Images, precision: Precision,
    include_integrated_log_likelihood: Literal[False]
) -> OptimalDisplacementReturn:
    ... # pragma: no cover
@overload
def compute_optimal_displacement(comparator: GeneratorType, templates: Templates, images: Images, precision: Precision,
    include_integrated_log_likelihood: Literal[True]
) -> tuple[OptimalDisplacementReturn, torch.Tensor]:
    ... # pragma: no cover
def compute_optimal_displacement(
    comparator: GeneratorType,
    templates: Templates,
    images: Images,
    precision: Precision,
    include_integrated_log_likelihood: bool,
) -> OptimalDisplacementReturn | tuple[OptimalDisplacementReturn, torch.Tensor]:
    """Compute cross-correlation between templates and images, returning optimal
    displacement, and (optionally) the integrated log likelihood for
    each template-image pair.

    Args:
        comparator (GeneratorType): A generator which returns one batch of cross-correlation
            likelihood computation per call
        templates (Templates): A stack of Templates, with the displacements-to-search
            already configured
        images (Images): A stack of Images
        ctf (CTF): Contrast transfer function, either singleton (applied to
            every image) or a tensor of per-point values (one grid per image)
        precision (Precision): The precision at which to return results (note that
            this is separate from the precision used for computing the cross-correlation
            likelihood)
        include_integrated_likelihood (bool): Whether to include integrated log likelihood
            in the return

    Returns:
        OptimalDisplacementReturn | tuple(OptimalDisplacementReturn, torch.Tensor): The
            optimal displacement for each image (if no ILL requested) or a tuple of (same, ILL). ILL is
            represented as a matrix of scores indexed as [image, template].
    """
    (torch_float_type, _, _) = Precision.get_dtypes(precision, default=Precision.DOUBLE)
    n_inplanes = templates.polar_grid.n_inplanes    

    if include_integrated_log_likelihood:
        _log_likelihood_MS = torch.zeros((images.n_images, templates.n_images), dtype=torch_float_type, device="cpu")

    cross_correlation_MSw = torch.zeros(
        (images.n_images, templates.n_images, n_inplanes),
        dtype=torch_float_type,
        device="cpu"
    )
    optimal_displacement_x_MSw = torch.zeros(
        (images.n_images, templates.n_images, n_inplanes),
        dtype=torch_float_type,
        device="cpu"
    )
    optimal_displacement_y_MSw = torch.zeros(
        (images.n_images, templates.n_images, n_inplanes),
        dtype=torch_float_type,
        device="cpu"
    )

    for batch in comparator:
        (t_start, t_end, i_start, i_end, cross_corr_msdw, ill) = batch
        values, nd = torch.max(cross_corr_msdw, dim=2)
        nd = nd.cpu()
        cross_correlation_MSw[i_start:i_end, t_start:t_end] = values.cpu()
        optimal_displacement_x_MSw[i_start:i_end, t_start:t_end] = templates.displacement_grid_angstrom[0,nd].cpu()
        optimal_displacement_y_MSw[i_start:i_end, t_start:t_end] = templates.displacement_grid_angstrom[1,nd].cpu()

        if ill is not None and include_integrated_log_likelihood:
            _log_likelihood_MS[i_start:i_end, t_start:t_end] = ill.cpu()

    ret = OptimalDisplacementReturn(
        cross_correlation_MSw,
        optimal_displacement_x_MSw,
        optimal_displacement_y_MSw,
    )

    if include_integrated_log_likelihood:
        log_likelihood_M = aggregate_ill(templates, _log_likelihood_MS, precision)
        return (ret, log_likelihood_M)

    return ret
