from __future__ import annotations

from typing import Literal, NamedTuple, overload, TYPE_CHECKING
import torch

if TYPE_CHECKING: # pragma: no cover
    from cryolike.likelihoods import GeneratorType

from cryolike.likelihoods.iteration.integrated_log_likelihood_aggregation import aggregate_ill
from cryolike.stacks import Templates, Images
from cryolike.util import (
    Precision,
)


class CrossCorrelationReturn(NamedTuple):
    """Class representing the cross-correlation values for each image, template
    pair at each possible displacement and inplane rotation angle.

    Attributes:
        cross_correlation_MSdw (torch.Tensor): Cross-correlation between each
            image-template pair, at each displacement and rotation. Indexed by
            [image, template, displacement, rotation].
    """
    cross_correlation_MSdw: torch.Tensor


@overload
def compute_cross_correlation_complete(comparator: GeneratorType, templates: Templates, images: Images, precision: Precision,
    include_integrated_log_likelihood: Literal[False]
) -> CrossCorrelationReturn:
    ... # pragma: no cover
@overload
def compute_cross_correlation_complete(comparator: GeneratorType, templates: Templates, images: Images, precision: Precision,
    include_integrated_log_likelihood: Literal[True]
) -> tuple[CrossCorrelationReturn, torch.Tensor]:
    ... # pragma: no cover
def compute_cross_correlation_complete(
    comparator: GeneratorType,
    templates: Templates,
    images: Images,
    precision: Precision,
    include_integrated_log_likelihood: bool,
) -> CrossCorrelationReturn | tuple[CrossCorrelationReturn, torch.Tensor]:
    """Compute cross-correlation between templates and images, returning a
    tensor identifying the cross-correlation for each image-template pair at
    every displacement and rotation, and (optionally) the integrated log
    likelihood of each image.

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
        CrossCorrelationReturn | tuple(CrossCorrelationReturn, torch.Tensor): The
            full cross correlation matrix for each image (if no ILL requested) or a tuple of (same, ILL).
            ILL is represented as a single score per image.
    """
    (torch_float_type, _, _) = Precision.get_dtypes(precision, default=Precision.DOUBLE)
    n_disp = templates.n_displacements
    n_inplanes = templates.polar_grid.n_inplanes    

    if include_integrated_log_likelihood:
        _log_likelihood_MS = torch.zeros((images.n_images, templates.n_images), dtype=torch_float_type, device="cpu")

    _cross_correlation_MSdw = torch.zeros(
        (images.n_images, templates.n_images, n_disp, n_inplanes),
        dtype=torch_float_type,
        device="cpu"
    )

    for batch in comparator:
        (t_start, t_end, i_start, i_end, cross_corr_msdw, ill) = batch
        if ill is not None and include_integrated_log_likelihood:
            _log_likelihood_MS[i_start:i_end, t_start:t_end] = ill.cpu()

        _cross_correlation_MSdw[i_start:i_end, t_start:t_end, :, :] = cross_corr_msdw.cpu()

    ret = CrossCorrelationReturn(cross_correlation_MSdw=_cross_correlation_MSdw)

    if include_integrated_log_likelihood:
        log_likelihood_M = aggregate_ill(templates, _log_likelihood_MS, precision)
        return (ret, log_likelihood_M)

    return ret
