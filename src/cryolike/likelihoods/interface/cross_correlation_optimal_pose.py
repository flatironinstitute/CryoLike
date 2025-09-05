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


class OptimalPoseReturn(NamedTuple):
    """Class representing the cross-correlation value
    and optimal pose (template, displacement, and rotation) for each image.

    Attributes:
        cross_correlation_M (torch.Tensor): Best cross-correlation for each
            image. Indexed by image number.
        optimal_template_M (torch.Tensor): The template generating the best
            cross-correlation for each image, indexed by image number.
        optimal_displacement_x_M (torch.Tensor): The optimal x-displacement
            for each image, indexed by image number
        optimal_displacement_y_M (torch.Tensor): The optimal y-displacement
            for each image, indexed by image number
        optimal_inplane_rotation_M (torch.Tensor): The optimal inplane
            rotation for each image, indexed by image number
    """
    cross_correlation_M: torch.Tensor
    optimal_template_M: torch.Tensor
    optimal_displacement_x_M: torch.Tensor
    optimal_displacement_y_M: torch.Tensor
    optimal_inplane_rotation_M: torch.Tensor



@overload
def compute_optimal_pose(comparator: GeneratorType, templates: Templates, images: Images, precision: Precision,
    include_integrated_log_likelihood: Literal[False]
) -> OptimalPoseReturn:
    ... # pragma: no cover
@overload
def compute_optimal_pose(comparator: GeneratorType, templates: Templates, images: Images, precision: Precision,
    include_integrated_log_likelihood: Literal[True]
) -> tuple[OptimalPoseReturn, torch.Tensor]:
    ... # pragma: no cover
def compute_optimal_pose(
    comparator: GeneratorType,
    templates: Templates,
    images: Images,
    precision: Precision,
    include_integrated_log_likelihood: bool
) -> OptimalPoseReturn | tuple[OptimalPoseReturn, torch.Tensor]:
    """Compute cross-correlation between templates and images, returning a
    tensor identifying the best cross-correlation for each image-template pair,
    as well as tensors identifying the identity of the optimal matching template
    and the x- and y-displacements and rotation producing the optimal match; and
    (optionally) the integrated log likelihood of each image.

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
        OptimalPoseReturn | tuple(OptimalPoseReturn, torch.Tensor): The
            score for each image against its optimal template, as well as the
            optimal template, displacements, and rotation producing this score
            (if no ILL requested); or a tuple of (same, ILL).
            ILL is represented as a single score per image.
    """
    (torch_float_type, _, _) = Precision.get_dtypes(precision, default=Precision.DOUBLE)

    if include_integrated_log_likelihood:
        _log_likelihood_MS = torch.zeros((images.n_images, templates.n_images), dtype=torch_float_type, device="cpu")

    _cross_correlation_MS = torch.zeros((images.n_images, templates.n_images), dtype=torch_float_type, device="cpu")
    _opt_disp_x_MS = torch.zeros((images.n_images, templates.n_images), dtype=torch_float_type, device="cpu")
    _opt_disp_y_MS = torch.zeros((images.n_images, templates.n_images), dtype=torch_float_type, device="cpu")
    _opt_inplane_rot_MS = torch.zeros((images.n_images, templates.n_images), dtype=torch_float_type, device="cpu")
    _gamma = to_torch(templates.polar_grid.theta_shell)

    for batch in comparator:
        (t_start, t_end, i_start, i_end, cross_corr_msdw, ill) = batch
        shape_dw = cross_corr_msdw.shape[2:]
        cross_correlation_msdw_cuda_optdw = cross_corr_msdw.flatten(start_dim=2, end_dim=3)
        values, indices_ravel = torch.max(cross_correlation_msdw_cuda_optdw, dim=2)
        _cross_correlation_MS[i_start:i_end, t_start:t_end] = values
        indices = torch.unravel_index(indices_ravel, shape_dw)
        nd = indices[0].cpu()
        nw = indices[1].cpu()
        _opt_disp_x_MS[i_start:i_end, t_start:t_end] = templates.displacement_grid_angstrom[0, nd]
        _opt_disp_y_MS[i_start:i_end, t_start:t_end] = templates.displacement_grid_angstrom[1, nd]
        _opt_inplane_rot_MS[i_start:i_end, t_start:t_end] = _gamma[nw]

        if ill is not None and include_integrated_log_likelihood:
            _log_likelihood_MS[i_start:i_end, t_start:t_end] = ill.cpu()

    # Finalize
    imgs_range = torch.arange(images.n_images).cpu()
    cross_correlation_M, optimal_template_M = torch.max(_cross_correlation_MS, dim=1)
    opt_disp_x_M = _opt_disp_x_MS[imgs_range, optimal_template_M].cpu()
    opt_disp_y_M = _opt_disp_y_MS[imgs_range, optimal_template_M].cpu()
    opt_inplane_rot_M = _opt_inplane_rot_MS[imgs_range, optimal_template_M].cpu()

    ret = OptimalPoseReturn(
        cross_correlation_M.cpu(), optimal_template_M.cpu(),
        opt_disp_x_M, opt_disp_y_M, opt_inplane_rot_M
    )

    if include_integrated_log_likelihood:
        log_likelihood_M = aggregate_ill(templates, _log_likelihood_MS, precision)
        return (ret, log_likelihood_M)

    return ret
