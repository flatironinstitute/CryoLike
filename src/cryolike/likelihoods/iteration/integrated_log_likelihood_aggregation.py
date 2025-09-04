import numpy as np
import torch

from cryolike.stacks import Templates
from cryolike.util import (
    Precision,
    to_torch,
)


def aggregate_ill(
    templates: Templates,
    log_likelihood_MS: torch.Tensor,
    precision: Precision,
) -> torch.Tensor:
    """Aggregates the per-image-template-pair integrated log likelihood into a single
    per-image likelihood using logsumexp.

    Args:
        templates (Templates): Template stack from the comparison, used to find the
            viewing-angle weights and the number of displacements (which assumes the
            templates are displaced; if the images are displaced, we will need to add
            an additional parameter)
        log_likelihood_MS (torch.Tensor): The per-pair log likelihood values
        precision (Precision): Precision for the computation

    Returns:
        torch.Tensor: An aggregate log likelihood per image
    """
    n_disp = templates.n_displacements  # NOTE: Assumes templates are displaced, not images. May need to revise
    n_inplanes = templates.polar_grid.n_inplanes
    
    log_viewing_weights = to_torch(templates.viewing_angles.weights_viewing, precision, "cpu")

    # assumption: log_viewing_weights should be tensor of n-templates size, so should broadcast
    # over the 2nd dimension of log_likelihood_MS
    log_likelihood_M = torch.logsumexp(log_likelihood_MS + log_viewing_weights, dim=1) \
                       - np.log(n_disp) \
                       - np.log(n_inplanes)
    assert isinstance(log_likelihood_M, torch.Tensor)
    return log_likelihood_M
