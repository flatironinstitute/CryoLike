from numpy import pi
import torch
from torch.testing import assert_close
import pytest
from unittest.mock import Mock

from cryolike.util import (
    Precision,
    to_torch,
)

from cryolike.likelihoods import template_first_comparator, compute_cross_correlation_complete

from cross_correlation_fixtures import (
    parameters,
    make_cases,
    make_polar_grid,
    make_viewing_angles,
    make_planewave_templates,
    viewing_angles_to_cartesian_displacements,
    get_planar_ctf,
    # p_xx_p_multiwave_iterative,
    p_xx_p_multiwave_vectorized,
)


param_matrix = make_cases()
@pytest.mark.parametrize("params", param_matrix)
def test_cross_correlation_PxxP_from_a_k_p(params: parameters):
    if (params.device == 'cuda' and not torch.cuda.is_available()):
        pytest.skip("Test cannot run because CUDA is not present.")

    (torch_float_type, _, _) = params.precision.get_dtypes(default=Precision.DOUBLE)
    with torch.device(params.device):
        torch.set_default_dtype(torch_float_type)
        ## More stuff to probably parametrize over?
        box_size = 2.0
        n_displacements_x = 3
        n_displacements_y = 5
        wavevector_planewave = params.wavevector
        template_ctf_angle = params.template_ctf_angle  # phi_S
        image_ctf_angle = params.image_ctf_angle        # phi_M
        displacement_planewave_image = torch.tensor([0.2, -0.18])
        ####

        polar_grid = make_polar_grid(params.n_pixels)
        viewing_angles = make_viewing_angles(params.device, torch_float_type)

        ### Set up numeric (cryolike-native) objects
        # CTFs are defined by the angles (and numerically also depend on the grid)
        planar_ctf_template = get_planar_ctf(polar_grid, template_ctf_angle, box_size, params.precision, params.device)
        planar_ctf_image = get_planar_ctf(polar_grid, image_ctf_angle, box_size, params.precision, params.device)
        templates = make_planewave_templates(wavevector_planewave, viewing_angles, polar_grid, params.precision)
        ## Images are a copy of the templates, with an applied displacement vector
        images = templates.to_images()
        images.phys_grid = Mock()
        images.phys_grid.n_pixels_total = params.n_pixels * params.n_pixels

        images.displace_fourier_images(
            x_displacements = displacement_planewave_image[0].item(),
            y_displacements = displacement_planewave_image[1].item()
        )
        images.apply_ctf(planar_ctf_image)

        templates.set_displacement_grid(
            max_displacement_pixels=params.max_displacement,
            n_displacements_x=n_displacements_x,
            n_displacements_y=n_displacements_y,
            pixel_size_angstrom=1.
        )

        iterator = template_first_comparator(
            torch.device(params.device),
            images,
            templates,
            planar_ctf_template,
            n_images_per_batch=5,
            n_templates_per_batch=5,
            return_integrated_likelihood=False,
            precision=params.precision
        )
        res = compute_cross_correlation_complete(iterator, templates, images, params.precision, False)

        # Analytic objects
        wavevector_templates = viewing_angles_to_cartesian_displacements(viewing_angles, wavevector_planewave).to(params.device)
        wavevector_images = wavevector_templates.clone() - displacement_planewave_image.to(wavevector_templates.device)
        analytic = p_xx_p_multiwave_vectorized(
            wavevector_templates,
            wavevector_images,
            template_ctf_angle,
            image_ctf_angle,
            to_torch(templates.polar_grid.theta_shell * -1., params.precision, params.device),
            templates.displacement_grid_angstrom.T.to(params.device),
            polar_grid.radius_max
        )

        assert_close(res.cross_correlation_MSdw, analytic.cpu())
        torch.set_default_dtype(torch.float32)

