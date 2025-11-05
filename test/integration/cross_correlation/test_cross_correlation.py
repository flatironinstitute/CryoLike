from numpy import pi
import torch
from torch.testing import assert_close
import pytest

from cryolike.likelihoods import (
    compute_cross_correlation_complete,
    template_first_comparator
)
from cryolike.util import (
    Precision,
    to_torch,
    get_device
)
from cross_correlation_fixtures import (
    parameters,
    make_cases,
    make_polar_grid,
    make_viewing_angles,
    make_planewave_templates,
    viewing_angles_to_cartesian_displacements,
    get_planar_ctf,
    planewave_planar_planewave_planar
)

param_matrix = make_cases()
@pytest.mark.parametrize("params", param_matrix)
def test_cross_correlation_PxxP_from_a_k_p(params: parameters):
    if (params.device == 'cuda' and not torch.cuda.is_available()):
        pytest.skip("Test cannot run because CUDA is not present.")
    _device = get_device(params.device)
    
    (torch_float_type, _, _) = params.precision.get_dtypes(default=Precision.DOUBLE)
    box_size = 2.0
    n_displacements_x = 3
    n_displacements_y = 5
    n_pixels = params.n_pixels
    pixel_size = box_size / n_pixels
    ####
    wavevector_planewave = params.wavevector
    angle_planar_ctf_template = - pi / 5.0 # phi_S
    angle_planar_ctf_image = + pi / 3.0  # phi_M
    displacement_planewave_image = torch.tensor([-0.17, -0.03], dtype=torch_float_type, device=_device) # delta_M
    ####
    
    polar_grid = make_polar_grid(params.n_pixels)
    viewing_angles = make_viewing_angles(_device, torch_float_type)
    planar_ctf_template = get_planar_ctf(polar_grid, angle_planar_ctf_template, box_size, params.precision, _device)
    planar_ctf_image = get_planar_ctf(polar_grid, angle_planar_ctf_image, box_size, params.precision, _device)
    templates = make_planewave_templates(wavevector_planewave, viewing_angles, polar_grid, params.precision)
    images = templates.to_images()
    images.displace_fourier_images(
        x_displacements = displacement_planewave_image[0].item(),
        y_displacements = displacement_planewave_image[1].item()
    )
    images.apply_ctf(planar_ctf_image)
    wavevector_planewave_templates = viewing_angles_to_cartesian_displacements(viewing_angles, wavevector_planewave).to(_device)
    wavevector_planewave_images = wavevector_planewave_templates.clone() - displacement_planewave_image
    _gammas = to_torch(polar_grid.theta_shell, params.precision, _device) * -1.0

    n_images = images.n_images
    n_templates = templates.n_images

    templates.set_displacement_grid(
        max_displacement_pixels=params.max_displacement / pixel_size,
        n_displacements_x=n_displacements_x,
        n_displacements_y=n_displacements_y,
        pixel_size_angstrom=pixel_size
    )

    iterator = template_first_comparator(
        device=_device,
        images=images,
        templates=templates,
        ctf=planar_ctf_template,
        n_images_per_batch=n_images,
        n_templates_per_batch=n_templates,
        return_integrated_likelihood=False,
        precision=params.precision
    )
    ret = compute_cross_correlation_complete(
        iterator,
        templates,
        images,
        params.precision,
        False
    )
    analytic = planewave_planar_planewave_planar(
        wavevector_planewave_templates,
        wavevector_planewave_images,
        _gammas,
        templates.displacement_grid_angstrom.T.to(params.device),
        torch.tensor(angle_planar_ctf_template, dtype=torch_float_type, device=params.device),
        torch.tensor(angle_planar_ctf_image, dtype=torch_float_type, device=params.device),
        polar_grid.radius_max
    )
    assert_close(ret.cross_correlation_MSdw.cpu(), analytic.cpu(), atol=params.abs_tolerance_cross_correlation, rtol=params.rel_tolerance_cross_correlation)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])