from numpy import pi
import torch
from torch.testing import assert_close
import pytest

from cryolike.cross_correlation_likelihood import CrossCorrelationLikelihood, conform_ctf
from cryolike.util import (
    CrossCorrelationReturnType,
    Precision,
    to_torch,
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
    
    (torch_float_type, _, _) = params.precision.get_dtypes(default=Precision.DOUBLE)
    box_size = 2.0
    n_displacements_x = 3
    n_displacements_y = 5
    wavevector_planewave = params.wavevector
    angle_planar_ctf_template = - pi / 5.0 # phi_S
    angle_planar_ctf_image = + pi / 3.0  # phi_M
    displacement_planewave_image = torch.tensor([-0.17, -0.03], dtype=torch_float_type, device=params.device) # delta_M
    ####
    
    polar_grid = make_polar_grid(params.n_pixels)
    viewing_angles = make_viewing_angles(params.device, torch_float_type)
    planar_ctf_template = get_planar_ctf(polar_grid, angle_planar_ctf_template, box_size, params.precision, params.device)
    planar_ctf_image = get_planar_ctf(polar_grid, angle_planar_ctf_image, box_size, params.precision, params.device)
    templates = make_planewave_templates(wavevector_planewave, viewing_angles, polar_grid, params.precision)
    images = templates.to_images()
    images.displace_images_fourier(
        x_displacements = displacement_planewave_image[0].item(),
        y_displacements = displacement_planewave_image[1].item()
    )
    images.apply_ctf(planar_ctf_image)
    wavevector_planewave_templates = viewing_angles_to_cartesian_displacements(viewing_angles, wavevector_planewave).to(params.device)
    wavevector_planewave_images = wavevector_planewave_templates.clone() - displacement_planewave_image
    
    ####
    cc = CrossCorrelationLikelihood(
        templates = templates,
        max_displacement = params.max_displacement,
        n_displacements_x = n_displacements_x,
        n_displacements_y = n_displacements_y,
        precision = params.precision,
        device = params.device,
        verbose = False
    )
    ctf_tensor = conform_ctf(to_torch(planar_ctf_template.ctf, params.precision, params.device), planar_ctf_template.anisotropy)
    res = cc._compute_cross_correlation_likelihood(
        device=torch.device(params.device),
        images_fourier = images.images_fourier,
        ctf=ctf_tensor,
        n_pixels_phys = params.n_pixels*params.n_pixels,
        n_templates_per_batch=viewing_angles.n_angles,
        n_images_per_batch=viewing_angles.n_angles,
        return_type=CrossCorrelationReturnType.FULL_TENSOR,
        return_integrated_likelihood=False,
    )
    analytic = planewave_planar_planewave_planar(
        wavevector_planewave_templates,
        wavevector_planewave_images,
        cc._gamma.to(params.device) * -1.0,
        torch.stack([cc.x_displacements_expt_scale, cc.y_displacements_expt_scale]).T.to(params.device),
        torch.tensor(angle_planar_ctf_template, dtype=torch_float_type, device=params.device),
        torch.tensor(angle_planar_ctf_image, dtype=torch_float_type, device=params.device),
        polar_grid.radius_max
    )
    assert_close(res.cross_correlation_SMdw, analytic.cpu(), atol=params.abs_tolerance_cross_correlation, rtol=params.rel_tolerance_cross_correlation)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])