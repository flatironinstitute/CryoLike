import numpy as np
import torch
from torch.testing import assert_close
import pytest
from numpy import pi

from cryolike.likelihoods.kernels.cross_correlation_likelihood_kernel import compute_image_norms, compute_cross_correlation, compute_template_norms
from cryolike.likelihoods.kernels.integrated_log_likelihood_kernel import ill_kernel
from cryolike.util import (
    CrossCorrelationReturnType,
    Precision,
    to_torch,
    fourier_bessel_transform
)
from cryolike.grids.polar_grid import PolarGrid
from cryolike.stacks.template import Templates

from cross_correlation_fixtures import (
    parameters,
    make_cases,
    make_polar_grid,
    make_viewing_angles,
    make_planewave_templates,
    viewing_angles_to_cartesian_displacements,
    get_planar_ctf,
)
from likelihood_fixtures import LogLikelihoodPlanarCTFPlanewaves


class CustomIdentityKernel:

    wavevector_identity: torch.Tensor
    polar_grid: PolarGrid
    device: torch.device

    def __init__(
        self,
        wavevector_identity: torch.Tensor,
        polar_grid: PolarGrid,
        device = torch.device('cpu'),
    ):
        self.wavevector_identity = wavevector_identity
        self.polar_grid = polar_grid
        self.device = device

    def __call__(
        self,
        polar_grid: PolarGrid,
        precision: Precision,
    ) -> torch.Tensor:
        wavevector_identity = to_torch(self.wavevector_identity, precision, self.device)
        _x_points = to_torch(polar_grid.x_points, precision, self.device)
        _y_points = to_torch(polar_grid.y_points, precision, self.device)
        kernel = torch.exp(
            2 * np.pi * 1j * (_x_points * wavevector_identity[0] + _y_points * wavevector_identity[1])
        ).reshape(polar_grid.n_shells, polar_grid.n_inplanes)
        # print("kernel", kernel.shape, kernel.dtype)
        return kernel


param_matrix = make_cases()
@pytest.mark.parametrize("params", param_matrix)
def test_likelihood_PxxP_from_a_k_p(params: parameters):
    if (params.device == 'cuda' and not torch.cuda.is_available()):
        pytest.skip("Test cannot run because CUDA is not present.")
    
    (torch_float_type, torch_complex_type, _) = params.precision.get_dtypes(default=Precision.DOUBLE)
    box_size = 2.0
    n_displacements_x = 3
    n_displacements_y = 5
    n_pixels = params.n_pixels
    n_pixels_total = params.n_pixels * params.n_pixels
    pixel_size = box_size / n_pixels
    ####
    wavevector_planewave = params.wavevector
    angle_planar_ctf_template = - pi / 5.0 # phi_S
    angle_planar_ctf_image = + pi / 3.0 # phi_M
    # displacement_planewave_image = torch.tensor([0.0, 0.0], dtype=torch_float_type, device=params.device) # delta_M
    displacement_planewave_image = torch.tensor([-0.17, -0.03], dtype=torch_float_type, device=params.device) # delta_M
    # displacement_planewave_image = torch.tensor([-0.01, -0.01], dtype=torch_float_type, device=params.device) # delta_M
    wavevector_planewave_identity = torch.tensor([0.01, 0.01], dtype=torch_float_type) * (2.0 * np.pi)
    
    _device = torch.device(params.device)
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
    
    images.normalize_images_fourier(ord = 2)
    templates.normalize_images_fourier(ord = 2)

    templates.set_displacement_grid(
        max_displacement_pixels=params.max_displacement / pixel_size,
        n_displacements_x=n_displacements_x,
        n_displacements_y=n_displacements_y,
        pixel_size_angstrom=pixel_size
    )

    n_images = images.n_images
    n_templates = templates.n_images
    
    ####
    identity_kernel = CustomIdentityKernel(
        wavevector_identity = wavevector_planewave_identity,
        polar_grid = polar_grid,
        device = _device,
    )
    assert callable(identity_kernel)
    ####
    # Now we try and recalculate this integral using the image and template structures.
    ####
    _gamma = to_torch(polar_grid.theta_shell, params.precision, _device) * -1.0
    displacements = templates.displacement_grid_angstrom.T
    
    log_likelihood_class = LogLikelihoodPlanarCTFPlanewaves(
        wavevector_planewave_templates=wavevector_planewave_templates,
        wavevector_planewave_images=wavevector_planewave_images,
        wavevector_planewave_identity=wavevector_planewave_identity.unsqueeze(0),
        angle_planar_ctf_template=angle_planar_ctf_template,
        angle_planar_ctf_image=angle_planar_ctf_image,
        gamma=_gamma,
        displacements=displacements,
        polar_grid=polar_grid,
        n_pixels=n_pixels_total,
        precision=params.precision,
        # device=_device
    )
    log_likelihood_analytical = log_likelihood_class.log_likelihood_final()
    
    integration_weights_sqrt = torch.sqrt(
        to_torch(
            templates.polar_grid.integration_weight_points,
            params.precision,
            _device
        )
    ).unsqueeze(0)  # natively nw, this makes it snw

    ctf_batch = planar_ctf_template.ctf
    
    Iss = templates.polar_grid.mask_integral
    sqrt_mask_points = to_torch(
        templates.polar_grid.mask_points,
        params.precision,
        _device
    ) * integration_weights_sqrt

    t_snw = to_torch(
        templates.images_fourier,
        params.precision,
        _device
    ) * integration_weights_sqrt
    i_mnw = to_torch(
        images.images_fourier,
        params.precision,
        _device
    ) * integration_weights_sqrt
    i_bessel_mnq = fourier_bessel_transform(i_mnw * ctf_batch).conj()
    t_bessel_sdnq = fourier_bessel_transform(
        integration_weights_sqrt.unsqueeze(0) * # was snw, now sdnw
        templates.project_images_over_displacements(0, n_templates, _device)
    )
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

    # (_, log_likelihood_SMDW) = cc._compute_cross_correlation_likelihood(
    #     device=_device,
    #     images_fourier=images.images_fourier,
    #     ctf=ctf_tensor,
    #     n_pixels_phys=params.n_pixels * params.n_pixels,
    #     n_templates_per_batch=viewing_angles.n_angles,
    #     n_images_per_batch=viewing_angles.n_angles,
    #     return_type=CrossCorrelationReturnType.FULL_TENSOR,
    #     return_integrated_likelihood=True,
    #     log_likelihood_keep_displacement_and_rotation=True
    # )
    # # print("log_likelihood_SMDW", log_likelihood_SMDW[:,:,0,0])

    log_likelihood_msdw = ill_kernel(
        Iss,
        n_pixels_total,
        sqrt_mask_points,
        t_snw * ctf_batch.unsqueeze(1),
        i_mnw,
        Ixx_msdw,
        Iyy_msdw,
        Ixy_msdw
    )
    assert_close(
        log_likelihood_msdw,
        log_likelihood_analytical, 
        atol=params.abs_tolerance_log_likelihood,
        rtol=params.rel_tolerance_log_likelihood
    )

    # likelihood_model = LikelihoodFourierModel(
    #     model=templates,
    #     polar_grid=polar_grid,
    #     box_size=box_size,
    #     n_pixels=params.n_pixels * params.n_pixels,
    #     precision=params.precision,
    #     device=_device,
    #     identity_kernel=identity_kernel,
    #     verbose=False
    # )
    # likelihood_optimal_pose = likelihood_model(
    #     images=images,
    #     template_indices=None,
    #     ctf=planar_ctf_template,
    #     verbose=False
    # )
    # print("likelihood_optimal_pose", likelihood_optimal_pose)
    # assert_close(
    #     likelihood_optimal_pose.cpu(),
    #     log_likelihood_SMDW[:, :, 0, 0][range(templates.n_images), range(templates.n_images)].cpu(),
    #     atol=1e-6,
    #     rtol=1e-6
    # )


if __name__ == '__main__':
    print('running test_cross_correlation_PxxP_from_a_k_p')
    # params = parameters.default()
    # test_likelihood_PxxP_from_a_k_p(params)
    # print('returning')
    pytest.main([__file__, "-v", "--tb=short"])
