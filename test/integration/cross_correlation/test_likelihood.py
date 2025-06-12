import numpy as np
import torch
from torch.testing import assert_close
import pytest
from numpy import pi

from cryolike.cross_correlation_likelihood import CrossCorrelationLikelihood, conform_ctf
from cryolike.likelihood import LikelihoodFourierModel
from cryolike.util import (
    CrossCorrelationReturnType,
    Precision,
    to_torch,
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
    n_displacements_x = 1#3
    n_displacements_y = 1#5
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
    images.displace_images_fourier(
        x_displacements = displacement_planewave_image[0].item(),
        y_displacements = displacement_planewave_image[1].item()
    )
    images.apply_ctf(planar_ctf_image)
    wavevector_planewave_templates = viewing_angles_to_cartesian_displacements(viewing_angles, wavevector_planewave).to(_device)
    wavevector_planewave_images = wavevector_planewave_templates.clone() - displacement_planewave_image
    
    images.normalize_images_fourier(ord = 2)
    templates.normalize_images_fourier(ord = 2)
    
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
    cc = CrossCorrelationLikelihood(
        templates = templates,
        max_displacement = params.max_displacement,
        n_displacements_x = n_displacements_x,
        n_displacements_y = n_displacements_y,
        identity_kernel = identity_kernel,
        precision = params.precision,
        device = _device,
        verbose = False
    )
    ctf_tensor = conform_ctf(to_torch(planar_ctf_template.ctf, params.precision, _device), planar_ctf_template.anisotropy)
    gamma = cc._gamma.to(_device) * -1.0
    displacements = torch.stack([cc.x_displacements_expt_scale, cc.y_displacements_expt_scale]).T
    
    log_likelihood_class = LogLikelihoodPlanarCTFPlanewaves(
        wavevector_planewave_templates=wavevector_planewave_templates,
        wavevector_planewave_images=wavevector_planewave_images,
        wavevector_planewave_identity=wavevector_planewave_identity.unsqueeze(0),
        angle_planar_ctf_template=angle_planar_ctf_template,
        angle_planar_ctf_image=angle_planar_ctf_image,
        gamma=gamma,
        displacements=displacements,
        polar_grid=polar_grid,
        n_pixels=params.n_pixels * params.n_pixels,
        precision=params.precision,
        # device=_device
    )
    log_likelihood_analytical = log_likelihood_class.log_likelihood_final()
    print("log_likelihood_analytical", log_likelihood_analytical[:,:,0,0])

    (_, log_likelihood_SMDW) = cc._compute_cross_correlation_likelihood(
        device=_device,
        images_fourier=images.images_fourier,
        ctf=ctf_tensor,
        n_pixels_phys=params.n_pixels * params.n_pixels,
        n_templates_per_batch=viewing_angles.n_angles,
        n_images_per_batch=viewing_angles.n_angles,
        return_type=CrossCorrelationReturnType.FULL_TENSOR,
        return_integrated_likelihood=True,
        log_likelihood_keep_displacement_and_rotation=True
    )
    # print("log_likelihood_SMDW", log_likelihood_SMDW[:,:,0,0])
    assert_close(
        log_likelihood_SMDW,
        log_likelihood_analytical, 
        atol=params.abs_tolerance_log_likelihood,
        rtol=params.rel_tolerance_log_likelihood
    )


    # from matplotlib import pyplot as plt
    # import matplotlib as mpl
    # mpl.rcParams['figure.dpi'] = 300

    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # _x = log_likelihood_analytical.flatten()
    # _y = log_likelihood_SMDW.flatten()
    # isfinite = torch.isfinite(_x) & torch.isfinite(_y)
    # _x = _x[isfinite]
    # _y = _y[isfinite]
    # ax.scatter(_x, _y, s=1, c='blue', alpha=0.5)
    # from scipy.stats import linregress
    # slope, intercept, r_value, p_value, std_err = linregress(_x, _y)
    # print("slope", slope, "intercept", intercept, "r_value", r_value, "p_value", p_value, "std_err", std_err)
    # _min_x = _x.min()
    # _max_x = _x.max()
    # ax.plot(
    #     [_min_x, _max_x],
    #     [slope * _min_x + intercept, slope * _max_x + intercept],
    #     color='red',
    #     linewidth=1,
    #     label=f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r_value**2:.2f}"
    # )
    # ax.legend()
    # ax.set_xlabel('log_likelihood_analytical')
    # ax.set_ylabel('log_likelihood_SMDW')
    # plt.savefig("test_likelihood_PxxP_from_a_k_p.png")
    

    likelihood_model = LikelihoodFourierModel(
        model=templates,
        polar_grid=polar_grid,
        box_size=box_size,
        n_pixels=params.n_pixels * params.n_pixels,
        precision=params.precision,
        device=_device,
        identity_kernel=identity_kernel,
        verbose=False
    )
    likelihood_optimal_pose = likelihood_model(
        images=images,
        template_indices=None,
        ctf=planar_ctf_template,
        verbose=False
    )
    # print("likelihood_optimal_pose", likelihood_optimal_pose)
    assert_close(
        likelihood_optimal_pose.cpu(),
        log_likelihood_SMDW[:, :, 0, 0][range(templates.n_images), range(templates.n_images)].cpu(),
        atol=1e-6,
        rtol=1e-6
    )


if __name__ == '__main__':
    print('running test_cross_correlation_PxxP_from_a_k_p')
    # params = parameters.default()
    # test_likelihood_PxxP_from_a_k_p(params)
    # print('returning')
    pytest.main([__file__, "-v", "--tb=short"])
