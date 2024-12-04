import numpy as np
import torch
from scipy.special import jv

from cryolike.grids import PolarGrid, FourierImages
from cryolike.stacks import Templates, Images
from cryolike.microscopy import CTF, ViewingAngles
from cryolike.cross_correlation_likelihood import CrossCorrelationLikelihood, conform_ctf
from cryolike.util import (
    CrossCorrelationReturnType,
    FloatArrayType,
    Precision,
    set_precision,
    to_torch,
)


def test_cross_correlation():
    
    box_size = 2.0
    n_pixels = 128
    pixel_size = box_size / n_pixels

    precision = Precision.SINGLE
    (torch_float_type, torch_complex_type, _) = set_precision(precision, Precision.DOUBLE)
    radius_max = n_pixels / (2.0 * np.pi) * np.pi / 2.0 #/ 2.0
    dist_radii = 1.0 / (2.0 * np.pi) * np.pi / 2.0 #/ 2.0
    n_inplanes = n_pixels * 2

    max_displacement = 0.0
    n_displacements_x = 1
    n_displacements_y = 1

    k = 0.2 / box_size * 2.0
    alpha = 0.01

    polar_grid = PolarGrid(
        radius_max = radius_max,
        dist_radii = dist_radii,
        n_inplanes = n_inplanes,
        uniform = True
    )
    radii = polar_grid.radius_shells
    thetas = polar_grid.theta_shell
    n_shells = polar_grid.n_shells

    ctf_alpha = np.sqrt(jv(0, alpha * radii))
    ctf_alpha_inte = jv(1, alpha * radius_max) * 2 * np.pi * radius_max / alpha

    def plane_bessel_plane_integral_0(
        radius_max: float = 48.0 / (2.0 * np.pi),
        delta_images: float | FloatArrayType = 0.1,
        omega_images: float | FloatArrayType = 0.0,
        delta_templates: float | FloatArrayType = 0.1,
        omega_templates: float | FloatArrayType = 0.0,
        delta_images_xy: FloatArrayType | None = None,
        delta_templates_xy: FloatArrayType | None = None,
        alpha : float = 1.0
    ):
        if delta_images_xy is None:
            delta_images_x = delta_images * np.cos(omega_images)
            delta_images_y = delta_images * np.sin(omega_images)
            delta_images_xy = np.stack([delta_images_x, delta_images_y], axis = 1)
        if delta_templates_xy is None:
            delta_templates_x = delta_templates * np.cos(omega_templates)
            delta_templates_y = delta_templates * np.sin(omega_templates)
            delta_templates_xy = np.stack([delta_templates_x, delta_templates_y], axis = 1)
        assert delta_images_xy is not None
        assert delta_templates_xy is not None
        delta_T_xy = delta_templates_xy[:,None,:] - delta_images_xy[None,:,:]
        delta_T = np.linalg.norm(delta_T_xy, axis = 2)

        a = alpha * radius_max
        b = 2 * np.pi * radius_max * delta_T
        c = np.maximum(a, b)
        d = np.minimum(a, b)
        I = 2 * np.pi * radius_max ** 2 * (d * jv(-1, d) * jv(0, c) - c * jv(-1, c) * jv(0, d)) / np.maximum(1e-12, c ** 2 - d ** 2)
        return I

    viewing_angles = ViewingAngles.from_viewing_distance(viewing_distance=8.0 / (4.0 * np.pi))
    polars_viewing = viewing_angles.polars
    azimus_viewing = viewing_angles.azimus
    n_viewings = viewing_angles.n_angles

    ##
    ## project wavevector to viewing angles
    ##
    cos_polars_viewing = torch.cos(polars_viewing)
    sin_polars_viewing = torch.sin(polars_viewing)
    cos_azimus_viewing = torch.cos(azimus_viewing)
    sin_azimus_viewing = torch.sin(azimus_viewing)

    normvector_templates_x = sin_polars_viewing * cos_azimus_viewing
    normvector_templates_y = sin_polars_viewing * sin_azimus_viewing
    normvector_templates_z = cos_polars_viewing
    normvector_templates = torch.stack([normvector_templates_x, normvector_templates_y, normvector_templates_z], dim = 1)

    wavevector = torch.tensor([0.0, 0.0, k], dtype = torch.float64)

    wavevector_templates = wavevector[None,:] - normvector_templates_z[:,None] * k * normvector_templates

    xvector_x_templates = cos_azimus_viewing * cos_polars_viewing
    xvector_y_templates = sin_azimus_viewing * cos_polars_viewing
    xvector_z_templates = - sin_polars_viewing
    xvector_templates = torch.stack([xvector_x_templates, xvector_y_templates, xvector_z_templates], dim = 1)
    wavevector_x_templates = torch.sum(wavevector_templates * xvector_templates, dim = 1)

    yvector_x_templates = - sin_azimus_viewing
    yvector_y_templates = cos_azimus_viewing
    yvector_z_templates = torch.zeros_like(sin_polars_viewing)
    yvector_templates = torch.stack([yvector_x_templates, yvector_y_templates, yvector_z_templates], dim = 1)
    wavevector_y_templates = torch.sum(wavevector_templates * yvector_templates, dim = 1)

    wavevector_xy_templates = torch.stack([wavevector_x_templates, wavevector_y_templates], dim = 1)

    ##
    ## Generate images from wavevector
    ##
    x_polar_grid = torch.from_numpy(polar_grid.x_points)
    y_polar_grid = torch.from_numpy(polar_grid.y_points)
    xy_polar_grid = torch.stack([x_polar_grid, y_polar_grid], dim = 1)
    images_fourier_wavevector = torch.exp(1j * torch.sum(wavevector_xy_templates.unsqueeze(1) * xy_polar_grid.unsqueeze(0), dim = 2))
    images_fourier_wavevector = images_fourier_wavevector.reshape(n_viewings, n_shells, n_inplanes) * ctf_alpha[None,:,None]

    weights = torch.from_numpy(polar_grid.weight_points).reshape(n_shells, n_inplanes).unsqueeze(0)

    fourier_imgs = FourierImages(images_fourier=images_fourier_wavevector.numpy(), polar_grid=polar_grid)
    im_wavevector = Images(fourier_images_data=fourier_imgs)

    wavevector_xy_templates = wavevector_xy_templates.numpy()

    I_true = plane_bessel_plane_integral_0(
        radius_max = radius_max,
        delta_templates_xy = wavevector_xy_templates,
        delta_images_xy = wavevector_xy_templates,
        alpha = alpha
    ) / ctf_alpha_inte

    def density_function(fourier_slices):
        kdotr = torch.sum(wavevector.unsqueeze(0).unsqueeze(0).cuda() * fourier_slices.cuda(), dim = 2)
        images_fourier = torch.exp(1j * 2 * np.pi * kdotr)
        # images_fourier = torch.exp(1j * k * fourier_slices[:,:,1])
        # images_fourier = fourier_slices[:,:,2]
        # images_fourier /= torch.norm(images_fourier, dim = 1, keepdim = True)
        return images_fourier.to(torch_complex_type)
    tp = Templates.generate_from_function(density_function, viewing_angles, polar_grid, precision=precision)
    assert tp.templates_fourier is not None

    im = Images.from_templates(templates = tp)
    im.transform_to_spatial(grid=(n_pixels, pixel_size), precision=precision)
    ctf = CTF(
        polar_grid = polar_grid,
        box_size = box_size, # in Angstrom
        anisotropy = False,
        ctf_descriptor = ctf_alpha
    )
    im.apply_ctf(ctf)

    cc = CrossCorrelationLikelihood(
        templates = tp,
        max_displacement = max_displacement,
        n_displacements_x = n_displacements_x,
        n_displacements_y = n_displacements_y,
        precision = precision,
        device = 'cuda',
        verbose = True
    )

    _imgs = im.images_fourier
    _ctf = conform_ctf(to_torch(ctf.ctf, precision, "cpu"), ctf.anisotropy)
    assert _imgs is not None

    res = cc._compute_cross_correlation_likelihood(
        device=torch.device("cuda"),
        images_fourier = _imgs,
        ctf=_ctf,
        n_pixels_phys = im.phys_grid.n_pixels[0] * im.phys_grid.n_pixels[1],
        n_templates_per_batch=128,
        n_images_per_batch=128,
        return_type=CrossCorrelationReturnType.OPTIMAL_DISPLACEMENT,
        return_integrated_likelihood=False,
    )
    cross_correlation_w0 = res.cross_correlation_SMw[:,:,0].numpy()
    print("cross_correlation_w0", cross_correlation_w0)
    print("I_true", I_true)

    assert np.allclose(cross_correlation_w0, I_true, atol = 1e-5)


if __name__ == '__main__':
    test_cross_correlation()
    print('Cross-correlation planewave tests passed!')