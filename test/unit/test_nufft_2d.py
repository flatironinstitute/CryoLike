import numpy as np
import torch

from cryolike.polar_grid import PolarGrid
from cryolike.cartesian_grid import CartesianGrid2D
from cryolike.nufft import fourier_polar_to_cartesian_phys, cartesian_phys_to_fourier_polar
from cryolike.util.enums import Precision, QuadratureType


def test_nufft_2d():

    nufft_eps_double = 1e-12
    nufft_eps_single = 1e-6
    box_size = 128.0
    n_pixels_tests = [128, 256, 512]

    x_1 = 0.1 * box_size
    y_1 = 0.1 * box_size
    x_2 = -0.2 * box_size
    y_2 = -0.2 * box_size

    atom_scale_factor = 4.0
    acc_tol = 1e-9

    print('Testing 2D to 2D NUFFT (2D Physical density <-> 2D Fourier slices in 2D fourier space)')
    for n_pixels in n_pixels_tests:
        
        pixel_size = box_size / n_pixels
        phys_grid = CartesianGrid2D(n_pixels = n_pixels, pixel_size = pixel_size)
        radius_max = np.pi * n_pixels / (2.0 * np.pi) / 2.0
        dist_radii = np.pi / 2.0 / (2.0 * np.pi)
        
        sigma_atom = pixel_size * atom_scale_factor
        sigma_atom_sq = sigma_atom ** 2 

        x_c = torch.from_numpy(phys_grid.x_pixels)
        y_c = torch.from_numpy(phys_grid.y_pixels)
        Gauss_1_xc = torch.exp(- ((x_c - x_1) ** 2 + (y_c - y_1) ** 2) / (2 * sigma_atom_sq))
        Gauss_1_xc /= (2 * np.pi * sigma_atom_sq)
        Gauss_2_xc = torch.exp(- ((x_c - x_2) ** 2 + (y_c - y_2) ** 2) / (2 * sigma_atom_sq))
        Gauss_2_xc /= (2 * np.pi * sigma_atom_sq)
        image_phys_true = (Gauss_1_xc + Gauss_2_xc) / 2.0 + 0j
        l2norm_image_phys_true = torch.sqrt(torch.sum(torch.abs(image_phys_true) ** 2 * pixel_size ** 2))
        
        for half_space in [True, False]:
        
            n_inplanes = n_pixels // 2 + 1 if half_space else n_pixels
            polar_grid = PolarGrid(
                radius_max = radius_max,
                dist_radii = dist_radii,
                n_inplanes = n_inplanes,
                quadrature = QuadratureType.GAUSS_JACOBI_BETA_1,
                uniform = True
            )
            theta = polar_grid.theta_shell
            radius = polar_grid.radius_shells
            n_shells = polar_grid.n_shells
            n_inplanes = polar_grid.n_inplanes

            x_1_scaled = x_1 * 2.0 / box_size * (2.0 * np.pi)
            y_1_scaled = y_1 * 2.0 / box_size * (2.0 * np.pi)
            x_2_scaled = x_2 * 2.0 / box_size * (2.0 * np.pi)
            y_2_scaled = y_2 * 2.0 / box_size * (2.0 * np.pi)

            x_p = torch.from_numpy(polar_grid.x_points)
            y_p = torch.from_numpy(polar_grid.y_points)
            r_p = torch.from_numpy(polar_grid.radius_points)

            sigma_atom_sq_scaled = sigma_atom_sq * (2.0 / box_size) ** 2
            Gauss_kp = - (2 * sigma_atom_sq_scaled * (np.pi * r_p) ** 2)
            # factor = (4 * sigma_atom_sq_scaled * np.pi ** 2)
            T_kp_1 = torch.exp(Gauss_kp - 1j * (x_p * x_1_scaled + y_p * y_1_scaled))
            T_kp_2 = torch.exp(Gauss_kp - 1j * (x_p * x_2_scaled + y_p * y_2_scaled))
            image_fourier_true = (T_kp_1 + T_kp_2) / 2.0
            l2norm_image_fourier_true = torch.sqrt(polar_grid.integrate(torch.abs(image_fourier_true) ** 2))

            for precision in [Precision.SINGLE, Precision.DOUBLE]:
                nufft_eps = nufft_eps_single if precision == Precision.SINGLE else nufft_eps_double
                for use_cuda in [False, True]:
                    if not use_cuda and precision == Precision.SINGLE:
                        continue
                    for image_repeat in [1, 2]:
                        
                        image_phys_true_repeat = image_phys_true.unsqueeze(0).repeat(image_repeat, 1, 1)
                        image_fourier_true_repeat = image_fourier_true.unsqueeze(0).repeat(image_repeat, 1)
                        
                        image_fourier_recover = cartesian_phys_to_fourier_polar(
                            grid_cartesian_phys = phys_grid,
                            grid_fourier_polar = polar_grid,
                            images_phys = image_phys_true_repeat,
                            eps = nufft_eps,
                            precision = precision,
                            use_cuda = use_cuda
                        )
                        l2norm_image_fourier_recover = torch.sqrt(polar_grid.integrate(torch.abs(image_fourier_recover) ** 2))
                        cross_correlation_for = (polar_grid.integrate(image_fourier_true_repeat * torch.conj(image_fourier_recover)) / l2norm_image_fourier_recover / l2norm_image_fourier_true).real
                        
                        image_phys_recover = fourier_polar_to_cartesian_phys(
                            grid_fourier_polar = polar_grid,
                            grid_cartesian_phys = phys_grid,
                            image_polar = image_fourier_true_repeat,
                            eps = nufft_eps,
                            precision = precision,
                            use_cuda = use_cuda
                        )
                        image_phys_recover_real = image_phys_recover.real
                        l2norm_image_phys_real_recover = torch.sqrt(torch.sum(np.abs(image_phys_recover_real) ** 2 * pixel_size ** 2, dim=(1, 2)))
                        cross_correlation_back = (torch.sum(image_phys_true_repeat.real * image_phys_recover_real.real * pixel_size ** 2, dim=(1,2)) / l2norm_image_phys_real_recover / l2norm_image_phys_true).real

                        image_phys_recover_two = fourier_polar_to_cartesian_phys(
                            grid_fourier_polar = polar_grid,
                            grid_cartesian_phys = phys_grid,
                            image_polar = image_fourier_recover,
                            eps = nufft_eps,
                            precision = precision,
                            use_cuda = use_cuda
                        )
                        l2norm_image_phys_real_recover_two = torch.sqrt(torch.sum(np.abs(image_phys_recover_two.real) ** 2 * pixel_size ** 2, dim=(1, 2)))
                        cross_correlation_two = (torch.sum(image_phys_true_repeat.real * image_phys_recover_two.real * pixel_size ** 2, dim=(1,2)) / l2norm_image_phys_true / l2norm_image_phys_real_recover_two).real
                        
                        assert np.allclose(cross_correlation_for, torch.ones_like(cross_correlation_for), atol = acc_tol)
                        assert np.allclose(cross_correlation_back, torch.ones_like(cross_correlation_back), atol = acc_tol)
                        assert np.allclose(cross_correlation_two, torch.ones_like(cross_correlation_two), atol = acc_tol)


if __name__ == "__main__":
    test_nufft_2d()
    print("2D NUFFT tests passed!")