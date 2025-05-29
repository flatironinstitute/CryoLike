import numpy as np
import torch

from cryolike.grids import PolarGrid, CartesianGrid2D, CartesianGrid3D, Volume, PhysicalVolume
from cryolike.microscopy.nufft import fourier_polar_to_cartesian_phys, volume_phys_to_fourier_points
from cryolike.util import Precision, QuadratureType


def test_nufft_3d():

    n_pixels_tests = [128, 256]
    box_size = 128.0

    nufft_eps_double = 1e-12
    nufft_eps_single = 1e-6
    x_1 = 0.1 * box_size
    y_1 = 0.1 * box_size
    x_2 = -0.2 * box_size
    y_2 = -0.2 * box_size
    flag_assert = False

    print('Testing 3D to 3D NUFFT (3D Physical density -> 2D Fourier slices in 3D fourier space)')
    for n_pixels in n_pixels_tests:

        pixel_size = box_size / n_pixels
        sigma_atom = pixel_size * 4.0
        sigma_atom_sq = sigma_atom ** 2

        # for half_space in [True, False]:
        for half_space in [False]:
            
            phys_grid = CartesianGrid2D(n_pixels = n_pixels, pixel_size = pixel_size)
            radius_max = n_pixels * np.pi / 2.0 / (2.0 * np.pi)
            dist_radii = np.pi / 2.0 / (2.0 * np.pi)
            # n_inplanes = n_pixels // 2 + 1 if half_space else n_pixels
            n_inplanes = n_pixels
            polar_grid = PolarGrid(
                radius_max = radius_max,
                dist_radii = dist_radii,
                n_inplanes = n_inplanes,
                quadrature = QuadratureType.GAUSS_JACOBI_BETA_1,
                uniform = True,
                half_space = half_space
            )
            theta = polar_grid.theta_shell
            radius = polar_grid.radius_shells
            n_shells = polar_grid.n_shells
            n_inplanes = polar_grid.n_inplanes

            phys_grid_3d = CartesianGrid3D(n_voxels = n_pixels, voxel_size = box_size / n_pixels)
            x_c = torch.from_numpy(phys_grid_3d.x_voxels)
            y_c = torch.from_numpy(phys_grid_3d.y_voxels)
            z_c = torch.from_numpy(phys_grid_3d.z_voxels)
            Gauss_1 = torch.exp(- ((x_c - x_1) ** 2 + (y_c - y_1) ** 2 + z_c ** 2) / (2 * sigma_atom_sq))
            Gauss_1 *= (2 * np.pi * sigma_atom_sq) ** (-1.5)
            Gauss_2 = torch.exp(- ((x_c - x_2) ** 2 + (y_c - y_2) ** 2 + z_c ** 2) / (2 * sigma_atom_sq))
            Gauss_2 *= (2 * np.pi * sigma_atom_sq) ** (-1.5)
            assert isinstance(Gauss_1, torch.Tensor)
            assert isinstance(Gauss_2, torch.Tensor)
            density_physical = (Gauss_1 + Gauss_2) / 2.0
            image_phys_true = torch.sum(density_physical, dim = 2) * pixel_size

            l1norm_image_phys_true = torch.sum(image_phys_true * pixel_size ** 2)
            l2norm_image_phys_true = torch.sqrt(torch.sum(torch.abs(image_phys_true) ** 2 * pixel_size ** 2))
            density_physical = density_physical + 0j
            
            x_p = torch.from_numpy(polar_grid.x_points) * (2.0 * np.pi)
            y_p = torch.from_numpy(polar_grid.y_points) * (2.0 * np.pi)
            r_p = torch.from_numpy(polar_grid.radius_points)
            sigma_atom_sq_scaled = sigma_atom_sq * (2.0 / box_size) ** 2
            Gauss_kp = - (2 * sigma_atom_sq_scaled * (np.pi * r_p) ** 2)
            T_kp_1 = torch.exp(Gauss_kp - 1j * (x_p * x_1 + y_p * y_1) * 2.0 / box_size) * (2 * np.pi * sigma_atom_sq_scaled)
            T_kp_2 = torch.exp(Gauss_kp - 1j * (x_p * x_2 + y_p * y_2) * 2.0 / box_size) * (2 * np.pi * sigma_atom_sq_scaled)
            image_fourier_true = (T_kp_1 + T_kp_2) / 2.0
            l2norm_image_fourier_true = np.sqrt(polar_grid.integrate(torch.abs(image_fourier_true) ** 2))

            voxel_size = box_size / n_pixels
            volume = Volume(density_physical_data=PhysicalVolume(
                density_physical = density_physical,
                voxel_size = voxel_size 
            ), box_size = box_size)
            fourier_slices_x = torch.from_numpy(polar_grid.x_points)
            fourier_slices_y = torch.from_numpy(polar_grid.y_points)
            fourier_slices_z = torch.zeros_like(fourier_slices_x)
            fourier_slices = torch.stack([fourier_slices_x, fourier_slices_y, fourier_slices_z], dim = 1)
            for precision in [Precision.SINGLE, Precision.DOUBLE]:
                nufft_eps = nufft_eps_single if precision == Precision.SINGLE else nufft_eps_double
                for device in ["cpu", "cuda"]:
                    if device == "cpu" and precision == Precision.SINGLE:
                        continue
                    image_fourier = volume_phys_to_fourier_points(
                        volume = volume,
                        fourier_slices = fourier_slices,
                        eps = nufft_eps,
                        input_device = device,
                        precision = precision
                    )
                    image_phys_recover = fourier_polar_to_cartesian_phys(
                        grid_fourier_polar = polar_grid,
                        grid_cartesian_phys = phys_grid,
                        image_polar = image_fourier,
                        eps = nufft_eps,
                        device = device,
                        precision = precision
                    )[0]
                    l2norm_image_phys_recover = torch.sqrt(torch.sum(torch.abs(image_phys_recover) ** 2 * pixel_size ** 2))
                    cross_correlation_back = (torch.sum(image_phys_true * image_phys_recover * pixel_size ** 2) / l2norm_image_phys_recover / l2norm_image_phys_true).real.item()

                    assert np.isclose(cross_correlation_back, 1.0, atol = 1e-6), f'Cross-correlation: {cross_correlation_back}'

            del volume
            del fourier_slices
            del phys_grid_3d
    
    
if __name__ == '__main__':
    test_nufft_3d()
    print('3D NUFFT tests passed!')