import numpy as np
import torch
from cryolike.grids import PolarGrid, CartesianGrid2D
from cryolike.stacks import Templates
from cryolike.metadata import ViewingAngles
from cryolike.util import Precision, AtomShape, AtomicModel

# TODO: parametrize to test on cpu, cuda
def test_displacement():

    precision = Precision.DOUBLE
    nufft_eps = 1e-12
    box_size = 160.0
    sigma_atom = 1.0
    n_pixels = 256
    atom_shape = AtomShape.GAUSSIAN
    pixel_size = box_size / n_pixels

    radius_max = n_pixels * np.pi / 2.0 / (2.0 * np.pi)
    dist_radii = np.pi / 2.0 / (2.0 * np.pi)
    n_inplanes = n_pixels

    true_displacement_x = 0.2 * box_size
    true_displacement_y = 0.1 * box_size
    atomic_coordinates = np.array(
        [[-0.2, -0.1, 0.0],
        [0.1, 0.2, 0.0]]
    ) * box_size
    atom_radii = sigma_atom
    atomic_model = AtomicModel(atomic_coordinates = atomic_coordinates.copy(), atom_radii = atom_radii, box_size = box_size, precision = precision)

    print("atomic model:")
    print(atomic_model.atomic_coordinates)
    print("atom radii:", atomic_model.atom_radii)
    print("box size:", atomic_model.box_size)

    cartesian_grid = CartesianGrid2D(
        n_pixels = n_pixels,
        pixel_size = pixel_size,
    )
    polar_grid = PolarGrid(
        radius_max = radius_max,
        dist_radii = dist_radii,
        n_inplanes = n_inplanes,
        uniform = True
    )
    n_shells = polar_grid.n_shells
    viewing_angles = ViewingAngles(
        torch.zeros(1, dtype = torch.float64), 
        torch.zeros(1, dtype = torch.float64), 
        torch.zeros(1, dtype = torch.float64)
    )
    theta = polar_grid.theta_shell
    radius = polar_grid.radius_shells

    ## Create displaced images
    tp = Templates.generate_from_positions(
        atomic_model=atomic_model, 
        viewing_angles=viewing_angles, 
        polar_grid=polar_grid, 
        box_size=box_size, 
        atom_shape=atom_shape, 
        precision=precision
    )
    templates = tp.images_fourier
    assert templates is not None
    templates_phys = tp.transform_to_spatial((n_pixels, pixel_size), precision=precision)
    images = tp.to_images()
    n_images = images.n_images
    x_displacements = np.ones(n_images, dtype=np.float64) * true_displacement_x
    y_displacements = np.ones(n_images, dtype=np.float64) * true_displacement_y
    images.displace_images_fourier(
        x_displacements=x_displacements,
        y_displacements=y_displacements,
        precision=precision
    )
    images.transform_to_spatial(grid=(n_pixels, pixel_size), precision=precision)
    images_phys_moveimage = images.images_phys.clone()

    ## Create image of displaced atomic model
    atomic_coordinates_moved = atomic_coordinates.copy() + np.array([true_displacement_x, true_displacement_y, 0.0])
    atomic_model_displaced = AtomicModel(atomic_coordinates=atomic_coordinates_moved, atom_radii=atom_radii, box_size=box_size, precision=precision)
    templates = Templates.generate_from_positions(atomic_model=atomic_model_displaced, viewing_angles=viewing_angles, polar_grid=polar_grid, box_size=box_size, atom_shape=atom_shape, precision=precision)
    templates_phys = templates.transform_to_spatial((n_pixels, pixel_size), precision=precision)
    images_phys_moveatom = templates_phys
    assert images_phys_moveatom is not None

    from cryolike.plot import plot_images
    plot_images(images_phys_moveimage, cartesian_grid, n_plots=1, show=False, filename="images_phys_moveimage.png")
    plot_images(images_phys_moveatom, cartesian_grid, n_plots=1, show=False, filename="images_phys_moveatom.png")

    ## Calculate cross-correlation
    cross_correlation = torch.sum(images_phys_moveimage * images_phys_moveatom).real / torch.sqrt(torch.sum(images_phys_moveimage.abs() ** 2) * torch.sum(images_phys_moveatom.abs() ** 2))
    print(f"Cross-correlation: {cross_correlation:.4f}")
    assert np.isclose(cross_correlation, 1.0, atol=1e-3)


if __name__ == "__main__":
    test_displacement()
    print("Displacement tests passed!")