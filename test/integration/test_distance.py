import numpy as np
import torch
from pathlib import Path

from cryolike.likelihoods import template_first_comparator, compute_optimal_pose
from cryolike.microscopy import CTF
from cryolike.metadata import LensDescriptor, ViewingAngles
from cryolike.stacks import Templates
from cryolike.util import Precision, AtomShape, AtomicModel
from cryolike.grids import PolarGrid

from time import time

# NOTE: This consistently runs out of memory for me;
# let's see if we can make it more efficient
# (I have switched to running on CPU for now)
def test_distance():

    use_cuda = False
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    precision = Precision.SINGLE
    snr = 0.1
    if precision == Precision.SINGLE:
        torch_float_type = torch.float32
    elif precision == Precision.DOUBLE:
        torch_float_type = torch.float64

    nufft_eps = 1e-12
    box_size = 32.0
    n_pixels = 256
    pixel_size = box_size / n_pixels
    atom_shape = AtomShape.HARD_SPHERE

    radius_max = n_pixels / (2.0 * np.pi) * np.pi / 2.0
    dist_radii = 1.0 / (2.0 * np.pi) * np.pi / 2.0
    n_inplanes = n_pixels * 2

    max_displacement = 1.5 * pixel_size
    n_displacements_x = 4
    n_displacements_y = 4

    true_displacement_x = 0.5 * pixel_size
    true_displacement_y = 0.5 * pixel_size
    true_rotation = 2.0 * np.pi / n_inplanes
    print("true_displacement_x: ", true_displacement_x)
    print("true_displacement_y: ", true_displacement_y)
    print("true_rotation: ", true_rotation)

    polar_grid = PolarGrid(
        radius_max = radius_max,
        dist_radii = dist_radii,
        n_inplanes = n_inplanes,
        uniform = True
    )
    n_shells = polar_grid.n_shells
    viewing_angles = ViewingAngles.from_viewing_distance(4.0 / (4.0 * np.pi))
    theta = polar_grid.theta_shell
    radius = polar_grid.radius_shells

    file_dir = Path(__file__).resolve()
    atomic_model = AtomicModel.read_from_pdb(pdb_file=str(file_dir.parent.parent.joinpath("data").joinpath("1uao.pdb")), box_size=box_size, use_protein_residue_model=True)
    tp = Templates.generate_from_positions(atomic_model, viewing_angles, polar_grid, box_size, atom_shape=atom_shape, precision=precision)

    n_images = tp.n_images
    # defocus = np.random.uniform(300.0, 900.0, n_images)
    defocus = np.linspace(300.0, 900.0, n_images)
    # Astigmatism = np.random.uniform(0, 20, n_images)
    Astigmatism = np.linspace(0, 20, n_images)
    defocusU = defocus + Astigmatism / 2
    defocusV = defocus - Astigmatism / 2
    defocusAng = np.zeros(n_images, dtype=np.float64)#np.random.uniform(-90, 90, n_images)
    phaseShift = np.zeros(n_images, dtype=np.float64)
    microscope = LensDescriptor(
        defocusU = defocusU, # in Angstrom
        defocusV = defocusV,  # in Angstrom
        defocusAngle = defocusAng, # in degrees, defocus angle
        sphericalAberration = 2.7,  # in mm, spherical aberration
        voltage = 300,  # in kV
        amplitudeContrast = 0.1,    # amplitude contrast
        phaseShift = phaseShift,   # in degrees
    )
    ctf = CTF(
        ctf_descriptor=microscope,
        polar_grid = polar_grid,
        box_size = box_size, # in Angstrom
        anisotropy = True
    )
    # image = Images.from_templates(templates = tp)
    image = tp.to_images()
    image.displace_fourier_images(
        x_displacements = true_displacement_x,
        y_displacements = true_displacement_y,
    )
    image.rotate_images_fourier_discrete(true_rotation)
    image.apply_ctf(ctf)
    image.transform_to_spatial(grid=(n_pixels, pixel_size), device=device, precision=precision)

    cross_correlation_true = torch.ones(n_images, dtype = torch_float_type)
    optimal_template_S_true = torch.arange(n_images, dtype = torch.long)
    optimal_displacement_x_S_true = torch.ones(n_images, dtype = torch_float_type) * true_displacement_x
    optimal_displacement_y_S_true = torch.ones(n_images, dtype = torch_float_type) * true_displacement_y
    optimal_inplane_rotation_S_true = torch.ones(n_images, dtype = torch_float_type) * true_rotation

    time_start = time()
    tp.set_displacement_grid(
        max_displacement / pixel_size, # we defined it in angstrom
        n_displacements_x=n_displacements_x,
        n_displacements_y=n_displacements_x,
        pixel_size_angstrom=pixel_size
    )
    iterator = template_first_comparator(
        device=device,
        templates=tp,
        images=image,
        ctf=ctf,
        n_images_per_batch=128,
        n_templates_per_batch=16,
        return_integrated_likelihood=False,
        precision=precision
    )
    time_end = time()
    print("Time (init): ", time_end - time_start)

    time_start = time()
    optimal_pose = compute_optimal_pose(iterator, tp, image, precision, False)
    time_end = time()
    print("Time (run): ", time_end - time_start)

    print("cc.cross_correlation_S_: ", optimal_pose.cross_correlation_M[:5])
    print("cc.optimal_template_S_: ", optimal_pose.optimal_template_M[:5])
    print("self.optimal_displacement_x_S_: ", optimal_pose.optimal_displacement_x_M[:5])
    print("self.optimal_displacement_y_S_: ", optimal_pose.optimal_displacement_y_M[:5])
    print("self.optimal_inplane_rotation_S_: ", optimal_pose.optimal_inplane_rotation_M[:5])

    assert torch.allclose(optimal_pose.cross_correlation_M, cross_correlation_true, atol = 1e-2)
    assert torch.allclose(optimal_pose.optimal_template_M, optimal_template_S_true)
    assert torch.allclose(optimal_pose.optimal_displacement_x_M, optimal_displacement_x_S_true, atol = 1e-2)
    assert torch.allclose(optimal_pose.optimal_displacement_y_M, optimal_displacement_y_S_true, atol = 1e-2)
    assert torch.allclose(optimal_pose.optimal_inplane_rotation_M, optimal_inplane_rotation_S_true, atol = 1e-2)
    
    
if __name__ == "__main__":
    test_distance()
    print("Distance tests passed!")
