import numpy as np
import torch
from pathlib import Path
from copy import deepcopy

from cryolike.grids import PolarGrid
from cryolike.stacks import Templates, Images
from cryolike.microscopy import CTF, LensDescriptor, ViewingAngles
from cryolike.util import (AtomShape, AtomicModel, CrossCorrelationReturnType, Precision, to_torch)
from cryolike.cross_correlation_likelihood import CrossCorrelationLikelihood, OptimalPoseReturn, OptimizedDisplacementAndRotationReturn
from cryolike.likelihood import calc_distance_optimal_templates_vs_physical_images

from time import time

def test_likelihood():

    use_cuda = True
    device = "cuda" if use_cuda else "cpu"
    precision = Precision.SINGLE
    snr = 1.0

    slope_tol = 0.05

    nufft_eps = 1e-12
    box_size = 32.0
    n_pixels = 256
    pixel_size = box_size / n_pixels
    atom_radii = 3.0

    radius_max = n_pixels / (2.0 * np.pi) * np.pi / 2.0 #/ 2.0
    dist_radii = 1.0 / (2.0 * np.pi) * np.pi / 2.0 #/ 2.0
    n_inplanes = 256
    # viewing_distance = 8.0 / (4.0 * np.pi)

    max_displacement = 4.0 * pixel_size
    n_displacements_x = 9
    n_displacements_y = 9

    # n_possible_displacements, possible_displacements_x, possible_displacements_y = get_possible_displacements(max_displacement, n_displacements)
    true_displacement_x = 1.0 * pixel_size
    true_displacement_y = 1.0 * pixel_size
    # true_displacement_x = possible_displacements_x[0]
    # true_displacement_y = possible_displacements_y[0]
    true_rotation = 2.0 * np.pi / n_inplanes #* 20

    polar_grid = PolarGrid(
        radius_max = radius_max,
        dist_radii = dist_radii,
        n_inplanes = n_inplanes,
        uniform = True
    )
    n_shells = polar_grid.n_shells

    n_images = 100
    polars_viewing = np.random.uniform(0.0, np.pi, n_images)
    azimus_viewing = np.random.uniform(0.0, 2.0 * np.pi, n_images)
    gammas_viewing = np.random.uniform(0.0, 2.0 * np.pi, n_images)

    viewing_angles = ViewingAngles(polars=polars_viewing, azimus=azimus_viewing, gammas=gammas_viewing)

    theta = polar_grid.theta_shell
    radius = polar_grid.radius_shells
    file_dir = Path(__file__).resolve()
    data_file = file_dir.parent.parent.joinpath("data").joinpath("1uao.pdb")
    atomic_model = AtomicModel.read_from_pdb(atom_radii = atom_radii, pdb_file = str(data_file.absolute()), atom_selection = "name CA", box_size = box_size)
    tp = Templates.generate_from_positions(atomic_model, viewing_angles, polar_grid, box_size, atom_shape=AtomShape.HARD_SPHERE,
        precision=precision)
    # tp.normalize_templates_fourier(ord = 2, use_max = False)

    n_images = tp.n_templates
    # defocus = np.random.uniform(300.0, 900.0, n_images)
    defocus = np.ones(n_images, dtype=np.float64) * 300.0
    Astigmatism = np.random.uniform(0, 20, n_images)
    defocusU = defocus + Astigmatism / 2
    defocusV = defocus - Astigmatism / 2
    defocusAng = np.zeros(n_images, dtype=np.float64)#np.random.uniform(-90, 90, n_images)
    phaseShift = np.zeros(n_images, dtype=np.float64)
    ctf_desc = LensDescriptor(
        defocusU = defocusU, # in Angstrom
        defocusV = defocusV,  # in Angstrom
        defocusAng = defocusAng, # in degrees, defocus angle
        sphericalAberration = 2.7,  # in mm, spherical aberration
        voltage = 300,  # in kV
        amplitudeContrast = 0.1,    # amplitude contrast
        phaseShift = phaseShift,   # in degrees
    )
    ctf = CTF(
        ctf_descriptor=ctf_desc,
        polar_grid = polar_grid,
        box_size = box_size, # in Angstrom
        anisotropy = True#True#
    )

    image = Images.from_templates(templates = tp)
    image.apply_ctf(ctf)
    image.displace_images_fourier(
        x_displacements = true_displacement_x,
        y_displacements = true_displacement_y,
        precision = precision,
    )
    image.rotate_images_fourier(true_rotation)
    image.transform_to_spatial(grid=(n_pixels, pixel_size), use_cuda = use_cuda)

    assert image.images_phys is not None
    image_true = deepcopy(image.images_phys.real)
    _, sigma_noise = image.add_noise_phys(snr = snr)
    image.transform_to_fourier(polar_grid, use_cuda = use_cuda)
    mean, std = image.center_physical_image_signal()
    image.transform_to_fourier(polar_grid, use_cuda = use_cuda)
    image.transform_to_spatial(grid=(n_pixels, pixel_size), use_cuda = use_cuda)
    image_true -= mean
    image_true /= std
    assert image.images_phys is not None
    sigma_noise = torch.sqrt(torch.mean((image.images_phys - image_true) ** 2, dim = (1, 2)))
    loglik_true = - n_pixels ** 2 / 2 - np.log(2 * np.pi * sigma_noise ** 2) * (n_pixels ** 2 / 2)

    cclik = CrossCorrelationLikelihood(
        templates = tp,
        max_displacement = max_displacement,
        n_displacements_x = n_displacements_x,
        n_displacements_y = n_displacements_y,
        precision = precision,
        device = device,
        verbose = True
    )

    _imgs = image.images_fourier
    _ctf = to_torch(ctf.ctf, precision, "cpu")
    assert _imgs is not None

    res = cclik._compute_cross_correlation_likelihood(
        device=torch.device("cuda"),
        images_fourier = _imgs,
        ctf = _ctf,
        n_pixels_phys = image.phys_grid.n_pixels[0] * image.phys_grid.n_pixels[1],
        n_images_per_batch=64,#128,#256,
        n_templates_per_batch=16,
        return_type=CrossCorrelationReturnType.OPTIMAL_POSE,
        return_integrated_likelihood=True
    )
    assert len(res) == 2
    (optimal_pose, log_likelihood_fourier_integrated) = res
    assert isinstance(optimal_pose, OptimalPoseReturn)
    assert isinstance(log_likelihood_fourier_integrated, torch.Tensor)
    A = np.vstack([log_likelihood_fourier_integrated, np.ones(len(log_likelihood_fourier_integrated))]).T
    m, c = np.linalg.lstsq(A, loglik_true, rcond=None)[0]
    assert np.isclose(m, 1.0, atol=slope_tol)

    res = calc_distance_optimal_templates_vs_physical_images(
        template = tp,
        image = image,
        ctf = ctf,
        mode = "fourier",
        template_indices = optimal_pose.optimal_template_S,
        displacements_x = optimal_pose.optimal_displacement_x_S,
        displacements_y = optimal_pose.optimal_displacement_y_S,
        inplane_rotations = optimal_pose.optimal_inplane_rotation_S,
        return_distance = False,
        return_likelihood = True,
        return_cross_correlation = True,
        precision = precision,
        use_cuda = True
    )
    assert isinstance(res, tuple)
    assert len(res) == 2
    log_likelihood_fourier_optimal, cross_correlation_fourier = res
    A = np.vstack([log_likelihood_fourier_optimal, np.ones(len(log_likelihood_fourier_optimal))]).T
    m, c = np.linalg.lstsq(A, loglik_true, rcond=None)[0]
    assert np.isclose(m, 1.0, atol=slope_tol)

    res = calc_distance_optimal_templates_vs_physical_images(
        template = tp,
        image = image,
        ctf = ctf,
        mode = "phys",
        template_indices = optimal_pose.optimal_template_S,
        displacements_x = optimal_pose.optimal_displacement_x_S,
        displacements_y = optimal_pose.optimal_displacement_y_S,
        inplane_rotations = optimal_pose.optimal_inplane_rotation_S,
        return_distance = False,
        return_likelihood = True,
        return_cross_correlation = True,
        precision = precision,
        use_cuda = True
    )
    assert isinstance(res, tuple)
    assert len(res) == 2
    log_likelihood_physical_optimal, cross_correlation_physical = res
    A = np.vstack([log_likelihood_physical_optimal, np.ones(len(log_likelihood_physical_optimal))]).T
    m, c = np.linalg.lstsq(A, loglik_true, rcond=None)[0]
    assert np.isclose(m, 1.0, atol=slope_tol)

    print("Likelihood test passed")

    # Just checking that all the branches work
    a = cclik._compute_cross_correlation_likelihood(
        device=torch.device("cuda"),
        images_fourier = _imgs,
        ctf = _ctf,
        n_pixels_phys = image.phys_grid.n_pixels[0] * image.phys_grid.n_pixels[1],
        n_images_per_batch=64,#128,#256,
        n_templates_per_batch=16,
        return_type=CrossCorrelationReturnType.OPTIMAL_DISPLACEMENT_AND_ROTATION,
        return_integrated_likelihood=False
    )
    assert isinstance(a, OptimizedDisplacementAndRotationReturn)

    b = cclik._compute_cross_correlation_likelihood(
        device=torch.device("cuda"),
        images_fourier = _imgs,
        ctf = _ctf,
        n_pixels_phys = image.phys_grid.n_pixels[0] * image.phys_grid.n_pixels[1],
        n_images_per_batch=64,#128,#256,
        n_templates_per_batch=16,
        return_type=CrossCorrelationReturnType.NONE,
        return_integrated_likelihood=True
    )
    assert isinstance(b, torch.Tensor)

    _ = cclik._compute_cross_correlation_likelihood(
        device=torch.device("cuda"),
        images_fourier = _imgs,
        ctf = _ctf,
        n_pixels_phys = image.phys_grid.n_pixels[0] * image.phys_grid.n_pixels[1],
        n_images_per_batch=64,#128,#256,
        n_templates_per_batch=16,
        return_type=CrossCorrelationReturnType.FULL_TENSOR,
        return_integrated_likelihood=False
    )

    _ = cclik._compute_cross_correlation_likelihood(
        device=torch.device("cuda"),
        images_fourier = _imgs,
        ctf = _ctf,
        n_pixels_phys = image.phys_grid.n_pixels[0] * image.phys_grid.n_pixels[1],
        n_images_per_batch=64,#128,#256,
        n_templates_per_batch=16,
        return_type=CrossCorrelationReturnType.OPTIMAL_DISPLACEMENT,
        return_integrated_likelihood=False
    )

    _ = cclik._compute_cross_correlation_likelihood(
        device=torch.device("cuda"),
        images_fourier = _imgs,
        ctf = _ctf,
        n_pixels_phys = image.phys_grid.n_pixels[0] * image.phys_grid.n_pixels[1],
        n_images_per_batch=64,#128,#256,
        n_templates_per_batch=16,
        return_type=CrossCorrelationReturnType.OPTIMAL_ROTATION,
        return_integrated_likelihood=False
    )

    _ = cclik._compute_cross_correlation_likelihood(
        device=torch.device("cuda"),
        images_fourier = _imgs,
        ctf = _ctf,
        n_pixels_phys = image.phys_grid.n_pixels[0] * image.phys_grid.n_pixels[1],
        n_images_per_batch=64,#128,#256,
        n_templates_per_batch=16,
        return_type=CrossCorrelationReturnType.OPTIMAL_DISPLACEMENT_AND_ROTATION,
        return_integrated_likelihood=False
    )
    print("Optional return types test passed")

if __name__ == "__main__":
    test_likelihood()
    print("All tests passed!")