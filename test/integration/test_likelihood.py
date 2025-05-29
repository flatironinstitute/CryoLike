import numpy as np
import torch
from pathlib import Path
from copy import deepcopy
from torch.testing import assert_close

from cryolike.grids import PolarGrid
from cryolike.stacks import Templates
from cryolike.microscopy import CTF
from cryolike.metadata import LensDescriptor, ViewingAngles
from cryolike.util import (AtomShape, AtomicModel, CrossCorrelationReturnType, Precision, to_torch, get_device)
from cryolike.cross_correlation_likelihood import CrossCorrelationLikelihood, OptimalPoseReturn, OptimizedDisplacementAndRotationReturn
from cryolike.likelihood import calc_likelihood_optimal_pose, calc_fourier_likelihood_images_given_optimal_pose#, calc_physical_likelihood_images_given_optimal_pose


## TODO: optimize and clean up this test
def test_likelihood():

    use_cuda = True
    device = "cuda" if use_cuda else "cpu"
    precision = Precision.SINGLE
    snr = 1.0

    slope_tol = 0.05

    nufft_eps = 1e-12
    box_size = 32.0
    n_pixels = 128
    pixel_size = box_size / n_pixels
    atom_radii = 3.0

    radius_max = n_pixels / (2.0 * np.pi) * np.pi / 2.0 #/ 2.0
    dist_radii = 1.0 / (2.0 * np.pi) * np.pi / 2.0 #/ 2.0
    n_inplanes = n_pixels

    max_displacement = 4.0 * pixel_size
    n_displacements_x = 9
    n_displacements_y = 9

    true_displacement_x = 1.0 * pixel_size
    true_displacement_y = 1.0 * pixel_size

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
    gammas_viewing_true = 2.0 * np.pi / n_inplanes * np.random.randint(0, n_inplanes, n_images)
    viewing_angles = ViewingAngles(polars=polars_viewing, azimus=azimus_viewing, gammas=gammas_viewing_true)

    theta = polar_grid.theta_shell
    radius = polar_grid.radius_shells
    file_dir = Path(__file__).resolve()
    data_file = file_dir.parent.parent.joinpath("data").joinpath("1uao.pdb")
    atom_shape = AtomShape.HARD_SPHERE
    atomic_model = AtomicModel.read_from_pdb(atom_radii = atom_radii, pdb_file = str(data_file.absolute()), atom_selection = "name CA", box_size = box_size)
    tp = Templates.generate_from_positions(atomic_model, viewing_angles, polar_grid, box_size, atom_shape=atom_shape, precision=precision)
    tp.normalize_images_fourier(ord = 2, use_max = False)

    n_images = tp.n_images
    # defocus = np.random.uniform(300.0, 900.0, n_images)
    defocus = np.ones(n_images, dtype=np.float64) * 900.0
    Astigmatism = np.random.uniform(0, 20, n_images)
    defocusU = defocus + Astigmatism / 2
    defocusV = defocus - Astigmatism / 2
    defocusAng = np.zeros(n_images, dtype=np.float64)#np.random.uniform(-90, 90, n_images)
    phaseShift = np.zeros(n_images, dtype=np.float64)
    ctf_desc = LensDescriptor(
        defocusU = defocusU, # in Angstrom
        defocusV = defocusV,  # in Angstrom
        defocusAngle = defocusAng, # in degrees, defocus angle
        sphericalAberration = 2.7,  # in mm, spherical aberration
        voltage = 300,  # in kV
        amplitudeContrast = 0.1,    # amplitude contrast
        phaseShift = phaseShift,   # in degrees
    )
    ctf = CTF(
        ctf_descriptor=ctf_desc,
        polar_grid = polar_grid,
        box_size = box_size, # in Angstrom
        anisotropy = True#False#
    )
    
    image = tp.to_images()
    image.displace_images_fourier(
        x_displacements = true_displacement_x,
        y_displacements = true_displacement_y,
        precision = precision,
    )
    image.rotate_images_fourier_discrete(gammas_viewing_true)
    image.apply_ctf(ctf)
    image.transform_to_spatial(grid=(n_pixels, pixel_size), device = device)
    mean, std = image.center_physical_image_signal()

    # _, cross_correlation_optimal_pose = calc_physical_likelihood_images_given_optimal_pose(
    #     images=image,
    #     model=atomic_model,
    #     atom_shape=atom_shape,
    #     viewing_angles=viewing_angles,
    #     x_displacements=true_displacement_x,
    #     y_displacements=true_displacement_y,
    #     ctf=ctf,
    #     precision=precision,
    #     device=device,
    #     verbose=False
    # )
    # assert_close(cross_correlation_optimal_pose, torch.ones_like(cross_correlation_optimal_pose), rtol=1e-6, atol=1e-6)
    
    image_true = deepcopy(image.images_phys.real)
    _, sigma_noise_added = image.add_noise_phys(snr = snr)
    mean, std = image.center_physical_image_signal()

    ## forward and backward FFT to consider bandlimit the physical images according to the polar grid of Fourier space
    image.transform_to_fourier(polar_grid, device = device)
    image.transform_to_spatial(grid=(n_pixels, pixel_size), device = device)

    image_true -= mean
    image_true /= std
    
    noise_variance = torch.sum((image.images_phys - image_true) ** 2, dim = (1, 2)) / n_pixels ** 2
    loglik_true = - n_pixels ** 2 / 2 - np.log(2 * np.pi * noise_variance.cpu()) * (n_pixels ** 2 / 2)

    # likelihood_optimal_pose, cross_correlation_optimal_pose = calc_physical_likelihood_images_given_optimal_pose(
    #     images=image,
    #     model=atomic_model,
    #     atom_shape=atom_shape,
    #     viewing_angles=viewing_angles,
    #     search_displacements=False,
    #     x_displacements=true_displacement_x,
    #     y_displacements=true_displacement_y,
    #     ctf=ctf,
    #     precision=precision,
    #     device=device,
    #     verbose=False
    # )
    # A = np.vstack([likelihood_optimal_pose, np.ones(len(likelihood_optimal_pose))]).T
    # m, c = np.linalg.lstsq(A, loglik_true, rcond=None)[0]
    # print("For calc_physical_likelihood_images_given_optimal_pose without search_displacements, slope is: ", m, " and intercept is: ", c)
    # assert np.isclose(m, 1.0, atol=slope_tol)
    
    displacements_x_grid = torch.linspace(-max_displacement, max_displacement, n_displacements_x, dtype=torch.float32, device=device)
    dicplacements_y_grid = torch.linspace(-max_displacement, max_displacement, n_displacements_y, dtype=torch.float32, device=device)
    # likelihood_optimal_pose_search_displacements, cross_correlation_optimal_pose_search_displacements = calc_physical_likelihood_images_given_optimal_pose(
    #     images=image,
    #     model=atomic_model,
    #     atom_shape=atom_shape,
    #     viewing_angles=viewing_angles,
    #     search_displacements=True,
    #     x_displacements=displacements_x_grid,
    #     y_displacements=dicplacements_y_grid,
    #     ctf=ctf,
    #     precision=precision,
    #     device=device,
    #     verbose=False
    # )
    # likelihood_optimal_pose_best_displacements = torch.amax(likelihood_optimal_pose_search_displacements, dim=1)
    # A = np.vstack([likelihood_optimal_pose_best_displacements, np.ones(len(likelihood_optimal_pose_best_displacements))]).T
    # m, c = np.linalg.lstsq(A, loglik_true, rcond=None)[0]
    # print("For calc_physical_likelihood_images_given_optimal_pose with search_displacements, slope is: ", m, " and intercept is: ", c)
    # assert np.isclose(m, 1.0, atol=slope_tol)

    likelihood_fourier_optimal_pose_search_displacements, cross_correlation_fourier_optimal_pose_search_displacements = calc_fourier_likelihood_images_given_optimal_pose(
        images=image,
        model=atomic_model,
        atom_shape=atom_shape,
        viewing_angles=viewing_angles,
        search_displacements=True,
        x_displacements=displacements_x_grid,
        y_displacements=dicplacements_y_grid,
        ctf=ctf,
        precision=precision,
        device=device,
        verbose=False
    )
    likelihood_fourier_optimal_pose_best_displacements = torch.amax(likelihood_fourier_optimal_pose_search_displacements, dim=1)
    A = np.vstack([likelihood_fourier_optimal_pose_best_displacements, np.ones(len(likelihood_fourier_optimal_pose_best_displacements))]).T
    m, c = np.linalg.lstsq(A, loglik_true, rcond=None)[0]
    print("For calc_fourier_likelihood_images_given_optimal_pose with search_displacements, slope is: ", m, " and intercept is: ", c)
    assert np.isclose(m, 1.0, atol=slope_tol)

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
    print("For _compute_cross_correlation_likelihood, slope is: ", m, " and intercept is: ", c)
    assert np.isclose(m, 1.0, atol=slope_tol)

    res = calc_likelihood_optimal_pose(
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
    print("For calc_likelihood_optimal_pose with mode fourier, slope is: ", m, " and intercept is: ", c)
    assert np.isclose(m, 1.0, atol=slope_tol)

    # res = calc_likelihood_optimal_pose(
    #     template = tp,
    #     image = image,
    #     ctf = ctf,
    #     mode = "phys",
    #     template_indices = optimal_pose.optimal_template_S,
    #     displacements_x = optimal_pose.optimal_displacement_x_S,
    #     displacements_y = optimal_pose.optimal_displacement_y_S,
    #     inplane_rotations = optimal_pose.optimal_inplane_rotation_S,
    #     return_distance = False,
    #     return_likelihood = True,
    #     return_cross_correlation = True,
    #     precision = precision,
    #     use_cuda = True
    # )
    # assert isinstance(res, tuple)
    # assert len(res) == 2
    # log_likelihood_physical_optimal, cross_correlation_physical = res
    # A = np.vstack([log_likelihood_physical_optimal, np.ones(len(log_likelihood_physical_optimal))]).T
    # m, c = np.linalg.lstsq(A, loglik_true, rcond=None)[0]
    # print("For calc_likelihood_optimal_pose with mode phys, slope is: ", m, " and intercept is: ", c)
    # assert np.isclose(m, 1.0, atol=slope_tol)

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
    # print("Optional return types test passed")


if __name__ == "__main__":
    test_likelihood()
    print("All tests passed")