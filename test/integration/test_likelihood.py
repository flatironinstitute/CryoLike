import numpy as np
import torch
from pathlib import Path
from copy import deepcopy
from torch.testing import assert_close
import pytest

from cryolike.grids import PolarGrid
from cryolike.likelihoods import (
    template_first_comparator,
    compute_optimal_pose,
    OptimalPoseReturn,
    compute_cross_correlation_complete,
    CrossCorrelationReturn,
    compute_optimal_displacement,
    OptimalDisplacementReturn,
    compute_optimal_displacement_and_rotation,
    OptimalDisplacementAndRotationReturn,
    compute_optimal_rotation,
    OptimalRotationReturn
)
from cryolike.stacks import Templates
from cryolike.microscopy import CTF
from cryolike.metadata import LensDescriptor, ViewingAngles
from cryolike.util import (AtomShape, AtomicModel, Precision)
from cryolike.likelihoods.likelihood import calc_likelihood_optimal_pose, calc_fourier_likelihood_images_given_optimal_pose#, calc_physical_likelihood_images_given_optimal_pose


## TODO: optimize and clean up this test
def test_likelihood():

    use_cuda = True
    device = "cuda" if use_cuda else "cpu"
    precision = Precision.SINGLE
    snr = 1.0

    slope_tol = 0.05

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
    n_images = 100
    polars_viewing = np.random.uniform(0.0, np.pi, n_images)
    azimus_viewing = np.random.uniform(0.0, 2.0 * np.pi, n_images)
    gammas_viewing_true = 2.0 * np.pi / n_inplanes * np.random.randint(0, n_inplanes, n_images)
    viewing_angles = ViewingAngles(polars=polars_viewing, azimus=azimus_viewing, gammas=gammas_viewing_true)

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
    image.displace_fourier_images(
        x_displacements = true_displacement_x,
        y_displacements = true_displacement_y,
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
    loglik_true = - n_pixels ** 2 / 2 - torch.log(2 * np.pi * noise_variance.cpu()) * (n_pixels ** 2 / 2)

    displacements_x_grid = torch.linspace(-max_displacement, max_displacement, n_displacements_x, dtype=torch.float32, device=device)
    displacements_y_grid = torch.linspace(-max_displacement, max_displacement, n_displacements_y, dtype=torch.float32, device=device)

    likelihood_fourier_optimal_pose_search_displacements, cross_correlation_fourier_optimal_pose_search_displacements = calc_fourier_likelihood_images_given_optimal_pose(
        images=image,
        model=atomic_model,
        atom_shape=atom_shape,
        viewing_angles=viewing_angles,
        search_displacements=True,
        x_displacements=displacements_x_grid,
        y_displacements=displacements_y_grid,
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

    tp.set_displacement_grid(
        max_displacement_pixels=max_displacement / pixel_size,
        n_displacements_x=n_displacements_x,
        n_displacements_y=n_displacements_y,
        pixel_size_angstrom=1.
    )
    assert image.images_fourier is not None

    iterator = template_first_comparator(
        torch.device(device),
        image,
        tp,
        ctf,
        n_images_per_batch=64,
        n_templates_per_batch=16,
        return_integrated_likelihood=True,
        precision=precision
    )

    res = compute_optimal_pose(
        iterator,
        tp,
        image,
        precision,
        include_integrated_log_likelihood=True
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
        template_indices = optimal_pose.optimal_template_M,
        displacements_x = optimal_pose.optimal_displacement_x_M,
        displacements_y = optimal_pose.optimal_displacement_y_M,
        inplane_rotations = optimal_pose.optimal_inplane_rotation_M,
        return_distance = False,
        return_likelihood = True,
        return_cross_correlation = True,
        precision = precision,
        use_cuda = True
    )
    assert isinstance(res, tuple)
    assert len(res) == 2
    log_likelihood_fourier_optimal, cross_correlation_fourier = res
    assert isinstance(log_likelihood_fourier_optimal, torch.Tensor)
    A = np.vstack([log_likelihood_fourier_optimal, np.ones(len(log_likelihood_fourier_optimal))]).T
    m, c = np.linalg.lstsq(A, loglik_true, rcond=None)[0]
    print("For calc_likelihood_optimal_pose with mode fourier, slope is: ", m, " and intercept is: ", c)
    assert np.isclose(m, 1.0, atol=slope_tol)


@pytest.mark.parametrize(
    "fn_res", [
        (compute_cross_correlation_complete, CrossCorrelationReturn),
        (compute_optimal_displacement, OptimalDisplacementReturn),
        (compute_optimal_displacement_and_rotation, OptimalDisplacementAndRotationReturn),
        (compute_optimal_rotation, OptimalRotationReturn)
    ],
)
def test_cross_correlation_return_types(fn_res):
    if (not torch.cuda.is_available()):
        pytest.skip("Test cannot run because CUDA is not present.")
    (fn, expected_type) = fn_res

    ### NOTE: **ALL** of the following until the iterator is built
    ### is just copied from above, this should be handled better
    device = "cuda"
    precision = Precision.SINGLE

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
    n_images = 100
    polars_viewing = np.random.uniform(0.0, np.pi, n_images)
    azimus_viewing = np.random.uniform(0.0, 2.0 * np.pi, n_images)
    gammas_viewing_true = 2.0 * np.pi / n_inplanes * np.random.randint(0, n_inplanes, n_images)
    viewing_angles = ViewingAngles(polars=polars_viewing, azimus=azimus_viewing, gammas=gammas_viewing_true)

    file_dir = Path(__file__).resolve()
    data_file = file_dir.parent.parent.joinpath("data").joinpath("1uao.pdb")
    atom_shape = AtomShape.HARD_SPHERE
    atomic_model = AtomicModel.read_from_pdb(atom_radii = atom_radii, pdb_file = str(data_file.absolute()), atom_selection = "name CA", box_size = box_size)
    tp = Templates.generate_from_positions(atomic_model, viewing_angles, polar_grid, box_size, atom_shape=atom_shape, precision=precision)
    tp.normalize_images_fourier(ord = 2, use_max = False)

    n_images = tp.n_images
    defocus = np.ones(n_images, dtype=np.float64) * 900.0
    Astigmatism = np.random.uniform(0, 20, n_images)
    defocusU = defocus + Astigmatism / 2
    defocusV = defocus - Astigmatism / 2
    defocusAng = np.zeros(n_images, dtype=np.float64)
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
    image.displace_fourier_images(
        x_displacements = true_displacement_x,
        y_displacements = true_displacement_y,
    )
    image.rotate_images_fourier_discrete(gammas_viewing_true)
    image.apply_ctf(ctf)
    image.transform_to_spatial(grid=(n_pixels, pixel_size), device = device)

    tp.set_displacement_grid(
        max_displacement_pixels=max_displacement / pixel_size,
        n_displacements_x=n_displacements_x,
        n_displacements_y=n_displacements_y,
        pixel_size_angstrom=1.
    )
    assert image.images_fourier is not None

    iterator = template_first_comparator(
        torch.device(device),
        image,
        tp,
        ctf,
        n_images_per_batch=64,
        n_templates_per_batch=16,
        return_integrated_likelihood=True,
        precision=precision
    )

    res = fn(
        iterator,
        tp,
        image,
        precision,
        False
    )
    assert isinstance(res, expected_type)


if __name__ == "__main__":
    test_likelihood()
    print("All tests passed")
