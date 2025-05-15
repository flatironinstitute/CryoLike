from numpy import pi
import torch
from torch.testing import assert_close
import time

from cryolike.cross_correlation_likelihood import CrossCorrelationLikelihood, conform_ctf
from cryolike.util import (
    CrossCorrelationReturnType,
    Precision,
)

from benchmark_fixtures import (
    parameters,
    make_batch_size_params,
    make_n_pixels_params,
    make_n_displacements_params,
    make_mock_polar_grid,
    make_mock_viewing_angles,
    make_mock_ctf,
    make_mock_templates
)


def benchmark_cross_correlation(params: parameters):
    
    (torch_float_type, _, _) = params.precision.get_dtypes(default=Precision.SINGLE)
    polar_grid = make_mock_polar_grid(params.n_pixels)
    viewing_angles = make_mock_viewing_angles(params.n_templates, params.precision)
    ctf = make_mock_ctf(polar_grid, True, params.precision)
    templates = make_mock_templates(polar_grid, viewing_angles, params.precision)
    images = templates.to_images()

    print(f"Running benchmark with case: ")
    print(f"  n_pixels: {params.n_pixels}")
    print(f"  n_templates: {params.n_templates}")
    print(f"  n_displacements_x: {params.n_displacements_x}")
    print(f"  n_displacements_y: {params.n_displacements_y}")
    print(f"  n_templates_per_batch: {params.n_templates_per_batch}")
    print(f"  n_images_per_batch: {params.n_images_per_batch}")
    print(f"  precision: {params.precision}")
    print(f"  device: {params.device}")
    
    ####
    cc = CrossCorrelationLikelihood(
        templates = templates,
        max_displacement = 1.0,
        n_displacements_x = params.n_displacements_x,
        n_displacements_y = params.n_displacements_y,
        precision = params.precision,
        device = params.device,
        verbose = False
    )
    _t_start = time.time()
    res = cc._compute_cross_correlation_likelihood(
        device=torch.device(params.device),
        images_fourier = images.images_fourier,
        ctf=ctf.ctf,
        n_pixels_phys = params.n_pixels*params.n_pixels,
        n_templates_per_batch=params.n_templates_per_batch,
        n_images_per_batch=params.n_images_per_batch,
        return_type=CrossCorrelationReturnType.OPTIMAL_POSE,
        return_integrated_likelihood=False,
    )#
    _t_end = time.time()
    _t_ms = (_t_end - _t_start) * 1000
    print(f"  Time taken: {_t_ms:.2f} ms")

    torch.cuda.empty_cache()
    return _t_ms
    


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        exit(1)
    
    params_batch_size = make_batch_size_params()
    for params in params_batch_size:
        torch.cuda.empty_cache()
        t_ms = benchmark_cross_correlation(params)
        # print(f"  n_images_per_batch: {params.n_images_per_batch}")
        # print(f"  n_templates_per_batch: {params.n_templates_per_batch}")

    params_n_pixels = make_n_pixels_params()
    for params in params_n_pixels:
        torch.cuda.empty_cache()
        t_ms = benchmark_cross_correlation(params)
        # print(f"  n_pixels: {params.n_pixels}")
    
    params_n_displacements = make_n_displacements_params()
    for params in params_n_displacements:
        torch.cuda.empty_cache()
        t_ms = benchmark_cross_correlation(params)
        # print(f"  n_displacements: {params.n_displacements_x * params.n_displacements_y}")

    return