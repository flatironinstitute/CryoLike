import os
from pathlib import Path
import torch
from torch.testing import assert_close
from unittest.mock import patch
from pytest import raises, mark
import numpy as np

from cryolike.util import stitch_log_likelihood_matrices
PKG = "cryolike.util.post_process_output"


@mark.parametrize("n_stacks", [(10), (None)])
def test_stitch_log_likelihood_matrices(tmp_path, n_stacks):
    n_templates = n_stacks
    n_image_stacks = n_stacks
    n_images_per_stack = 1024

    if n_stacks is None:
        # Create files to read for stack counts
        n_image_stacks_internal = 11
        n_templates_internal = 9
        templates_dir = os.path.join(tmp_path, 'templates')
        image_dir = os.path.join(tmp_path, 'images', 'phys')
        os.makedirs(templates_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        templates_file_list = np.array([f"filename{x}" for x in range(n_templates_internal)])
        np.save(os.path.join(templates_dir, "template_file_list.npy"), templates_file_list)
        for i in range(n_image_stacks_internal):
            fn = Path(image_dir) / f"file_{i}.img"
            fn.touch()
    else:
        n_templates_internal = n_templates
        n_image_stacks_internal = n_image_stacks

    for i_template in range(n_templates_internal):
        for i_stack in range(n_image_stacks_internal):
            output_directory_likelihood = os.path.join(tmp_path, 'likelihood', 'template%d'%i_template, 'log_likelihood')
            os.makedirs(output_directory_likelihood, exist_ok=True)
            mock_likelihood = torch.arange(n_images_per_stack * i_stack, n_images_per_stack * (i_stack+1)).float()
            torch.save(mock_likelihood, os.path.join(output_directory_likelihood, f'log_likelihood_fourier_S_stack_{i_stack:06}.pt'))
            torch.save(mock_likelihood, os.path.join(output_directory_likelihood, f'log_likelihood_phys_S_stack_{i_stack:06}.pt'))
            torch.save(mock_likelihood, os.path.join(output_directory_likelihood, f'log_likelihood_S_stack_{i_stack:06}.pt'))
    
    if n_stacks is None:
        stitch_log_likelihood_matrices(output_directory=tmp_path)
    else:
        stitch_log_likelihood_matrices(n_templates, n_image_stacks, tmp_path)
    
    opt_phys_ll_matrix = torch.load(os.path.join(tmp_path, 'likelihood_matrix', 'optimal_physical_log_likelihood_matrix.pt'), map_location='cpu', weights_only=True)
    opt_fourier_ll_matrix = torch.load(os.path.join(tmp_path, 'likelihood_matrix', 'optimal_fourier_log_likelihood_matrix.pt'), map_location='cpu', weights_only=True)
    int_fourier_ll_matrix = torch.load(os.path.join(tmp_path, 'likelihood_matrix', 'integrated_fourier_log_likelihood_matrix.pt'), map_location='cpu', weights_only=True)

    assert opt_phys_ll_matrix.shape == (n_templates_internal, n_image_stacks_internal * n_images_per_stack)
    assert opt_fourier_ll_matrix.shape == (n_templates_internal, n_image_stacks_internal * n_images_per_stack)
    assert int_fourier_ll_matrix.shape == (n_templates_internal, n_image_stacks_internal * n_images_per_stack)

    for i_template in range(n_templates_internal):
        for i_stack in range(n_image_stacks_internal):
            assert_close(opt_phys_ll_matrix[i_template, i_stack*n_images_per_stack:(i_stack+1)*n_images_per_stack], torch.arange(n_images_per_stack * i_stack, n_images_per_stack * (i_stack+1)).float())
            assert_close(opt_fourier_ll_matrix[i_template, i_stack*n_images_per_stack:(i_stack+1)*n_images_per_stack], torch.arange(n_images_per_stack * i_stack, n_images_per_stack * (i_stack+1)).float())
            assert_close(int_fourier_ll_matrix[i_template, i_stack*n_images_per_stack:(i_stack+1)*n_images_per_stack], torch.arange(n_images_per_stack * i_stack, n_images_per_stack * (i_stack+1)).float())


def test_stitch_log_likelihood_matrices_throws_on_single_core():
    with patch(f"{PKG}.cpu_count") as cpu_count:
        cpu_count.return_value = 1
        with raises(RuntimeError, match="single core"):
            stitch_log_likelihood_matrices()
