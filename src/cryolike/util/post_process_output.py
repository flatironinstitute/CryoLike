import torch
import glob
import numpy as np
from multiprocessing import Pool, cpu_count
import os


# TODO: Currently unused internally & API is not documented.
# Needs review.
# If kept, convert os. to pathlib &
# try to centralize file-naming operations in the file mgr directory.

def _torch_load_weights_only_true(file_path):
    return torch.load(file_path, map_location='cpu', weights_only=True)


def _concatenate_image_batches(file_list, num_workers):
    with Pool(num_workers) as pool:
        tensors = pool.map(_torch_load_weights_only_true, file_list)
    concatenated_tensor = torch.cat(tensors, dim=0)
    return concatenated_tensor


def stitch_log_likelihood_matrices(n_templates: int = 0, n_image_stacks: int = 0, output_directory: str = ''):
    n_cpus = cpu_count()
    if n_cpus == 1:
        raise RuntimeError("This function is not useful for single core machines")

    if n_templates <= 0:
        n_templates = len(np.load(os.path.join(output_directory,'templates/template_file_list.npy')))
    if n_image_stacks <= 0:
        n_image_stacks = len(glob.glob(os.path.join(output_directory, 'images', 'phys/*')))

    n_jobs = n_templates * n_image_stacks
    num_workers = min(n_cpus, n_jobs)

    opt_fourier_list = []
    phys_list = []
    int_fourier_list = []
    for i_template in range(n_templates):
        folder_log_likelihood = os.path.join(output_directory, 'likelihood', 'template%d'%i_template, 'log_likelihood')
        for i_stack in range(n_image_stacks):
            opt_fourier_list.append(os.path.join(folder_log_likelihood, f'log_likelihood_fourier_S_stack_{i_stack:06}.pt'))
            phys_list.append(os.path.join(folder_log_likelihood, f'log_likelihood_phys_S_stack_{i_stack:06}.pt'))
            int_fourier_list.append(os.path.join(folder_log_likelihood, f'log_likelihood_S_stack_{i_stack:06}.pt'))
    opt_phys_ll_matrix = _concatenate_image_batches(opt_fourier_list, num_workers).reshape(n_templates, -1)
    opt_fourier_ll_matrix = _concatenate_image_batches(phys_list, num_workers).reshape(n_templates, -1)
    int_fourier_ll_matrix = _concatenate_image_batches(int_fourier_list, num_workers).reshape(n_templates, -1)

    output_directory_likelihood_matrix = os.path.join(output_directory, 'likelihood_matrix')
    os.makedirs(output_directory_likelihood_matrix, exist_ok=True)
    torch.save(opt_phys_ll_matrix, os.path.join(output_directory_likelihood_matrix, 'optimal_physical_log_likelihood_matrix.pt'))
    torch.save(opt_fourier_ll_matrix, os.path.join(output_directory_likelihood_matrix, 'optimal_fourier_log_likelihood_matrix.pt'))
    torch.save(int_fourier_ll_matrix, os.path.join(output_directory_likelihood_matrix, 'integrated_fourier_log_likelihood_matrix.pt'))
