import torch
import glob
import numpy as np
from multiprocessing import Pool, cpu_count
import os


def _torch_load_weights_only_true(file_path):
    return torch.load(file_path, map_location='cpu', weights_only=True)


def _concatenate_image_batches(file_list, num_workers):
    with Pool(num_workers) as pool:
        tensors = pool.map(_torch_load_weights_only_true, file_list)
    concatenated_tensor = torch.cat(tensors, dim=0)
    return concatenated_tensor


def stitch_log_likelihood_matrices(n_templates: int = 0, n_image_stacks: int = 0, output_directory: str = '',phys=False,opt=False,integrated=False, cc=False):
    if phys == False and opt == False and integrated == False and cc == False:
       raise ValueError('at least one of physical, integrated or optimal log likelihoods must be post processed.') 
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
    cc_list = []
    for i_template in range(n_templates):
        folder_log_likelihood = os.path.join(output_directory, 'likelihood', 'template%d'%i_template, 'log_likelihood')
        folder_cc = os.path.join(output_directory, 'likelihood', 'template%d'%i_template, 'cross_correlation')
        for i_stack in range(n_image_stacks):
            if opt == True:
                opt_fourier_list.append(os.path.join(folder_log_likelihood, f'log_likelihood_fourier_S_stack_{i_stack:06}.pt'))
            if phys == True:
                phys_list.append(os.path.join(folder_log_likelihood, f'log_likelihood_phys_S_stack_{i_stack:06}.pt'))
            if integrated == True:
                int_fourier_list.append(os.path.join(folder_log_likelihood, f'log_likelihood_S_stack_{i_stack:06}.pt'))
            if cc == True:
                cc_list.append(os.path.join(folder_cc, f'cross_correlation_S_stack_{i_stack:06}.pt'))
    if phys == True:
        opt_phys_ll_matrix = _concatenate_image_batches(phys_list, num_workers).reshape(n_templates, -1)
    if opt == True:
        opt_fourier_ll_matrix = _concatenate_image_batches(opt_fourier_list, num_workers).reshape(n_templates, -1)
    if integrated == True:
        int_fourier_ll_matrix = _concatenate_image_batches(int_fourier_list, num_workers).reshape(n_templates, -1)
    if cc == True:
        cc_matrix = _concatenate_image_batches(cc_list, num_workers).reshape(n_templates, -1)

    output_matrix_directory = os.path.join(output_directory, 'concatenated_matrices')
    os.makedirs(output_matrix_directory, exist_ok=True)
    if phys == True:
        torch.save(opt_phys_ll_matrix, os.path.join(output_matrix_directory, 'optimal_physical_log_likelihood_matrix.pt'))
    if opt == True:
        torch.save(opt_fourier_ll_matrix, os.path.join(output_matrix_directory, 'optimal_fourier_log_likelihood_matrix.pt'))
    if integrated == True:
        torch.save(int_fourier_ll_matrix, os.path.join(output_matrix_directory, 'integrated_fourier_log_likelihood_matrix.pt'))
    if cc == True:
        torch.save(cc_matrix, os.path.join(output_matrix_directory, 'cross_correlation_matrix.pt'))
