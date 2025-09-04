import torch
from multiprocessing import Pool, cpu_count
from typing import Sequence
from os import PathLike

from cryolike.file_mgmt import PostProcessFileManager


def _torch_load_weights_only_true(file_path: PathLike | str):
    return torch.load(file_path, map_location='cpu', weights_only=True) # pragma: no cover
    # no-covering this because the coverage tool doesn't realize it's being called as a callback


def _concatenate_image_batches(file_list: Sequence[PathLike | str], num_workers: int):

    with Pool(num_workers) as pool:
        tensors = pool.map(_torch_load_weights_only_true, file_list)
    concatenated_tensor = torch.cat(tensors, dim=0)
    return concatenated_tensor


def stitch_log_likelihood_matrices(
    n_templates: int = 0,
    n_image_stacks: int = 0,
    output_directory: str = 'output',
    batch_directory: str = '',
    template_directory: str = '',
    opt=False,
    phys=False,
    integrated=False,
    cc=False
):
    n_cpus = cpu_count()
    if not (phys or opt or integrated or cc):
        raise ValueError("At least one output type must be selected.")
    if n_cpus == 1:
        raise RuntimeError("This function is not useful for single core machines")

    file_mgr = PostProcessFileManager(output_directory, batch_directory, template_directory)

    (n_t, n_imgs) = file_mgr.confirm_counts(n_templates, n_image_stacks)
    n_jobs = n_t * n_imgs
    num_workers = min(n_cpus, n_jobs)

    file_lists = file_mgr.get_source_lists(n_t, n_imgs, opt, phys, integrated, cc)
    targets = file_mgr.get_output_targets()

    if opt:
        torch.save(
            _concatenate_image_batches(
                file_lists.FourierStacks,
                num_workers
            ).reshape(n_t, -1),
            targets.FourierMatrix
        )
    if phys:
        torch.save(
            _concatenate_image_batches(
                file_lists.PhysStacks,
                num_workers
            ).reshape(n_t, -1),
            targets.PhysMatrix
        )
    if integrated:
        torch.save(
            _concatenate_image_batches(
                file_lists.IntegratedStacks,
                num_workers
            ).reshape(n_t, -1),
            targets.IntegratedMatrix
        )
    if cc:
        torch.save(
            _concatenate_image_batches(
                file_lists.CrossCorrelationStacks,
                num_workers
            ).reshape(n_t, -1),
            targets.CrossCorrelationMatrix
        )
