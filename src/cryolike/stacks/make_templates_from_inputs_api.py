from typing import Sequence
import torch
import numpy as np
import os

from .template import Templates
from cryolike.util import Precision, AtomicModel, check_cuda, get_cuda_bool
from cryolike.grids import Volume, PhysicalVolume
from cryolike.plot import plot_images, plot_power_spectrum
from cryolike.metadata import ImageDescriptor

MRC_EXTENSIONS = ['.mrc', '.mrcs', '.map']
PDB_EXTENSIONS = ['.pdb']


def _set_up_directories(folder_output: str, output_plots: bool) -> str | None:
    """Safely create the requested output directory and, optionally,
    the directory for storing plots. If outputting plots wasn't requested,
    return None as the plot output directory, which is a signal elsewhere
    that plots aren't requested.

    Args:
        folder_output (str): Target for the output templates
        output_plots (bool): Whether plots were requested, in which case
            a child directory will be created for them and returned.

    Returns:
        (str | None): None, if no plots requested; else, the directory
            where plots should be written.
    """

    os.makedirs(folder_output, exist_ok = True)
    if output_plots:
        folder_output_plots = os.path.join(folder_output, 'plots')
        os.makedirs(folder_output_plots, exist_ok = True)
    return folder_output_plots if output_plots else None


def _make_plotter_fn(plot_output_dir: str | None):
    def _no_op(tp: Templates, params: ImageDescriptor, name: str):
        ...  # "function body intentionally left blank"
    if plot_output_dir is None:
        return _no_op
    def _generate_plots(tp: Templates, params: ImageDescriptor, name: str):
        if not tp.has_fourier_images():
            return
        plot_images(tp.images_fourier, grid=tp.polar_grid, n_plots=16, filename=os.path.join(plot_output_dir, "templates_fourier_%s.png" % name), show=False)
        plot_power_spectrum(source=tp, filename_plot=os.path.join(plot_output_dir, "power_spectrum_%s.png" % name), show=False)
        templates_phys = tp.transform_to_spatial(grid=params.cartesian_grid, max_to_transform=16)
        plot_images(templates_phys, grid=params.cartesian_grid, n_plots=16, filename=os.path.join(plot_output_dir, "templates_phys_%s.png" % name), show=False)
    return _generate_plots


def _get_input_name(input: str | torch.Tensor | np.ndarray, iteration_count: int) -> tuple[str, str]:
    # TODO: optional verbosity flag to avoid print statements
    if isinstance(input, str):
        print(f"Processing {input}...")
        return os.path.splitext(os.path.basename(input))
    else: # input must be in-memory array
        print(f"Processing numpy array or torch tensor [input {iteration_count}]...")
        name = f"tensor_{iteration_count}"
        return (name, '')


def _make_templates_from_mrc_file(
    mrc_file: str,
    descriptor: ImageDescriptor,
    torch_float_type: torch.dtype,
    device: torch.device,
    verbose: bool
) -> Templates:
    use_cuda = get_cuda_bool(device)
    volume = Volume.from_mrc(filename = mrc_file)
    if volume.density_physical is None:
        raise ValueError(f"Can't happen: parsing mrc file {mrc_file} did not generate a physical density.")
    volume.density_physical = volume.density_physical.to(torch_float_type).to(device)
    return Templates.generate_from_physical_volume(
        volume=volume,
        polar_grid=descriptor.polar_grid,
        viewing_angles=descriptor.viewing_angles,
        precision=descriptor.precision,
        verbose=verbose,
        use_cuda=use_cuda
    )


def _make_templates_from_pdb_file(
    pdb_file: str,
    descriptor: ImageDescriptor,
    verbose: bool,
    use_cuda: bool = True
) -> Templates:
    if not descriptor.is_compatible_with_pdb():
        raise ValueError("Attempting to read templates from PDB file, but the atom_radii parameter or use_protein_residue_model=True is not set.")
    box_size = descriptor.cartesian_grid.box_size
    if len(np.unique(box_size)) != 1:
        raise ValueError("Reading an atomic model from a PDB file with a non-square box size is not yet supported.")
    _box_edge_length: float = box_size[0]
    atomic_model = AtomicModel.read_from_pdb(
        pdb_file = pdb_file,
        atom_selection = descriptor.atom_selection,
        atom_radii = descriptor.atom_radii,
        box_size = _box_edge_length,
        use_protein_residue_model = descriptor.use_protein_residue_model
    )
    return Templates.generate_from_positions(
        atomic_model=atomic_model,
        viewing_angles=descriptor.viewing_angles,
        polar_grid=descriptor.polar_grid,
        box_size=box_size,
        atom_shape=descriptor.atom_shape,
        precision=descriptor.precision,
        verbose = verbose,
        use_cuda = use_cuda
    )


def _make_templates_from_memory_array(
    input: np.ndarray | torch.Tensor,
    descriptor: ImageDescriptor,
    torch_float_type: torch.dtype,
    device: torch.device,
    verbose: bool,
) -> Templates:
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)
    # TODO NOTE: hard-coded to assume square pixels mapped into cubic voxels
    density_physical_data = PhysicalVolume(density_physical=input, voxel_size=descriptor.cartesian_grid.pixel_size[0])
    # TODO: Might be easier to just take the first dimension of the box size
    volume = Volume(density_physical_data=density_physical_data, box_size=descriptor.get_3d_box_size())
    if (volume.density_physical is None):
        raise ValueError("Unreachable: creating a volume with explicit physical density did not preserve physical density.")
    volume.density_physical = volume.density_physical.to(torch_float_type).to(device)
    return Templates.generate_from_physical_volume(
        volume,
        descriptor.polar_grid,
        descriptor.viewing_angles,
        precision=descriptor.precision,
        verbose=verbose,
        use_cuda=get_cuda_bool(device)
    )


def _inputs_include_pdb_files(inputs: Sequence):
    for x in inputs:
        if not isinstance(x, str):
            continue
        (_, ext) = os.path.splitext(os.path.basename(x))
        if ext in PDB_EXTENSIONS:
            return True
    return False


def _make_raw_template(
    input: str | np.ndarray | torch.Tensor,
    iteration_cnt: int,
    descriptor: ImageDescriptor,
    t_float: torch.dtype,
    device: torch.device,
    verbose: bool
):
    use_cuda = get_cuda_bool(device)
    (name, extension) = _get_input_name(input, iteration_cnt)
    if isinstance(input, str):
        # TODO: it might be better to do a more reliable test
        if extension in MRC_EXTENSIONS:
            print("mrc_name:", name)
            tp = _make_templates_from_mrc_file(input, descriptor, t_float, device, verbose)
        elif extension in PDB_EXTENSIONS:
            print(f"pdb_name: {name}")
            tp = _make_templates_from_pdb_file(input, descriptor, verbose, use_cuda)
        else:
            raise ValueError("Unknown input format")
    elif isinstance(input, np.ndarray) or isinstance(input, torch.Tensor):
        tp = _make_templates_from_memory_array(input, descriptor, t_float, device, verbose)
    else:
        raise ValueError("Unknown input format")
    return (tp, name)


def _get_template_output_filename(folder_output: str, name: str) -> str:
    template_file = os.path.join(folder_output, f"templates_fourier_{name}.pt")
    same_name_count = 0
    while (os.path.exists(template_file)):
        same_name_count += 1
        template_file = os.path.join(folder_output, f"templates_fourier_{name}_{same_name_count}.pt")
    return template_file


# TODO: expose parameters for template normalization
def make_templates_from_inputs(
    list_of_inputs: Sequence[str | np.ndarray | torch.Tensor],
    image_parameters_file: str,
    output_plots: bool = True,
    folder_output: str = "./templates/",
    verbose: bool = False,
    use_cuda: bool = True
):
    """Parse a series of inputs to internal pytorch tensor representation, then save to an output directory.

    Args:
        list_of_inputs (list): List of inputs. Can be paths to pdb files, paths to mrc/mrcs/map files,
            or numpy arrays or torch tensors.
        image_parameters_file (str): Path to a saved image parameters file (ImageDescriptor)
        output_plots (bool, optional): Whether to output plots of the parsed Templates. Defaults to True.
        folder_output (str, optional): Directory in which to write the generated Template data.
            Defaults to "./templates/".
        verbose (bool, optional): Whether to provide verbose output. Defaults to False.
        use_cuda (bool, optional): Whether to use cuda. Defaults to True.

    Raises:
        ValueError: If any inputs have an unrecognized file extension or are neither string
            nor array type.

    Returns:
        None. By side effect, all parsed templates will be written to an output directory as
            Pytorch Tensors.
    """
    if len(list_of_inputs) == 0:
        return
    descriptor = ImageDescriptor.load(image_parameters_file)
    precision = Precision.from_str(descriptor.precision)
    (t_float, _, _) = precision.get_dtypes(default=Precision.SINGLE)
    device = check_cuda(use_cuda)

    if _inputs_include_pdb_files(list_of_inputs) and not descriptor.is_compatible_with_pdb():
        raise ValueError("To process PDB files, you must either set an atom_radii or set use_protein_residue_model=True.")

    plots_output_dir = _set_up_directories(folder_output, output_plots)
    plotter_fn = _make_plotter_fn(plots_output_dir)
    template_file_list = []
    for i, input in enumerate(list_of_inputs):
        (tp, name) = _make_raw_template(input, i, descriptor, t_float, device, verbose)
        tp.normalize_images_fourier(ord=2, use_max=False)
        plotter_fn(tp, descriptor, name)
        template_file = _get_template_output_filename(folder_output, name)
        torch.save(tp.images_fourier, template_file)
        template_file_list.append(template_file)

    np.save(os.path.join(folder_output, 'template_file_list.npy'), template_file_list)
