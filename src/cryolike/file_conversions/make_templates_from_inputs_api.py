from __future__ import annotations

from typing import Sequence, TYPE_CHECKING
import torch
import numpy as np
from pathlib import Path

if TYPE_CHECKING: # pragma: no cover
    from cryolike.file_mgmt import TEMPLATE_INPUT_DESC, TemplateOutputFiles

from cryolike.stacks.template import Templates
from cryolike.metadata import ImageDescriptor
from cryolike.util import Precision, AtomicModel, get_device, InputFileType
from cryolike.file_mgmt import TemplateFileManager
from cryolike.grids import Volume, PhysicalVolume
from cryolike.plot import plot_images, plot_power_spectrum


def _make_plotter_fn(output_plots: bool, params: ImageDescriptor):
    def _no_op(tp: Templates, names: TemplateOutputFiles):
        ...  # "function body intentionally left blank"
    if not output_plots:
        return _no_op
    def _generate_plots(tp: Templates, names: TemplateOutputFiles):
        if not tp.has_fourier_images():
            return
        
        plot_images(tp.images_fourier, grid=tp.polar_grid, n_plots=16, filename=str(names.fourier_images_plot_file.absolute()), show=False)
        plot_power_spectrum(source=tp, filename_plot=str(names.power_plot_file.absolute()), show=False)
        templates_phys = tp.transform_to_spatial(grid=params.cartesian_grid, max_to_transform=16)
        plot_images(templates_phys, grid=params.cartesian_grid, n_plots=16, filename=str(names.physical_images_plot_file.absolute()), show=False)
    return _generate_plots


def _make_templates_from_mrc_file(
    mrc_file: str,
    descriptor: ImageDescriptor,
    torch_float_type: torch.dtype,
    device: torch.device,
    verbose: bool
) -> Templates:
    volume = Volume.from_mrc(filename = mrc_file)
    if volume.density_physical is None:
        raise ValueError(f"Can't happen: parsing mrc file {mrc_file} did not generate a physical density.")
    volume.density_physical = volume.density_physical.to(torch_float_type).to(device)
    return Templates.generate_from_physical_volume(
        volume=volume,
        polar_grid=descriptor.polar_grid,
        viewing_angles=descriptor.viewing_angles,
        precision=descriptor.precision,
        verbose=verbose
    )


def _make_templates_from_pdb_file(
    pdb_file: str,
    descriptor: ImageDescriptor,
    verbose: bool
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
        verbose = verbose
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
        verbose=verbose
    )


def _make_raw_template(
    input: TEMPLATE_INPUT_DESC,
    descriptor: ImageDescriptor,
    t_float: torch.dtype,
    device: torch.device,
    verbose: bool
):
    (src, _, ftype) = input
    if ftype == InputFileType.MRC:
        assert isinstance(src, Path)
        src_fn = str(src.absolute())
        tp = _make_templates_from_mrc_file(src_fn, descriptor, t_float, device, verbose)
    elif ftype == InputFileType.PDB:
        assert isinstance(src, Path)
        src_fn = str(src.absolute())
        tp = _make_templates_from_pdb_file(src_fn, descriptor, verbose)
    elif ftype == InputFileType.MEM:
        assert isinstance(src, torch.Tensor) or isinstance(src, np.ndarray)
        tp = _make_templates_from_memory_array(src, descriptor, t_float, device, verbose)
    else:
        raise ValueError("Unknown input format")

    return tp


def _make_template_maker_fn(
        descriptor: ImageDescriptor,
        t_float: torch.dtype,
        device: torch.device,
        verbose: bool = False
):
    def maker(input: TEMPLATE_INPUT_DESC) -> Templates:
        tp = _make_raw_template(input, descriptor, t_float, device, verbose)
        tp.normalize_images_fourier(ord=2, use_max=False)
        return tp
    return maker


# TODO: expose parameters for template normalization
def make_templates_from_inputs(
    list_of_inputs: Sequence[str | np.ndarray | torch.Tensor],
    image_parameters_file: str,
    output_plots: bool = True,
    folder_output: str = "./templates/",
    verbose: bool = False
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
    device = get_device(None)
    reader_fn = _make_template_maker_fn(descriptor, t_float, device, verbose)
    plotter_fn = _make_plotter_fn(output_plots, descriptor)
    
    filemgr = TemplateFileManager(folder_output, output_plots, list_of_inputs, reader_fn, plotter_fn, verbose)

    if filemgr.inputs_include_pdb_files() and not descriptor.is_compatible_with_pdb():
        raise ValueError("To process PDB files, you must either set an atom_radii or set use_protein_residue_model=True.")

    filemgr.process_inputs()
    filemgr.save_file_list()
