from typing import Callable, Literal, NamedTuple
import torch
import numpy as np
import os

from .template import Templates
from cryolike.util import Precision, set_precision, AtomicModel, AtomShape, check_cuda, interpret_precision
from cryolike.grids import PolarGrid, CartesianGrid2D, Volume, PhysicalVolume
from cryolike.plot import plot_images, plot_power_spectrum
from cryolike.microscopy import parse_parameters, ParsedParameters, save_parameters, print_parameters, ViewingAngles

# TODO: Rebase off templates-images-refactor branch

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
    def _no_op(tp: Templates, params: ParsedParameters, name: str):
        ...  # "function body intentionally left blank"
    if plot_output_dir is None:
        return _no_op
    def _generate_plots(tp: Templates, params: ParsedParameters, name: str):
        if tp.templates_fourier is None or len(tp.templates_fourier) == 0:
            return
        plot_images(tp.templates_fourier, grid=tp.polar_grid, n_plots=16, filename=os.path.join(plot_output_dir, "templates_fourier_%s.png" % name), show=False)
        plot_power_spectrum(source=tp, filename_plot=os.path.join(plot_output_dir, "power_spectrum_%s.png" % name), show=False)
        phys_grid = CartesianGrid2D(n_pixels=params.n_voxels[:2], pixel_size=params.voxel_size[:2])
        templates_phys = tp.transform_to_spatial(phys_grid=phys_grid, n_templates_stop=16, save_to_class=False)
        plot_images(templates_phys, grid=phys_grid, n_plots=16, filename=os.path.join(plot_output_dir, "templates_phys_%s.png" % name), show=False)
    return _generate_plots


class _TemplateConfig(NamedTuple):
    device: torch.device
    plotter_fn: Callable[[Templates, ParsedParameters, str], None]
    torch_float_type: torch.dtype
    params: ParsedParameters
    polar_grid: PolarGrid
    viewing_angles: ViewingAngles


def _make_templates_config(
    n_voxels: int, # number of voxels in each dimension
    voxel_size: float, # in Angstrom
    viewing_distance: float, # angular distance between two viewing angles
    resolution_factor: float, # 1.0 for full resolution at nyquist frequency, 0.5 for half resolution at double nyquist frequency
    precision: Literal['single'] | Literal['double'] | Precision,
    n_inplanes: int | None, # number of inplane angles
    atom_radii: float | None, # radius of atoms
    atom_selection: str, # selection of atoms
    use_protein_residue_model: bool, # use residue model
    atom_shape: str | AtomShape, # shape of atoms, either 'hard-sphere' or 'gaussian'
    output_plots: bool,
    folder_output: str,
) -> _TemplateConfig:
    device = check_cuda(True)   # TODO: Refactor to allow user specification
    plots_output_dir = _set_up_directories(folder_output, output_plots)
    plotter_fn = _make_plotter_fn(plots_output_dir)
    _precision = interpret_precision(precision)
    (torch_float_type, _, _) = set_precision(_precision, default=Precision.SINGLE)
    
    params = parse_parameters(n_voxels = n_voxels, voxel_size = voxel_size, resolution_factor = resolution_factor, precision = _precision, viewing_distance = viewing_distance, n_inplanes = n_inplanes, atom_radii = atom_radii, atom_selection = atom_selection, use_protein_residue_model = use_protein_residue_model, atom_shape = atom_shape)
    save_parameters(params, os.path.join(folder_output, "parameters.npz"))
    print_parameters(params)
    
    polar_grid = PolarGrid(
        radius_max = params.radius_max,
        dist_radii = params.dist_radii,
        n_inplanes = params.n_inplanes,
        uniform = True
    )
    viewing_angles = ViewingAngles.from_viewing_distance(viewing_distance)

    return _TemplateConfig(device, plotter_fn, torch_float_type, params, polar_grid, viewing_angles)


def _get_input_name(input: str, iteration_count: int) -> tuple[str, str]:
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
    cfg: _TemplateConfig,
    verbose: bool,
    use_cuda: bool
) -> Templates:
    volume = Volume.from_mrc(filename = mrc_file)
    if volume.density_physical is None:
        raise ValueError(f"Can't happen: parsing mrc file {mrc_file} did not generate a physical density.")
    volume.density_physical = volume.density_physical.to(cfg.torch_float_type).to(cfg.device)
    return Templates.generate_from_physical_volume(
        volume=volume,
        polar_grid=cfg.polar_grid,
        viewing_angles=cfg.viewing_angles,
        precision=cfg.params.precision,
        verbose=verbose,
        use_cuda=use_cuda
    )


def _make_templates_from_pdb_file(
    pdb_file: str,
    cfg: _TemplateConfig,
    use_protein_residue_model: bool,
    verbose: bool
) -> Templates:
    if cfg.params.atom_radii is None:
        raise ValueError("Attempting to read templates from PDB file, but the atom_radii parameter was not set.")
    if len(np.unique(cfg.params.box_size)) != 1:
        raise ValueError("Reading an atomic model from a PDB file with a non-square box size is not yet supported.")
    _box_edge_length: float = cfg.params.box_size[0]
    atomic_model = AtomicModel.read_from_pdb(
        pdb_file = pdb_file,
        atom_selection = cfg.params.atom_selection,
        atom_radii = cfg.params.atom_radii,
        box_size = _box_edge_length,
        use_protein_residue_model = use_protein_residue_model
    )
    return Templates.generate_from_positions(
        atomic_model=atomic_model,
        viewing_angles=cfg.viewing_angles,
        polar_grid=cfg.polar_grid,
        box_size=cfg.params.box_size,
        atom_shape=cfg.params.atom_shape,
        precision=cfg.params.precision,
        verbose = verbose
    )


def _make_templates_from_memory_array(
    input: np.ndarray | torch.Tensor,
    cfg: _TemplateConfig,
    verbose: bool,
    use_cuda: bool
) -> Templates:
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)
    density_physical_data = PhysicalVolume(density_physical=input, voxel_size=cfg.params.voxel_size)
    volume = Volume(density_physical_data=density_physical_data, box_size=cfg.params.box_size)
    if (volume.density_physical is None):
        raise ValueError("Unreachable: creating a volume with explicit physical density did not preserve physical density.")
    volume.density_physical = volume.density_physical.to(cfg.torch_float_type).to(cfg.device)
    return Templates.generate_from_physical_volume(
        volume,
        cfg.polar_grid,
        cfg.viewing_angles,
        precision=cfg.params.precision,
        verbose=verbose,
        use_cuda=use_cuda
    )


# TODO QUERY: Would it be valid for list_of_inputs to include a mix of file types and/or mix of files and arrays?
# TODO QUERY: Do we need to expose use_protein_residue_model? Shouldn't this be True for all PDB files and irrelevant otherwise?
def make_templates_from_inputs(
    list_of_inputs: list,
    n_voxels: int,
    voxel_size: float,
    viewing_distance: float,
    resolution_factor: float = 1.0,
    precision: Literal['single'] | Literal['double'] | Precision = Precision.SINGLE,
    n_inplanes: int | None = None,
    atom_radii: float | None = None,
    atom_selection: str = "name CA",
    use_protein_residue_model: bool = True, # TODO QUERY: Why should this ever not be true?
    atom_shape: Literal['hard-sphere'] | Literal['gaussian'] | AtomShape = AtomShape.GAUSSIAN,
    output_plots: bool = True,
    folder_output: str = "./templates/",
    verbose: bool = False,
    use_cuda: bool = True
):
    """Parse a series of inputs to internal pytorch tensor representation, then save to an output directory.

    Args:
        list_of_inputs (list): List of inputs. Can be paths to pdb files, paths to mrc/mrcs/map files,
            or numpy arrays or torch tensors.
        n_voxels (int): Number of voxels in each dimension. This is a scalar because only cubic
            volumes are presently supported.
        voxel_size (float): Size of each voxel, in Angstrom. Voxels are assumed cubic.
        viewing_distance (float): Angular distance between two viewing angles
        resolution_factor (float, optional): 1.0 for full resolution at nyquist frequency, 0.5
            for half resolution at double nyquist frequency. Defaults to 1.0.
        precision ('single' | 'double' | Precision, optional): Desired precision of output image tensors.
            Defaults to single precision.
        n_inplanes (int | None, optional): Number of inplane angles. If unset, will be automatically
            set during parameter parsing.
        atom_radii (float | None, optional): Radius of atoms, in Angstrom. Only meaningful for PDB
            inputs. If unset, will be set automatically during parameter parsing.
        atom_selection (str, optional): Which atoms to use from the model file. Only required
            for PDB inputs. Defaults to "name CA" (which assumes a protein residue model).
        use_protein_residue_model (bool, optional): Only meaningful for PDB inputs, where it
            should be set to True (the default).
        atom_shape ('hard-sphere' | 'gaussian' | AtomShape): Shape of atoms in the file. Only
            meaningful for PDB inputs. Defaults to Gaussian.
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
    cfg = _make_templates_config(
            n_voxels=n_voxels,
            voxel_size=voxel_size,
            viewing_distance=viewing_distance,
            resolution_factor=resolution_factor,
            precision=precision,
            n_inplanes=n_inplanes,
            atom_radii=atom_radii,
            atom_selection=atom_selection,
            use_protein_residue_model=use_protein_residue_model,
            atom_shape=atom_shape,
            output_plots=output_plots,
            folder_output=folder_output
        )

    template_file_list = []
    for i, input in enumerate(list_of_inputs):
        (name, extension) = _get_input_name(input, i)
        if isinstance(input, str):
            if extension in ['.mrc', '.mrcs', '.map']:
                print("mrc_name:", name)
                tp = _make_templates_from_mrc_file(input, cfg, verbose, use_cuda)
            elif extension in ['.pdb']:
                print(f"pdb_name: {name}")
                tp = _make_templates_from_pdb_file(input, cfg, use_protein_residue_model, verbose)
            else:
                raise ValueError("Unknown input format")
        elif isinstance(input, np.ndarray) or isinstance(input, torch.Tensor):
            tp = _make_templates_from_memory_array(input, cfg, verbose)
        else:
            raise ValueError("Unknown input format")
        tp.normalize_templates_fourier(ord=2, use_max=False)
        cfg.plotter_fn(tp, cfg.params, name)
        template_file = os.path.join(folder_output, f"templates_fourier_{name}.pt")
        same_name_count = 0
        while (os.path.exists(template_file)):
            same_name_count += 1
            template_file = os.path.join(folder_output, f"templates_fourier_{name}_{same_name_count}.pt")
        torch.save(tp.templates_fourier, template_file)
        
        template_file_list.append(template_file)
    np.save(os.path.join(folder_output, 'template_file_list.npy'), template_file_list)
