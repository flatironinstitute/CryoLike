from cryolike.viewing_angles import ViewingAngles
import torch
import numpy as np
import os
from cryolike.util.enums import Precision
from cryolike.util.typechecks import set_precision

from cryolike.polar_grid import PolarGrid
from cryolike.cartesian_grid import CartesianGrid2D
from cryolike.volume import Volume, PhysicalVolume
from cryolike.atomic_model import AtomicModel
from cryolike.template import Templates
from cryolike.plot import plot_images, plot_power_spectrum
from cryolike.parameters import parse_parameters, save_parameters, print_parameters

def make_templates_from_inputs(
    list_of_inputs : list, # list of filenames of pdb, or mrc files, or list of numpy arrays or torch tensors
    n_voxels : int, # number of voxels in each dimension
    voxel_size : float, # in Angstrom
    resolution_factor : float = 1.0, # 1.0 for full resolution at nyquist frequency, 0.5 for half resolution at double nyquist frequency
    precision : str | Precision = 'single', # 'single' or 'double'
    viewing_distance : float | None = None, # angular distance between two viewing angles
    n_inplanes : int | None = None, # number of inplane angles
    atom_radii : float | None = None, # radius of atoms
    atom_selection : str = "name CA", # selection of atoms
    use_protein_residue_model : bool = True, # use residue model
    atom_shape : str = "gaussian", # shape of atoms, either 'hard-sphere' or 'gaussian'
    flag_plots : bool = True,
    folder_output : str | None = None,
    verbose : bool = False
):
    if folder_output is None:
        folder_output = './templates/'
    assert folder_output is not None
    os.makedirs(folder_output, exist_ok = True)
    if flag_plots:
        folder_output_plots = os.path.join(folder_output, 'plots')
        os.makedirs(folder_output_plots, exist_ok = True)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("CUDA is available, using GPU")
    else:
        print("CUDA is not available, using CPU")
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if isinstance(precision, str):
        if precision == 'single':
            precision = Precision.SINGLE
        elif precision == 'double':
            precision = Precision.DOUBLE
        else:
            raise ValueError("Unknown precision")
    (torch_float_type, _, _) = set_precision(precision, default=Precision.SINGLE)
    assert isinstance(precision, Precision)
    
    params = parse_parameters(n_voxels = n_voxels, voxel_size = voxel_size, resolution_factor = resolution_factor, precision = precision, viewing_distance = viewing_distance, n_inplanes = n_inplanes, atom_radii = atom_radii, atom_selection = atom_selection, use_protein_residue_model = use_protein_residue_model, atom_shape = atom_shape)
    save_parameters(params, os.path.join(folder_output, "parameters.npz"))
    print_parameters(params)
    
    assert isinstance(precision, Precision)
    
    polar_grid = PolarGrid(
        radius_max = params.radius_max,
        dist_radii = params.dist_radii,
        n_inplanes = params.n_inplanes,
        uniform = True
    )
    viewing_angles = ViewingAngles.from_viewing_distance(viewing_distance)
    template_file_list = []
    for i, input in enumerate(list_of_inputs):
        name = None
        if isinstance(input, str):
            if input.endswith('.mrc') or input.endswith('.mrcs') or input.endswith('.map'):
                mrc_file = input
                print("Processing %s..." % mrc_file)
                name = os.path.splitext(os.path.basename(mrc_file))[0]
                print("mrc_name:", name)
                volume = Volume.from_mrc(filename = mrc_file)
                assert volume.density_physical is not None
                volume.density_physical = volume.density_physical.to(torch_float_type).to(device)
                tp = Templates.generate_from_physical_volume(volume=volume, polar_grid=polar_grid, viewing_angles=viewing_angles, precision=params.precision, verbose=verbose)
            elif input.endswith('.pdb'):
                pdb_file = input
                print("Processing %s..." % pdb_file)
                name = os.path.splitext(os.path.basename(pdb_file))[0]
                print("pdb_name:", name)
                atomic_model = AtomicModel.read_from_pdb(pdb_file = pdb_file, atom_selection = params.atom_selection, atom_radii = params.atom_radii, box_size = params.box_size, use_protein_residue_model = use_protein_residue_model)
                tp = Templates.generate_from_positions(atomic_model, viewing_angles, polar_grid, box_size=params.box_size, atom_shape=params.atom_shape, precision=params.precision, verbose = verbose)
            else:
                raise ValueError("Unknown input format")
        elif isinstance(input, np.ndarray) or isinstance(input, torch.Tensor):
            print("Processing numpy array or torch tensor...")
            if isinstance(input, np.ndarray):
                input = torch.from_numpy(input)
            name = "tensor%d" % i
            density_physical_data = PhysicalVolume(density_physical=input, voxel_size=params.voxel_size, voxel_grid=None)
            volume = Volume(density_physical_data=density_physical_data, box_size=params.box_size)
            assert volume.density_physical is not None
            volume.density_physical = volume.density_physical.to(torch_float_type).to(device)
            tp = Templates.generate_from_physical_volume(volume, polar_grid, viewing_angles, precision=params.precision, verbose=verbose)
        else:
            raise ValueError("Unknown input format")
        tp.normalize_templates_fourier(ord=2, use_max=False)
        if flag_plots:
            assert tp.templates_fourier is not None
            plot_images(tp.templates_fourier.cpu().numpy(), polar_grid=tp.polar_grid, n_plots=16, filename=os.path.join(folder_output_plots, "templates_fourier_%s.png" % name), show=False)
            plot_power_spectrum(image_or_template=tp, filename_plot=os.path.join(folder_output_plots, "power_spectrum_%s.png" % name), show=False)
            phys_grid = CartesianGrid2D(n_pixels=params.n_voxels[:2], pixel_size=params.voxel_size[:2])
            templates_phys = tp.transform_to_spatial(phys_grid=phys_grid, n_templates_stop=16, use_cuda=use_cuda, save_to_class=False)
            plot_images(templates_phys.cpu().numpy(), phys_grid=phys_grid, n_plots=16, filename=os.path.join(folder_output_plots, "templates_phys_%s.png" % name), show=False)
        template_file = os.path.join(folder_output, "templates_fourier_%s.pt" % name)
        torch.save(tp.templates_fourier, template_file)
        
        template_file_list.append(template_file)
        np.save(os.path.join(folder_output, 'template_file_list.npy'), template_file_list)