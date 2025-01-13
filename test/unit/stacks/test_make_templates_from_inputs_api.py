from unittest.mock import patch, MagicMock, Mock
from pytest import raises
import os
import torch
import numpy as np
import numpy.testing as npt
from torch.testing import assert_close

from cryolike.util import Precision
from cryolike.stacks.make_templates_from_inputs_api import (
    _make_plotter_fn,
    _make_templates_config,
    _get_input_name,
    _make_templates_from_mrc_file,
    _make_templates_from_pdb_file,
    _make_templates_from_memory_array,
    make_templates_from_inputs
)

PKG = "cryolike.stacks.make_templates_from_inputs_api"

# Make a default TemplateConfig

def test_set_up_directories():
    # Imports local to this function! Oh my!
    # They're an ugly workaround to patch the module-level 'import os'
    # in make_templates_from_imports.
    # There may be a better solution for this issue.
    import os as local_os
    out_dir = local_os.path.join("my", "output", "directory")
    plot_dir = local_os.path.join(out_dir, "plots")
    mock_mkdirs = Mock()
    local_os.makedirs = mock_mkdirs
    from cryolike.stacks.make_templates_from_inputs_api import _set_up_directories

    _set_up_directories(out_dir, False)
    mock_mkdirs.assert_called_once_with(out_dir, exist_ok = True)
    mock_mkdirs.reset_mock()

    _set_up_directories(out_dir, True)
    assert mock_mkdirs.call_count == 2
    mock_mkdirs.assert_called_with(plot_dir, exist_ok = True)


@patch(f"{PKG}.CartesianGrid2D")
@patch(f"{PKG}.plot_power_spectrum")
@patch(f"{PKG}.plot_images")
def test_make_plotter_fn(mock_plot_images: Mock, mock_plot_pspec: Mock, mock_cart_grid: Mock):
    plotter = _make_plotter_fn("output_dir")
    params = Mock()
    params.n_voxels = [0, 1, 2]
    params.voxel_size = [0., 1., 2.]
    name = 'plotName'

    mock_tp = Mock()
    mock_tp.templates_fourier = None
    
    # if there are no fourier templates, this should be a no-op
    plotter(mock_tp, params, name)
    mock_plot_images.assert_not_called()
    mock_tp.templates_fourier = []
    plotter(mock_tp, params, name)
    mock_plot_images.assert_not_called()

    # This time we should see some activity
    mock_tp.templates_fourier = [1, 2, 3]
    plotter(mock_tp, params, name)
    assert mock_plot_images.call_count == 2
    mock_plot_pspec.assert_called_once()
    n_taken = mock_tp.transform_to_spatial.call_args[1]['n_templates_stop']
    assert mock_plot_images.call_args[1]['n_plots'] == n_taken
    assert mock_plot_pspec.call_args[1]['filename_plot'] == os.path.join("output_dir", f"power_spectrum_{name}.png")


@patch(f"{PKG}.plot_images")
def test_make_plotter_fn_no_op_when_no_output_dir(mock_plot: Mock):
    mock_plot.side_effect = ValueError("If this function is called, the test failed.")
    plotter = _make_plotter_fn(None)
    plotter(Mock(), Mock(), "")


@patch(f"{PKG}.ViewingAngles")
@patch(f"{PKG}.PolarGrid")
@patch(f"{PKG}.print_parameters")
@patch(f"{PKG}.save_parameters")
@patch(f"{PKG}.parse_parameters")
@patch(f"{PKG}._set_up_directories")
def test_make_templates_config(
    mock_dirs: Mock,
    mock_parse: Mock,
    mock_save: Mock,
    mock_print: Mock,
    mock_polarGrid: MagicMock,
    mock_viewingAngles: MagicMock
):
    mock_dirs.return_value = None
    mocked_parsed = Mock()
    mock_parse.return_value = mocked_parsed
    
    ret = _make_templates_config(
        n_voxels=3,
        voxel_size=3.0,
        viewing_distance=0.5,
        resolution_factor=1.0,
        precision=Precision.DOUBLE,
        n_inplanes=15,
        atom_radii=2.5,
        atom_selection="my-atoms",
        use_protein_residue_model=True,
        atom_shape='hard-sphere',
        output_plots=False,
        folder_output='my/folder/'
    )

    mock_dirs.assert_called_once_with('my/folder/', False)
    mock_parse.assert_called_once_with(
        n_voxels = 3,
        voxel_size = 3.0,
        resolution_factor = 1.0,
        precision = Precision.DOUBLE,
        viewing_distance = 0.5,
        n_inplanes = 15,
        atom_radii = 2.5,
        atom_selection = "my-atoms",
        use_protein_residue_model = True,
        atom_shape = 'hard-sphere'
    )
    # TOOO: Check if this breaks on non-nix filesystems
    mock_save.assert_called_once_with(mocked_parsed, "my/folder/parameters.npz")
    mock_print.assert_called_once()


@patch("builtins.print")
def test_get_input_name(mock_print: Mock):
    file = "myfile.pdb"
    (name, ext) = _get_input_name(file, 0)
    assert name == "myfile"
    assert ext == ".pdb"
    mock_print.assert_called_once_with("Processing myfile.pdb...")
    mock_print.reset_mock()

    data = [1, 2, 3]
    (name, ext) = _get_input_name(data, 4) # type: ignore
    assert name == "tensor_4"
    assert ext == ''
    mock_print.assert_called_once()


## NOTE: The following tests to the individual make_templates functions
# are UNIT tests for this file; they DO NOT TEST the underlying calls to
# the Volume and Templates classes, which are supposed to be tested separately.
# The tests here only confirm that the parameters have all been dispatched
# correctly.
# i.e. these tests don't show there won't be bugs, only that when they are,
# the problem is probably somewhere else.

def _get_mock_config():
    mock_cfg = Mock()
    mock_cfg.device = "cpu"
    mock_cfg.plotter_fn = Mock()
    mock_cfg.torch_float_type = torch.float64
    mock_cfg.polar_grid = Mock()
    mock_cfg.viewing_angles = Mock()

    params = Mock()
    params.atom_radii = 2.0
    params.atom_selection = "default_atoms"
    params.atom_shape = "gaussian"
    params.box_size = np.array([3., 3., 3.])
    params.precision = Precision.DOUBLE
    params.voxel_size = np.array([1., 1., 1.])
    mock_cfg.params = params

    return mock_cfg


@patch(f"{PKG}.Templates")
@patch(f"{PKG}.Volume")
def test_make_templates_from_mrc_file(mock_vol: Mock, mock_templates: Mock):
    cfg = _get_mock_config()
    volume = Mock()
    mock_vol.from_mrc = Mock(return_value=volume)
    mock_template_obj = Mock()
    mock_templates.generate_from_physical_volume = Mock(return_value=mock_template_obj)

    filename = "my_file.mrc"

    # Catch the case where a physical density didn't get populated
    volume.density_physical = None
    with raises(ValueError, match=f"{filename} did not generate a physical density"):
        _ = _make_templates_from_mrc_file(filename, cfg, True)
    mock_vol.from_mrc.reset_mock()

    volume.density_physical = torch.arange(4)   # integer dtype
    res = _make_templates_from_mrc_file(filename, cfg, True)
    assert res == mock_template_obj
    mock_vol.from_mrc.assert_called_once()
    assert volume.density_physical.dtype == cfg.torch_float_type
    assert volume.density_physical.device == torch.device(cfg.device)
    mock_templates.generate_from_physical_volume.assert_called_once_with(
        volume=volume,
        polar_grid=cfg.polar_grid,
        viewing_angles=cfg.viewing_angles,
        precision=cfg.params.precision,
        verbose=True
    )


@patch(f"{PKG}.Templates")
@patch(f"{PKG}.AtomicModel")
def test_make_templates_from_pdb_file(mock_atomic: Mock, mock_templates: Mock):
    cfg = _get_mock_config()
    expected_edge_length = cfg.params.box_size[0]
    expected_filename = 'file.pdb'
    use_res_model = False
    verbose = True
    mock_atomic_model = Mock()
    mock_atomic.read_from_pdb = Mock(return_value = mock_atomic_model)

    res = _make_templates_from_pdb_file(expected_filename, cfg, use_res_model, verbose)

    mock_atomic.read_from_pdb.assert_called_once_with(
        pdb_file = expected_filename,
        atom_selection = cfg.params.atom_selection,
        atom_radii = cfg.params.atom_radii,
        box_size = expected_edge_length,
        use_protein_residue_model = use_res_model
    )
    mock_templates.generate_from_positions.assert_called_once_with(
        atomic_model = mock_atomic_model,
        viewing_angles = cfg.viewing_angles,
        polar_grid = cfg.polar_grid,
        box_size = cfg.params.box_size,
        atom_shape = cfg.params.atom_shape,
        precision = cfg.params.precision,
        verbose = verbose
    )


def test_make_templates_from_pdb_file_throws_on_no_atom_radii():
    cfg = _get_mock_config()
    cfg.params.atom_radii = None
    with raises(ValueError, match="atom_radii parameter was not set"):
        _ = _make_templates_from_pdb_file('filename', cfg, False, False)


def test_make_templates_from_pdb_file_throws_on_non_cubic_box_size():
    cfg = _get_mock_config()
    cfg.params.box_size = np.array([1., 2., 3.])
    with raises(ValueError, match="non-square box size"):
        _ = _make_templates_from_pdb_file('filename', cfg, False, False)


@patch(f"{PKG}.Templates.generate_from_physical_volume")
@patch(f"{PKG}.Volume")
@patch(f"{PKG}.PhysicalVolume")
def test_make_templates_from_memory_array(mock_physvol: Mock, mock_vol: Mock, mock_gen: Mock):
    mock_density_physical_data = Mock()
    mock_volume_returned = Mock()
    # Note: we could easily use the actual implementation for this
    # However, we wouldn't be able to *assert* anything on the result
    # So better not to imply that we have test coverage of it
    mock_physvol.return_value = mock_density_physical_data
    mock_vol.return_value = mock_volume_returned

    cfg = _get_mock_config()
    input = np.arange(5)
    verbose = False

    # test the unreachable case where physical density is none
    mock_volume_returned.density_physical = None
    with raises(ValueError, match="did not preserve physical"):
        # converting this to numpy to imply coverage for the silent "else" branch
        _ = _make_templates_from_memory_array(torch.from_numpy(input), cfg, verbose)

    mock_physvol.reset_mock()
    mock_vol.reset_mock()
    density_to_use = torch.tensor([4, 6, 2])
    mock_volume_returned.density_physical = density_to_use.clone()

    res = _make_templates_from_memory_array(input, cfg, verbose)
    mock_physvol.assert_called_once()
    used_input = mock_physvol.call_args[1]['density_physical']
    assert_close(used_input, torch.from_numpy(input))
    assert isinstance(used_input, torch.Tensor)

    mock_vol.assert_called_once_with(
        density_physical_data=mock_density_physical_data,
        box_size=cfg.params.box_size
    )
    mock_gen.assert_called_once()
    used_volume = mock_gen.call_args[0][0]
    assert_close(used_volume.density_physical, density_to_use.to(torch.float64))
    assert used_volume.density_physical.dtype == cfg.torch_float_type
    assert used_volume.density_physical.device == torch.device(cfg.device)


@patch(f"{PKG}.np.save")
@patch(f"{PKG}.torch.save")
@patch(f"{PKG}._make_templates_from_memory_array")
@patch(f"{PKG}._make_templates_from_pdb_file")
@patch(f"{PKG}._make_templates_from_mrc_file")
@patch(f"{PKG}._make_templates_config")
@patch(f"{PKG}.os.path.exists")
@patch("builtins.print")
def test_make_templates_from_inputs(
    mock_print: Mock,
    mock_exists: Mock,
    mock_get_cfg: Mock,
    mock_mrc: Mock,
    mock_pdb: Mock,
    mock_mem: Mock,
    mock_torch_save: Mock,
    mock_np_save: Mock
):
    mock_template = Mock()
    mock_template.normalize_templates_fourier = Mock()
    assert isinstance(mock_template.normalize_templates_fourier, Mock)
    mock_mrc.return_value = mock_template
    mock_pdb.return_value = mock_template
    mock_mem.return_value = mock_template

    cfg = _get_mock_config()
    mock_get_cfg.return_value = cfg


    mrc_inputs = ["mrc1.mrc", "mrc2.mrcs", "mrc3.map"]
    pdb_inputs = ["pdb.pdb"]
    numpy_input = np.arange(3)
    torch_input = torch.arange(6)
    array_inputs = [numpy_input, torch_input]
    inputs = []
    inputs.extend(mrc_inputs)
    inputs.extend(pdb_inputs)
    inputs.extend(array_inputs)

    output_folder = "my/templates"
    expected_names = [
        "templates_fourier_mrc1.pt",
        "templates_fourier_mrc2.pt",
        "templates_fourier_mrc3.pt",
        "templates_fourier_pdb_3.pt",
        "templates_fourier_tensor_4.pt",
        "templates_fourier_tensor_5.pt"
    ]
    fake_existing_basenames = ["pdb", "pdb_1", "pdb_2"]
    fake_existing_files = [os.path.join(output_folder, f"templates_fourier_{x}.pt") for x in fake_existing_basenames]
    mock_exists.side_effect = lambda x: x in fake_existing_files

    kwargs = {
        'n_voxels': -1,
        'voxel_size': -1.,
        'viewing_distance': -1.,
        'resolution_factor': -1.,
        'precision': Precision.DOUBLE,
        'n_inplanes': -5,
        'atom_radii': -1.,
        'atom_selection': "my-atoms",
        'use_protein_residue_model': False,
        'atom_shape': 'hard-sphere',
        'output_plots': False,
        'folder_output': output_folder,
    }

    make_templates_from_inputs(list_of_inputs=inputs, verbose=True, **kwargs)

    # assert template creation called with all passed parameters
    mock_get_cfg.assert_called_once_with(**kwargs)

    # assert dispatch functions called right number of times with right arguments
    assert mock_mrc.call_count == len(mrc_inputs)
    inputs_seen: list[str] = [call_args[0][0] for call_args in mock_mrc.call_args_list]
    for i in range(len(inputs_seen)):
        assert inputs_seen[i] == mrc_inputs[i]

    assert mock_pdb.call_count == len(pdb_inputs)
    inputs_seen: list[str] = [call_args[0][0] for call_args in mock_pdb.call_args_list]
    for i in range(len(inputs_seen)):
        assert inputs_seen[i] == pdb_inputs[i]

    assert mock_mem.call_count == len(array_inputs)
    npt.assert_allclose(mock_mem.call_args_list[0][0][0], array_inputs[0])
    assert_close(mock_mem.call_args_list[1][0][0], array_inputs[1])

    # assert normalization carried out on every pass
    assert mock_template.normalize_templates_fourier.call_count == len(inputs)

    # assert plotting carried out on each pass
    assert cfg.plotter_fn.call_count == len(inputs)
    
    # assert each template saved
    assert mock_torch_save.call_count == len(inputs)
    save_paths: list[str] = [ca[0][1] for ca in mock_torch_save.call_args_list]
    for i in range(len(save_paths)):
        assert save_paths[i] == os.path.join(output_folder, expected_names[i])
    
    # assert full template file list returned
    mock_np_save.assert_called_once()
    (list_save_name, file_list) = mock_np_save.call_args[0]
    assert list_save_name == os.path.join(output_folder, "template_file_list.npy")
    assert len(file_list) == len(expected_names)
    for i in range(len(save_paths)):
        assert save_paths[i] == file_list[i]


@patch(f"{PKG}._make_templates_config")
def test_make_templates_from_inputs_no_ops_on_empty_input_list(mock_cfg: Mock):
    no_op_result = make_templates_from_inputs(
        list_of_inputs=[],
        n_voxels = -1,
        voxel_size=-1.,
        viewing_distance=-1.
    )
    mock_cfg.assert_not_called()
    assert no_op_result is None


@patch(f"{PKG}._make_templates_config")
def test_make_templates_from_inputs_throws_on_bad_input_types(mock_get_cfg: Mock):
    bad_file = "input.xml"
    with raises(ValueError, match="Unknown input format"):
        _ = make_templates_from_inputs(
            list_of_inputs=[bad_file],
            n_voxels=-1,
            voxel_size=-1.,
            viewing_distance=-1.
        )
    
    obviously_not_data = 5.0
    with raises(ValueError, match="Unknown input format"):
        _ = make_templates_from_inputs(
            list_of_inputs=[obviously_not_data],
            n_voxels=-1,
            voxel_size=-1.,
            viewing_distance=-1.
    )

