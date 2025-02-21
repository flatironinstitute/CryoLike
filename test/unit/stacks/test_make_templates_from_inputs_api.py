from unittest.mock import patch, Mock
from pytest import raises, mark
import os
import torch
import numpy as np
from torch.testing import assert_close

from cryolike.util import Precision
from cryolike.stacks.make_templates_from_inputs_api import (
    _make_plotter_fn,
    _get_input_name,
    _make_templates_from_mrc_file,
    _make_templates_from_pdb_file,
    _make_templates_from_memory_array,
    _make_raw_template,
    _get_template_output_filename,
    make_templates_from_inputs,
    _set_up_directories,
)

PKG = "cryolike.stacks.make_templates_from_inputs_api"


def _get_mock_img_desc():
    img_desc = Mock()
    img_desc.cartesian_grid = Mock()
    img_desc.cartesian_grid.box_size = np.array([3., 3.,])
    img_desc.cartesian_grid.pixel_size = np.array([2., 2.])
    img_desc.box_size_return = np.ones((3,)) * img_desc.cartesian_grid.box_size[0]
    img_desc.get_3d_box_size = Mock(return_value=img_desc.box_size_return)
    img_desc.polar_grid = Mock()
    img_desc.viewing_angles = Mock()
    img_desc.atom_radii = 2.0
    img_desc.atom_selection = "default_atoms"
    img_desc.atom_shape = "gaussian"
    img_desc.precision = Precision.DOUBLE
    img_desc.use_protein_residue_model = True

    return img_desc


FLOAT_T = torch.float64
DEV = torch.device("cpu")


@mark.parametrize("with_plots", (True, False))
def test_set_up_directories(tmp_path, with_plots):
    out_dir = tmp_path / "my" / "output" / "directory"
    plot_dir = out_dir / "plots"

    res = _set_up_directories(out_dir, with_plots)
    assert os.path.exists(out_dir)
    if with_plots:
        assert os.path.exists(plot_dir)
        assert res == str(plot_dir)
    else:
        assert not os.path.exists(plot_dir)
        assert res is None


@patch(f"{PKG}.plot_power_spectrum")
@patch(f"{PKG}.plot_images")
def test_make_plotter_fn(plot_images: Mock, plot_pspec: Mock):
    plotter = _make_plotter_fn("output_dir")
    img_desc = Mock()
    img_desc.cartesian_grid = Mock()
    name = 'plotName'

    tp = Mock()
    tp.images_fourier = torch.tensor([])
    tp.has_fourier_images = lambda: len(tp.images_fourier) > 0

    # if there are no fourier templates, this should be a no-op
    plotter(tp, img_desc, name)
    plot_images.assert_not_called()

    # This time we should see some activity
    tp.images_fourier = [1, 2, 3]
    plotter(tp, img_desc, name)
    assert plot_images.call_count == 2
    plot_pspec.assert_called_once()
    n_taken = tp.transform_to_spatial.call_args[1]['max_to_transform']
    assert plot_images.call_args[1]['n_plots'] == n_taken
    assert plot_pspec.call_args[1]['filename_plot'] == os.path.join("output_dir", f"power_spectrum_{name}.png")


@patch(f"{PKG}.plot_images")
def test_make_plotter_fn_no_op_when_no_output_dir(plot: Mock):
    plot.side_effect = ValueError("If this function is called, the test failed.")
    plotter = _make_plotter_fn(None)
    plotter(Mock(), Mock(), "")


@patch("builtins.print")
def test_get_input_name(_print: Mock):
    file = "myfile.pdb"
    (name, ext) = _get_input_name(file, 0)
    assert name == "myfile"
    assert ext == ".pdb"
    _print.assert_called_once_with("Processing myfile.pdb...")
    _print.reset_mock()

    data = [1, 2, 3]
    (name, ext) = _get_input_name(data, 4) # type: ignore
    assert name == "tensor_4"
    assert ext == ''
    _print.assert_called_once()


## NOTE: The following tests to the individual make_templates functions
# are UNIT tests for this file; they DO NOT TEST the underlying calls to
# the Volume and Templates classes, which are supposed to be tested separately.
# The tests here only confirm that the parameters have all been dispatched
# correctly.
# i.e. these tests don't show there won't be bugs, only that when they are,
# the problem is probably somewhere else.


@patch(f"{PKG}.Templates")
@patch(f"{PKG}.Volume")
def test_make_templates_from_mrc_file(vol: Mock, templates: Mock):
    descriptor = _get_mock_img_desc()
    volume = Mock()
    vol.from_mrc = Mock(return_value=volume)
    tp = Mock()
    templates.generate_from_physical_volume = Mock(return_value=tp)

    filename = "my_file.mrc"

    # Catch the case where a physical density didn't get populated
    volume.density_physical = None
    with raises(ValueError, match=f"{filename} did not generate a physical density"):
        _ = _make_templates_from_mrc_file(filename, descriptor, FLOAT_T, DEV, True)
    vol.from_mrc.reset_mock()

    volume.density_physical = torch.arange(4)   # integer dtype
    res = _make_templates_from_mrc_file(filename, descriptor, FLOAT_T, DEV, True)
    assert res == tp
    vol.from_mrc.assert_called_once()
    assert volume.density_physical.dtype == FLOAT_T
    assert volume.density_physical.device == DEV
    templates.generate_from_physical_volume.assert_called_once_with(
        volume=volume,
        polar_grid=descriptor.polar_grid,
        viewing_angles=descriptor.viewing_angles,
        precision=descriptor.precision,
        verbose=True
    )


@patch(f"{PKG}.Templates")
@patch(f"{PKG}.AtomicModel")
def test_make_templates_from_pdb_file(atomicModel: Mock, templates: Mock):
    descriptor = _get_mock_img_desc()
    expected_edge_length = descriptor.cartesian_grid.box_size[0]
    expected_filename = 'file.pdb'
    verbose = True
    _atomic_model = Mock()
    atomicModel.read_from_pdb = Mock(return_value = _atomic_model)

    _ = _make_templates_from_pdb_file(expected_filename, descriptor, verbose)

    atomicModel.read_from_pdb.assert_called_once_with(
        pdb_file = expected_filename,
        atom_selection = descriptor.atom_selection,
        atom_radii = descriptor.atom_radii,
        box_size = expected_edge_length,
        use_protein_residue_model = descriptor.use_protein_residue_model
    )
    templates.generate_from_positions.assert_called_once_with(
        atomic_model = _atomic_model,
        viewing_angles = descriptor.viewing_angles,
        polar_grid = descriptor.polar_grid,
        box_size = descriptor.cartesian_grid.box_size,
        atom_shape = descriptor.atom_shape,
        precision = descriptor.precision,
        verbose = verbose
    )


def test_make_templates_from_pdb_file_throws_on_incompatible():
    descriptor = _get_mock_img_desc()
    descriptor.is_compatible_with_pdb = Mock(return_value=False)
    with raises(ValueError, match="is not set"):
        _ = _make_templates_from_pdb_file('filename', descriptor, False)


def test_make_templates_from_pdb_file_throws_on_non_cubic_box_size():
    descriptor = _get_mock_img_desc()
    descriptor.cartesian_grid.box_size = np.array([1., 2., 3.])
    with raises(ValueError, match="non-square box size"):
        _ = _make_templates_from_pdb_file('filename', descriptor, False)


@patch(f"{PKG}.Templates.generate_from_physical_volume")
@patch(f"{PKG}.Volume")
@patch(f"{PKG}.PhysicalVolume")
def test_make_templates_from_memory_array_with_lost_physical_volume(physvol: Mock, vol: Mock, generate: Mock):
    density_physical_data = Mock()
    volume_returned = Mock()
    # Note: we could easily use the actual implementation for this
    # However, we wouldn't be able to *assert* anything on the result
    # So better not to imply that we have test coverage of it
    physvol.return_value = density_physical_data
    vol.return_value = volume_returned

    descriptor = _get_mock_img_desc()
    input = np.arange(5)
    verbose = False

    # test the unreachable case where physical density is none
    volume_returned.density_physical = None
    with raises(ValueError, match="did not preserve physical"):
        # converting this to numpy to imply coverage for the silent "else" branch
        _ = _make_templates_from_memory_array(torch.from_numpy(input), descriptor, FLOAT_T, DEV, verbose)


@patch(f"{PKG}.Templates.generate_from_physical_volume")
@patch(f"{PKG}.Volume")
@patch(f"{PKG}.PhysicalVolume")
def test_make_templates_from_memory_array(physvol: Mock, vol: Mock, generate: Mock):
    density_physical_data = Mock()
    volume_returned = Mock()
    physvol.return_value = density_physical_data
    vol.return_value = volume_returned

    descriptor = _get_mock_img_desc()
    input = np.arange(5)
    verbose = False

    density_to_use = torch.tensor([4, 6, 2])
    volume_returned.density_physical = density_to_use.clone()

    _ = _make_templates_from_memory_array(input, descriptor, FLOAT_T, DEV, verbose)
    physvol.assert_called_once()
    used_input = physvol.call_args[1]['density_physical']
    assert_close(used_input, torch.from_numpy(input))
    assert isinstance(used_input, torch.Tensor)

    vol.assert_called_once_with(
        density_physical_data=density_physical_data,
        box_size=descriptor.box_size_return
    )
    generate.assert_called_once()
    used_volume = generate.call_args[0][0]
    assert_close(used_volume.density_physical, density_to_use.to(torch.float64))
    assert used_volume.density_physical.dtype == FLOAT_T
    assert used_volume.density_physical.device == DEV


@patch(f"{PKG}._make_templates_from_memory_array")
@patch(f"{PKG}._make_templates_from_pdb_file")
@patch(f"{PKG}._make_templates_from_mrc_file")
@patch("builtins.print")
def test_make_raw_template(_print: Mock, _mrc: Mock, _pdb: Mock, _mem: Mock):
    mrc_inputs = ["mrc1.mrc", "mrc2.mrcs", "mrc3.map"]
    pdb_inputs = ["pdb.pdb"]
    numpy_input = np.arange(3)
    torch_input = torch.arange(6)
    array_inputs = [numpy_input, torch_input]
    inputs = []
    [inputs.extend(x) for x in [mrc_inputs, pdb_inputs, array_inputs]]
    desc = _get_mock_img_desc()
    t_float = FLOAT_T
    dev = DEV
    verbose = True

    for i, input in enumerate(inputs):
        exp_name = os.path.splitext(input)[0] if isinstance(input, str) else f"tensor_{i}"
        (tp, name) = _make_raw_template(input, i, desc, t_float, dev, verbose)
        assert name == exp_name

    assert _mrc.call_count == len(mrc_inputs)
    assert _pdb.call_count == len(pdb_inputs)
    assert _mem.call_count == len(array_inputs)
    # one for the stated output filename of each input file, plus one for each input
    assert _print.call_count == len(mrc_inputs) + len(pdb_inputs) + len(inputs)


@mark.parametrize("existing_file_count", [0, 3])
def test_get_template_output_filename(existing_file_count: int):
    folder_out = "myfolder/"
    name = "outname"
    file_hits = [True] * existing_file_count
    file_hits.append(False)
    expected_prefix = f"templates_fourier_{name}"
    expected_ordinal = f"_{existing_file_count}" if existing_file_count > 0 else ""
    expected_name = f"{expected_prefix}{expected_ordinal}.pt"
    expected_result = os.path.join(folder_out, expected_name)

    with patch(f"{PKG}.os.path.exists") as exists:
        exists.side_effect = file_hits
        result = _get_template_output_filename(folder_out, name)
        assert result == expected_result


@patch(f"{PKG}.np.save")
@patch(f"{PKG}.torch.save")
@patch(f"{PKG}._make_plotter_fn")
@patch(f"{PKG}._set_up_directories")
def test_make_templates_from_inputs(setup_dirs: Mock, make_plotter: Mock, t_save: Mock, np_save: Mock):
    desc = _get_mock_img_desc()
    plots_dir = 'my-plots-dir'
    setup_dirs.side_effect = lambda x, y: None if not y else plots_dir
    plotter_fn = Mock()
    make_plotter.return_value = plotter_fn
    inputs = ['f1', 'f2']

    with patch(f"{PKG}.ImageDescriptor.load") as load:
        load.return_value = desc
        with patch(f"{PKG}._make_raw_template") as make_tp:
            mock_templates = Mock()
            make_tp.side_effect = lambda input, a, b, c, d, e: (mock_templates, input)

            make_templates_from_inputs(inputs, 'params')

            n_iter = len(inputs)
            assert t_save.call_count == n_iter
            assert isinstance(mock_templates.normalize_images_fourier, Mock )
            mock_templates.normalize_images_fourier.assert_called_with(ord=2, use_max=False)
            assert mock_templates.normalize_images_fourier.call_count == 2
            assert np_save.call_count == 1


@patch(f"{PKG}.check_cuda")
def test_make_templates_from_inputs_no_op_on_empty_input_list(check_cuda: Mock):
    inputs = []
    make_templates_from_inputs(inputs, 'some filename')
    check_cuda.assert_not_called()


def test_make_templates_from_inputs_throws_on_unprocessable_pdb():
    inputs = ['some_pdb_file.pdb']
    descriptor = _get_mock_img_desc()
    descriptor.is_compatible_with_pdb = Mock(return_value=False)

    with patch(f"{PKG}.ImageDescriptor.load") as load:
        load.return_value = descriptor
        with raises(ValueError, match="To process PDB files"):
            make_templates_from_inputs(inputs, 'some filename')


@mark.parametrize("inputlist", [["file.txt"], [-6.]])
@patch(f"{PKG}._set_up_directories")
def test_make_templates_from_inputs_throws_on_unknown_type(setup_dirs, inputlist):
    descriptor = _get_mock_img_desc()
    with patch(f"{PKG}.ImageDescriptor.load") as load:
        load.return_value = descriptor
        with raises(ValueError, match="Unknown input format"):
            make_templates_from_inputs(list(inputlist), 'descriptor fn', output_plots=False)
        load.assert_called_once_with('descriptor fn')


@patch(f"{PKG}.ImageDescriptor")
def test_make_templates_from_inputs_no_ops_on_empty_input_list(img_desc: Mock):
    descriptor = _get_mock_img_desc()
    img_desc.load = Mock(return_value=descriptor)

    no_op_result = make_templates_from_inputs(
        list_of_inputs=[],
        image_parameters_file="filename.npz"
    )
    img_desc.load.assert_not_called()
    assert no_op_result is None


@patch(f"{PKG}.ImageDescriptor")
def test_make_templates_from_inputs_throws_on_bad_input_types(img_desc: Mock):
    bad_file = "input.xml"
    descriptor = _get_mock_img_desc()
    img_desc.load = Mock(return_value=descriptor)

    with raises(ValueError, match="Unknown input format"):
        _ = make_templates_from_inputs(
            list_of_inputs=[bad_file],
            image_parameters_file="filename.npz"
        )
    
    obviously_not_data = 5.0
    with raises(ValueError, match="Unknown input format"):
        _ = make_templates_from_inputs(
            list_of_inputs=[obviously_not_data], # type: ignore
            image_parameters_file="filename.npz"
    )

