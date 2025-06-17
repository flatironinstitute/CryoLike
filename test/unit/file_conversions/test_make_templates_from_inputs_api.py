from unittest.mock import patch, Mock
from pytest import raises, mark
import os
import torch
import numpy as np
from torch.testing import assert_close
from pathlib import Path

from cryolike.util import Precision, InputFileType, get_device
from cryolike.file_conversions.make_templates_from_inputs_api import (
    _make_plotter_fn,
    _make_templates_from_mrc_file,
    _make_templates_from_pdb_file,
    _make_templates_from_memory_array,
    _make_raw_template,
    _make_template_maker_fn,
    make_templates_from_inputs,
)

PKG = "cryolike.file_conversions.make_templates_from_inputs_api"


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


@patch(f"{PKG}.plot_power_spectrum")
@patch(f"{PKG}.plot_images")
def test_make_plotter_fn(plot_images: Mock, plot_pspec: Mock):
    plotter = _make_plotter_fn(True, _get_mock_img_desc())
    tp = Mock()
    tp.images_fourier = torch.tensor([])
    tp.has_fourier_images = lambda: len(tp.images_fourier) > 0

    # if there are no fourier templates, this should be a no-op
    plotter(tp, Mock())
    plot_images.assert_not_called()

    # This time we should see some activity
    tp.images_fourier = [1, 2, 3]
    plotter(tp, Mock())
    assert plot_images.call_count == 2
    plot_pspec.assert_called_once()
    n_taken = tp.transform_to_spatial.call_args[1]['max_to_transform']
    assert plot_images.call_args[1]['n_plots'] == n_taken


@patch(f"{PKG}.plot_images")
def test_make_plotter_fn_no_op_when_no_output_dir(plot: Mock):
    plot.side_effect = ValueError("If this function is called, the test failed.")
    plotter = _make_plotter_fn(False, _get_mock_img_desc())
    plotter(Mock(), Mock())


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
@mark.parametrize("centering", [True, False])
def test_make_templates_from_pdb_file(atomicModel: Mock, templates: Mock, centering: bool):
    descriptor = _get_mock_img_desc()
    expected_edge_length = descriptor.cartesian_grid.box_size[0]
    expected_filename = 'file.pdb'
    verbose = True
    _atomic_model = Mock()
    atomicModel.read_from_traj = Mock(return_value = _atomic_model)

    _ = _make_templates_from_pdb_file(expected_filename, descriptor, centering, DEV, verbose)

    atomicModel.read_from_traj.assert_called_once_with(
        top_file = expected_filename,
        atom_selection = descriptor.atom_selection,
        atom_radii = descriptor.atom_radii,
        box_size = expected_edge_length,
        centering = centering,
        use_protein_residue_model = descriptor.use_protein_residue_model
    )

    templates.generate_from_positions.assert_called_once_with(
        atomic_model = _atomic_model,
        viewing_angles = descriptor.viewing_angles,
        polar_grid = descriptor.polar_grid,
        box_size = descriptor.cartesian_grid.box_size,
        atom_shape = descriptor.atom_shape,
        compute_device = DEV,
        output_device = get_device("cpu"),
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
    mrc_input = (Path("mrc1.mrc"), "foo", InputFileType.MRC)
    pdb_input = (Path("pdb1.pdb"), "foo", InputFileType.PDB)
    numpy_input = np.arange(3)
    torch_input = torch.arange(6)
    array_inputs = [(numpy_input, 'foo', InputFileType.MEM), 
                    (torch_input, 'foo', InputFileType.MEM)]
    desc = _get_mock_img_desc()
    t_float = FLOAT_T
    dev = DEV
    verbose = True

    _make_raw_template(mrc_input, desc, t_float, dev, verbose)
    assert _mrc.call_count == 1

    _make_raw_template(pdb_input, desc, t_float, dev, verbose)
    assert _pdb.call_count == 1

    for x in array_inputs:
        _make_raw_template(x, desc, t_float, dev, verbose)
    assert _mem.call_count == len(array_inputs)


def test_make_raw_template_throws_on_bad_type():
    input = ('foo', 'bar', -1)
    with raises(ValueError, match="Unknown input format"):
        _make_raw_template(input, Mock(), Mock(), Mock(), False) # type: ignore


@patch(f"{PKG}._make_raw_template")
def test_make_template_maker(_make_raw: Mock):
    _tp = Mock()
    _make_raw.return_value = _tp
    
    desc = _get_mock_img_desc()
    t_float = torch.float64
    device = torch.device('cpu')
    verbose = True
    input = ('foo', 'bar', 'baz')

    sut = _make_template_maker_fn(desc, t_float, device, verbose)
    returned_template = sut(input) # type: ignore
    _make_raw.assert_called_once_with(input, desc, t_float, device, verbose)
    assert returned_template == _tp
    _tp.normalize_images_fourier.assert_called_once()


@patch(f"{PKG}.TemplateFileManager")
@patch(f"{PKG}._make_plotter_fn")
@patch(f"{PKG}._make_template_maker_fn")
def test_make_templates_from_inputs(make_reader: Mock, make_plotter: Mock, mgr_ctor: Mock):
    desc = _get_mock_img_desc()
    output_dir = 'my-plots-dir'
    plotter_fn = Mock()
    make_plotter.return_value = plotter_fn
    reader_fn = Mock()
    make_reader.return_value = reader_fn
    inputs = ['f1', 'f2']

    filemgr = Mock()
    filemgr.inputs_include_pdb_files = Mock(return_value=False)
    mgr_ctor.return_value = filemgr

    with patch(f"{PKG}.ImageDescriptor.load") as load:
        load.return_value = desc
        make_templates_from_inputs(inputs, 'params', False, output_dir)
    mgr_ctor.assert_called_once()
    calls = mgr_ctor.call_args[0]
    assert calls[0] == output_dir
    assert calls[1] == False
    assert calls[2] == inputs
    assert calls[3] == reader_fn
    assert calls[4] == plotter_fn
    
    filemgr.process_inputs.assert_called_once()
    filemgr.save_file_list.assert_called_once()


@patch(f"{PKG}.TemplateFileManager")
def test_make_templates_from_inputs_throws_on_unprocessable_pdb(mgr_ctor: Mock):
    inputs = ['some_pdb_file.pdb']
    mgr = Mock()
    mgr.inputs_include_pdb_files = Mock(return_value=True)
    mgr_ctor.return_value = mgr

    descriptor = _get_mock_img_desc()
    descriptor.is_compatible_with_pdb = Mock(return_value=False)

    with patch(f"{PKG}.ImageDescriptor.load") as load:
        load.return_value = descriptor
        with raises(ValueError, match="To process PDB files"):
            make_templates_from_inputs(inputs, 'some filename')


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

