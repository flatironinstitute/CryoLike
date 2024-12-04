from pytest import raises
from typing import Any
from unittest.mock import patch, Mock

import numpy as np
import numpy.testing as npt

from cryolike.convert_particle_stacks.particle_stacks_metadata import (
    _Metadata,
    _batchify,
)

PKG = "cryolike.convert_particle_stacks.particle_stacks_metadata"


target_vals = {
    'defocusU': 1.0,
    'defocusV': 2.0,
    'defocusAngle': 3.0,
    'sphericalAberration': 4.0,
    'voltage': 5.0,
    'amplitudeContrast': 6.0,
    'phaseShift': 7.0,
    'ctfBfactor': 8.0,
    'ctfScalefactor': 9.0,
}

target_vals_ndarray = { k: np.array([target_vals[k]]) for k in target_vals.keys() }

keys_not_enforced = ['sphericalAberration', 'voltage', 'amplitudeContrast', 'ctfBfactor', 'ctfScalefactor']


def test_metadata_init_with_floats():
    obj = _Metadata(
        defocusU = target_vals['defocusU'],
        defocusV = target_vals['defocusV'],
        defocusAngle = target_vals['defocusAngle'],
        sphericalAberration = target_vals['sphericalAberration'],
        voltage = target_vals['voltage'],
        amplitudeContrast = target_vals['amplitudeContrast'],
        phaseShift = target_vals['phaseShift'],
        ctfBfactor = target_vals['ctfBfactor'],
        ctfScalefactor = target_vals['ctfScalefactor']
    )
    assert obj.defocus_is_degree == True
    assert obj.phase_shift_is_degree == True
    for k in target_vals.keys():
        val = getattr(obj, k)
        if k not in keys_not_enforced:
            assert isinstance(val, np.ndarray)
            npt.assert_equal(val, target_vals_ndarray[k])
        else:
            assert val == target_vals[k]


def test_metadata_init_with_arrays():
    obj = _Metadata(
        defocusU = target_vals_ndarray['defocusU'],
        defocusV = target_vals_ndarray['defocusV'],
        defocusAngle = target_vals_ndarray['defocusAngle'],
        sphericalAberration = target_vals_ndarray['sphericalAberration'],
        voltage = target_vals_ndarray['voltage'],
        amplitudeContrast = target_vals_ndarray['amplitudeContrast'],
        phaseShift = target_vals_ndarray['phaseShift'],
        ctfBfactor = target_vals_ndarray['ctfBfactor'],
        ctfScalefactor = target_vals_ndarray['ctfScalefactor'],
        defocus_is_degree=False,
        phase_shift_is_degree=False
    )
    assert obj.defocus_is_degree == False
    assert obj.phase_shift_is_degree == False
    for k in target_vals_ndarray.keys():
        val = getattr(obj, k)
        npt.assert_equal(val, target_vals_ndarray[k])


mock_vals = {
    "defocusU": np.array([1., 2., 3.]),
    "defocusV": np.array([2., 3., 4.]),
    "defocusAngle": np.array([3., 4., 5.]),
    "sphericalAberration": np.array([4., 4., 4.,]),
    "voltage": np.array([5., 5., 5.,]),
    "amplitudeContrast": np.array([6., 6., 6.]),
    "phaseShift": np.array([7., 8., 9.]),
    "ctfBfactor": np.array([8., 8., 8.]),
    "ctfScalefactor": np.array([9., 10., 11.]),
    "cs_files": np.array(['a', 'b', 'c']),
    "cs_idx": np.array(['a', 'b', 'c']),
    "cs_pixel_size": np.array([2.0, 4.0])
}

mock_star_datablock = {
    "DefocusU": mock_vals["defocusU"],
    "DefocusV": mock_vals["defocusV"],
    "DefocusAngle": mock_vals["defocusAngle"],
    "SphericalAberration": mock_vals["sphericalAberration"],
    "Voltage": mock_vals["voltage"],
    "AmplitudeContrast": mock_vals["amplitudeContrast"],
    "PhaseShift": mock_vals["phaseShift"],
    "CtfBfactor": mock_vals["ctfBfactor"],
    "CtfScalefactor": mock_vals["ctfScalefactor"],
}
star_datablock_conditional_keys = ["AmplitudeContrast", "PhaseShift", "CtfBfactor", "CtfScalefactor"]


def get_mock_read_star_file(include_conditonals: bool):
    datablock = mock_star_datablock.copy()
    if not include_conditonals:
        for x in star_datablock_conditional_keys:
            datablock.pop(x, None)
    return lambda x: (datablock, None)


def compare_result_to_mock(res: _Metadata, use_starfile_defaults: bool = False, use_cs_fs: bool = False, cs_fs_with_pixels: bool = True):
    npt.assert_array_equal(res.defocusU, mock_vals["defocusU"])
    npt.assert_array_equal(res.defocusV, mock_vals["defocusV"])
    npt.assert_array_equal(res.defocusAngle, mock_vals["defocusAngle"])
    trimmed_sa = np.array([mock_vals['sphericalAberration'][0]])
    trimmed_v = np.array([mock_vals['voltage'][0]])
    npt.assert_array_equal(res.sphericalAberration, trimmed_sa)
    npt.assert_array_equal(res.voltage, trimmed_v)

    if use_starfile_defaults:
        assert res.amplitudeContrast == 0.1
        npt.assert_array_equal(res.phaseShift, np.zeros_like(res.defocusU))
        assert res.ctfBfactor == 0.0
        assert res.ctfScalefactor == 1.0
    else:
        trimmed_ac = np.array([mock_vals["amplitudeContrast"][0]])
        npt.assert_array_equal(res.amplitudeContrast, trimmed_ac)
        npt.assert_array_equal(res.phaseShift, mock_vals["phaseShift"])

    if use_cs_fs:
        assert res.cs_files is not None
        assert res.cs_idxs is not None
        npt.assert_array_equal(res.cs_files, mock_vals["cs_files"])
        npt.assert_array_equal(res.cs_idxs, mock_vals["cs_idx"])
        if not cs_fs_with_pixels:
            assert res.cs_pixel_size is None
        else:
            assert res.cs_pixel_size is not None
            npt.assert_array_equal(res.cs_pixel_size, np.array(mock_vals["cs_pixel_size"][0]))
    else:
        assert res.cs_files is None
        assert res.cs_idxs is None
        assert res.cs_pixel_size is None


@patch("builtins.print")
@patch(f"{PKG}.read_star_file")
def test_from_star_file(mock_read: Mock, mock_print: Mock):
    mock_read.side_effect = get_mock_read_star_file(include_conditonals=True)
    res = _Metadata.from_star_file('filename', defocus_is_degree=False, phase_shift_is_degree=False)
    mock_print.assert_not_called()
    compare_result_to_mock(res)

    assert res.ctfBfactor == 8.0
    assert isinstance(res.ctfScalefactor, np.ndarray)
    npt.assert_array_equal(res.ctfScalefactor, np.array([9., 10., 11.]))
    assert not res.defocus_is_degree
    assert not res.phase_shift_is_degree


@patch("builtins.print")
@patch(f"{PKG}.read_star_file")
def test_from_star_file_with_defaults(mock_read: Mock, mock_print: Mock):
    mock_read.side_effect = get_mock_read_star_file(include_conditonals=False)
    res = _Metadata.from_star_file('doesnt_matter')
    compare_result_to_mock(res, use_starfile_defaults=True)
    assert mock_print.call_count == 4


mock_cryosparc = {
    "ctf/df1_A": mock_vals["defocusU"],
    "ctf/df2_A": mock_vals["defocusV"],
    "ctf/df_angle_rad": mock_vals["defocusAngle"],
    "ctf/cs_mm": mock_vals["sphericalAberration"],
    "ctf/accel_kv": mock_vals["voltage"],
    "ctf/amp_contrast": mock_vals["amplitudeContrast"],
    "ctf/phase_shift_rad": mock_vals["phaseShift"],
    "blob/path": mock_vals["cs_files"],
    "blob/idx": mock_vals["cs_idx"],
    "blob/psize_A": mock_vals["cs_pixel_size"]
}


@patch(f"{PKG}.path")
def test_from_cryospark_file_throws_if_no_file(mock_path: Mock):
    mock_path.exists = Mock(side_effect=lambda x: False)
    with raises(ValueError, match="file not found"):
        _ = _Metadata.from_cryospark_file("doesn't matter")


@patch(f"{PKG}.np")
@patch(f"{PKG}.path")
def test_from_cryospark_file_no_fs_data(mock_path: Mock, mock_np: Mock):
    mock_path.exists = lambda x: True
    mock_np.load = Mock(side_effect=lambda x: mock_cryosparc)
    res = _Metadata.from_cryospark_file("filename", get_fs_data=False)
    compare_result_to_mock(res)


@patch(f"{PKG}.np")
@patch(f"{PKG}.path")
def test_from_cryospark_file_with_fs_data(mock_path: Mock, mock_np: Mock):
    mock_path.exists = lambda x: True
    mock_np.load = Mock(side_effect=lambda x: mock_cryosparc)
    res = _Metadata.from_cryospark_file("filename", get_fs_data=True)
    compare_result_to_mock(res, use_cs_fs=True)


@patch(f"{PKG}.np")
@patch(f"{PKG}.path")
def test_from_cryospark_file_with_fs_data_no_pixel_size(mock_path: Mock, mock_np: Mock):
    mock_path.exists = lambda x: True
    def mock_cryosparc_no_ps(_: str):
        datablock: dict[str, Any] = mock_cryosparc.copy()
        datablock["blob/psize_A"] = None
        return datablock
    mock_np.load = Mock(side_effect=mock_cryosparc_no_ps)
    res = _Metadata.from_cryospark_file("filename", get_fs_data=True)
    compare_result_to_mock(res, use_cs_fs=True, cs_fs_with_pixels=False)


def test_batchify_for_floats():
    f = 5.
    res = _batchify(f, 0, 15)
    assert res == f


def test_batchify_for_arrays():
    a = np.array(range(10), dtype=np.float32)
    res = _batchify(a, 3, 6)
    npt.assert_allclose(res, np.array(range(3, 6, 1), dtype=np.float32))


def test_take_range():
    sut = _Metadata(
        defocusU = mock_vals["defocusU"],
        defocusV = mock_vals["defocusV"],
        defocusAngle = mock_vals["defocusAngle"],
        sphericalAberration = mock_vals["sphericalAberration"],
        voltage = mock_vals["voltage"],
        amplitudeContrast = mock_vals["amplitudeContrast"][0],
        phaseShift = mock_vals["phaseShift"],
        ctfBfactor = mock_vals["ctfBfactor"],
        ctfScalefactor = mock_vals["ctfScalefactor"],
    )
    res = sut.take_range(1, 10)
    npt.assert_array_equal(res.defocusU, sut.defocusU[1:])
    assert res.amplitudeContrast == sut.amplitudeContrast # s.b. scalar


@patch(f"{PKG}.np")
@patch(f"{PKG}.read_star_file")
def test_save_params_star(mock_read: Mock, mock_np: Mock):
    mock_np.savez_compressed = Mock()
    assert isinstance(mock_np.savez_compressed, Mock)
    mock_read.side_effect = get_mock_read_star_file(include_conditonals=True)
    metadata = _Metadata.from_star_file('filename')
    mock_im = Mock()
    output_fn = "output-file"
    n_images = 10

    metadata.save_params_star(output_fn, n_images, mock_im, 0, 10)

    mock_np.savez_compressed.assert_called_once_with(
        output_fn,
        n_images=n_images,
        stack_start=0,
        stack_end=10,
        defocusU=metadata.defocusU,
        defocusV=metadata.defocusV,
        defocusAng=metadata.defocusAngle,
        sphericalAberration=metadata.sphericalAberration,
        voltage=metadata.voltage,
        amplitudeContrast=metadata.amplitudeContrast,
        ctfBfactor=metadata.ctfBfactor,
        ctfScalefactor=metadata.ctfScalefactor,
        phaseShift=metadata.phaseShift,
        box_size=mock_im.box_size,
        n_pixels=mock_im.phys_grid.n_pixels,
        pixel_size=mock_im.phys_grid.pixel_size,
        defocus_angle_is_degree=metadata.defocus_is_degree,
        phase_shift_is_degree=metadata.phase_shift_is_degree
    )


@patch(f"{PKG}.read_star_file")
def test_save_params_star_checks_required_keys(mock_read: Mock):
    mock_read.side_effect = get_mock_read_star_file(include_conditonals=True)
    metadata = _Metadata.from_star_file('filename')
    mock_im = Mock()
    output_fn = "output-file"
    n_images = 10

    ctfBfactor = metadata.ctfBfactor
    metadata.ctfBfactor = None
    with raises(ValueError, match="save_params_star function requires"):
        metadata.save_params_star(output_fn, n_images, mock_im, 0, 10)
    metadata.ctfBfactor = ctfBfactor

    metadata.ctfScalefactor = None
    with raises(ValueError, match="save_params_star function requires"):
        metadata.save_params_star(output_fn, n_images, mock_im, 0, 10)


@patch(f"{PKG}.read_star_file")
def test_save_params_star_checks_stack_length(mock_read: Mock):
    mock_read.side_effect = get_mock_read_star_file(include_conditonals=True)
    metadata = _Metadata.from_star_file('filename')
    mock_im = Mock()
    output_fn = "output-file"
    n_images = 10

    with raises(ValueError, match="non-positive"):
        metadata.save_params_star(output_fn, n_images, mock_im, 5, 5)


@patch(f"{PKG}.np")
@patch(f"{PKG}.read_star_file")
def test_save_params(mock_read: Mock, mock_np: Mock):
    mock_np.savez_compressed = Mock()
    assert isinstance(mock_np.savez_compressed, Mock)
    mock_read.side_effect = get_mock_read_star_file(include_conditonals=True)
    metadata = _Metadata.from_star_file('filename')
    mock_im = Mock()
    output_fn = "output-file"
    n_images = 10

    metadata.save_params(output_fn, n_images, mock_im)

    mock_np.savez_compressed.assert_called_once_with(
        output_fn,
        n_images=n_images,
        defocusU=metadata.defocusU,
        defocusV=metadata.defocusV,
        defocusAng=metadata.defocusAngle,
        sphericalAberration=metadata.sphericalAberration,
        voltage=metadata.voltage,
        amplitudeContrast=metadata.amplitudeContrast,
        phaseShift=metadata.phaseShift,
        box_size=mock_im.box_size,
        n_pixels=mock_im.phys_grid.n_pixels,
        pixel_size=mock_im.phys_grid.pixel_size,
        defocus_angle_is_degree=metadata.defocus_is_degree,
        phase_shift_is_degree=metadata.phase_shift_is_degree
    )
