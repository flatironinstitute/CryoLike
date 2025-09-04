from unittest.mock import patch, Mock
from pytest import raises, mark
import numpy as np
import numpy.testing as npt

from cryolike.util import Precision, AtomShape
from cryolike.grids import CartesianGrid2D, PolarGrid
from cryolike.metadata.lens_descriptor import LensDescriptor, RELION_FIELDS, ALL_ANGLE_FIELDS

PKG = "cryolike.metadata.lens_descriptor"

defaults = {
    'defocusU': np.array([1., 2., 3.]),
    'defocusV': np.array([3., 4., 5.]),
    'defocusAngle': np.array([90., 180., 135.]),
    'phaseShift': np.array([45., 135., 60.]),
    'sphericalAberration': 3.2,
    'voltage': 320.,
    'amplitudeContrast': 0.5,
    'angleRotation': np.array([0., .5, .6]),
    'angleTilt': np.array([0., .1, .2]),
    'anglePsi': np.array([0., .3, .6]),
    'ref_pixel_size': np.array([2.5, 2.5]),
    'files': np.array(['file 1', 'file 2']),
    'idxs': np.array([[1, 4, 5], [2, 3, 7]]),
    'ctfBfactor': 12.,
    'ctfScalefactor': np.array([.5])
}


def test_ctor():
    default_res = LensDescriptor()
    for x in ['defocusU', 'defocusV', 'defocusAngle', 'phaseShift']:
        r = getattr(default_res, x)
        assert isinstance(r, np.ndarray)
        assert len(r) > 0
    for x in ['sphericalAberration', 'voltage', 'amplitudeContrast']:
        r = getattr(default_res, x)
        assert isinstance(r, float)
    for x in ['files', 'idxs', 'ref_pixel_size', 'ctfBfactor', 'ctfScalefactor', 'angleRotation', 'angleTilt', 'anglePsi']:
        r = getattr(default_res, x)
        assert r is None

    res = LensDescriptor(**defaults, defocusAngle_degree=False, phaseShift_degree=False)
    for x in defaults.keys():
        r = getattr(res, x)
        if x == 'files':
            expected = defaults[x]
            assert len(expected) == len(r)
            for i in range(len(expected)):
                assert expected[i] == r[i]
        else:
            npt.assert_allclose(r, defaults[x])
    

def test_ctor_converts_angles():
    res = LensDescriptor(
        defocusAngle=defaults['defocusAngle'],
        phaseShift=defaults['phaseShift'],
        defocusAngle_degree=True, # the default
        phaseShift_degree=True # the default
    )
    npt.assert_allclose(res.phaseShift, np.radians(defaults['phaseShift']))
    npt.assert_allclose(res.defocusAngle, np.radians(defaults['defocusAngle']))


def test_ctor_throws_on_bad_defocus_sizes():
    wrong_length = np.concatenate([defaults['defocusV'], [2., 2.]])
    with raises(ValueError, match="must have the same size"):
        _ = LensDescriptor(defocusU=defaults['defocusU'], defocusV=wrong_length)


def test_ctor_throws_on_nonunique_fields():
    nonunique = np.array([1., 2.])
    with raises(ValueError, match="multiple distinct values"):
        _ = LensDescriptor(sphericalAberration=nonunique)
    with raises(ValueError, match="multiple distinct values"):
        _ = LensDescriptor(voltage=nonunique)
    with raises(ValueError, match="multiple distinct values"):
        _ = LensDescriptor(amplitudeContrast=nonunique)


def test_batch_whole():
    sut = LensDescriptor(**defaults, defocusAngle_degree=False, phaseShift_degree=False)
    whole = sut.batch_whole()

    npt.assert_allclose(whole.defocusU, defaults['defocusU'])
    npt.assert_allclose(whole.defocusV, defaults['defocusV'])
    npt.assert_allclose(whole.defocusAngle, defaults['defocusAngle'])
    npt.assert_allclose(whole.phaseShift, defaults['phaseShift'])


def test_get_slice():
    sut = LensDescriptor(**defaults, defocusAngle_degree=False, phaseShift_degree=False)
    max_len = len(defaults['defocusU'])
    head_slice = sut.get_slice(0, 1)
    thick_slice = sut.get_slice(max_len - 1, max_len + 2)

    head_defU_len = len(head_slice.defocusU)
    thick_defU_len = len(thick_slice.defocusU)
    assert  head_defU_len == 1
    assert thick_defU_len == 1

    for field in ['defocusV', 'defocusAngle', 'phaseShift']:
        orig = getattr(sut, field)
        v = getattr(head_slice, field)
        assert len(v) == head_defU_len
        npt.assert_allclose(v, orig[0])
        v = getattr(thick_slice, field)
        assert len(v) == thick_defU_len
        npt.assert_allclose(v, orig[-1])    


def test_get_selections():
    sut = LensDescriptor(**defaults, defocusAngle_degree=False, phaseShift_degree=False)
    selections = np.array([0, 2])
    result = sut.get_selections(selections)

    for field in ['defocusU', 'defocusV', 'defocusAngle', 'phaseShift']:
        orig = getattr(sut, field)
        res = getattr(result, field)
        npt.assert_allclose(res[0], orig[0])
        npt.assert_allclose(res[-1], orig[-1])


def test_serialization_roundtrip():
    sut = LensDescriptor(**defaults)
    rt = LensDescriptor.from_dict(sut.to_dict())

    for f in defaults.keys():
        orig = getattr(sut, f)
        res = getattr(rt, f)
            # Have to special-case this b/c it's string-valued,
            # so assert_allclose fails
        if f == 'files':
            assert len(orig) == len(res)
            for i in range(len(orig)):
                assert orig[i] == res[i]
        else:
            npt.assert_allclose(orig, res)


@patch(f"{PKG}.np.load")
def test_from_cryosparc_file(load: Mock):
    putative_file = "myfile.cs"
    ret_data = {
        'ctf/df1_A': defaults['defocusU'],
        'ctf/df2_A': defaults['defocusV'],
        'ctf/df_angle_rad': defaults['defocusAngle'],
        'ctf/cs_mm': np.ones((5,)) * defaults['sphericalAberration'],
        'ctf/accel_kv': np.ones((5,)) * defaults['voltage'],
        'ctf/amp_contrast': np.ones((5,)) * defaults['amplitudeContrast'],
        'ctf/phase_shift_rad': defaults['phaseShift'],
        'blob/path': defaults['files'],
        'blob/idx': defaults['idxs'],
        'blob/psize_A': defaults['ref_pixel_size']
    }
    load.return_value = ret_data

    res = LensDescriptor.from_cryosparc_file(putative_file, True)
    load.assert_called_once_with(putative_file)
    for x in defaults.keys():
        if x in ['ctfBfactor', 'ctfScalefactor', 'angleRotation', 'angleTilt', 'anglePsi']:
            continue
        orig = defaults[x]
        actual = getattr(res, x)
        if x == 'files':
            assert len(orig) == len(actual)
            for i in range(len(orig)):
                assert orig[i] == actual[i]
        else:
            npt.assert_allclose(orig, actual)

    ret_data.pop('blob/psize_A')
    res = LensDescriptor.from_cryosparc_file(putative_file, True)
    assert res.ref_pixel_size is None

    res = LensDescriptor.from_cryosparc_file(putative_file, False)
    assert res.files is None
    assert res.idxs is None
    assert res.ref_pixel_size is None


@patch(f"{PKG}.read_star_file")
def test_from_starfile(read: Mock):
    file = "myfile.star"
    ret_data = {
        'DefocusU': defaults['defocusU'],
        'DefocusV': defaults['defocusV'],
        'DefocusAngle': defaults['defocusAngle'],
        'SphericalAberration': defaults['sphericalAberration'],
        'Voltage': np.ones((5,)) * defaults['voltage'],
        'AmplitudeContrast': defaults['amplitudeContrast'],
        'PhaseShift': defaults['phaseShift'],
        'CtfBfactor': defaults['ctfBfactor'],
        'CtfScalefactor': defaults['ctfScalefactor']
    }
    read.return_value = (ret_data, None)

    res = LensDescriptor.from_starfile(file, False, False)
    read.assert_called_once_with(file)
    for f in defaults.keys():
        if f in ['files', 'idxs', 'ref_pixel_size', 'angleRotation', 'angleTilt', 'anglePsi']:
            continue
        orig = defaults[f]
        actual = getattr(res, f)
        npt.assert_allclose(orig, actual)

    # Check defaults in case where the values weren't set in file
    for f in ["AmplitudeContrast", "PhaseShift", "CtfBfactor", "CtfScalefactor"]:
        ret_data.pop(f)

    res = LensDescriptor.from_starfile(file, False, False)
    assert res.amplitudeContrast == 0.1
    npt.assert_allclose(res.phaseShift, np.zeros_like(res.defocusU))
    assert res.ctfBfactor == 0.0
    assert res.ctfScalefactor == 1.0



FIX_INDEXED_STARFILE_VALUES = {
    'DefocusU': np.array([1., 1., 1.]),
    'DefocusV': np.array([2., 2., 2.]),
    'DefocusAngle': np.array([30., 30., 30.]),
    'PhaseShift': np.array([60., 60., 60.]),
    'SphericalAberration': np.array([3., 3., 3.]),
    'Voltage': np.array([4., 4., 4.]),
    'AmplitudeContrast': np.array([5., 5., 5.]),
    'AngleRot': np.array([120., 120., 120.]),
    'AngleTilt': np.array([90., 90., 90.]),
    'AnglePsi': np.array([45., 45., 45.]),
    'ImagePixelSize': np.array([.1, .1, .1]),
    'CtfBfactor': np.array([12., 12., 12.]),
    'CtfScalefactor': np.array([13., 13., 13.]),
    'ImageName': np.array(['000027@file1.mrc', '000001@sub/file2.mrc', '000005@file1.mrc']),
    'expected files': ['file1.mrc', 'sub/file2.mrc', 'file1.mrc'],
    'expected idxs': np.array([26, 0, 4])
}
DEFAULTABLES = []
SKIPPABLES = []
REQUIRED = []
for x in RELION_FIELDS:
    if x.defaultable:
        DEFAULTABLES.append(x.relion_field)
    if not x.required:
        SKIPPABLES. append(x.relion_field)
    if x.required and not x.defaultable:
        REQUIRED.append(x.relion_field)


def load_mock_indexed_starfile_reader(read: Mock, include_optionals: bool = True, include_required: bool = True):
    mock_dict = {}
    mock_fields = []
    for k in FIX_INDEXED_STARFILE_VALUES.keys():

        if not include_optionals and (k in DEFAULTABLES or k in SKIPPABLES):
            continue
        if not include_required and k in REQUIRED:
            continue
        mock_dict[k] = FIX_INDEXED_STARFILE_VALUES[k]
        mock_fields.append(k)

    read.return_value = (mock_dict, mock_fields)


@mark.parametrize('incl_opt', [(True), (False)])
@patch(f"{PKG}.read_star_file")
def test_from_indexed_starfile(read: Mock, incl_opt: bool):
    load_mock_indexed_starfile_reader(read, include_optionals=incl_opt)
    file = "myfile.star"
    sut = LensDescriptor.from_indexed_starfile(file)

    scalar_fields = ['sphericalAberration', 'voltage', 'amplitudeContrast']
    for x in RELION_FIELDS:
        val = getattr(sut, x.descriptor_field)
        if not incl_opt and not x.required:
            assert val is None
            continue
        if not incl_opt and x.defaultable:
            if isinstance(x.default, tuple):
                assert x.default[0] == 'expand'
                npt.assert_allclose(val, np.ones_like(sut.defocusU) * x.default[1])
            else:
                assert val == x.default
            continue
        if x.descriptor_field in scalar_fields:
            assert val == FIX_INDEXED_STARFILE_VALUES[x.relion_field][0]
            continue
        if x.relion_field in ALL_ANGLE_FIELDS:
            to_rad = np.radians(FIX_INDEXED_STARFILE_VALUES[x.relion_field])
            npt.assert_allclose(val, to_rad)
            continue
        if x.relion_field == 'ImagePixelSize':
            npt.assert_allclose(val, FIX_INDEXED_STARFILE_VALUES['ImagePixelSize'][0])
            continue
        npt.assert_allclose(val, FIX_INDEXED_STARFILE_VALUES[x.relion_field])

    # Confirm: correctly returns file names and indices
    assert sut.files is not None
    assert sut.idxs is not None
    for i, v in enumerate(FIX_INDEXED_STARFILE_VALUES['expected files']):
        assert sut.files[i] == v
    npt.assert_allclose(sut.idxs, FIX_INDEXED_STARFILE_VALUES['expected idxs'])


@patch(f"{PKG}.read_star_file")
def test_from_indexed_starfile_ignores_angles_when_asked(read: Mock):
    load_mock_indexed_starfile_reader(read, include_optionals=True)
    file = "myfile.star"
    sut = LensDescriptor.from_indexed_starfile(file, persist_angles=False)
    assert sut.anglePsi is None
    assert sut.angleRotation is None
    assert sut.angleTilt is None
    # make sure we didn't delete the non-optional ones
    assert sut.defocusAngle is not None
    assert sut.phaseShift is not None


@patch(f"{PKG}.read_star_file")
def test_from_indexed_starfile_throws_on_missing_fields(read: Mock):
    load_mock_indexed_starfile_reader(read, include_required=False)
    file = "myfile.star"
    with raises(ValueError, match=f"Unable to parse Relion-formatted starfile {file}"):
        _ = LensDescriptor.from_indexed_starfile(file)
