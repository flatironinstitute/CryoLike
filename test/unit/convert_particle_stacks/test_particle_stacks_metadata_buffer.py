from pytest import raises

import numpy as np
import numpy.testing as npt
import torch

from cryolike.convert_particle_stacks.particle_stacks_buffers import (
    _Metadata,
    _MetadataBuffer,
    _pop_batch,
)

mock_vals = {
    "defocusU": np.array([1., 2., 3.]),
    "defocusV": np.array([2., 3., 4.]),
    "defocusAngle": np.array([3., 4., 5.]),
    "sphericalAberration": np.array([4., 4., 4.,]),
    "voltage": np.array([5., 5., 5.,]),
    "amplitudeContrast": np.array([6., 6., 6.]),
    "phaseShift": np.array([7., 8., 9.]),
}

def fix_make_metadatabuffer():
    meta = _Metadata(
        defocusU= mock_vals["defocusU"],
        defocusV= mock_vals["defocusV"],
        defocusAngle= mock_vals["defocusAngle"],
        sphericalAberration= mock_vals["sphericalAberration"],
        voltage= mock_vals["voltage"],
        amplitudeContrast= mock_vals["amplitudeContrast"],
        phaseShift= mock_vals["phaseShift"],
    )
    buffer = _MetadataBuffer(meta)
    return buffer


def assert_pop_batch(orig, head, tail, batch: int):
    assert type(head) == type(orig)
    assert type(head) == type(tail)
    assert len(head) + len(tail) == len(orig)
    assert len(head) == min(len(orig), batch)
    for i, val in enumerate(head):
        assert orig[i] == val
    for i, val in enumerate(tail):
        assert orig[i + len(head)] == val

def test_pop_batch():
    batch_sizes = [0, 4, 8, 10, 11]
    torch_list = torch.arange(10)
    float_list = np.array(np.arange(10), dtype=np.float32)
    cmplx_list = np.array(np.arange(10), dtype=np.complex128)

    for src_list in [torch_list, float_list, cmplx_list]:
        for batch_length in batch_sizes:
            (head, tail) = _pop_batch(src_list, batch_length)
            assert_pop_batch(src_list, head, tail, batch_length)


def test_metadata_buffer_init():
    sut = fix_make_metadatabuffer()
    for x in ['defocusU', 'defocusV', 'defocusAngle', 'phaseShift']:
        ary = getattr(sut, x)
        assert len(ary) == 0
    for x in ['sphericalAberration', 'voltage', 'amplitudeContrast']:
        val = getattr(sut, x)
        assert isinstance(val, float)
        assert val == mock_vals[x][0]


def test_metadata_buffer_ensure_size_consistency():
    sut = fix_make_metadatabuffer()
    sut.defocusU = np.array(range(3), dtype=np.float32)
    sut.defocusV = np.array(range(3), dtype=np.float32)
    sut.defocusAngle = np.array(range(3), dtype=np.float32)
    sut.phaseShift = np.array(range(3), dtype=np.float32)
    sut.stack_size = 3

    # try changing the size of different fields
    for x in ['defocusU', 'defocusV', 'defocusAngle', 'phaseShift']:
        ary = getattr(sut, x)
        setattr(sut, x, np.append(ary, 1.0))
        with raises(ValueError, match="Inconsistent size"):
            sut._ensure_size_consistency()
        setattr(sut, x, ary)
    
    # should pass
    sut._ensure_size_consistency()


def test_metadata_buffer_make_copy():
    fix = fix_make_metadatabuffer()
    defU = np.array([1., 1.])
    defV = np.array([2., 2.])
    defA = np.array([3., 3.])
    phase = np.array([4., 4.])

    res = fix.make_copy(defU, defV, defA, phase)
    npt.assert_equal(res.defocusU, defU)
    npt.assert_equal(res.defocusV, defV)
    npt.assert_equal(res.defocusAngle, defA)
    npt.assert_equal(res.phaseShift, phase)

    assert res.stack_size == len(defU)
    assert res.stack_size != fix.stack_size

    for x in ["sphericalAberration", "voltage", "amplitudeContrast"]:
        assert getattr(fix, x) == getattr(res, x)


def test_metadata_buffer_append_batch_when_empty():
    sut = fix_make_metadatabuffer()
    assert sut.stack_size == 0
    defU = np.array([1., 1.])
    defV = np.array([2., 2.])
    defA = np.array([3., 3.])
    phase = np.array([4., 4.])

    sut.append_batch(defU, defV, defA, phase)
    assert sut.stack_size == len(defU)
    npt.assert_array_equal(sut.defocusU, defU)
    npt.assert_array_equal(sut.defocusV, defV)
    npt.assert_array_equal(sut.defocusAngle, defA)
    npt.assert_array_equal(sut.phaseShift, phase)


def test_metadata_buffer_append_batch_not_empty():
    sut = fix_make_metadatabuffer()
    assert sut.stack_size == 0
    defU = np.array([1., 1.])
    defV = np.array([2., 2.])
    defA = np.array([3., 3.])
    phase = np.array([4., 4.])
    sut.defocusU = defU
    sut.defocusV = defV
    sut.defocusAngle = defA
    sut.phaseShift = phase
    sut.stack_size = 2

    # Intentionally in reverse order
    sut.append_batch(phase, defA, defV, defU)
    assert sut.stack_size == 2 * len(defU)
    npt.assert_array_equal(sut.defocusU, np.concatenate((defU, phase), axis=0))
    npt.assert_array_equal(sut.defocusV, np.concatenate((defV, defA), axis=0))
    npt.assert_array_equal(sut.defocusAngle, np.concatenate((defA, defV), axis=0))
    npt.assert_array_equal(sut.phaseShift, np.concatenate((phase, defU), axis=0))


def test_metadata_buffer_pop_batch():
    sut = fix_make_metadatabuffer()
    assert sut.stack_size == 0
    defU = np.array([1., 1., 2.])
    defV = np.array([2., 2., 3.])
    defA = np.array([3., 3., 4.])
    phase = np.array([4., 4., 5.])
    sut.defocusU = defU
    sut.defocusV = defV
    sut.defocusAngle = defA
    sut.phaseShift = phase
    sut.stack_size = 3

    popped = sut.pop_batch(1)
    assert sut.stack_size == 2
    assert popped.stack_size == 1
    npt.assert_array_equal(popped.defocusU, defU[:1])
    npt.assert_array_equal(popped.defocusV, defV[:1])
    npt.assert_array_equal(popped.defocusAngle, defA[:1])
    npt.assert_array_equal(popped.phaseShift, phase[:1])

    npt.assert_array_equal(sut.defocusU, defU[1:])
    npt.assert_array_equal(sut.defocusV, defV[1:])
    npt.assert_array_equal(sut.defocusAngle, defA[1:])
    npt.assert_array_equal(sut.phaseShift, phase[1:])


