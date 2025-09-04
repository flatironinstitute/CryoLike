from pytest import raises, mark

import numpy as np
import numpy.testing as npt

from cryolike.metadata.buffered_lens_descriptor import (
    LensDescriptor,
    LensDescriptorBuffer,
    BatchableLensFields
)

mock_vals = {
    "defocusU": np.array([1., 2., 3.]),
    "defocusV": np.array([2., 3., 4.]),
    "defocusAngle_deg": np.array([30., 60., 90.]),
    "sphericalAberration": np.array([4., 4., 4.,]),
    "voltage": np.array([5., 5., 5.,]),
    "amplitudeContrast": np.array([6., 6., 6.]),
    "phaseShift_deg": np.array([90., 135., 180.]),
}
mock_vals["defocusAngle"] = np.radians(mock_vals["defocusAngle_deg"])
mock_vals["phaseShift"] = np.radians(mock_vals["phaseShift_deg"])


mock_buffer_vals = {
    "defocusU": np.array([1.,1.]),
    "defocusV": np.array([2., 2.]),
    "defocusAngle": np.array([3., 3.]),
    "phaseShift": np.array([4., 4.])
}
mock_buffer_optionals = {
    "angleRotation": np.array([5., 5.]),
    "angleTilt": np.array([6., 6.]),
    "anglePsi": np.array([7., 7.])
}


def fix_make_buffer():
    parent = LensDescriptor(
        defocusU= mock_vals["defocusU"],
        defocusV= mock_vals["defocusV"],
        defocusAngle= mock_vals["defocusAngle_deg"],
        sphericalAberration= mock_vals["sphericalAberration"],
        voltage= mock_vals["voltage"],
        amplitudeContrast= mock_vals["amplitudeContrast"],
        phaseShift= mock_vals["phaseShift_deg"],
        defocusAngle_degree=True,
        phaseShift_degree=True
    )
    buffer = LensDescriptorBuffer(parent)
    return buffer


def fix_get_buffer_seed(with_optionals: bool = False):
    suffix = np.array([12., 12., 12.])
    vals_dict = {}

    for k in mock_buffer_vals.keys():
        vals_dict[k] = np.concatenate((mock_buffer_vals[k], suffix), axis=0)
    
    for k in mock_buffer_optionals.keys():
        vals_dict[k] = None if not with_optionals else np.concatenate((mock_buffer_optionals[k], suffix), axis=0)
    return BatchableLensFields(**vals_dict)        


def fix_seed_buffer(sut: LensDescriptorBuffer, seed: BatchableLensFields):
    for field in BatchableLensFields._fields:
        seed_val = getattr(seed, field)
        if isinstance(seed_val, np.ndarray):
            seed_val = seed_val.copy()
        setattr(sut, field, seed_val)
    sut.stack_size = seed.defocusU.shape[0]


def fix_make_batch(with_optionals: bool = False):
    if with_optionals:
        batch = BatchableLensFields(**mock_buffer_vals, **mock_buffer_optionals)
    else:
        batch = BatchableLensFields(
            **mock_buffer_vals,
            angleRotation=None,
            angleTilt=None,
            anglePsi=None
        )
    return batch


def test_buffer_init():
    sut = fix_make_buffer()
    assert sut.stack_size == 0
    for x in BatchableLensFields._fields:
        ary = getattr(sut, x)
        if x in ["angleRotation", "angleTilt", "anglePsi"]:
            assert ary is None
        else:
            assert len(ary) == 0


@mark.parametrize("with_opts", [(False), (True)])
def test_buffer_ensure_size_consistency(with_opts):
    sut = fix_make_buffer()
    seed = fix_get_buffer_seed(with_optionals=with_opts)
    fix_seed_buffer(sut, seed)

    # try changing the size of different fields
    for x in BatchableLensFields._fields:
        ary = getattr(sut, x)
        if ary is None:
            continue
        setattr(sut, x, np.append(ary, 1.0))
        with raises(ValueError, match="Inconsistent size"):
            sut._ensure_size_consistency()
        setattr(sut, x, ary)
    
    # try populating some but not all of the optionals
    for x in ['angleRotation', 'angleTilt', 'anglePsi']:
        ary = getattr(sut, x)
        if ary is None:
            setattr(sut, x, np.ones(sut.defocusU.shape[0]))
            with raises(ValueError, match="Some, but not all,"):
                sut._ensure_size_consistency()
            setattr(sut, x, None)
        else:
            setattr(sut, x, None)
            with raises(ValueError, match="optional buffers are set"):
                sut._ensure_size_consistency()
            setattr(sut, x, ary)

    # should pass
    sut._ensure_size_consistency()


def test_update_parent():
    sut = fix_make_buffer()
    old_parent = sut.parent_descriptor
    new_parent = LensDescriptor()
    sut.update_parent(new_parent)
    assert sut.parent_descriptor == new_parent
    assert sut.parent_descriptor != old_parent


def test_update_parent_throws_on_nonempty():
    sut = fix_make_buffer()
    seed = fix_get_buffer_seed()
    fix_seed_buffer(sut, seed)

    new_parent = LensDescriptor()
    old_parent = sut.parent_descriptor
    # should not throw: this is a no-op
    sut.update_parent(old_parent)

    with raises(ValueError, match="buffer is not empty"):
        sut.update_parent(new_parent)


def _check_equality(batch: BatchableLensFields, sut: LensDescriptorBuffer):
    assert sut.stack_size == len(batch.defocusU)
    for fieldname in BatchableLensFields._fields:
        batch_val = getattr(batch, fieldname)
        sut_val = getattr(sut, fieldname)
        if batch_val is not None:
            npt.assert_array_equal(sut_val, batch_val)
        else:
            assert sut_val is None


def _check_extended(sut: LensDescriptorBuffer, seed: BatchableLensFields, batch: BatchableLensFields):
    for field in BatchableLensFields._fields:
        batch_val = getattr(batch, field)
        seed_val = getattr(seed, field)
        sut_val = getattr(sut, field)
        if batch_val is None:
            assert sut_val is None
            continue
        npt.assert_array_equal(sut_val, np.concatenate((seed_val, batch_val), axis=0))
    assert sut.stack_size == seed.defocusU.shape[0] + batch.defocusU.shape[0]


def _check_pop_equal(batch_size: int, sut: LensDescriptorBuffer, popped: LensDescriptorBuffer, seed: BatchableLensFields):
    seed_size = seed.defocusU.shape[0]
    assert sut.stack_size + popped.stack_size == seed_size
    assert popped.stack_size == batch_size
    for field in BatchableLensFields._fields:
        seed_val = getattr(seed, field)
        sut_val = getattr(sut, field)
        pop_val = getattr(popped, field)
        if seed_val is None:
            assert sut_val is None
            assert pop_val is None
            continue
        npt.assert_array_equal(pop_val, seed_val[:batch_size])
        npt.assert_array_equal(sut_val, seed_val[batch_size:])


@mark.parametrize("with_optionals", [(False), (True)])
def test_buffer_enqueue_when_empty(with_optionals):
    sut = fix_make_buffer()
    assert sut.stack_size == 0
    assert sut.is_empty()

    batch = fix_make_batch(with_optionals=with_optionals)
    sut.enqueue(batch)
    _check_equality(batch, sut)


@mark.parametrize("with_optionals", [(False), (True)])
def test_buffer_enqueue_not_empty(with_optionals):
    sut = fix_make_buffer()
    assert sut.stack_size == 0

    seed = fix_get_buffer_seed(with_optionals=with_optionals)
    fix_seed_buffer(sut, seed)
    seeded_size = sut.stack_size
    assert seeded_size > 0

    batch = fix_make_batch(with_optionals=with_optionals)
    sut.enqueue(batch)

    assert sut.stack_size > seeded_size
    _check_extended(sut, seed, batch)


@mark.parametrize("with_optionals", [(False), (True)])
def test_pop_batch(with_optionals):
    sut = fix_make_buffer()
    assert sut.stack_size == 0
    seed = fix_get_buffer_seed(with_optionals=with_optionals)
    fix_seed_buffer(sut, seed)
    assert not sut.is_empty()

    popped = sut.pop_batch(1)
    _check_pop_equal(1, sut, popped, seed)


@mark.parametrize("with_opts", [(False), (True)])
def test_to_dict(with_opts):
    sut = fix_make_buffer()
    seed = fix_get_buffer_seed(with_optionals=with_opts)
    fix_seed_buffer(sut, seed)

    as_dict = sut.to_dict()
    for field in BatchableLensFields._fields:
        sut_val = getattr(sut, field)
        if sut_val is None:
            assert as_dict[field] is None
        else:
            npt.assert_allclose(sut_val, as_dict[field])
    npt.assert_allclose(np.squeeze(as_dict["sphericalAberration"]), mock_vals["sphericalAberration"])
