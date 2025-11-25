from pytest import mark, raises, skip
from unittest.mock import Mock, patch
from math import ceil
from torch.cuda import OutOfMemoryError, is_available

from cryolike.util import Precision

from cryolike.run_likelihood import (
    _BatchConfig,
    _get_smallest_batch,
    _run_stack_loop
)

PKG = "cryolike.run_likelihood"

# NOTE: Present testing covers only the batch size discovery behavior.
# It does not actually assert over likelihood file configuration,
# displacements, correct configuration for the implemented cross-correlation
# return types or kernels, or that the run-stack-loop function is calling
# kernels with correct values.


def test_run_stack_loop_skip_exist():
    if not is_available():
        skip("Test cannot run because CUDA is not present.")
    file_conf = Mock()
    outputs = Mock()
    image_desc = Mock()
    sizes = Mock()
    tp = Mock()
    kernel = Mock()
    file_conf.outputs_exist = Mock()
    assert isinstance(file_conf.outputs_exist, Mock)
    file_conf.outputs_exist.side_effect = [True, True, False]
    file_conf.load_img_stack = Mock(return_value=(Mock(), Mock()))
    assert isinstance(file_conf.load_img_stack, Mock)

    with patch('builtins.print') as _:
        with patch(f"{PKG}.template_first_comparator") as __:
            _run_stack_loop(
                3,
                True,
                file_conf,
                outputs,
                image_desc,
                sizes,
                tp,
                Precision.DOUBLE,
                False,
                kernel
            )
            file_conf.write_outputs.assert_called_once()
            kernel.assert_called_once()

    kernel.reset_mock()
    file_conf.write_outputs.reset_mock()
    with patch('builtins.print') as _:
        with patch(f"{PKG}.template_first_comparator") as __:
            _run_stack_loop(
                3,
                False,
                file_conf,
                outputs,
                image_desc,
                sizes,
                tp,
                Precision.DOUBLE,
                False,
                kernel
            )
            assert kernel.call_count == 3
            assert file_conf.write_outputs.call_count == 3


def test_run_stack_loop_throws_on_oom_with_no_discovery():
    file_conf = Mock()
    outputs = Mock()
    image_desc = Mock()
    sizes = _BatchConfig(False, 10, 10, 100)
    tp = Mock()
    kernel = Mock(side_effect=OutOfMemoryError)
    file_conf.load_img_stack = Mock(return_value=(Mock(), Mock()))
    assert isinstance(file_conf.load_img_stack, Mock)

    with patch('builtins.print') as _:
        with patch(f"{PKG}.template_first_comparator") as __:
            with raises(OutOfMemoryError):
                _run_stack_loop(
                    3,
                    False,
                    file_conf,
                    outputs,
                    image_desc,
                    sizes,
                    tp,
                    Precision.DOUBLE,
                    False,
                    kernel
                )


def test_run_stack_loop_batch_size_discovery():
    file_conf = Mock()
    outputs = Mock()
    image_desc = Mock()
    tp = Mock()
    kernel = Mock()
    mock_imgs_1 = Mock()
    mock_imgs_1.n_images = 100
    mock_imgs_2 = Mock()
    mock_imgs_2.n_images = 300
    file_conf.load_img_stack = Mock(side_effect=[(mock_imgs_1, Mock()), (mock_imgs_2, Mock())])
    assert isinstance(file_conf.load_img_stack, Mock)

    kernel.side_effect = [OutOfMemoryError, Mock(), OutOfMemoryError, Mock()]
    # With these settings, we expect to wind up with:
    #  1. Fail: lower template batch size to 5
    #  2. Succeed: set template batch size bounds
    #  3. Fail: Set image batch size ceiling = 300
    #     and image batch to 200
    #  4. Succeed: End loop
    sizes = _BatchConfig(True, 10, 100, 10)

    with patch('builtins.print') as _:
        with patch(f"{PKG}.template_first_comparator") as __:
            _run_stack_loop(
                2,
                False,
                file_conf,
                outputs,
                image_desc,
                sizes,
                tp,
                Precision.DOUBLE,
                True,
                kernel
            )
            # attempted loop 4x and finished it twice
            assert kernel.call_count == 4
            assert file_conf.write_outputs.call_count == 2
            assert sizes.t_batch == 5
            assert sizes.t_hard_ceiling == 10
            assert sizes.t_hard_floor == 5
            assert sizes.i_batch == 200
            assert sizes.i_hard_ceiling == 300
            assert sizes.i_hard_floor == 100


@mark.parametrize('total,batch', [
    (60,19),
    (60,15),
    (60,60),
    (60,120),
    (60,1)
])
def test_get_smallest_batch(total: int, batch: int):
    batch_count = ceil(total / batch)
    smallest_batch = _get_smallest_batch(batch, total)
    new_batch_count = ceil(total / smallest_batch)
    # three conditions for success:
    # 1. the proposed value has to result in the same # of batches
    # 2. A smaller batch value would result in more batches
    # 3. the proposed value should not exceed the original one
    
    assert new_batch_count == batch_count
    if smallest_batch == 1:
        return
    using_lower_batch = ceil(total / (smallest_batch - 1))
    assert using_lower_batch > batch_count
    assert smallest_batch <= batch


def test_get_smallest_batch_throws_on_negative_values():
    with raises(ValueError):
        _get_smallest_batch(0, 4)
    with raises(ValueError):
        _get_smallest_batch(3, -2)


@mark.parametrize("with_discovery, with_guesses", [(True, False), (False, True)])
def test_batchconfig_initialization(with_discovery: bool, with_guesses: bool):
    t_batch = 5 if with_guesses else None
    i_batch = 7 if with_guesses else None

    sut = _BatchConfig(with_discovery, t_batch, i_batch, 75)
    assert sut.t_total == 75
    assert sut.active == with_discovery
    assert sut.last_worked == False
    if with_guesses:
        assert sut.t_batch == t_batch
        assert sut.i_batch == i_batch
    else:
        assert sut.t_batch == -1
        assert sut.i_batch == -1
    assert sut.i_hard_floor == 0
    for x in [sut.i_hard_ceiling, sut.t_hard_ceiling, sut.t_hard_floor]:
        assert x == -1


def test_batchconfig_initialization_throws_on_ambiguous_invocation():
    with raises(ValueError):
        _ = _BatchConfig(do_discovery=False, t_batch=15, i_batch=None, t_total=60)
    with raises(ValueError):
        _ = _BatchConfig(do_discovery=False, t_batch=None, i_batch=15, t_total=60)


def test_batchconfig_initialization_throws_on_nonpositive_batch_sizes():
    with raises(ValueError):
        _ = _BatchConfig(True, None, 0, 60)
    with raises(ValueError):
        _ = _BatchConfig(True, -3, None, 60)


@mark.parametrize("i,t", [(1, 1), (5, None), (None, 10)])
def test_guess_initial_batches(i: int | None, t: int | None):
    # Values chosen just to make sure they don't reproduce the actual guess rule
    t_total = t * 100 if t is not None else 100
    i_total = i if i is not None else 800
    sut = _BatchConfig(True, t, i, t_total)
    sut.guess_initial_batches(i_total)

    # Note: this tests both that we honor the presets and that we
    # generate guesses
    if i is not None:
        assert sut.i_batch == i
    if t is not None:
        assert sut.t_batch == t
    assert sut.i_batch > 0
    assert sut.t_batch > 0


def test_adjust_batches_up_no_op_when_not_discovering():
    sut = _BatchConfig(False, 1, 1, 1)
    sut.last_worked = True
    with patch(f"{PKG}._get_smallest_batch") as fn:
        sut.adjust_batches_up(40)
        fn.assert_not_called()


def test_adjust_batches_up_no_op_when_failed_last_cycle():
    sut = _BatchConfig(True, 1, 1, 1)
    assert not sut.last_worked
    with patch(f"{PKG}._get_smallest_batch") as fn:
        sut.adjust_batches_up(40)
        fn.assert_not_called()
        assert sut.last_worked


def test_adjust_batches_up_throws_on_bad_batches():
    sut = _BatchConfig(True, 1, 1, 1)
    sut.last_worked = True
    # in practice, this shouldn't happen through usual channels
    sut.i_batch = 0
    with raises(ValueError):
        sut.adjust_batches_up(40)
    sut.i_batch = 1
    sut.t_batch = -1
    with raises(ValueError):
        sut.adjust_batches_up(40)


def test_adjust_batches_up_increases_template_size():
    # These numbers should yield ceil(60/13) = 5 batches.
    t_count = 60
    t_batch_initial = 13
    i_batch_initial = 50
    sut = _BatchConfig(True, t_batch_initial, i_batch_initial, t_count)
    with patch(f"{PKG}._get_smallest_batch") as fn:
        sut.adjust_batches_up(40)
        fn.assert_not_called()
    # "last_worked" flag should now be active
    assert sut.t_hard_floor == -1
    assert sut.t_hard_ceiling == -1

    sut.adjust_batches_up(40)

    # Floor should be set to 12 (smallest batch yielding 5 batches)
    # & the new t_batch should be 15, which yields 4 batches.
    assert sut.t_hard_floor == 12
    assert sut.t_batch == 15
    # Confirm we short-circuited adjustments to the image batch size
    assert sut.i_batch == 50 # the amount passed in the test ctor
    assert sut.i_hard_floor == 0


def test_adjust_batches_up_increases_image_size_without_ceiling():
    # NOTE: Also tests that we set ceiling to max if full batch worked
    t_count = 60
    t_batch_init = 120
    i_batch_init = 50
    i_max = 5000
    sut = _BatchConfig(True, t_batch_init, i_batch_init, t_count)
    # manually flip the toggle so that we actually do stuff
    sut.last_worked = True

    # With no i_hard_ceiling, we go for broke and try to use
    # a full batch for the images.
    # While we're here, also check that we set template hard
    # ceiling to the requested batch if a single-run batch size
    # worked.
    assert sut.t_hard_ceiling == -1
    assert sut.i_hard_ceiling == -1
    sut.adjust_batches_up(i_max)
    assert sut.t_hard_ceiling >= t_count
    assert sut.i_batch == i_max
    assert sut.i_batch > i_batch_init


def test_adjust_batches_up_increases_image_size_with_ceiling():
    t_count = 60
    t_batch_init = 60
    i_batch_init = 50
    i_batch_ceil = 100
    i_max = 5000
    sut = _BatchConfig(True, t_batch_init, i_batch_init, t_count)
    sut.last_worked = True
    sut.i_hard_ceiling = i_batch_ceil

    sut.adjust_batches_up(i_max)
    assert sut.i_batch == i_batch_init + (i_batch_ceil - i_batch_init) // 2
    assert sut.i_batch > i_batch_init
    assert sut.i_batch < i_max


def test_adjust_batches_up_no_op_if_ceiling_precludes_image_increase():
    t_count = 60
    t_batch_init = 60
    i_batch_init = 100
    i_batch_ceil = 180
    i_max = 200
    sut = _BatchConfig(True, t_batch_init, i_batch_init, t_count)
    sut.last_worked = True
    sut.i_hard_ceiling = i_batch_ceil

    # numbers above should give 2 batches for either init or ceil size
    # this should be a no-op for the size increase
    sut.adjust_batches_up(i_max)
    assert sut.i_batch == i_batch_init


def test_find_higher_template_size_throws_on_full_batch():
    t_count = 60
    t_batch = 60
    sut = _BatchConfig(True, t_batch, None, t_count)
    with raises(ValueError):
        sut._find_higher_template_size()


def test_find_higher_template_size_no_op_on_inactive():
    t_count = 60
    t_batch = 60
    sut = _BatchConfig(False, t_batch, 5, t_count)
    # Assert: nothing thrown
    # (could mock out _get_smallest_batch to assert not called, if feeling pedantic)
    sut._find_higher_template_size()


def test_adjust_batches_down_no_op_when_not_discovering():
    sut = _BatchConfig(False, 5, 5, 5)
    sut.last_worked = True  # use as signal that we have no-oped
    sut.adjust_batches_down()
    # Because the first step of actual batch down-adjustment is to
    # set this flag, this serves as a check that we did no-op
    assert sut.last_worked


def test_adjust_batches_down_throws_if_cannot_go_lower():
    sut = _BatchConfig(True, 1, 1, 5)
    with raises(ValueError, match="Unfixable"):
        sut.adjust_batches_down()


def test_adjust_batches_down_lowers_t_batch_if_no_floor():
    t_count = 60
    t_batch = t_count // 4
    i_batch = 1
    sut = _BatchConfig(True, t_batch, i_batch, t_count)
    sut.last_worked = True

    assert sut.t_batch == t_batch
    assert sut.t_hard_floor < 0
    sut.adjust_batches_down()
    assert sut.last_worked == False
    assert sut.t_hard_ceiling == t_batch
    assert sut.t_batch < t_batch
    assert sut.t_batch == t_count // 5


def test_adjust_batches_down_sets_bounds_if_t_is_one():
    t_count = 60
    t_batch = 1
    i_batch = 10
    sut = _BatchConfig(True, t_batch, i_batch, t_count)

    assert sut.t_hard_floor <= 0
    assert sut.t_hard_ceiling <= 0
    sut.adjust_batches_down()
    assert sut.t_hard_ceiling == 1
    assert sut.t_hard_floor == 1
    assert sut.i_batch <= i_batch


def test_adjust_batches_down_lowers_i_batch_if_t_has_floor():
    t_count = 60
    t_batch = 15
    i_batch = 100
    sut = _BatchConfig(True, t_batch, i_batch, t_count)
    sut.t_hard_floor = t_batch

    assert sut.i_hard_ceiling == -1
    sut.adjust_batches_down()

    assert sut.i_hard_ceiling == i_batch
    assert sut.i_batch == i_batch // 2


def test_adjust_batches_down_lowers_i_if_both_have_floor():
    t_count = 60
    t_batch = 15
    i_batch = 100
    i_floor = 50
    sut = _BatchConfig(True, t_batch, i_batch, t_count)
    sut.t_hard_floor = t_batch
    sut.i_hard_floor = i_floor

    sut.adjust_batches_down()
    assert sut.i_batch == (i_batch + i_floor) // 2
