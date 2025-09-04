from pytest import raises, mark
import numpy as np
import torch

from cryolike.util.typechecks import (
    ensure_positive,
    ensure_positive_finite,
    ensure_integer,
    is_integral_torch_tensor
)

positives = [1.5, 2, np.array([1.5, 2.5, 3.5]), np.array([1, 2, 3]), torch.tensor([1.5, 2.5, 3.5])]
nonpositives = [
    -1.5, 0.,
    -2, 0,
    np.array([1.5, -2.5]), np.array([1.5, 0.]),
    np.array([1, 2, -3]), np.array([1, 2, 0]),
    torch.tensor([1.5, -2.5, 3.5]), torch.tensor([1.5, 0.0])
]

def test_ensure_positive():
    for x in positives:
        ensure_positive(x, '')
    for x in nonpositives:
        with raises(ValueError, match='Invalid value'):
            ensure_positive(x, '')


def test_ensure_positive_finite():
    passes = [1., 1]
    fail_nonpositive = [-1, -1.5]
    fail_nonfinite = [np.inf, np.nan -np.nan]
    for x in passes:
        ensure_positive_finite(x, '')
    for x in fail_nonpositive:
        with raises(ValueError, match='positive value'):
            ensure_positive_finite(x, '')
    for x in fail_nonfinite:
        with raises(ValueError, match='finite value'):
            ensure_positive_finite(x, '')


def test_ensure_integer():
    passes = [1, 2, 4, 4.0]
    fails = [1.2, 5.4]
    for x in passes:
        _ = ensure_integer(x, '')
    for x in fails:
        with raises(ValueError, match='must be an integer'):
            _ = ensure_integer(x, '')


def test_is_integral_torch_tensor():
    passes = [torch.tensor([4, 5])]
    fails = [torch.tensor([1.5, 2.5]), torch.tensor([1.0 + 1j, 0.0 - 0.5j])]

    for x in passes:
        res = is_integral_torch_tensor(x)
        assert res
    for x in fails:
        res = is_integral_torch_tensor(x)
        assert not res
