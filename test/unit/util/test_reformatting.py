import numpy as np
from pytest import raises
from unittest.mock import patch
from io import StringIO

from cryolike.util.reformatting import project_descriptor, project_scalar, project_vector
from cryolike.grids.cartesian_grid import TargetType

def test_project_scalar_matches_input_type():
    my_float = 1.0
    float_res = project_scalar(my_float, 3)
    assert issubclass(float_res.dtype.type, np.floating)
    my_int = 4
    int_res = project_scalar(my_int, 3)
    assert issubclass(int_res.dtype.type, np.integer)

def test_project_scalar_returns_right_value():
    val = 3.5
    res = project_scalar(val, 5)
    assert np.all(res == val)

def test_project_scalar_returns_right_vector_size():
    res_2 = project_scalar(5, 2)
    assert res_2.shape == (2,)
    res_3 = project_scalar(5, 3)
    assert res_3.shape == (3,)

def test_project_vector_squeezes_for_ndarray():
    vector = np.array([[5, 4]])
    res = project_vector(vector, 2, 'label')
    np.testing.assert_allclose(res, [5, 4])
    assert res.shape == (2,)

def test_project_vector_errors_on_multidimensional_array():
    vector = np.array([[5, 5], [4, 4]])
    with raises(ValueError, match="multi-dimensional"):
        _ = project_vector(vector, 2, 'label')

def test_project_vector_uses_scalar_for_unit_length_vector():
    v = np.array([5])
    res = project_vector(v, 3, 'label')
    assert np.all(res == 5)
    assert res.shape == (3,)
    v = [5]
    res = project_vector(v, 3, 'label')
    assert np.all(res == 5)
    assert res.shape == (3,)

def test_project_vector_errors_if_too_short():
    v = [1, 2]
    with raises(ValueError, match="3-dimensional"):
        _ = project_vector(v, 3, 'short')

@patch('sys.stdout', new_callable=StringIO)
def test_project_vector_warns_on_too_long(mock_stdout):
    v = [1, 2, 3]
    _ = project_vector(v, 2, 'foo')
    assert mock_stdout.getvalue() == 'Warning: foo is more than 2-dimensional. Ignoring higher dimensions...\n'

def test_project_vector_returns_first_n_items():
    v = [1, 2, 3]
    res = project_vector(v, 2, 'foo')
    np.testing.assert_allclose(res, v[:2])
    res2 = project_vector(np.array(v), 2, 'foo')
    np.testing.assert_allclose(res2, v[:2])


def check_projection(res: np.ndarray, vals: list[int] | list[float], shape: tuple[int], target: TargetType):
    np.testing.assert_allclose(res, vals)
    assert res.shape == shape
    if target == TargetType.INT:
        assert issubclass(res.dtype.type, np.integer)
    elif target == TargetType.FLOAT:
        assert issubclass(res.dtype.type, np.floating)
    else:
        raise ValueError("Unreachable: unsupported tuple value")

def test_project_descriptor_works_on_scalar_input():
    dims = 3
    res = project_descriptor(1, 'int', dims, TargetType.INT)
    check_projection(res, [1, 1, 1], (dims,), TargetType.INT)
    res = project_descriptor(1., 'float', dims, TargetType.FLOAT)
    check_projection(res, [1, 1, 1], (dims,), TargetType.FLOAT)

def test_project_descriptor_works_on_vector_input():
    dims = 3
    base_v = [1, 2, 3]
    res = project_descriptor(base_v, 'int list', dims, TargetType.INT)
    check_projection(res, base_v, (dims,), TargetType.INT)
    res = project_descriptor(np.array(base_v), 'int array', dims, TargetType.INT)
    check_projection(res, base_v, (dims,), TargetType.INT)
    res = project_descriptor([float(v) for v in base_v], 'float list', dims, TargetType.FLOAT)
    check_projection(res, base_v, (dims,), TargetType.FLOAT)
    res = project_descriptor(np.array([float(v) for v in base_v]), 'float array', dims, TargetType.FLOAT)
    check_projection(res, base_v, (dims,), TargetType.FLOAT)

def test_project_descriptor_converts_integer_valued_floats():
    v = [1., 2., 3.]
    res = project_descriptor(v, 'floats', 3, TargetType.INT)
    assert issubclass(res.dtype.type, np.integer)

def test_project_descriptor_raises_on_noninteger_values():
    v = [1., 1.5, 2.0]
    with raises(ValueError, match="must have integer values"):
        _ = project_descriptor(v, 'float list', 3, TargetType.INT)

def test_project_descriptor_raises_on_nonpositive_values():
    v = [-1, -2, -3]
    with raises(ValueError, match="must have positive values"):
        _ = project_descriptor(v, 'negative list', 3, TargetType.INT)

def test_project_descriptor_converts_ints_to_float():
    v = np.array([1, 2, 3]).astype(int)
    res = project_descriptor(v, 'ints', 3, TargetType.FLOAT)
    assert issubclass(res.dtype.type, np.floating)