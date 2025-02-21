from unittest.mock import patch, Mock
from pytest import raises, mark
import numpy as np
import numpy.testing as npt

from cryolike.util.io import load_file, save_descriptors

PKG = "cryolike.util.io"

not_array = 5
is_str = 'string'
singleton = np.array([5])
one_val = np.array([5., 5.])
multi_val = np.array([1., 2., 3.])
empty_ary = np.array([])
none_val = None
none_ary = np.array([None])
singleton_str = np.array(['string'])
one_str = np.array(['string', 'string'])
multi_str = np.array(['string1', 'string2'])


@mark.parametrize("value", [not_array, is_str, singleton, one_val, multi_val, empty_ary, none_val, none_ary, singleton_str, one_str, multi_str])
def test_load_file(tmp_path, value):
    fname = tmp_path / "myfile.npz"
    np.savez_compressed(fname, v=value)

    result = load_file(fname)
    if value is None:
        assert result['v'] is None
    elif isinstance(value, np.ndarray):
        if value.dtype.type is np.str_:
            for i, v in enumerate(result['v']):
                assert v == value[i]
        elif value.size == 0:
            assert result['v'] is None
        else:
            if value[0] is None:
                assert result['v'] is None
            else:
                npt.assert_allclose(result['v'], value)
    else:
        assert result['v'] == value


def test_load_file_throws_on_no_name():
    with raises(FileNotFoundError, match="requires a file name"):
        _ = load_file("")


def test_load_file_throws_on_not_npz():
    filename = "some_file.txt"
    with raises(ValueError, match=".npz files"):
        _ = load_file(filename)


@patch(f"{PKG}.np.savez_compressed")
def test_save_descriptors(save: Mock):
    my_name = "file_does_not_exist.npz"
    dict_one = {'a': 1, 'b': 2}
    dict_two = {'c': 3}
    dict_three = {'d': 4}

    save_descriptors(my_name, dict_one, dict_two, dict_three)
    save.assert_called_once()
    call = save.call_args
    assert call[0][0] == my_name
    call_dict = call[1]
    assert len(call_dict.keys()) == 4
    assert call_dict['a'] == 1
    assert call_dict['b'] == 2
    assert call_dict['c'] == 3
    assert call_dict['d'] == 4


@patch(f"{PKG}.os.path.exists")
def test_save_descriptors_raises_on_existing_file(exists: Mock):
    my_name = "file.npz"
    exists.return_value = True
    with raises(ValueError, match="already exists"):
        save_descriptors(str(my_name), {'key': 'value'})


def test_save_descriptors_raises_on_key_loss():
    dict_one = {'a': 1, 'b': 2}
    dict_two = {'a': 3, 'c': 5}
    non_extant_file = "file_does_not_exist.npz"
    with raises(ValueError, match="Duplicate keys"):
        save_descriptors(non_extant_file, dict_one, dict_two)


@patch(f"{PKG}.np.savez_compressed")
def test_save_descriptors_no_op_on_empty_args(save: Mock):
    save_descriptors("my_file.npz")
    save.assert_not_called()
