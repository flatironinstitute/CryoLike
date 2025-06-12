from pathlib import Path
from pytest import raises, mark
from unittest.mock import patch, Mock

from cryolike.file_mgmt.file_base import (
    make_dir,
    check_files_exist,
    get_input_filename
)


@mark.parametrize('path_exists,with_branch', [(True, True),
                                             (True, False),
                                             (False, True),
                                             (False, False)])
def test_make_dir(tmp_path, path_exists, with_branch):
    branch = '' if not with_branch else 'child'
    target_path = tmp_path / branch
    if path_exists:
        target_path.mkdir(exist_ok=True)
    
    make_dir(tmp_path, branch)
    assert target_path.is_dir()


def test_make_dir_with_extant_file(tmp_path):
    branch = 'myfile.txt'
    target_path = tmp_path / branch
    target_path.write_text("")
    with raises(FileExistsError, match="already exists"):
        make_dir(tmp_path, branch)


@mark.parametrize('missing_fcount', [(0), (1), (3)])
def test_check_files_exist(tmp_path, missing_fcount):
    files = ["file1.txt", "file2.txt", "file3.txt", "file4.txt"]
    all_files = [tmp_path / f for f in files]
    missing_files = []
    for _ in range(missing_fcount):
        missing_files.append(tmp_path / files.pop())
    for f in files:
        p = tmp_path / f
        p.write_text("")
    
    (res_missing, res_missing_list) = check_files_exist(all_files)
    assert res_missing == (missing_fcount == 0)
    assert len(res_missing_list) == missing_fcount
    for f in missing_files:
        assert str(f) in res_missing_list


@mark.parametrize('i,ftype', [
    (5, 'fourier'),
    (12, 'params'),
    (97, 'phys'),
    (102, 'unpermitted')
])
def test_get_input_filename(tmp_path, i, ftype):
    if ftype == 'unpermitted':
        with raises(NotImplementedError, match='Impermissible file type'):
            get_input_filename(tmp_path, i, ftype) # type: ignore
        return

    i_stack = f'stack_{i:06}'
    if ftype == 'fourier':
        expected_name = f"particles_fourier_{i_stack}.pt"
    if ftype == 'params':
        expected_name = f"particles_fourier_{i_stack}.npz"
    if ftype == 'phys':
        expected_name = f"particles_phys_{i_stack}.pt"
    
    res = get_input_filename(tmp_path, i, ftype)
    assert res == tmp_path / expected_name
