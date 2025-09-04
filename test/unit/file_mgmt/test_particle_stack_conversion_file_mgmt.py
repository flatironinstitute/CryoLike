from pathlib import Path
from pytest import raises, mark
from unittest.mock import patch, Mock
from numpy import array as np_array
import numpy.testing as npt

from cryolike.file_mgmt import make_dir
from cryolike.file_mgmt.particle_stack_conversion_file_mgmt import (
    ParticleConversionFileManager,
    CryosparcJobPath,
    get_filenames_and_indices,
    ensure_input_files_exist
)

PKG = "cryolike.file_mgmt.particle_stack_conversion_file_mgmt"

def test_get_filenames_and_indices(tmp_path):
    files = ["does-not-exist", b"file1", ">file2", b"file1"]
    (tmp_path / "file1").write_text("")
    (tmp_path / "file2").write_text("")


    cryosparc_folder = str(tmp_path)
    files =[b"does-not-exist", "file1", b">file2", "file1"]
    expected_file_roots = ["does-not-exist", "file1", "file2"]
    expected_files = [f"{cryosparc_folder}/{expected_file_roots[i]}" for i in range(len(expected_file_roots))]
    indices = np_array([1, 3, 2, 4])
    expected_returns = [
        (expected_files[1], np_array([3, 4]), np_array([1, 3])),
        (expected_files[2], np_array([2]), np_array([2]))
    ]

    lens_desc = Mock()
    lens_desc.files = files
    lens_desc.idxs = indices


    with patch("builtins.print") as _print:
        res = get_filenames_and_indices(lens_desc, str(tmp_path))
        # Assert: skips non-extant files
        _print.assert_called_once_with(f"File {expected_files[0]} does not exist, skipping...")

    # Assert: returns mrc_file, indices tuples
    for exp_row in expected_returns:
        match = None
        for result_row in res:
            if result_row[0] == exp_row[0]:
                match = result_row
                break
        assert match is not None
        assert match[0] == exp_row[0]
        npt.assert_equal(match[1], exp_row[1])
        npt.assert_equal(match[2], exp_row[2])


def test_get_filenames_and_indices_throws_when_no_files():
    files = [b"myfiles/foo", b"myfiles/bar"]
    lens_desc = Mock()
    lens_desc.files = files
    lens_desc.idxs = [1, 5]

    with patch("builtins.print") as _print:
        with raises(ValueError, match="None of the MRC files"):
            _ = get_filenames_and_indices(lens_desc)


def test_ensure_input_files_exist(tmp_path):
    base = ["file_1.mrc", "file_2.mrc"]
    extra_fn = "missing.txt"
    final_fns = []
    for x in base:
        fn = tmp_path / x
        fn.write_text("")
        final_fns.append(str(fn))
    ensure_input_files_exist(final_fns)

    final_fns.append(extra_fn)
    with raises(ValueError):
        ensure_input_files_exist(final_fns)


def _job_paths_fixture(tmp_path: Path, job_number: int, restack_exist: bool = False, downsample_exist: bool = False, omit_job_dir: bool = False):
    cryo_dir = tmp_path / "THIS_SHOULD_NOT_BE_VISIBLE"
    job_dir = cryo_dir / f"J{job_number}"
    if (restack_exist):
        make_dir(job_dir, "restack")
    if (downsample_exist):
        make_dir(job_dir, "downsample")
    if not omit_job_dir:
        make_dir(job_dir, '')

    return CryosparcJobPath(str(cryo_dir), job_number)


@mark.parametrize("incl_restack,incl_downsample", [(True, True), (True, False), (False, True), (False, False)])
def test_job_paths_config(tmp_path, incl_restack: bool, incl_downsample: bool):
    job_number = 12
    if not (incl_downsample or incl_restack):
        with raises(ValueError, match="Error: directory not found"):
            _ = _job_paths_fixture(tmp_path, job_number, incl_restack, incl_downsample)
        return
    sut = _job_paths_fixture(tmp_path, job_number, incl_restack, incl_downsample)

    assert sut.folder_type == 'restacked' if incl_restack else 'downsample'
    root_dir = sut.file_cs.parent.parent
    assert sut.file_cs == root_dir / f"J{job_number}" / f"J{job_number}_passthrough_particles.cs"


def test_job_paths_fails_if_not_job_dir(tmp_path):
    job_number = 12
    with raises(ValueError, match="Error: folder not found"):
        _ = _job_paths_fixture(tmp_path, job_number, omit_job_dir=True)
    

@mark.parametrize("use_downsample", [(True), (False)])
def test_job_path_get_mrc_filename(tmp_path, use_downsample: bool):
    job_number = 14
    i_file = 63
    expected_type = "downsample" if use_downsample else "restacked"
    expected_file = f"batch_{i_file:06}_{expected_type}.mrc"

    sut = _job_paths_fixture(tmp_path, job_number, not use_downsample, use_downsample)
    res = sut.get_mrc_filename(i_file)

    assert res == sut.mrc_folder / expected_file


####

def _get_file_mgr(tmp_path: Path):
    folder_base = tmp_path / "YOU_SHOULD_NOT_SEE_THIS"
    
    return ParticleConversionFileManager(str(folder_base))


def test_init_output_folders(tmp_path) -> None:
    folder_base = tmp_path / "YOU_SHOULD_NOT_SEE_THIS"
    expected_base = Path(folder_base)
    expected_plots = expected_base / "plots"
    expected_fft = expected_base / "fft"
    expected_phys = expected_base / "phys"
    expected = [expected_base, expected_plots, expected_fft, expected_phys]
    for p in expected:
        assert not p.exists()

    _ = ParticleConversionFileManager(str(folder_base))

    for p in expected:
        assert p.is_dir()


def test_get_output_filenames(tmp_path) -> None:
    i_stack = 10
    expected_phys_stack = "particles_phys_stack_000010.pt"
    expected_fourier_stack = "particles_fourier_stack_000010.pt"
    expected_params = "particles_fourier_stack_000010.npz"

    sut = _get_file_mgr(tmp_path)
    res = sut.get_output_filenames(i_stack)

    assert res.phys_stack == sut._out_phys / expected_phys_stack
    assert res.fourier_stack == sut._out_fft / expected_fourier_stack
    assert res.params_filename == sut._out_fft / expected_params


def test_get_plot_filenames(tmp_path) -> None:
    i_stack = 10
    suffix = f"stack_000010.png"
    expected_phys = f"particles_phys_{suffix}"
    expected_four = f"particles_fourier_{suffix}"
    expected_power = f"power_spectrum_{suffix}"

    sut = _get_file_mgr(tmp_path)
    (res_phys, res_four, res_power) = sut.get_plot_filenames(i_stack)

    assert res_phys == str(sut._out_plots / expected_phys)
    assert res_four == str(sut._out_plots / expected_four)
    assert res_power == str(sut._out_plots / expected_power)


@mark.parametrize("job_number,file_count", [(2,13), (12,1)])
def test_read_job_dir(tmp_path, job_number: int, file_count: int):
    sut = _get_file_mgr(tmp_path)

    seed_dir = tmp_path / "ERROR_IF_SEEN" / f"J{job_number}"
    make_dir(seed_dir, "restack")
    expected_files = []
    for i in range(file_count):
        p = seed_dir / "restack" / f"batch_{i:06}_restacked.mrc"
        expected_files.append(str(p))
        p.write_text("")
    with patch("builtins.print") as _print:
        (_, files) = sut.read_job_dir(seed_dir.parent, job_number)
        assert _print.call_count == 1
        assert len(files) == file_count
        for i, got in enumerate(files):
            assert got == expected_files[i]


@mark.parametrize("overall,overwrite", [(None, False), (12, True)])
def test_write_batch(tmp_path, overall: int | None, overwrite: bool):
    sut = _get_file_mgr(tmp_path)
    stack_cnt = 12
    stack_size = 37
    phys_batch = Mock()
    phys_batch.shape = [stack_size, 33]
    fourier_batch = Mock()
    img_desc = Mock()
    lens_batch = Mock()

    out_fns = sut.get_output_filenames(stack_cnt)

    expected_phys_stack_fn = out_fns.phys_stack
    expected_four_stack_fn = out_fns.fourier_stack
    expected_params_fn = out_fns.params_filename

    with (
        patch("builtins.print") as m_print,
        patch(f"{PKG}.save_combined_params") as m_combo,
        patch(f"{PKG}.save") as m_save
    ):
        sut.write_batch(stack_cnt, phys_batch, fourier_batch, img_desc, lens_batch, overwrite, overall)
        assert m_print.call_count == 1
        assert m_save.call_count == 2
        assert m_combo.call_count == 1
        
        args = m_save.call_args_list
        assert args[0][0][0] == phys_batch
        assert args[0][0][1] == expected_phys_stack_fn
        assert args[1][0][0] == fourier_batch
        assert args[1][0][1] == expected_four_stack_fn

        args = m_combo.call_args
        assert args[0][0] == str(expected_params_fn)
        assert args[0][1] == img_desc
        assert args[0][2] == lens_batch
        assert args[1]['n_imgs_this_stack'] == stack_size
        assert args[1]['overall_batch_start'] == overall
        assert args[1]['overwrite'] == overwrite
