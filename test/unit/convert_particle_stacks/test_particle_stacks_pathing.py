from pytest import raises
from typing import Callable
from unittest.mock import call, patch, Mock

from cryolike.convert_particle_stacks.particle_stacks_pathing import (
    OutputFolders,
    JobPaths
)

PKG = "cryolike.convert_particle_stacks.particle_stacks_pathing"

def mock_join(x: str, y: str):
    return f"{x}/{y}"


@patch(f"{PKG}.makedirs")
@patch(f"{PKG}.path")
def test_init_output_folders(mock_path: Mock, mock_mkdirs: Mock) -> None:
    mock_path.join = lambda x, y: mock_join(x, y)
    folder = "BASE"
    plots_out = mock_path.join(folder, 'plots')
    particles_fft_out = mock_path.join(folder, 'fft')
    particles_phys_out = mock_path.join(folder, 'phys')
    _ = OutputFolders(folder)
    calls = [call(plots_out, exist_ok=True), call(particles_fft_out, exist_ok=True), call(particles_phys_out, exist_ok=True)]
    mock_mkdirs.assert_has_calls(calls)


@patch(f"{PKG}.makedirs")
@patch(f"{PKG}.path")
def test_output_folders_get_output_filenames(mock_path: Mock, mock_mkdirs: Mock) -> None:
    mock_path.join = lambda x, y: mock_join(x, y)
    folder = "BASE"
    sut = OutputFolders(folder)
    i_stack = 5
    (phys_stack, fourier_stack, params_file) = sut.get_output_filenames(i_stack)
    assert phys_stack == mock_path.join(sut.folder_output_particles_phys, f"particles_phys_stack_{i_stack:06}.pt")
    assert fourier_stack == mock_path.join(sut.folder_output_particles_fft, f"particles_fourier_stack_{i_stack:06}.pt")
    assert params_file == mock_path.join(sut.folder_output_particles_fft, f"particles_fourier_stack_{i_stack:06}.npz")


def _job_paths_mock_fixture(mock_path: Mock, exists: None | Callable[[str], bool] = None):
    mock_path.join = lambda x, y: mock_join(x, y)
    if exists is None:
        exists = lambda x: True
    mock_path.exists = Mock(side_effect=exists)
    job_number = 5
    folder_cryosparc = "CRYOSPARC_FOLDER"
    return (job_number, folder_cryosparc, mock_path)


@patch(f"{PKG}.path")
def test_init_job_paths_all_exist_with_restack(mock_path: Mock):
    (job_number, folder_cryosparc, mock_path) = _job_paths_mock_fixture(mock_path)

    sut = JobPaths(folder_cryosparc, job_number)
    assert sut.folder_type == "restacks"
    assert sut.file_cs == f"{folder_cryosparc}/J{job_number}/J{job_number}_passthrough_particles.cs"


@patch(f"{PKG}.path")
def test_init_job_paths_all_exist_no_restack(mock_path: Mock):
    exists = lambda x: False if "restack" in x else True
    (job_number, folder_cryosparc, mock_path) = _job_paths_mock_fixture(mock_path, exists)

    sut = JobPaths(folder_cryosparc, job_number)
    assert sut.folder_type == "downsample"
    assert sut.file_cs == f"{folder_cryosparc}/J{job_number}/J{job_number}_passthrough_particles.cs"


@patch(f"{PKG}.path")
def test_init_job_paths_errors_on_no_job_folder(mock_path: Mock):
    exists = lambda x: False
    (job_number, folder_cryosparc, mock_path) = _job_paths_mock_fixture(mock_path, exists)
    assert isinstance(mock_path.exists, Mock)

    with raises(ValueError, match="folder not found"):
        _ = JobPaths(folder_cryosparc, job_number)
        mock_path.exists.assert_called_once()


@patch(f"{PKG}.path")
def test_init_job_paths_errors_on_no_restack_or_downsample(mock_path: Mock):
    def mock_exists(x):
        if "restack" in x: return False
        if "downsample" in x: return False
        return True
    (job_number, folder_cryosparc, mock_path) = _job_paths_mock_fixture(mock_path, mock_exists)
    assert isinstance(mock_path.exists, Mock)

    with raises(ValueError, match="and"):
        sut = JobPaths(folder_cryosparc, job_number)
        calls = [call(sut.restacks_folder), call(sut.downsample_folder)]
        mock_path.exists.assert_has_calls(calls)


@patch(f"{PKG}.path")
def test_job_path_get_mrc_filename(mock_path: Mock):
    (job_number, folder_cryosparc, mock_path) = _job_paths_mock_fixture(mock_path)
    sut = JobPaths(folder_cryosparc, job_number)

    i_file = 10
    batch_str = f"batch_{i_file:06}"

    assert sut.folder_type == "restacks"
    res1 = sut.get_mrc_filename(i_file)

    sut.folder_type = "downsample"
    res2 = sut.get_mrc_filename(i_file)

    assert res1 == f"{sut.restacks_folder}/batch_{i_file}_restacked.mrc"
    assert res2 == f"{sut.downsample_folder}/batch_{i_file:06}_downsample.mrc"


@patch(f"{PKG}.path")
def test_job_path_get_mrc_filename_errors_on_bad_type(mock_path: Mock):
    (job_number, folder_cryosparc, mock_path) = _job_paths_mock_fixture(mock_path)
    sut = JobPaths(folder_cryosparc, job_number)
    sut.folder_type = "unsupported"     # type: ignore
    with raises(NotImplementedError, match="Impossible value"):
        sut.get_mrc_filename(5)


@patch("builtins.print")
@patch(f"{PKG}.path")
def test_job_path_get_mrc_filename_when_path_not_exist(mock_path: Mock, mock_print: Mock):
    exists = lambda x: False if "batch" in x else True
    (job_number, folder_cryosparc, mock_path) = _job_paths_mock_fixture(mock_path, exists)
    sut = JobPaths(folder_cryosparc, job_number)
    path = sut.get_mrc_filename(5)
    mock_print.assert_called_once()
    assert path is None

