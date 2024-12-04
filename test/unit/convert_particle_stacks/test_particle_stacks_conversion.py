from pytest import raises
from unittest.mock import call, patch, Mock

import torch
import numpy as np
import numpy.testing as npt

from cryolike.microscopy import ParsedParameters
from cryolike.util import Precision
from cryolike.convert_particle_stacks.particle_stacks_conversion import (
    _do_image_normalization,
    _get_unbuffered_batch,
    convert_particle_stacks_from_star_files,
    _drain_buffer,
    _make_Images_from_mrc_file,
    _get_filenames_and_image_selection_indices,
    _Metadata,
    convert_particle_stacks_from_cryosparc,
    convert_particle_stacks_from_cryosparc_restack,
)

PKG = "cryolike.convert_particle_stacks.particle_stacks_conversion"


#### Fixtures

fix_parameters = ParsedParameters(
    n_voxels = np.array([3, 3, 3]),
    voxel_size = np.array([1., 1., 1.]),
    box_size = np.array([5., 5.]),
    radius_max = 10.0,
    dist_radii = 1.0,
    n_inplanes = 5,
    precision = Precision.SINGLE
)


def configure_mock_OutputFolders(mock_OutputFolders: Mock):
    mock_output_filenames = Mock()
    mock_output_filenames.phys_stack = "phys_stack"
    mock_output_filenames.fourier_stack = "fourier_stack"
    mock_output_filenames.params_filename = "params_filename"

    mock_OutputFolders.folder_output_plots = "output_plots"
    mock_OutputFolders.folder_output_particles_fft = "output_fft"
    mock_OutputFolders.folder_output_particles_phys = "output_phys"
    mock_OutputFolders.get_output_filenames = Mock(side_effect=lambda x: mock_output_filenames)

    return mock_OutputFolders


def configure_mock_Metadata(mock_Metadata: Mock):
    mock_meta = Mock()
    mock_batch = Mock()
    mock_batch.save_params_star = Mock()
    mock_meta.batch = mock_batch
    mock_meta.take_range = Mock(side_effect=lambda x, y: mock_batch)

    def get_mock_meta(star_file, defocus_is_degree, phase_shift_is_degree):
        return mock_meta

    mock_Metadata.meta = mock_meta
    mock_Metadata.from_star_file = Mock(side_effect=get_mock_meta)

    return mock_Metadata


def make_mock_image_stack(length: int):
    mock_im = Mock()
    mock_im.images_phys = torch.arange(length)
    mock_im.n_images = length
    mock_im.phys_grid = Mock()
    mock_im.phys_grid.pixel_size = length
    return mock_im


def configure_mock_images(mock_Images: Mock):
    mock_from_mrc = Mock(side_effect=[make_mock_image_stack(7), make_mock_image_stack(5)])

    mock_Images.from_mrc = mock_from_mrc
    return mock_Images



#### Tests

## Generic functions

@patch("builtins.print")
def test_do_image_normalization(mock_print: Mock):
    im = Mock()
    im.images_phys = np.array(range(5))
    im.images_fourier = np.array(range(5))
    im.downsample_images_phys = Mock()
    im.center_physical_image_signal = Mock()
    im.transform_to_fourier = Mock()
    im.normalize_images_fourier = Mock()

    polar_grid = Mock()
    precision = Precision.DEFAULT

    # test no downsample
    _do_image_normalization(im, polar_grid, precision)
    assert isinstance(im.downsample_images_phys, Mock)
    im.downsample_images_phys.assert_not_called()
    assert im.center_physical_image_signal.call_count == 1
    im.transform_to_fourier.assert_called_once_with(polar_grid=polar_grid, precision=precision, use_cuda=True)
    im.normalize_images_fourier.assert_called_once_with(ord=2, use_max=False)

    # test with downsample
    _do_image_normalization(im, polar_grid, precision, 3)
    im.downsample_images_phys.assert_called_once_with(3)

    assert mock_print.call_count == 4


@patch(f"{PKG}.PhysicalImages")
@patch(f"{PKG}.Images")
def test_get_unbuffered_batch(mock_Images: Mock, mock_PhysicalImages: Mock):
    mock_Images.return_value = Mock()
    mock_im = Mock()
    mock_im.images_phys = list(range(9))
    mock_im.n_images = 9
    mock_im.phys_grid = Mock()
    pixel_size = 0.5
    mock_im.phys_grid.pixel_size = pixel_size

    mock_PhysicalImages.side_effect = ['a', 'b']

    batch_size = 5
    # test with full batch
    (s, e, obj) = _get_unbuffered_batch(0, batch_size, mock_im)
    assert s == 0
    assert e == batch_size

    mock_PhysicalImages.assert_called_once_with(images_phys=mock_im.images_phys[0:5], pixel_size=pixel_size)
    mock_Images.assert_called_once_with(phys_images_data='a')
    assert obj == mock_Images.return_value

    # test with partial batch
    (s, e, obj) = _get_unbuffered_batch(1, batch_size, mock_im)
    assert s == batch_size
    assert e == mock_im.n_images
    mock_Images.assert_called_with(phys_images_data='b')


@patch(f"{PKG}.Images")
def test_make_images_from_mrc_file(mock_images: Mock):
    mock_Images_obj = Mock()
    mock_Images_obj.select_images = Mock()
    assert isinstance(mock_Images_obj.select_images, Mock)
    mock_images.from_mrc = Mock(return_value=mock_Images_obj)
    assert isinstance(mock_images.from_mrc, Mock)

    mrc_file_path = "path"
    pixel_size = np.array([1., 1.])
    selected_imgs = np.array([1, 3, 5])

    ret = _make_Images_from_mrc_file(
        mrc_file_path=mrc_file_path,
        selected_imgs=selected_imgs,
        _pixel_size=pixel_size
    )

    assert ret == mock_Images_obj
    mock_images.from_mrc.assert_called_once_with(mrc_file_path, pixel_size=pixel_size)
    mock_Images_obj.select_images.assert_called_once_with(selected_imgs)


## convert_particle_stacks_from_star_files (star-file function)

@patch(f"{PKG}.torch")
@patch(f"{PKG}._plot_images")
@patch(f"{PKG}._do_image_normalization")
@patch(f"{PKG}._get_unbuffered_batch")
@patch(f"{PKG}.Images")
@patch(f"{PKG}._Metadata")
@patch(f"{PKG}.path")
@patch(f"{PKG}.OutputFolders")
@patch(f"{PKG}.ensure_parameters")
@patch("builtins.print")
def test_convert_particle_stacks(
    mock_print: Mock,
    mock_ensure_params: Mock,
    mock_OutputFolders: Mock,
    mock_path: Mock,
    mock_Metadata: Mock,
    mock_Images: Mock,
    mock_get_batch: Mock,
    mock_do_norm: Mock,
    mock_plot: Mock,
    mock_torch: Mock,
):
    # Most of these mocks are to prevent any actual behavior on the filesystem.
    mock_ensure_params.side_effect = lambda x: fix_parameters
    mock_OutputFolders = configure_mock_OutputFolders(mock_OutputFolders)
    mock_OutputFolders.side_effect = lambda x: mock_OutputFolders
    mock_path.exists = lambda x: True
    mock_Metadata = configure_mock_Metadata(mock_Metadata)
    mock_Images = configure_mock_images(mock_Images)
    mock_get_batch.side_effect = lambda i_batch, batch_size, im: (int(i_batch * batch_size), min(int((i_batch + 1 ) * batch_size), im.n_images), Mock())

    mock_torch.save = Mock(side_effect=lambda x, y: None)

    # Call convert_particle_stacks_from_star_files with 2 files. One has 7 images and one has 5.
    # With batch_size = 5, we expect the first file to take 2 stacks and the
    # second file to make one more stack.
    convert_particle_stacks_from_star_files(
        params_input='any params file name',
        particle_file_list = ["particle one", "particle two"],
        star_file_list = ["star_one", "star_two"],
        folder_output="folder_output",
        batch_size=5,
        flag_plots=False
    )
    # Asert that we read from each of the listed star files.
    mock_Metadata.from_star_file.assert_has_calls([
        call("star_one", defocus_is_degree=True, phase_shift_is_degree=True),
        call("star_two", defocus_is_degree=True, phase_shift_is_degree=True)
    ])
    # assert that we read from each of the listed MRC files.
    assert isinstance(mock_Images.from_mrc, Mock)
    mock_Images.from_mrc.assert_has_calls([
        call("particle one", pixel_size=None),
        call("particle two", pixel_size=None)
    ])

    # Assert that we fetched the output file structure once for each of the 3
    # expected passes (stacks 0, 1, and 2).
    assert isinstance(mock_OutputFolders.get_output_filenames, Mock)
    mock_OutputFolders.get_output_filenames.assert_has_calls([call(0), call(1), call(2)])
    # Assert that we called the normalization the right number of times.
    # NOTE: A more precise test would control the values returned for im and im_batch
    # and assert that those show up in the call to _do_image_normalization.
    assert mock_do_norm.call_count == 3

    # Further assert that we went through the loop the way we thought we did:
    # i.e. that we called take_range (to batch the metadata values) with the expected
    # ranges, and that we called save_params the right number of times.
    assert isinstance(mock_Metadata.meta.take_range, Mock)
    assert mock_Metadata.meta.take_range.call_count == 3
    mock_Metadata.meta.take_range.assert_has_calls([call(0, 5), call(5, 7), call(0, 5)])
    assert mock_Metadata.meta.batch.save_params_star.call_count == 3


@patch(f"{PKG}.torch")
@patch(f"{PKG}._plot_images")
@patch(f"{PKG}._do_image_normalization")
@patch(f"{PKG}._get_unbuffered_batch")
@patch(f"{PKG}.Images")
@patch(f"{PKG}._Metadata")
@patch(f"{PKG}.path")
@patch(f"{PKG}.OutputFolders")
@patch(f"{PKG}.ensure_parameters")
@patch("builtins.print")
def test_convert_particle_stacks_when_max_batch_exceeds_data(
    mock_print: Mock,
    mock_ensure_params: Mock,
    mock_OutputFolders: Mock,
    mock_path: Mock,
    mock_Metadata: Mock,
    mock_Images: Mock,
    mock_get_batch: Mock,
    mock_do_norm: Mock,
    mock_plot: Mock,
    mock_torch: Mock,
):
    mock_ensure_params.side_effect = lambda x: fix_parameters
    mock_OutputFolders = configure_mock_OutputFolders(mock_OutputFolders)
    mock_OutputFolders.side_effect = lambda x: mock_OutputFolders
    mock_path.exists = lambda x: True

    # mock_meta = get_mock_metadata()
    # mock_Metadata.from_star_file = Mock(side_effect=get_mock_from_star_file_side_effect(mock_meta))
    mock_Metadata = configure_mock_Metadata(mock_Metadata)
    mock_Images = configure_mock_images(mock_Images)
    mock_get_batch.side_effect = lambda i_batch, batch_size, im: (int(i_batch * batch_size), min(int((i_batch + 1 ) * batch_size), im.n_images), Mock())

    mock_torch.save = Mock(side_effect=lambda x, y: None)

    convert_particle_stacks_from_star_files(
        params_input='any params file name',
        particle_file_list = ["particle one"],
        star_file_list = ["star_one"],
        folder_output="folder_output",
        batch_size=5000,
        pixel_size=1.0,
        flag_plots=False
    )
    # Assertions largely match the test above, except that here we are assuming
    # a batch size that much exceeds the image count
    # (which is 7, since we only use the first mrc file return.)
    mock_Metadata.from_star_file.assert_called_once_with(
        "star_one",
        defocus_is_degree=True,
        phase_shift_is_degree=True
    )
    assert isinstance(mock_Images.from_mrc, Mock)
    mock_Images.from_mrc.assert_called_once()

    assert isinstance(mock_OutputFolders.get_output_filenames, Mock)
    mock_OutputFolders.get_output_filenames.assert_called_once_with(0)
    assert mock_do_norm.call_count == 1

    assert isinstance(mock_Metadata.meta.take_range, Mock)
    assert mock_Metadata.meta.take_range.call_count == 1
    mock_Metadata.meta.take_range.assert_called_once_with(0, 7)
    assert mock_Metadata.meta.batch.save_params_star.call_count == 1

    mock_print.assert_called() # for reporting the pixel size and params values


def test_convert_particle_stacks_throws_on_nonpositive_pixel_size():
    with raises(ValueError):
        convert_particle_stacks_from_star_files("any params", pixel_size=-1.)


def test_convert_particle_stacks_throws_on_bad_parameters():
    with raises(ValueError):
        convert_particle_stacks_from_star_files(params_input="Non-extant file")


@patch(f"{PKG}.path")
@patch(f"{PKG}.OutputFolders")
@patch(f"{PKG}.ensure_parameters")
def test_convert_particle_stacks_throws_on_missing_file(
    mock_ensure_params: Mock,
    mock_OutputFolders: Mock,
    mock_path: Mock
):
    mock_ensure_params.side_effect = lambda x: fix_parameters
    mock_OutputFolders = configure_mock_OutputFolders(mock_OutputFolders)
    mock_OutputFolders.side_effect = lambda x: mock_OutputFolders
    mock_path.exists = Mock(side_effect=[False])

    with raises(ValueError, match="file not found"):
        convert_particle_stacks_from_star_files(
            params_input='any params file name',
            particle_file_list = ["particle one", "particle two"],
            star_file_list = ["star_one", "star_two"],
            folder_output="folder_output",
            batch_size=5,
            flag_plots=False
        )


#################
## from cryosparc

# Testing this here b/c we'll want to mock it out in subsequent unit tests
@patch(f"{PKG}.torch")
@patch(f"{PKG}._plot_images")
@patch("builtins.print")
def test_drain_buffer(mock_print: Mock, mock_plot: Mock, mock_torch: Mock):
    mock_batch = Mock()
    mock_batch.save_params = Mock()
    mock_buffer = Mock()
    mock_buffer.pop_batch = Mock(side_effect=lambda x: mock_batch)

    batch_size = 15
    available_size = 10

    mock_img_buffer = Mock()
    mock_phys_batch = Mock()
    mock_fourier_batch = Mock()
    mock_phys_batch.shape = (available_size, 265)
    mock_img_buffer.pop_imgs = Mock(side_effect=lambda x: (mock_phys_batch, mock_fourier_batch))

    mock_output_dirs = Mock()
    mock_fns = Mock()
    mock_fns.phys_stack = "phys stack"
    mock_fns.fourier_stack = "fourier stack"
    mock_fns.params_filename = "params name"
    mock_output_dirs.get_output_filenames = Mock(side_effect=lambda x: mock_fns)

    mock_Images = Mock()
    mock_torch.save = Mock()

    # Act
    i_restack = 17
    flag_plots = True
    _drain_buffer(
        batch_size,
        i_restack,
        mock_buffer,
        mock_img_buffer,
        mock_output_dirs,
        mock_Images,
        flag_plots
    )

    mock_buffer.pop_batch.assert_called_once_with(batch_size)
    mock_img_buffer.pop_imgs.assert_called_once_with(batch_size)
    mock_print.assert_called_once_with(f"Stacking {available_size} images")
    mock_plot.assert_called_once_with(flag_plots, mock_Images, mock_output_dirs, i_restack, phys_img=mock_phys_batch, fourier_img=mock_fourier_batch)
    assert isinstance(mock_torch.save, Mock)
    mock_torch.save.assert_has_calls([call(mock_phys_batch, mock_fns.phys_stack), call(mock_fourier_batch, mock_fns.fourier_stack)])

    mock_batch.save_params.assert_called_with(mock_fns.params_filename, available_size, mock_Images)


@patch(f"{PKG}.path")
@patch("builtins.print")
def test_get_filenames_and_image_selection_indices(
    mock_print: Mock,
    mock_path: Mock,
):
    mock_path.join = Mock(side_effect=lambda x, y: f"{x}/{y}")
    mock_path.exists = Mock(side_effect=lambda x: False if "does-not-exist" in x else True)
    cryosparc_folder = "cryosparc"
    files = np.array([b"does-not-exist", b"file1", b">file2", b"file1"])
    expected_file_roots = ["does-not-exist", "file1", "file2"]
    indices = np.array([[1], [3], [2], [4]])

    mock_metadata = Mock()
    mock_metadata.cs_files = files
    mock_metadata.cs_idxs = indices

    res = _get_filenames_and_image_selection_indices(mock_metadata, cryosparc_folder)

    expected_files = [f"{cryosparc_folder}/{expected_file_roots[i]}" for i in range(len(expected_file_roots))]
    # Assert: skips non-extant files
    mock_print.assert_called_once_with(f"File {expected_files[0]} does not exist, skipping...")

    # Assert: skips first character of filename if that's a >
    assert isinstance(mock_path.exists, Mock)
    mock_path.exists.assert_has_calls(
        [call(expected_files[0]), call(expected_files[1]), call(expected_files[2])],
        any_order=True
    )

    # Assert: returns mrc_file, indices tuples
    assert res[0][0] == "cryosparc/file2"
    assert res[1][0] == "cryosparc/file1"
    npt.assert_equal(res[0][1], np.array([[2]]))
    npt.assert_equal(res[1][1], np.array([[3], [4]]))


def configure_mock_Metadata_cs(
    mock_Metadata: Mock,
    files: list[str],
    idxs: list[int],
    use_empty_pixel_size: bool = False
):
    fixture_metadata = _Metadata(
        defocusU=np.arange(1000, dtype="float"),
        defocusV=np.arange(1000, dtype="float"),
        defocusAngle=np.arange(1000, dtype="float"),
        phaseShift=np.arange(1000, dtype="float"),
        sphericalAberration=1.,
        voltage=1.,
        amplitudeContrast=1.,
        cs_files=np.array(files),
        cs_idxs=np.array(idxs),
        cs_pixel_size=None if use_empty_pixel_size else np.array([1., 1.])
    )
    def mock_from_cs(filename, get_fs_data=False):
        return fixture_metadata
    mock_Metadata.from_cryospark_file = Mock(side_effect=mock_from_cs)
    return (mock_Metadata, fixture_metadata)


def make_mock_Images(length: int):
    mock_Images = Mock()
    mock_Images.images_phys = length
    mock_Images.images_fourier = length
    mock_Images.n_images = length
    return mock_Images


def mock_drain(batch_size, i_restack, metadata_buffer, img_buffer, _output_dirs, im, flag_plots):
    img_buffer.stack_size -= batch_size
    img_buffer.stack_size = max(img_buffer.stack_size, 0)


def configure_mock_imgbuffer(mock_ImgBuffer: Mock):
    mock_ImgBuffer.stack_size = 0
    def append(size: int):
        mock_ImgBuffer.stack_size += size
    mock_ImgBuffer.append_imgs = Mock(side_effect=lambda x, y: append(x))
    mock_ImgBuffer.side_effect = lambda: mock_ImgBuffer
    return mock_ImgBuffer


@patch(f"{PKG}._drain_buffer")
@patch(f"{PKG}._do_image_normalization")
@patch(f"{PKG}._make_Images_from_mrc_file")
@patch(f"{PKG}._get_filenames_and_image_selection_indices")
@patch(f"{PKG}.ImgBuffer")
@patch(f"{PKG}._Metadata")
@patch(f"{PKG}.path")
@patch(f"{PKG}.OutputFolders")
@patch(f"{PKG}.ensure_parameters")
@patch("builtins.print")
def test_convert_particle_stacks_from_cryosparc(
    mock_print: Mock,
    mock_ensure_params: Mock,
    mock_OutputFolders: Mock,
    mock_path: Mock,
    mock_Metadata: Mock,
    mock_ImgBuffer: Mock,
    mock_get_fns: Mock,
    mock_get_imgs: Mock,
    mock_do_norm: Mock,
    mock_drain_buffer: Mock
):
    mock_ensure_params.side_effect = lambda x: fix_parameters
    mock_OutputFolders = configure_mock_OutputFolders(mock_OutputFolders)
    mock_OutputFolders.side_effect = lambda x: mock_OutputFolders
    mock_path.exists = lambda x: True
    mock_ImgBuffer = configure_mock_imgbuffer(mock_ImgBuffer)
    mock_drain_buffer.side_effect = mock_drain

    batch_size = 5
    # expect file 1 to have 3 imgs, file 2 to have 10 imgs, and file 3 to have 4
    # with batch size 3 this means we should process 1, then 2 (now @ 13 files),
    # drain twice (now @ 3), then process file 3 (now 7), drain one more batch,
    # then drain remainder
    nonunique_files = ["file 1", "file 2", "empty file", "file 3", "file 2"]
    nonunique_idxs = [1, 2, 0, 3, 2]
    (mock_Metadata, fix_meta) = configure_mock_Metadata_cs(mock_Metadata, nonunique_files, nonunique_idxs)

    mock_get_imgs.side_effect = [make_mock_Images(3), make_mock_Images(10), make_mock_Images(4)]
    indices_result = [1 + np.arange(3), 6 + np.arange(10), 18 + np.arange(4)]
    mock_get_fns_result = [
        ("cryosparc folder/file 1", indices_result[0] ),
        ("cryosparc folder/file 2", indices_result[1] ),
        ("cryosparc folder/file 3", indices_result[2] )
    ]
    mock_get_fns.return_value = mock_get_fns_result


    convert_particle_stacks_from_cryosparc(
        params_input="any file name",
        file_cs="cryosparc file",
        folder_cryosparc="cryosparc folder",
        folder_output="output folder",
        batch_size=batch_size,
        downsample_physical=2
    )

    # assert: processes all files
    mock_get_fns.assert_called_once_with(fix_meta, "cryosparc folder")
    assert mock_get_imgs.call_count == 3
    get_imgs_calls = mock_get_imgs.call_args_list
    for i, c in enumerate(get_imgs_calls):
        positional_params = c[0]
        assert positional_params[0] == mock_get_fns_result[i][0]
        npt.assert_equal(positional_params[1], mock_get_fns_result[i][1])
    
    # assert: normalizes & batches each non-None img
    assert mock_do_norm.call_count == 3
    assert isinstance(mock_ImgBuffer.append_imgs, Mock)
    mock_ImgBuffer.append_imgs.assert_has_calls([call(3, 3), call(10, 10), call(4, 4)])

    # assert: calls drain_buffer for complete batches
    assert mock_drain_buffer.call_count == 4
    drain_calls = mock_drain_buffer.call_args_list
    for i in range(4):
        assert drain_calls[i][0][1] == i

    mock_print.assert_called()


@patch(f"{PKG}._drain_buffer")
@patch(f"{PKG}._do_image_normalization")
@patch(f"{PKG}._make_Images_from_mrc_file")
@patch(f"{PKG}._get_filenames_and_image_selection_indices")
@patch(f"{PKG}.ImgBuffer")
@patch(f"{PKG}._Metadata")
@patch(f"{PKG}.path")
@patch(f"{PKG}.OutputFolders")
@patch(f"{PKG}.ensure_parameters")
@patch("builtins.print")
def test_convert_particle_stacks_from_cryosparc_quits_at_max_stacks(
    mock_print: Mock,
    mock_ensure_params: Mock,
    mock_OutputFolders: Mock,
    mock_path: Mock,
    mock_Metadata: Mock,
    mock_ImgBuffer: Mock,
    mock_get_fns: Mock,
    mock_get_imgs: Mock,
    mock_do_norm: Mock,
    mock_drain_buffer: Mock
):
    mock_ensure_params.side_effect = lambda x: fix_parameters
    mock_OutputFolders = configure_mock_OutputFolders(mock_OutputFolders)
    mock_OutputFolders.side_effect = lambda x: mock_OutputFolders
    mock_path.exists = lambda x: True
    mock_ImgBuffer = configure_mock_imgbuffer(mock_ImgBuffer)
    mock_drain_buffer.side_effect = mock_drain

    batch_size = 5
    nonunique_files = ["file 1"]
    nonunique_idxs = [1]
    (mock_Metadata, _) = configure_mock_Metadata_cs(mock_Metadata, nonunique_files, nonunique_idxs)

    mock_get_fns.return_value = [("foo", 12 + np.arange(30))]
    mock_get_imgs.side_effect = [make_mock_Images(30)]

    convert_particle_stacks_from_cryosparc(
        params_input="any file name",
        file_cs="cryosparc file",
        pixel_size=1.,
        folder_cryosparc="cryosparc folder",
        folder_output="output folder",
        batch_size=batch_size,
        downsample_physical=2,
        n_stacks_max=1
    )
    mock_drain_buffer.assert_called_once()


def test_convert_particle_stacks_from_cryosparc_throws_on_bad_params():
    with raises(ValueError):
        convert_particle_stacks_from_cryosparc(
            params_input="Non-extant file",
            file_cs="does not matter"
        )


@patch(f"{PKG}._Metadata")
@patch(f"{PKG}.path")
@patch(f"{PKG}.OutputFolders")
@patch(f"{PKG}.ensure_parameters")
@patch("builtins.print")
def test_convert_particle_stacks_from_cryosparc_throws_on_missing_pixel_size(
    mock_print: Mock,
    mock_ensure_params: Mock,
    mock_OutputFolders: Mock,
    mock_path: Mock,
    mock_Metadata: Mock,
):
    mock_ensure_params.side_effect = lambda x: fix_parameters
    mock_OutputFolders = configure_mock_OutputFolders(mock_OutputFolders)
    mock_OutputFolders.side_effect = lambda x: mock_OutputFolders
    mock_path.exists = lambda x: True
    (mock_Metadata, _) = configure_mock_Metadata_cs(mock_Metadata, ["file"], [1], use_empty_pixel_size=True)
    with raises(ValueError, match="Pixel size was not set"):
        convert_particle_stacks_from_cryosparc(
            params_input="Non-extant file",
            file_cs="does not matter"
        )


############################
# from cryosparc, restacking


@patch(f"{PKG}._drain_buffer")
@patch(f"{PKG}._do_image_normalization")
@patch(f"{PKG}.Images")
@patch(f"{PKG}.ImgBuffer")
@patch(f"{PKG}._Metadata")
@patch(f"{PKG}.JobPaths")
@patch(f"{PKG}.OutputFolders")
@patch(f"{PKG}.ensure_parameters")
@patch("builtins.print")
def test_convert_particle_stacks_from_cs_restack(
    mock_print: Mock,
    mock_ensure_params: Mock,
    mock_OutputFolders: Mock,
    mock_JobPath: Mock,
    mock_Metadata: Mock,
    mock_ImgBuffer: Mock,
    mock_Images: Mock,
    mock_do_norm: Mock,
    mock_drain_buffer: Mock
):
    mock_ensure_params.side_effect = lambda x: fix_parameters
    mock_OutputFolders = configure_mock_OutputFolders(mock_OutputFolders)
    mock_OutputFolders.side_effect = lambda x: mock_OutputFolders
    mock_JobPath.side_effect = lambda x, y: mock_JobPath
    mock_JobPath.file_cs = "cs file"
    files = ["file1", "file2", "file3", None, "file4"]
    mock_JobPath.get_mrc_filename = Mock(side_effect=[x for x in files])
    mock_ImgBuffer = configure_mock_imgbuffer(mock_ImgBuffer)
    mock_drain_buffer.side_effect = mock_drain
    (mock_Metadata, fix_meta) = configure_mock_Metadata_cs(mock_Metadata, [], [])

    mock_imgs = [make_mock_Images(3), make_mock_Images(10), make_mock_Images(4)]
    mock_Images.from_mrc = Mock(side_effect=[x for x in mock_imgs])

    job_number = 16
    batch_size = 5
    downsample = 2
    convert_particle_stacks_from_cryosparc_restack(
        params_input="does not matter",
        folder_cryosparc="cryosparc folder",
        job_number=job_number,
        folder_output="output folder",
        batch_size=batch_size,
        downsample_physical=downsample,
        pixel_size=1.
    )

    # Assert we fetched the right metadata
    mock_JobPath.assert_called_once_with("cryosparc folder", job_number)
    mock_Metadata.from_cryospark_file.assert_called_once_with("cs file")

    # Assert we processed all the files
    assert isinstance(mock_Images.from_mrc, Mock)
    assert mock_Images.from_mrc.call_count == 3
    from_mrc_calls = mock_Images.from_mrc.call_args_list
    for i in range(mock_Images.from_mrc.call_count):
        expected_path = files[i]
        assert from_mrc_calls[i][0][0] == expected_path

    # Assert we called normalization for each image batch
    assert mock_do_norm.call_count == 3
    norm_calls = mock_do_norm.call_args_list
    for i in range(len(norm_calls)):
        assert norm_calls[i][0][0] == mock_imgs[i]
        assert norm_calls[i][0][3] == downsample

    # Assert we appended the right number of images each time
    mock_ImgBuffer.append_imgs.assert_has_calls([call(3, 3), call(10, 10), call(4, 4)])

    # TODO: It might actually be worth mocking the metadata buffer
    # so we can assert that the right stuff got appended...

    # assert _drain_buffer called the right number of times
    assert mock_drain_buffer.call_count == 4
    drain_calls = mock_drain_buffer.call_args_list
    for i in range(len(drain_calls)):
        assert drain_calls[i][0][1] == i
    mock_print.assert_called()


@patch(f"{PKG}._drain_buffer")
@patch(f"{PKG}._do_image_normalization")
@patch(f"{PKG}.Images")
@patch(f"{PKG}.ImgBuffer")
@patch(f"{PKG}._Metadata")
@patch(f"{PKG}.JobPaths")
@patch(f"{PKG}.OutputFolders")
@patch(f"{PKG}.ensure_parameters")
@patch("builtins.print")
def test_convert_particle_stacks_from_cs_restack_honors_max_stacks(
    mock_print: Mock,
    mock_ensure_params: Mock,
    mock_OutputFolders: Mock,
    mock_JobPath: Mock,
    mock_Metadata: Mock,
    mock_ImgBuffer: Mock,
    mock_Images: Mock,
    mock_do_norm: Mock,
    mock_drain_buffer: Mock
):
    mock_ensure_params.side_effect = lambda x: fix_parameters
    mock_OutputFolders = configure_mock_OutputFolders(mock_OutputFolders)
    mock_OutputFolders.side_effect = lambda x: mock_OutputFolders
    mock_JobPath.side_effect = lambda x, y: mock_JobPath
    mock_JobPath.file_cs = "cs file"
    files = ["file1", "file2", "file3", None, "file4"]
    mock_JobPath.get_mrc_filename = Mock(side_effect=[x for x in files])
    mock_ImgBuffer = configure_mock_imgbuffer(mock_ImgBuffer)
    mock_drain_buffer.side_effect = mock_drain
    (mock_Metadata, fix_meta) = configure_mock_Metadata_cs(mock_Metadata, [], [])

    mock_imgs = [make_mock_Images(3), make_mock_Images(10), make_mock_Images(4)]
    mock_Images.from_mrc = Mock(side_effect=[x for x in mock_imgs])

    job_number = 16
    batch_size = 5
    downsample = 2
    convert_particle_stacks_from_cryosparc_restack(
        params_input="does not matter",
        folder_cryosparc="cryosparc folder",
        job_number=job_number,
        folder_output="output folder",
        batch_size=batch_size,
        downsample_physical=downsample,
        pixel_size=1.,
        n_stacks_max=1
    )

    mock_drain_buffer.assert_called_once()


@patch(f"{PKG}.ensure_parameters")
@patch("builtins.print")
def test_convert_particle_stacks_from_cs_restack_throws_on_nonpositive_pixel_size(
    mock_print: Mock,
    mock_ensure_params: Mock
):
    mock_ensure_params.side_effect = lambda x: fix_parameters
    with raises(ValueError, match="pixel size must be positive"):
        convert_particle_stacks_from_cryosparc_restack(
            params_input="does not matter"
        )
