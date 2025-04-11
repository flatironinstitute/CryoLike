from pytest import raises, mark
import torch
from unittest.mock import call, patch, Mock
from os import path

import numpy as np
import numpy.testing as npt

from cryolike.convert_particle_stacks.particle_stacks_converter import (
    _ensure_files_exist,
    _get_filenames_and_indices,
    StarfileInput,
    Indexed,
    SequentialCryosparc,
)

# the "get_base_converter" helper is used throughout here, because it
# wraps the constructor of the ParticleStackConverter object we actually
# want to test, but prevents creating a bunch of output directories on
# the filesystem.
from conversion_fixtures import (
    FIX_IMG_DESC,
    FIX_OUTPUT_DIR,
    get_base_converter,
    make_datasource,
    make_mock_imagestack,
    make_lens_descriptor
)

PKG = "cryolike.convert_particle_stacks.particle_stacks_converter"

## Generic functions

@patch(f"{PKG}.path.exists")
def test_ensure_files_exist(exists: Mock):
    good = ['file1', 'file2', 'file3']
    exists.side_effect = lambda x: x in good

    passes = ['file1', 'file3']
    _ensure_files_exist(passes)
    assert exists.call_count == 2

    with raises(ValueError, match='file not found'):
        _ensure_files_exist(['badfile', 'file2'])


@patch(f"{PKG}.path")
@patch("builtins.print")
def test_get_filenames_and_indices(_print: Mock, path: Mock):
    path.join = Mock(side_effect=lambda x, y: f"{x}/{y}")
    path.basename = Mock(side_effect=lambda x: x) # no need to do anything fancier for this example
    path.exists = Mock(side_effect=lambda x: False if "does-not-exist" in x else True)
    cryosparc_folder = "cryosparc"
    files = np.array([b"does-not-exist", b"file1", b">file2", b"file1"])
    expected_file_roots = ["does-not-exist", "file1", "file2"]
    indices = np.array([1, 3, 2, 4])

    metadata = Mock()
    metadata.files = files
    metadata.idxs = indices

    res = _get_filenames_and_indices(metadata, cryosparc_folder)

    expected_files = [f"{cryosparc_folder}/{expected_file_roots[i]}" for i in range(len(expected_file_roots))]
    # Assert: skips non-extant files
    _print.assert_called_once_with(f"File {expected_files[0]} does not exist, skipping...")

    # Assert: skips first character of filename if that's a >
    assert isinstance(path.exists, Mock)
    path.exists.assert_has_calls(
        [call(expected_files[0]), call(expected_files[1]), call(expected_files[2])],
        any_order=True
    )

    expected_returns = [
        (expected_files[1], np.array([3, 4]), np.array([1, 3])),
        (expected_files[2], np.array([2]), np.array([2]))
    ]

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


@patch(f"{PKG}.path")
@patch("builtins.print")
def test_get_filenames_and_indices_throws_when_no_files(_print: Mock, path: Mock):
    path.exists = Mock(return_value=False)
    assert isinstance(path.exists, Mock)
    files = np.array([b"myfiles/foo", b"myfiles/bar"])
    lens_desc = Mock()
    lens_desc.files = files
    lens_desc.idxs = np.array([1, 5])

    with raises(ValueError, match="None of the MRC files"):
        _ = _get_filenames_and_indices(lens_desc)
    args = [x[0][0] for x in path.exists.call_args_list]
    assert "myfiles/foo" in args
    assert "myfiles/bar" in args


def test_ctor_defaults():
    res = get_base_converter()
    assert res.inputs_buffer is not None
    assert len(res.inputs_buffer) == 0
    assert res.img_desc == FIX_IMG_DESC
    assert res.out_dirs.folder_output_particles_fft.startswith(FIX_OUTPUT_DIR)
    assert res.max_stacks == -1
    assert res.pixel_size is None


@patch("builtins.print")
def test_ctor_params(_print: Mock):
    n_stacks_max = 20
    pixel_size = 5.
    downsample = 2
    flag_plots = False

    res = get_base_converter(
        n_stacks_max=n_stacks_max,
        pixel_size=pixel_size,
        downsample_factor=downsample,
        flag_plots=flag_plots
    )
    assert res.max_stacks == n_stacks_max
    assert res.pixel_size is not None
    npt.assert_allclose(res.pixel_size, np.ones((2,)) * pixel_size)
    assert res.downsample_factor == downsample
    assert res.output_plots == flag_plots
    _print.assert_called_once()
    _print.reset_mock()

    # Test with float-array pixel size
    np_pixel_size = np.ones((3,)) * 2.
    res = get_base_converter(
        pixel_size=np_pixel_size
    )
    assert res.pixel_size is not None


def test_ctor_errors():
    with raises(ValueError, match="Invalid value for pixel size"):
        _ = get_base_converter(pixel_size=-2.)
    
    with raises(NotImplementedError):
        _ = get_base_converter(skip_exist=True)


def test_can_load_cryosparc():
    sut = get_base_converter()
    
    # passes with empty buffer
    assert sut._can_load_cryosparc()

    # passes when only starfile items
    sut.inputs_buffer.append(make_datasource('starfile', 'one'))
    sut.inputs_buffer.append(make_datasource('starfile', 'two'))
    assert sut._can_load_cryosparc()

    # fails when contains non-starfile items
    sut.inputs_buffer.append(make_datasource('indexed_cryosparc', 'three'))
    assert not sut._can_load_cryosparc()


@mark.parametrize("ref_pixel_size,manual_pixel_size", [(None, None), (1., None), (None, 1.), (1., 1.), (1., 2.)])
def test_confirm_pixel_size(ref_pixel_size, manual_pixel_size):#, _print: Mock):
    lens_desc = Mock()
    lens_desc.ref_pixel_size = ref_pixel_size
    sut = get_base_converter(pixel_size=manual_pixel_size)
    # How this is set depends on the input file; we'd like to keep the test a bit more isolated
    sut.lens_desc = lens_desc

    if ref_pixel_size is None and manual_pixel_size is None:
        with raises(ValueError, match="Pixel size was never set"):
            sut._confirm_pixel_size(False)
        return
    if ref_pixel_size is not None and manual_pixel_size is not None:
        if ref_pixel_size != manual_pixel_size:
            with raises(ValueError, match="does not match record pixel size"):
                sut._confirm_pixel_size(False)
            with patch("builtins.print") as _print:
                sut._confirm_pixel_size(True)
                assert sut.pixel_size is not None
                npt.assert_allclose(sut.pixel_size, ref_pixel_size)
                _print.assert_called_once()

    sut._confirm_pixel_size(False)
    assert sut.pixel_size is not None

    if manual_pixel_size is None or manual_pixel_size == ref_pixel_size:
        npt.assert_allclose(sut.pixel_size, ref_pixel_size)
    elif ref_pixel_size is None:
        npt.assert_allclose(sut.pixel_size, manual_pixel_size)


@patch(f"{PKG}._ensure_files_exist")
def test_prepare_star_files(_ensure_files: Mock):
    def f(files: list[str]):
        if 'badfile' in files:
            raise ValueError('file not found')

    _ensure_files.side_effect = f
    badlist = ['goodfile', 'badfile']
    goodlist = ['file_1', 'file_2']
    goodlist_mrc = [f"{x}.mrc" for x in goodlist]
    goodlist_star = [f"{x}.star" for x in goodlist]
    sut = get_base_converter()

    with raises(ValueError, match="not found"):
        sut.prepare_star_files(badlist, goodlist)
    with raises(ValueError, match="not found"):
        sut.prepare_star_files(goodlist, badlist)

    sut.prepare_star_files(goodlist_mrc, goodlist_star)
    sut.prepare_star_files(goodlist_mrc, goodlist_star, False, False)
    assert len(sut.inputs_buffer) == 4
    res: list[StarfileInput] = []
    for x in sut.inputs_buffer:
        assert x[0] == 'starfile'
        assert isinstance(x[1], StarfileInput)
        res.append(x[1])
    assert res[0].defocus_is_degree
    assert res[0].phase_shift_is_degree
    assert res[0].particle_file == goodlist_mrc[0]
    assert res[0].star_file == goodlist_star[0]

    assert not res[3].defocus_is_degree
    assert not res[3].phase_shift_is_degree
    assert res[3].particle_file == goodlist_mrc[1]
    assert res[3].star_file == goodlist_star[1]


@patch(f"{PKG}._get_filenames_and_indices")
@patch(f"{PKG}.LensDescriptor.from_indexed_starfile")
def test_prepare_indexed_starfile(from_file: Mock, get_names: Mock):
    lens_desc = Mock()
    lens_desc.ref_pixel_size = .4
    from_file.return_value = lens_desc

    files_indices = [
        make_datasource('indexed_starfile', 'file1')[1],
        make_datasource('indexed_starfile', 'file2')[1]
    ]
    get_names.return_value = files_indices

    sut = get_base_converter()
    sut.prepare_indexed_star_file("starfile")

    assert sut.lens_desc == lens_desc
    assert sut.lens_desc_buffer.parent_descriptor == lens_desc
    assert len(sut.inputs_buffer) == 2
    recs: list[Indexed] = []
    for x in sut.inputs_buffer:
        assert x[0] == 'indexed_starfile'
        assert isinstance(x[1], Indexed)
        recs.append(x[1])
    for i, r in enumerate(recs):
        row_exp = files_indices[i]
        assert isinstance(row_exp, Indexed)
        assert r.mrc_file == files_indices[i][0]
        npt.assert_allclose(r.selected_img_indices, row_exp[1])
        npt.assert_allclose(r.selected_lensdesc_indices, row_exp[2])


def test_prepare_indexed_starfile_fails_if_nonempty_buffer():
    sut = get_base_converter()
    sut.inputs_buffer.append(make_datasource('indexed_starfile', 'foo'))
    with raises(ValueError, match="Refusing to batch"):
        _ = sut.prepare_indexed_star_file('some-starfile')


@mark.parametrize("override", [(True), (False)])
def test_prepare_indexed_starfile_tests_pixel_size(override: bool):
    sut = get_base_converter()
    sut._confirm_pixel_size = Mock(side_effect=ValueError("Mocked error"))

    with patch(f"{PKG}.LensDescriptor.from_indexed_starfile") as mock:
        with raises(ValueError, match="Mocked error"):
            sut.prepare_indexed_star_file("str", "str", ignore_manual_pixel_size=override)
        sut._confirm_pixel_size.assert_called_once_with(override)



@patch(f"{PKG}._get_filenames_and_indices")
@patch(f"{PKG}.LensDescriptor.from_cryosparc_file")
def test_prepare_indexed_cryosparc(from_file: Mock, get_names: Mock):
    lens_desc = Mock()
    lens_desc.ref_pixel_size = None
    lens_desc.files = 'file1'
    lens_desc.idxs = 12
    from_file.return_value = lens_desc
    manual_pixel_size = 5.

    files_indices = [
        ("file1", np.array([5,3,6,4,7,8,9,1,0,2]), np.arange((10))),
        ("file2", np.array([22, 23, 76]), np.arange((3)) + 10)
    ]
    get_names.return_value = files_indices
    file_cs = "my_cs_file"
    folder_cryosparc = "my_cryosparc_folder"

    sut = get_base_converter(pixel_size=manual_pixel_size)

    sut.prepare_indexed_cryosparc(file_cs, folder_cryosparc)
    assert sut.lens_desc == lens_desc
    assert sut.lens_desc_buffer.parent_descriptor == lens_desc
    assert len(sut.inputs_buffer) == 2
    recs: list[Indexed] = []
    for x in sut.inputs_buffer:
        assert x[0] == 'indexed_cryosparc'
        assert isinstance(x[1], Indexed)
        recs.append(x[1])
    for i, r in enumerate(recs):
        assert r.mrc_file == files_indices[i][0]
        npt.assert_allclose(r.selected_img_indices, files_indices[i][1])
    assert sut.pixel_size is not None
    npt.assert_allclose(sut.pixel_size, np.ones((2,)) * manual_pixel_size)
    from_file.assert_called_once_with(file_cs, get_fs_data=True)
    get_names.assert_called_once_with(lens_desc, folder_cryosparc)


def test_prepare_indexed_cryosparc_fails_if_nonempty_buffer():
    sut = get_base_converter()
    sut.inputs_buffer.append(make_datasource('indexed_cryosparc', 'foo'))
    with raises(ValueError, match="Refusing to batch"):
        _ = sut.prepare_indexed_cryosparc('csfile', 'csfolder')


@mark.parametrize("override", [(True), (False)])
def test_prepare_indexed_cryosparc_tests_pixel_size(override: bool):
    sut = get_base_converter()
    sut._confirm_pixel_size = Mock(side_effect=ValueError("Mocked error"))

    with patch(f"{PKG}.LensDescriptor.from_cryosparc_file") as mock:
        with raises(ValueError, match="Mocked error"):
            sut.prepare_indexed_cryosparc("str", "str", ignore_manual_pixel_size=override)
        sut._confirm_pixel_size.assert_called_once_with(override)


@patch(f"{PKG}.LensDescriptor.from_cryosparc_file")
def test_prepare_indexed_cryosparc_fails_if_missing_fs_data(from_file: Mock):
    lens_desc = Mock()
    lens_desc.files = None
    lens_desc.ref_pixel_size = 2.
    from_file.return_value = lens_desc
    sut = get_base_converter()

    with raises(AssertionError):
        sut.prepare_indexed_cryosparc('str', 'str')
    
    lens_desc.files = 'not none'
    lens_desc.idxs = None
    with raises(AssertionError):
        sut.prepare_indexed_cryosparc('str', 'str')


@patch(f"{PKG}.LensDescriptor.from_cryosparc_file")
@patch(f"{PKG}.JobPaths")
def test_sequential_cryosparc(JobPath: Mock, from_file: Mock):
    JobPath.side_effect = lambda x, y: JobPath
    JobPath.file_cs = "cs file"
    files = ["file1", "file2", "file3", None, "file4"]
    JobPath.get_mrc_filename = Mock(side_effect=[x for x in files])

    lens_desc = Mock()
    from_file.return_value = lens_desc
    folder_cryosparc = 'my_folder'
    job_number = 12

    sut = get_base_converter()


    sut.prepare_sequential_cryosparc(folder_cryosparc, job_number)

    JobPath.assert_called_once_with(folder_cryosparc, job_number)
    assert sut.lens_desc == lens_desc
    assert sut.lens_desc_buffer.parent_descriptor == lens_desc
    assert JobPath.get_mrc_filename.call_count == 4
    assert len(sut.inputs_buffer) == 3
    
    for i, x in enumerate(sut.inputs_buffer):
        assert x[0] == 'sequential_cryosparc'
        assert isinstance(x[1], SequentialCryosparc)
        assert x[1].mrc_file == files[i]


def test_prepare_sequential_cryosparc_fails_if_nonempty_buffer():
    sut = get_base_converter()
    sut.inputs_buffer.append(make_datasource('indexed_cryosparc', 'foo'))
    with raises(ValueError, match="Refusing to batch"):
        _ = sut.prepare_sequential_cryosparc('csfolder')


@patch("builtins.print")
def test_normalize_images(_print: Mock):
    sut = get_base_converter()
    im = Mock()
    im.images_phys = np.array(range(5))
    im.images_fourier = np.array(range(5))
    im.downsample_images_phys = Mock()
    im.center_physical_image_signal = Mock()
    im.transform_to_fourier = Mock()
    im.normalize_images_fourier = Mock()


    # test no downsample
    sut._transform_and_normalize_images(im)
    assert isinstance(im.downsample_images_phys, Mock)
    im.downsample_images_phys.assert_not_called()
    assert im.center_physical_image_signal.call_count == 1
    im.transform_to_fourier.assert_called_once_with(polar_grid=FIX_IMG_DESC.polar_grid, precision=FIX_IMG_DESC.precision)
    im.normalize_images_fourier.assert_called_once_with(ord=2, use_max=False)

    # test with downsample
    sut.downsample_factor = 3
    sut.downsample_type = "mean"
    sut._transform_and_normalize_images(im)
    im.downsample_images_phys.assert_called_once_with(3, "mean")

    assert _print.call_count == 4


@patch(f"{PKG}.LensDescriptor.from_starfile")
@patch(f"{PKG}.Images.from_mrc")
def test_load_starfile(from_mrc: Mock, from_starfile: Mock):
    sut = get_base_converter(pixel_size=1.)
    assert not sut._must_flush_buffer

    img_cnt = 10
    lens_desc = make_lens_descriptor(img_cnt)
    from_starfile.return_value = lens_desc

    stack = make_mock_imagestack(img_cnt)
    from_mrc.return_value = stack

    r = StarfileInput(particle_file="myfile.mrc", star_file="myfile.star", defocus_is_degree=True, phase_shift_is_degree=True)
    sut._load_starfile(r)
    assert sut.images_buffer.stack_size == img_cnt
    assert sut.lens_desc_buffer.stack_size == img_cnt
    assert sut.lens_desc_buffer.parent_descriptor == lens_desc
    assert sut._must_flush_buffer

    from_mrc.assert_called_once()
    assert from_mrc.call_args[0][0] == "myfile.mrc"
    from_starfile.assert_called_once()
    assert from_starfile.call_args[0][0] == "myfile.star"


@patch(f"{PKG}.Images.from_mrc")
def test_load_indexed_cryosparc(from_mrc: Mock):
    sut = get_base_converter(pixel_size=1.)
    img_cnt = 10
    stack = make_mock_imagestack(img_cnt)
    from_mrc.return_value = stack
    lens_desc = make_lens_descriptor(img_cnt)
    sut.lens_desc = lens_desc

    selected_indices = np.array([1, 3, 5])
    selections_cnt = len(selected_indices)

    r = Indexed(mrc_file="myfile.mrc", selected_img_indices=selected_indices, selected_lensdesc_indices=np.array([0, 1, 2]))
    sut._load_indexed(r)

    assert sut.images_buffer.stack_size == selections_cnt
    assert sut.lens_desc_buffer.stack_size == selections_cnt
    assert not sut._must_flush_buffer
    from_mrc.assert_called_once()
    assert from_mrc.call_args[0][0] == "myfile.mrc"


@patch(f"{PKG}.Images.from_mrc")
def test_load_sequential_cryosparc(from_mrc: Mock):
    sut = get_base_converter(pixel_size=1.)
    img_cnt = 10
    stack = make_mock_imagestack(img_cnt)
    from_mrc.return_value = stack
    lens_desc = make_lens_descriptor(img_cnt)
    sut.lens_desc = lens_desc

    r = SequentialCryosparc(mrc_file="myfile.mrc")

    sut._load_sequential_cryosparc(r)
    assert sut._stack_start_file == img_cnt
    assert not sut._must_flush_buffer
    assert sut.images_buffer.stack_size == img_cnt
    assert sut.lens_desc_buffer.stack_size == img_cnt
    from_mrc.assert_called_once()
    assert from_mrc.call_args[0][0] == "myfile.mrc"


@patch(f"{PKG}.save_combined_params")
@patch(f"{PKG}.torch.save")
@patch(f"builtins.print")
def test_write_batch(_print: Mock, tsave: Mock, csave: Mock):
    sut = get_base_converter(pixel_size=1.)
    sut._plot_imgs = Mock()
    stack_size = 10
    stack = make_mock_imagestack(stack_size)
    sut.images_buffer.append_imgs(stack)
    lens_desc = make_lens_descriptor(stack_size)
    sut.lens_desc = lens_desc
    sut.lens_desc_buffer.update_parent(sut.lens_desc)
    sut.lens_desc_buffer.enqueue(lens_desc.batch_whole())

    assert sut.images_buffer.stack_size == stack_size

    batch_size = 6
    sut._write_batch(batch_size)
    _print.assert_called_with("Stacking 6 images")
    assert tsave.call_count == 2
    sut._plot_imgs.assert_called_once()
    csave.assert_called_once()
    call_args = csave.call_args
    assert call_args[1]['n_imgs_this_stack'] == batch_size
    assert call_args[1]['overall_batch_start'] is None

    # repeat with the overall counter
    sut._plot_imgs.reset_mock()
    _print.reset_mock()
    sut.output_plots = False

    sut._write_batch(batch_size, True)
    sut._plot_imgs.assert_not_called()
    realized_size = stack_size - batch_size
    _print.assert_called_with(f"Stacking {realized_size} images")
    call_args = csave.call_args
    assert call_args[1]['n_imgs_this_stack'] == realized_size
    assert call_args[1]['overall_batch_start'] == 0
    assert sut._stack_absolute_index == realized_size


def test_write_batch_throws_on_mismatched_lengths():
    sut = get_base_converter(pixel_size=1.)
    stack_size = 100
    stack = make_mock_imagestack(stack_size)
    sut.images_buffer.append_imgs(stack)
    lens_desc = make_lens_descriptor(stack_size//2)
    sut.lens_desc = lens_desc
    sut.lens_desc_buffer.update_parent(sut.lens_desc)
    sut.lens_desc_buffer.enqueue(lens_desc.batch_whole())

    batch_size = (stack_size // 2) + 15
    with raises(ValueError, match="Length mismatch"):
        sut._write_batch(batch_size)


def test_convert_stacks():
    sut = get_base_converter(pixel_size=1., overwrite=False)
    img_cnt = 8
    batch_size = 5
    # With these numbers, we will load a stack, write 6 (have 3 left),
    # load another stack (now at 11), write 5, (6), write 5 (1), load 8 (9),
    # write 5 (4), then write 4
    stack = make_mock_imagestack(img_cnt)
    sut._load_starfile = Mock()
    sut._load_indexed = Mock()
    sut._load_indexed.side_effect = lambda x: sut.images_buffer.append_imgs(stack)
    sut._load_sequential_cryosparc = Mock()
    sut._load_sequential_cryosparc.side_effect = lambda x: sut.images_buffer.append_imgs(stack)
    sut._write_batch = Mock()
    sut._write_batch.side_effect = lambda x, y: sut.images_buffer.pop_imgs(x)

    row1 = make_datasource("indexed_cryosparc", "file1", 10)
    row2 = make_datasource("sequential_cryosparc", "file2")
    row3 = make_datasource("indexed_starfile", "file3", 10)
    sut.inputs_buffer.append(row1)
    sut.inputs_buffer.append(row2)
    sut.inputs_buffer.append(row3)

    # Act
    sut.convert_stacks(batch_size)

    # Assert
    assert sut.i_stacks == 5
    sut._load_starfile.assert_not_called()
    sut._load_indexed.assert_called_with(row3[1])
    assert sut._load_indexed.call_count == 2
    sut._load_sequential_cryosparc.assert_called_once_with(row2[1])
    assert sut._write_batch.call_count == sut.i_stacks


def test_convert_stacks_full_flushes_buffer_for_starfiles():
    sut = get_base_converter(pixel_size=1.)
    # With these numbers, if we were completely buffering, we should see
    # (10) -> 2 -> (12) -> 4 -> 0 = 3 calls to _write.
    # We will instead observe 4 because we write the 2 images rather than
    # re-buffering.
    img_cnt = 10
    batch_size = 8
    stack = make_mock_imagestack(img_cnt)
    def load_star():
        sut.images_buffer.append_imgs(stack)
        sut._must_flush_buffer = True
    sut._load_starfile = Mock()
    sut._load_starfile.side_effect = lambda x: load_star()
    sut._write_batch = Mock()
    sut._write_batch.side_effect = lambda x, y: sut.images_buffer.pop_imgs(x)

    row1 = make_datasource("starfile", "file1")
    row2 = make_datasource("starfile", "file2")
    sut.inputs_buffer.append(row1)
    sut.inputs_buffer.append(row2)

    sut.convert_stacks(batch_size)

    assert sut.i_stacks == 4
    assert sut._write_batch.call_count == sut.i_stacks
    assert sut._load_starfile.call_count == 2


def test_convert_stacks_full_flushes_buffer_for_cryosparc_when_requested():
    sut = get_base_converter(pixel_size=1.)
    img_cnt = 10
    batch_size = 8
    stack = make_mock_imagestack(img_cnt)
    sut._load_sequential_cryosparc = Mock()
    sut._load_sequential_cryosparc.side_effect = lambda x: sut.images_buffer.append_imgs(stack)
    sut._write_batch = Mock()
    sut._write_batch.side_effect = lambda x, y: sut.images_buffer.pop_imgs(x)

    row1 = make_datasource("sequential_cryosparc", "file1")
    row2 = make_datasource("sequential_cryosparc", "file2")
    sut.inputs_buffer.append(row1)
    sut.inputs_buffer.append(row2)

    sut.convert_stacks(batch_size, never_combine_input_files=True)

    assert sut.i_stacks == 4
    assert sut._write_batch.call_count == sut.i_stacks
    assert sut._load_sequential_cryosparc.call_count == 2
    calls = sut._load_sequential_cryosparc.call_args_list
    assert calls[0][0][0] == row1[1]
    assert calls[1][0][0] == row2[1]


@patch("builtins.print")
@patch(f"{PKG}.torch.save")
def test_convert_stacks_aborts_on_max_stacks(tsave: Mock, _print: Mock):
    img_cnt = 10
    batch_size = 3
    stack = make_mock_imagestack(img_cnt)
    
    sut = get_base_converter(pixel_size=1.)
    sut.max_stacks = 2
    sut._load_sequential_cryosparc = Mock()
    sut._load_sequential_cryosparc.side_effect = lambda x: sut.images_buffer.append_imgs(stack)
    sut._write_batch = Mock()
    sut._write_batch.side_effect = lambda x, y: sut.images_buffer.pop_imgs(x)
    sut.inputs_buffer.append(make_datasource("sequential_cryosparc", "file"))

    sut.convert_stacks(batch_size)

    assert sut.i_stacks == sut.max_stacks
    assert sut._write_batch.call_count == sut.max_stacks
    assert sut.images_buffer.stack_size > 0


@patch("builtins.print")
def test_convert_stacks_warns_on_empty_buffer(_print: Mock):
    sut = get_base_converter()
    sut.convert_stacks()
    _print.assert_called_once_with("Warning: you must prepare input files before running convert_stacks.")


@patch("builtins.print")
def test_convert_stacks_throws_on_no_pixel_size(_print: Mock):
    sut = get_base_converter()
    sut.inputs_buffer.append(make_datasource("sequential_cryosparc", "some-name"))
    # In this case, the pixel_size should have been set either at converter initialization
    # or while parsing the Cryosparc file. Since we skipped that, there isn't a pixel size,
    # and we expect an error
    with raises(ValueError, match="Pixel size was never set"):
        sut.convert_stacks()


@patch(f"{PKG}.plot_power_spectrum")
@patch(f"{PKG}.plot_images")
def test_plot_imgs(plot_imgs: Mock, plot_ps: Mock):
    sut = get_base_converter()
    phys = torch.arange(10) * 1.
    four = torch.arange(10) * 4.
    plot_root = sut.out_dirs.folder_output_plots

    # confirms: 1) plot image count limited by data size; 2) correct suffix
    sut._plot_imgs(phys, four)
    plot_imgs_calls = plot_imgs.call_args_list
    phys_call = plot_imgs_calls[0]
    npt.assert_allclose(phys_call[0][0], phys)
    assert phys_call[1]['grid'] == sut.img_desc.cartesian_grid
    assert phys_call[1]['n_plots'] == phys.shape[0]
    assert phys_call[1]['filename'] == path.join(plot_root, f"particles_phys_stack_{sut.i_stacks:06}.png")

    four_call = plot_imgs_calls[1]
    npt.assert_allclose(four_call[0][0], four)
    assert four_call[1]['grid'] == sut.img_desc.polar_grid
    assert four_call[1]['n_plots'] == four.shape[0]
    assert four_call[1]['filename'] == path.join(plot_root, f"particles_fourier_stack_{sut.i_stacks:06}.png")

    plot_ps.assert_called_once()
    ps_args = plot_ps.call_args
    assert ps_args[1]['filename_plot'] == path.join(plot_root, f"power_spectrum_stack_{sut.i_stacks:06}.png")

    # Try with different max imgs to plot & i_count
    plot_imgs.reset_mock()
    sut.i_stacks = 5
    sut.max_imgs_to_plot = phys.shape[0] - 2
    sut._plot_imgs(phys, four)
    four_call_kwargs = plot_imgs.call_args[1]
    assert four_call_kwargs['n_plots'] == sut.max_imgs_to_plot
    assert four_call_kwargs['filename'] == path.join(plot_root, f"particles_fourier_stack_{sut.i_stacks:06}.png")
