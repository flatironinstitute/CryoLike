from pytest import raises, mark
from unittest.mock import patch, Mock
from typing import Literal

import numpy as np
import numpy.testing as npt

from cryolike.file_conversions.particle_stacks_converter import (
    StarfileInput,
    Indexed,
    SequentialCryosparc,
)

from conversion_fixtures import (
    FIX_IMG_DESC,
    get_base_converter,
    make_datasource,
    make_mock_imagestack,
    make_lens_descriptor
)

PKG = "cryolike.file_conversions.particle_stacks_converter"

## Generic functions

def test_ctor_defaults(tmp_path):
    res = get_base_converter(tmp_path)
    assert res.inputs_buffer is not None
    assert len(res.inputs_buffer) == 0
    assert res.img_desc == FIX_IMG_DESC
    assert res.max_stacks == -1
    assert res.pixel_size is None


def test_ctor_params(tmp_path):
    n_stacks_max = 20
    pixel_size = 5.
    downsample = 2
    flag_plots = False

    with patch('builtins.print') as _print:
        res = get_base_converter(
            tmp_path,
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

    # Test with float-array pixel size
    np_pixel_size = np.ones((3,)) * 2.
    res = get_base_converter(
        tmp_path,
        pixel_size=np_pixel_size
    )
    assert res.pixel_size is not None


def test_ctor_errors(tmp_path):
    with raises(ValueError, match="Invalid value for pixel size"):
        _ = get_base_converter(tmp_path, pixel_size=-2.)
    
    with raises(NotImplementedError):
        _ = get_base_converter(tmp_path, skip_exist=True)


def test_can_load_cryosparc(tmp_path):
    sut = get_base_converter(tmp_path)
    
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
def test_confirm_pixel_size(tmp_path, ref_pixel_size, manual_pixel_size):#, _print: Mock):
    lens_desc = Mock()
    lens_desc.ref_pixel_size = ref_pixel_size
    sut = get_base_converter(tmp_path, pixel_size=manual_pixel_size)
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


def test_prepare_star_files(tmp_path):
    badlist = ['goodfile', 'badfile']
    goodlist = ['file_1', 'file_2']
    goodlist_mrc = [f"{x}.mrc" for x in goodlist]
    goodlist_star = [f"{x}.star" for x in goodlist]
    all_files = []
    all_files.extend(goodlist_mrc)
    all_files.extend(goodlist_star)

    sut = get_base_converter(tmp_path, seed_files = all_files)
    goodlist_mrc_final = [str(sut.filemgr._out_base / x) for x in goodlist_mrc]
    goodlist_star_final = [str(sut.filemgr._out_base / x) for x in goodlist_star]


    with raises(ValueError, match="not found"):
        sut.prepare_star_files(badlist, goodlist)
    with raises(ValueError, match="not found"):
        sut.prepare_star_files(goodlist, badlist)

    sut.prepare_star_files(goodlist_mrc_final, goodlist_star_final)
    sut.prepare_star_files(goodlist_mrc_final, goodlist_star_final, False, False)
    assert len(sut.inputs_buffer) == 4
    res: list[StarfileInput] = []
    for x in sut.inputs_buffer:
        assert x[0] == 'starfile'
        assert isinstance(x[1], StarfileInput)
        res.append(x[1])
    assert res[0].defocus_is_degree
    assert res[0].phase_shift_is_degree
    assert res[0].particle_file == goodlist_mrc_final[0]
    assert res[0].star_file == goodlist_star_final[0]

    assert not res[3].defocus_is_degree
    assert not res[3].phase_shift_is_degree
    assert res[3].particle_file == goodlist_mrc_final[1]
    assert res[3].star_file == goodlist_star_final[1]


### Prepare indexed

@mark.parametrize("ftype,manual_pixel_size,mrc_folder",
    [('starfile', None, ''),
     ('cryosparc', 5., 'cryosparc_folder')]
)
def test_prepare_indexed(tmp_path,
    ftype: Literal['starfile'] | Literal['cryosparc'],
    manual_pixel_size: float | None,
    mrc_folder: str,
):
    srctype = 'indexed_starfile' if ftype == 'starfile' else 'indexed_cryosparc'
    files_indices = [
        make_datasource(srctype, 'file1')[1],
        make_datasource(srctype, 'file2')[1]
    ]

    with (
        patch(f"{PKG}.get_filenames_and_indices") as get_names,
        patch(f"{PKG}.LensDescriptor") as mock_ld
    ):
        get_names.return_value = files_indices

        lens_desc = Mock()
        lens_desc.ref_pixel_size = None if manual_pixel_size is not None else .4
        lens_desc.files = 'file1'
        lens_desc.idxs = 12
        mock_ld.from_indexed_starfile = Mock(return_value=lens_desc)
        mock_ld.from_cryosparc_file = Mock(return_value=lens_desc)

        sut = get_base_converter(tmp_path, pixel_size=manual_pixel_size)
        sut.prepare_indexed_file("starfile", 'starfile', mrc_folder)

    assert sut.lens_desc == lens_desc
    assert sut.lens_desc_buffer.parent_descriptor == lens_desc
    assert len(sut.inputs_buffer) == 2
    if manual_pixel_size is not None:
        assert sut.pixel_size is not None
        npt.assert_allclose(sut.pixel_size, np.ones((2,)) * manual_pixel_size)

    recs: list[Indexed] = []
    for x in sut.inputs_buffer:
        assert x[0] == 'indexed'
        assert isinstance(x[1], Indexed)
        recs.append(x[1])
    for i, r in enumerate(recs):
        assert r.mrc_file == files_indices[i][0]
        row_exp = files_indices[i]
        assert isinstance(row_exp, Indexed)
        npt.assert_allclose(r.selected_img_indices, row_exp[1])
        npt.assert_allclose(r.selected_lensdesc_indices, row_exp[2])


def test_prepare_indexed_throws_on_bad_type(tmp_path):
    sut = get_base_converter(tmp_path)
    with raises(NotImplementedError, match="Unallowed"):
        _ = sut.prepare_indexed_file('filename', "DISALLOWED_FILE_TYPE") # type: ignore


@mark.parametrize("ftype", [('starfile'), ('cryosparc')])
def test_prepare_indexed_fails_if_nonempty_buffer(tmp_path, ftype):
    sut = get_base_converter(tmp_path)
    sut.inputs_buffer.append(make_datasource('indexed_starfile', 'foo'))
    with raises(ValueError, match="Refusing to batch"):
        _ = sut.prepare_indexed_file('some-starfile', ftype)


@mark.parametrize("ftype,override", [('starfile', True), ('starfile', False),
                                     ('cryosparc', True), ('cryosparc', False)])
def test_prepare_indexed_tests_pixel_size(tmp_path, ftype, override: bool):
    sut = get_base_converter(tmp_path)
    sut._confirm_pixel_size = Mock(side_effect=ValueError("Mocked error"))

    with patch(f"{PKG}.LensDescriptor") as mock:
        mock.from_indexed_starfile = Mock()
        mock.from_cryosparc = Mock()
        with raises(ValueError, match="Mocked error"):
            sut.prepare_indexed_file("str", ftype, "str", ignore_manual_pixel_size=override)
        sut._confirm_pixel_size.assert_called_once_with(override)


def test_prepare_indexed_fails_if_cryosparc_missing_fs_data(tmp_path):
    lens_desc = Mock()
    lens_desc.files = None
    lens_desc.ref_pixel_size = 2.
    sut = get_base_converter(tmp_path)

    with patch(f"{PKG}.LensDescriptor.from_cryosparc_file") as from_file:
        from_file.return_value = lens_desc
        with raises(AssertionError):
            sut.prepare_indexed_file('str', 'cryosparc', 'str')
    
        lens_desc.files = 'not none'
        lens_desc.idxs = None
        with raises(AssertionError):
            sut.prepare_indexed_file('str', 'cryosparc', 'str')


### Prepare sequential cryosparc

@mark.parametrize("buffer_empty", [(True), (False)])
def test_sequential_cryosparc(tmp_path, buffer_empty):
    lens_desc = Mock()
    folder_cryosparc = 'my_folder'
    job_number = 12

    sut = get_base_converter(tmp_path)

    if not buffer_empty:
        sut.inputs_buffer.append(make_datasource('indexed_cryosparc', 'foo'))
        with raises(ValueError, match="Refusing to batch"):
            _ = sut.prepare_sequential_cryosparc('folder')
        return

    files = ["file_1", "file_2", "file_3"]
    sut.filemgr = Mock()
    sut.filemgr.read_job_dir = Mock(return_value=("lens_desc_filename", files))
    with patch(f"{PKG}.LensDescriptor.from_cryosparc_file") as from_file:
        from_file.return_value = lens_desc
        sut.prepare_sequential_cryosparc(folder_cryosparc, job_number)

    assert sut.lens_desc == lens_desc
    assert sut.lens_desc_buffer.parent_descriptor == lens_desc
    assert len(sut.inputs_buffer) == len(files)
    
    for i, x in enumerate(sut.inputs_buffer):
        assert x[0] == 'sequential_cryosparc'
        assert isinstance(x[1], SequentialCryosparc)
        assert x[1].mrc_file == files[i]


def test_normalize_images(tmp_path):
    sut = get_base_converter(tmp_path)
    im = Mock()
    im.images_phys = np.array(range(5))
    im.images_fourier = np.array(range(5))
    im.downsample_images_phys = Mock()
    im.center_physical_image_signal = Mock()
    im.transform_to_fourier = Mock()
    im.normalize_images_fourier = Mock()

    # test no downsample
    with patch("builtins.print") as _print:
        sut._normalize_and_center_images(im)
        assert isinstance(im.downsample_images_phys, Mock)
        im.downsample_images_phys.assert_not_called()
        assert im.center_physical_image_signal.call_count == 1
        im.transform_to_fourier.assert_called_once_with(polar_grid=FIX_IMG_DESC.polar_grid, precision=FIX_IMG_DESC.precision)
        im.normalize_images_fourier.assert_called_once_with(ord=2, use_max=False)

        # test with downsample
        sut.downsample_factor = 3
        sut.downsample_type = "mean"
        sut._normalize_and_center_images(im)
        im.downsample_images_phys.assert_called_once_with(sut.downsample_factor, sut.downsample_type)

        assert _print.call_count == 4


def test_load_starfile(tmp_path):
    sut = get_base_converter(tmp_path, pixel_size=1.)
    assert not sut._must_flush_buffer

    with (
        patch(f"{PKG}.LensDescriptor.from_starfile") as from_starfile,
        patch(f"{PKG}.Images.from_mrc") as from_mrc
    ):
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


def test_load_indexed(tmp_path):
    sut = get_base_converter(tmp_path, pixel_size=1.)
    img_cnt = 10
    stack = make_mock_imagestack(img_cnt)
    with patch(f"{PKG}.Images.from_mrc") as from_mrc:
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


def test_load_sequential_cryosparc(tmp_path):
    sut = get_base_converter(tmp_path, pixel_size=1.)
    img_cnt = 10
    stack = make_mock_imagestack(img_cnt)
    with patch(f"{PKG}.Images.from_mrc") as from_mrc:
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


def test_emit_batch(tmp_path):
    sut = get_base_converter(tmp_path, pixel_size=1.)
    sut._plot_imgs = Mock()
    sut.filemgr = Mock()
    sut.filemgr.write_batch = Mock()

    stack_size = 10
    stack = make_mock_imagestack(stack_size)
    sut.images_buffer.append_imgs(stack)
    lens_desc = make_lens_descriptor(stack_size)
    sut.lens_desc = lens_desc
    sut.lens_desc_buffer.update_parent(sut.lens_desc)
    sut.lens_desc_buffer.enqueue(lens_desc.batch_whole())

    assert sut.images_buffer.stack_size == stack_size

    batch_size = 6
    sut._emit_batch(batch_size)
    sut._plot_imgs.assert_called_once()
    sut.filemgr.write_batch.assert_called_once()

    # repeat with the overall counter
    sut._plot_imgs.reset_mock()
    sut.filemgr.write_batch.reset_mock()
    sut.output_plots = False

    sut._emit_batch(batch_size, True)
    sut._plot_imgs.assert_not_called()
    realized_size = stack_size - batch_size
    sut.filemgr.write_batch.assert_called_once()
    call_args = sut.filemgr.write_batch.call_args
    assert call_args[1]['overall_batch_start'] == 0
    assert sut._stack_absolute_index == realized_size


def test_write_batch_throws_on_mismatched_lengths(tmp_path):
    sut = get_base_converter(tmp_path, pixel_size=1.)
    stack_size = 100
    stack = make_mock_imagestack(stack_size)
    sut.images_buffer.append_imgs(stack)
    lens_desc = make_lens_descriptor(stack_size//2)
    sut.lens_desc = lens_desc
    sut.lens_desc_buffer.update_parent(sut.lens_desc)
    sut.lens_desc_buffer.enqueue(lens_desc.batch_whole())

    batch_size = (stack_size // 2) + 15
    with raises(ValueError, match="Length mismatch"):
        sut._emit_batch(batch_size)


def test_convert_stacks(tmp_path):
    sut = get_base_converter(tmp_path, pixel_size=1., overwrite=False)
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
    sut._emit_batch = Mock()
    sut._emit_batch.side_effect = lambda x, y: sut.images_buffer.pop_imgs(x)

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
    assert sut._emit_batch.call_count == sut.i_stacks


@mark.parametrize('mode',[('starfile'), ('cryosparc_with_request')])
def test_convert_stacks_full_flushes_buffer(tmp_path, mode):
    sut = get_base_converter(tmp_path, pixel_size=1.)
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
    sut._load_starfile = Mock(side_effect = lambda x: load_star())
    sut._load_sequential_cryosparc = Mock(side_effect = lambda x: sut.images_buffer.append_imgs(stack))
    sut._emit_batch = Mock(side_effect = lambda x, y: sut.images_buffer.pop_imgs(x))

    if mode == 'starfile':
        rowtype = 'starfile'
    elif mode == 'cryosparc_with_request':
        rowtype = 'sequential_cryosparc'
    else:
        raise NotImplementedError('Not a known mode')
    row1 = make_datasource(rowtype, "file1")
    row2 = make_datasource(rowtype, "file2")
    sut.inputs_buffer.append(row1)
    sut.inputs_buffer.append(row2)

    if mode == 'starfile':
        sut.convert_stacks(batch_size)
    elif mode == 'cryosparc_with_request':
        sut.convert_stacks(batch_size, never_combine_input_files=True)

    assert sut.i_stacks == 4
    assert sut._emit_batch.call_count == sut.i_stacks
    if mode == 'starfile':
        assert sut._load_starfile.call_count == 2
        calls = sut._load_starfile.call_args_list
    if mode == 'cryosparc_with_request':
        assert sut._load_sequential_cryosparc.call_count == 2
        calls = sut._load_sequential_cryosparc.call_args_list
    assert calls[0][0][0] == row1[1]
    assert calls[1][0][0] == row2[1]


def test_convert_stacks_aborts_on_max_stacks(tmp_path):
    img_cnt = 10
    batch_size = 3
    stack = make_mock_imagestack(img_cnt)
    
    sut = get_base_converter(tmp_path, pixel_size=1.)
    sut.max_stacks = 2
    sut._load_sequential_cryosparc = Mock()
    sut._load_sequential_cryosparc.side_effect = lambda x: sut.images_buffer.append_imgs(stack)
    sut._emit_batch = Mock()
    sut._emit_batch.side_effect = lambda x, y: sut.images_buffer.pop_imgs(x)
    sut.inputs_buffer.append(make_datasource("sequential_cryosparc", "file"))

    with patch('builtins.print') as _print:
        sut.convert_stacks(batch_size)

    assert sut.i_stacks == sut.max_stacks
    assert sut._emit_batch.call_count == sut.max_stacks
    assert sut.images_buffer.stack_size > 0


def test_convert_stacks_warns_on_empty_buffer(tmp_path):
    sut = get_base_converter(tmp_path)
    with patch('builtins.print') as _print:
        sut.convert_stacks()
        _print.assert_called_once_with("Warning: you must prepare input files before running convert_stacks.")


def test_convert_stacks_throws_on_no_pixel_size(tmp_path):
    sut = get_base_converter(tmp_path)
    sut.inputs_buffer.append(make_datasource("sequential_cryosparc", "some-name"))
    with patch('builtins.print') as _print:
        # In this case, the pixel_size should have been set either at converter initialization
        # or while parsing the Cryosparc file. Since we skipped that, there isn't a pixel size,
        # and we expect an error
        with raises(ValueError, match="Pixel size was never set"):
            sut.convert_stacks()


@mark.parametrize("limit_out_cnt", [(True), (False)])
def test_plot_imgs(tmp_path, limit_out_cnt):
    sut = get_base_converter(tmp_path)
    phys = Mock()
    phys.shape = [10, 24]
    four = Mock()
    four.shape = [10, 48]
    sut.filemgr = Mock()
    fns = (Mock(), Mock(), Mock())
    sut.filemgr.get_plot_filenames = Mock(return_value=fns)

    with (
        patch(f"{PKG}.plot_power_spectrum") as plot_ps,
        patch(f"{PKG}.plot_images") as plot_imgs
    ):
        if limit_out_cnt:
            sut.max_imgs_to_plot = phys.shape[0] - 2
            sut._plot_imgs(phys, four)
            four_call_kwargs = plot_imgs.call_args[1]
            assert four_call_kwargs['n_plots'] == sut.max_imgs_to_plot
            assert four_call_kwargs['filename'] == fns[1]
        else:
            assert sut.max_imgs_to_plot > phys.shape[0]
            sut._plot_imgs(phys, four)
            plot_imgs_calls = plot_imgs.call_args_list
            phys_call = plot_imgs_calls[0]
            assert phys_call[0][0] == phys
            assert phys_call[1]['grid'] == sut.img_desc.cartesian_grid
            assert phys_call[1]['n_plots'] == phys.shape[0]
            assert phys_call[1]['filename'] == fns[0]

            four_call = plot_imgs_calls[1]
            assert four_call[0][0] == four
            assert four_call[1]['grid'] == sut.img_desc.polar_grid
            assert four_call[1]['n_plots'] == four.shape[0]
            assert four_call[1]['filename'] == fns[1]

            plot_ps.assert_called_once()
            ps_args = plot_ps.call_args
            assert ps_args[1]['filename_plot'] == fns[2]
