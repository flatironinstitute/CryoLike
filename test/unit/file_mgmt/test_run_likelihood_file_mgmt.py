from pathlib import Path
from pytest import raises, mark
from typing import Literal
from unittest.mock import patch, Mock

from cryolike.util import OutputConfiguration
from cryolike.file_mgmt import get_input_filename, make_dir
from cryolike.file_mgmt.run_likelihood_file_mgmt import (
    LikelihoodFileManager
)

PKG = "cryolike.file_mgmt.run_likelihood_file_mgmt"

def _fix_get_path_root(tmp_path: Path, type: Literal['in'] | Literal['out'] | Literal['tmp']):
    if type == 'in':
        return tmp_path / "SHOULD_NOT_BE_VISIBLE_IN"
    elif type == 'out':
        return tmp_path / "SHOULD_NOT_BE_VISIBLE_OUT"
    else:
        return tmp_path / "SHOULD_NOT_BE_VISIBLE_TEMPLATE"


def _fix_get_mgr(tmp_path: Path,
    inputs_to_create: int = 3,
    create_phys: bool = False,
    i_template: int | None = None,
    with_inputs_missing: bool = False
):
    out_root = _fix_get_path_root(tmp_path, 'out')
    in_root = _fix_get_path_root(tmp_path, 'in')
    template_root = _fix_get_path_root(tmp_path, 'tmp')

    if inputs_to_create is not None and not with_inputs_missing:
        fft_root = in_root / 'fft'
        phys_root = in_root / 'phys'
        make_dir(in_root, 'fft')
        make_dir(in_root, 'phys')
        paths: list[Path] = []
        for i in range(inputs_to_create):
            paths.append(get_input_filename(fft_root, i, 'fourier'))
            paths.append(get_input_filename(fft_root, i, 'params'))
            if create_phys:
                paths.append(get_input_filename(phys_root, i, 'phys'))
        for x in paths:
            x.write_text("")
    args = {
        'folder_output': str(out_root),
        'folder_templates': str(template_root),
        'folder_particles': str(in_root),
        'n_stacks_to_process': inputs_to_create,
        'phys_needed': create_phys
    }
    if i_template is not None:
        args['i_template'] = i_template
    mgr = LikelihoodFileManager(**args)
    return mgr


def test_init_output_folders(tmp_path: Path):
    i_template = 12
    expected_outdir = _fix_get_path_root(tmp_path, 'out') / f"template{i_template}"

    sut = _fix_get_mgr(tmp_path, 0, i_template=i_template)

    assert sut._output_log_likelihood.is_dir()
    assert sut._output_log_likelihood == expected_outdir / "log_likelihood"
    assert sut._output_cross_correlation.is_dir()
    assert sut._output_cross_correlation == expected_outdir / "cross_correlation"
    assert sut._output_optimal_pose.is_dir()
    assert sut._output_optimal_pose == expected_outdir / "optimal_pose"


def test_template_filename(tmp_path):
    expected_template_dir = _fix_get_path_root(tmp_path, 'tmp')

    sut = _fix_get_mgr(tmp_path, 0)
    res = sut._get_template_filename()

    assert res == expected_template_dir / "template_file_list.npy"


@mark.parametrize("with_phys,should_fail", [(False, False), (True, False), (True, True)])
def test_check_input_files_exist(tmp_path, with_phys, should_fail):
    if should_fail:
        with raises(ValueError, match="Files not found"):
            _ = _fix_get_mgr(tmp_path, create_phys=with_phys, with_inputs_missing=should_fail)
        return
    _ = _fix_get_mgr(tmp_path, create_phys=with_phys)


@mark.parametrize("max_disp_p,pixel_size,i_template", [
    (4.5, 12., 0),
    (6.,  22., 1),
    (17., 3.3, 2)
])
def test_load_template(tmp_path, max_disp_p, pixel_size, i_template):
    file_list = ["file1", "file2", "file3"]
    params_input = Mock()
    floattype = Mock()
    _img_desc = Mock()
    _img_desc.precision.get_dtypes = Mock(return_value=(floattype, None, None))
    _img_desc.cartesian_grid.pixel_size = [pixel_size, pixel_size + 1]
    expected_max_disp = max_disp_p * pixel_size

    sut = _fix_get_mgr(tmp_path, 0)
    expected_fn = sut._get_template_filename()

    with (
        patch(f"{PKG}.ImageDescriptor") as _id,
        patch(f"{PKG}.np_load") as _npload,
        patch(f"{PKG}.load") as _load,
        patch(f"{PKG}.FourierImages") as _fi,
        patch(f"{PKG}.Templates") as _tp,
        patch("builtins.print") as _print
    ):
        _id.ensure = Mock(return_value=_img_desc)
        _npload.return_value = file_list

        res = sut.load_template(params_input, max_disp_p, i_template)

        _id.ensure.assert_called_once_with(params_input)
        _npload.assert_called_once_with(expected_fn, allow_pickle=True)
        _load.assert_called_once_with(file_list[i_template], weights_only=True)
        _fi.assert_called_once_with(_load.return_value, _img_desc.polar_grid)
        _tp.assert_called_once_with(
            fourier_data=_fi.return_value,
            phys_data = _img_desc.cartesian_grid,
            viewing_angles = _img_desc.viewing_angles
        )
        assert res[0] == _tp.return_value
        assert res[1] == _img_desc
        assert res[2] == floattype
        assert res[3] == expected_max_disp


@mark.parametrize("compatible", [(True), (False)])
def test_load_img_stack(tmp_path, compatible: bool):
    i_stack = 12
    image_desc = Mock()
    image_desc.is_compatible_with = Mock(return_value=compatible)

    sut = _fix_get_mgr(tmp_path, 0)
    expected_fourier_input = sut._get_input_filename(i_stack, "fourier")
    expected_param_input = sut._get_input_filename(i_stack, "params")

    with (
        patch(f"{PKG}.load") as _load,
        patch(f"{PKG}.load_combined_params") as _load_combined,
        patch(f"{PKG}.FourierImages") as _fi,
        patch(f"{PKG}.Images") as _ctor,
        patch(f"{PKG}.CTF") as _ctf_ctor,
        patch("builtins.print") as _print
    ):
        _stack_img_desc = Mock()
        _stack_lens_desc = Mock()
        _stack_img_desc.cartesian_grid = Mock()
        _stack_img_desc.cartesian_grid.box_size = [10, 15]
        _load_combined.return_value = (_stack_img_desc, _stack_lens_desc)
        if not compatible:
            with raises(ValueError, match="Incompatible"):
                sut.load_img_stack(i_stack, image_desc)
            return
        (res_im, res_ctf) = sut.load_img_stack(i_stack, image_desc)
        _load.assert_called_once_with(expected_fourier_input, weights_only=True)
        _load_combined.assert_called_once_with(expected_param_input)
        _fi.assert_called_once_with(_load.return_value, _stack_img_desc.polar_grid)
        _ctor.assert_called_once_with(fourier_data=_fi.return_value, phys_data=_stack_img_desc.cartesian_grid)
        _ctf_ctor.assert_called_once_with(
            ctf_descriptor=_stack_lens_desc,
            polar_grid=_stack_img_desc.polar_grid,
            box_size=_stack_img_desc.cartesian_grid.box_size[0],
            anisotropy=True
        )
        assert res_im == _ctor.return_value
        assert res_ctf == _ctf_ctor.return_value


def test_load_phys_stack(tmp_path):
    i_stack = 6
    image_desc = Mock()
    image_desc.cartesian_grid = Mock()
    image_desc.cartesian_grid.pixel_size = 12

    sut = _fix_get_mgr(tmp_path)
    expected_input_fn = sut._get_input_filename(i_stack, 'phys')
    with (
        patch(f"{PKG}.load") as _load,
        patch(f"{PKG}.PhysicalImages") as phys_img,
        patch(f"{PKG}.Images") as ctor
    ):
        res = sut.load_phys_stack(i_stack, image_desc)
        _load.assert_called_once_with(expected_input_fn, weights_only=True)
        phys_img.assert_called_once_with(_load.return_value, pixel_size=image_desc.cartesian_grid.pixel_size)
        ctor.assert_called_once_with(phys_data=phys_img.return_value, fourier_data=None)
        assert res == ctor.return_value


@mark.parametrize('already_done', [(False), (True)])
def test_save_displacements(tmp_path, already_done):
    sut = _fix_get_mgr(tmp_path)
    assert not sut._displacements_saved
    x_disp = Mock()
    y_disp = Mock()
    if already_done:
        sut._displacements_saved = True

    with (
        patch(f"{PKG}.save") as _save,
        patch(f"{PKG}.stack") as _stack
    ):
        sut.save_displacements(x_disp, y_disp)
        if already_done:
            _save.assert_not_called()
        else:
            _save.assert_called_once()
            assert sut._displacements_saved == True


# Maps the possible output file field names to the OutputConfiguration fields that
# would have to be set True for that file to be emitted.
FIELD_MAPS = {
    "cross_corr_pose_file": "cross_correlation_pose",
    "integrated_pose_file": "integrated_likelihood_fourier",
    "optimal_fourier_pose_likelihood_file": "optimal_fourier_pose_likelihood",
    "optimal_phys_pose_likelihood_file": "optimal_phys_pose_likelihood",
    "cross_corr_file": "optimal_pose",
    "template_indices_file": "optimal_pose",
    "x_displacement_file": "optimal_pose",
    "y_displacement_file": "optimal_pose",
    "inplane_rotation_file": "optimal_pose"
}

(ALL_OUTPUTS, _) = OutputConfiguration.make_all_possible_configs()

def test_get_output_filenames(tmp_path):
    sut = _fix_get_mgr(tmp_path, 0)

    i_stack = 15
    expected_suffix = f"stack_{i_stack:06}.pt"
    expecteds = {
        "cross_corr_pose_file": sut._output_cross_correlation / f"cross_correlation_pose_smdw_{expected_suffix}",
        "integrated_pose_file": sut._output_log_likelihood / f"log_likelihood_integrated_fourier_{expected_suffix}",
        "optimal_fourier_pose_likelihood_file": sut._output_log_likelihood / f"log_likelihood_optimal_fourier_{expected_suffix}",
        "optimal_phys_pose_likelihood_file": sut._output_log_likelihood / f"log_likelihood_optimal_physical_{expected_suffix}",
        "cross_corr_file": sut._output_cross_correlation / f"cross_correlation_{expected_suffix}",
        "template_indices_file": sut._output_optimal_pose / f"optimal_template_{expected_suffix}",
        "x_displacement_file": sut._output_optimal_pose / f"optimal_displacement_x_{expected_suffix}",
        "y_displacement_file": sut._output_optimal_pose / f"optimal_displacement_y_{expected_suffix}",
        "inplane_rotation_file": sut._output_optimal_pose / f"optimal_inplane_rotation_{expected_suffix}"
    }

    for outtype in ALL_OUTPUTS:
        res = sut._get_output_filenames(i_stack, outtype)
        if outtype.cross_correlation_pose:
            assert res.cross_corr_pose_file == expecteds['cross_corr_pose_file']
            for k in expecteds.keys():
                if k == 'cross_corr_pose_file': continue
                assert getattr(res, k) is None
            return
        for k in expecteds.keys():
            expected = expecteds[k] if getattr(outtype, FIELD_MAPS[k]) else None
            actual = getattr(res, k)
            assert expected == actual


def test_outputs_exist(tmp_path):
    i_stack = 26
    sut = _fix_get_mgr(tmp_path, 0)
    
    for outtype in ALL_OUTPUTS:
        files_obj = sut._get_output_filenames(i_stack, outtype)
        files: list[Path] = [getattr(files_obj, x) for x in files_obj._fields if getattr(files_obj, x) is not None]
        if len(files) == 0:
            # okay, no output files requested. This is allowed to happen if & only if
            # the requested output configuration didn't set any of the fields we
            # actually can do output for. So check all the FIELD_MAPS targets (i.e. the
            # fields on the OutputConfiguration object which map to a file we know about)
            # and assert that they're False.
            all_fields = set(FIELD_MAPS.values())
            for i in all_fields:
                assert not getattr(outtype, i)
            continue
        last_file = files.pop(-1)

        for x in files:
            x.write_text("")
        assert not sut.outputs_exist(i_stack, outtype)
        last_file.write_text("")
        assert sut.outputs_exist(i_stack, outtype)
        # cleanup
        for x in files:
            x.unlink()
        last_file.unlink()


def test_write_outputs(tmp_path):
    i_stack = 662
    out_data = Mock()
    sut = _fix_get_mgr(tmp_path, 0)

    for outtype in ALL_OUTPUTS:
        expected_fns = sut._get_output_filenames(i_stack, outtype)
        expected_call_count = 0

        with patch(f"{PKG}.save") as save:
            save.reset_mock()
            sut.write_outputs(i_stack, outtype, out_data)
            if outtype.cross_correlation_pose:
                save.assert_called_once_with(out_data.full_pose, expected_fns.cross_corr_pose_file)
                continue
            if outtype.integrated_likelihood_fourier:
                expected_call_count += 1
                save.assert_any_call(out_data.ll_fourier_integrated, expected_fns.integrated_pose_file)
            if outtype.optimal_fourier_pose_likelihood:
                expected_call_count += 1
                save.assert_any_call(out_data.ll_optimal_fourier_pose, expected_fns.optimal_fourier_pose_likelihood_file)
            if outtype.optimal_phys_pose_likelihood:
                expected_call_count += 1
                save.assert_any_call(out_data.ll_optimal_phys_pose, expected_fns.optimal_phys_pose_likelihood_file)
            if outtype.optimal_pose:
                expected_call_count += 5
                save.assert_any_call(out_data.optimal_pose.cross_correlation_S, expected_fns.cross_corr_file)
                save.assert_any_call(out_data.optimal_pose.optimal_template_S, expected_fns.template_indices_file)
                save.assert_any_call(out_data.optimal_pose.optimal_displacement_x_S, expected_fns.x_displacement_file)
                save.assert_any_call(out_data.optimal_pose.optimal_displacement_y_S, expected_fns.y_displacement_file)
                save.assert_any_call(out_data.optimal_pose.optimal_inplane_rotation_S, expected_fns.inplane_rotation_file)
            assert save.call_count == expected_call_count
