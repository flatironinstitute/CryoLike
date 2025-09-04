from pathlib import Path
from pytest import raises, mark
from typing import Literal, NamedTuple
from unittest.mock import patch, Mock
import numpy as np

from cryolike.util import OutputConfiguration
from cryolike.file_mgmt import get_input_filename, make_dir
from cryolike.file_mgmt.post_processing_file_mgmt import (
    PostProcessFileManager
)
from cryolike.file_mgmt.run_likelihood_file_mgmt import LikelihoodOutputFiles

PKG = "cryolike.file_mgmt.post_processing_file_mgmt"

class Tree(NamedTuple):
    output: Path
    batch: Path
    template: Path
    particles: Path
    matrix: Path


def _fix_get_mgr(
    tmp_path: Path,
    output_directory: str,
    batch_directory: str = '',
    template_directory: str = '',
    particles_directory: str = '',
):
    expected_out = tmp_path / 'SHOULD_NOT_BE_VISIBLE' / output_directory
    out_matrix_root = expected_out / "likelihood_matrix"

    ret_tree = Tree(
        output=expected_out,
        batch=Path(batch_directory) if len(batch_directory) > 0 else expected_out / 'likelihood',
        template=Path(template_directory) if len(template_directory) > 0 else expected_out / 'templates',
        particles=Path(particles_directory) if len(particles_directory) > 0 else expected_out / 'particles',
        matrix=out_matrix_root
    )

    mgr = PostProcessFileManager(
        output_directory=str(expected_out),
        batch_directory=batch_directory,
        template_directory=template_directory,
        particles_directory=particles_directory
    )

    return (mgr, ret_tree)


@mark.parametrize('explicit', [(False), (True)])
def test_init(tmp_path: Path, explicit: bool):
    if explicit:
        mgr, expected = _fix_get_mgr(tmp_path, 'out_dir', 'batch_dir', 'template_dir', 'particles_dir')
    else:
        mgr, expected = _fix_get_mgr(tmp_path, 'out_dir')

    assert mgr._output_directory == expected.output
    assert mgr._batch_directory == expected.batch
    assert mgr._template_root == expected.template
    assert mgr._particles_root == expected.particles
    assert mgr._output_matrix_root == expected.matrix


@mark.parametrize('n_templates,n_stacks', [(3,4), (3, 0), (0, 4), (0 ,0)])
def test_confirm_counts(tmp_path: Path, n_templates: int, n_stacks: int):
    (mgr, expected) = _fix_get_mgr(tmp_path, 'output')
    seeded_templates = 5
    seeded_stacks = 2
    assert seeded_templates != n_templates
    assert seeded_stacks != n_stacks

    # make sure we are in the tmp directory, since we'll be making files
    assert expected.template.is_relative_to(tmp_path)
    make_dir(expected.template, '')
    # seed template file
    template_ary = np.arange(seeded_templates)
    np.save(expected.template / 'template_file_list.npy', template_ary)

    # seed stacks
    make_dir(expected.particles, 'phys')
    root = expected.particles / 'phys'
    for i in range(seeded_stacks):
        (root / f'file{i}').touch()

    ret_templates, ret_stack = mgr.confirm_counts(n_templates, n_stacks)
    if n_templates > 0:
        assert ret_templates == n_templates
    else:
        assert ret_templates == seeded_templates
    if n_stacks > 0:
        assert ret_stack == n_stacks
    else:
        assert ret_stack == seeded_stacks


@mark.parametrize('inclusions,files_exist', [('a',True), ('b',False)])
def test_get_source_lists(tmp_path: Path, inclusions: str, files_exist: bool):
    if inclusions == 'a':
        phys = True
        opt = True
        integrated = False
        cc = False
    elif inclusions == 'b':
        phys = False
        opt = False
        integrated = True
        cc = True
    else:
        raise NotImplementedError()

    (mgr, expected) = _fix_get_mgr(tmp_path, 'out')

    def mock_filenames(i_template: int, i_s: int):
        string = f"{i_template}-{i_s}.txt"

        return LikelihoodOutputFiles(
            cross_corr_pose_file=None,
            integrated_pose_file=None if not integrated else expected.batch / string,
            optimal_fourier_pose_likelihood_file=None if not opt else expected.batch / string,
            optimal_phys_pose_likelihood_file=None if not phys else expected.batch / string,
            cross_corr_file=None if not cc else expected.batch / string,
            template_indices_file=None,
            x_displacement_file=None,
            y_displacement_file=None,
            inplane_rotation_file=None
        )


    n_stacks = 3
    templates_fixed_val = 5
    expected_list = []
    for i in range(n_stacks):
        expected_list.append(expected.batch / f"{templates_fixed_val}-{i}.txt")

    with patch(f"{PKG}.LikelihoodFileManager") as llmgr:
        mock_mgr = Mock()
        mock_mgr._get_output_filenames = Mock(
            side_effect=lambda i_s, _: mock_filenames(templates_fixed_val, i_s)
        )
        llmgr.return_value = mock_mgr

        if files_exist:
            make_dir(expected.batch, '')
            for l in expected_list:
                Path(l).touch()

        with patch('builtins.print') as _print:
            res = mgr.get_source_lists(2, n_stacks, phys, opt, integrated, cc)

            if not files_exist:
                _print.assert_called()
            else:
                _print.assert_not_called()
            
            if inclusions == 'a':
                for (i, v) in enumerate(expected_list):
                    assert res.PhysStacks[i] == v
                for (i, v) in enumerate(expected_list):
                    assert res.FourierStacks[i] == v
                assert len(res.CrossCorrelationStacks) == 0
                assert len(res.IntegratedStacks) == 0
            else:
                assert len(res.PhysStacks) == 0
                assert len(res.FourierStacks) == 0
                for (i, v) in enumerate(expected_list):
                    assert res.IntegratedStacks[i] == v
                for (i, v) in enumerate(expected_list):
                    assert res.CrossCorrelationStacks[i] == v


def test_gest_source_lists_throws_on_bad_counts(tmp_path: Path):
    (mgr, _) = _fix_get_mgr(tmp_path, 'out')
    with raises(ValueError, match="Number of templates/stacks"):
        mgr.get_source_lists(0, 5)
    with raises(ValueError, match="Number of templates/stacks"):
        mgr.get_source_lists(5, 0)


def test_get_output_targets(tmp_path: Path):
    (mgr, expected) = _fix_get_mgr(tmp_path, 'out')
    res = mgr.get_output_targets()

    assert res.FourierMatrix == expected.matrix / 'optimal_fourier_log_likelihood_matrix.pt'
    assert res.PhysMatrix == expected.matrix / 'optimal_physical_log_likelihood_matrix.pt'
    assert res.IntegratedMatrix == expected.matrix / 'integrated_fourier_log_likelihood_matrix.pt'
    assert res.CrossCorrelationMatrix == expected.matrix / 'cross_correlation_matrix.pt'
