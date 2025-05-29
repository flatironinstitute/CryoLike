from collections.abc import Sequence
from unittest.mock import patch, Mock
from pytest import raises, mark
import os
import torch
import numpy as np
from pathlib import Path

from cryolike.util import InputFileType

from cryolike.file_mgmt.template_creation_file_mgmt import (
    TemplateFileManager,
    _get_input_file_type,
    MRC_EXTENSIONS,
    PDB_EXTENSIONS
)

PKG = "cryolike.file_mgmt.template_creation_file_mgmt"


def _get_file_mgr(
    tmp_path: Path,
    list_of_inputs: Sequence[str | torch.Tensor | np.ndarray],
    output_plots: bool = False,
    verbose: bool = False,
    make_input_files: bool = True,
    file_seeds: list[tuple[str, int]] = [],
    template_maker_fn: Mock = Mock(),
    plotter_fn: Mock = Mock()
):
    folder_output = tmp_path / "YOU_SHOULD_NOT_SEE_THIS"
    if make_input_files:
        final_inputs = []
        for x in list_of_inputs:
            if isinstance(x, str):
                f = tmp_path / x
                f.write_text("")
                final_inputs.append(str(f))
            else:
                final_inputs.append(x)
        list_of_inputs = final_inputs

    mgr = TemplateFileManager(
        str(folder_output),
        output_plots,
        list_of_inputs,
        template_maker_fn,
        plotter_fn,
        verbose
    )

    for f, cnt in file_seeds:
        (folder_output / f"templates_fourier_{f}.pt").write_text("")
        for i in range(cnt):
            (folder_output / f"templates_fourier_{f}_{i}.pt").write_text("")

    return mgr


@mark.parametrize("with_plots", (True, False))
def test_set_up_directories(tmp_path, with_plots):
    out_dir = tmp_path / "my" / "output" / "directory"
    plot_dir = out_dir / "plots"

    _ = TemplateFileManager(out_dir, with_plots, [], Mock(), Mock()) # type: ignore

    assert os.path.exists(out_dir)
    if with_plots:
        assert os.path.exists(plot_dir)
    else:
        assert not os.path.exists(plot_dir)


@mark.parametrize("type,list", (
    [InputFileType.MRC, MRC_EXTENSIONS],
    [InputFileType.PDB, PDB_EXTENSIONS],
))
def test_get_input_file_type(type: InputFileType, list: list[str]):
    for x in list:
        fn = Path("my") / "tmp" / f"file_name.{x}"
        res = _get_input_file_type(fn)
        assert res == type


def test_get_input_file_type_throws_on_bad_type():
    fn = Path("my") / "badfile.xlsx"
    with raises(ValueError, match="Unsupported"):
        _get_input_file_type(fn)


@mark.parametrize("verbose", (True, False))
def test_parse_inputs(tmp_path, verbose: bool):
    file_input_names = ["mrc1.mrc", "mrc2.mrcs", "mrc3.map", "pdb.pdb"]
    tensor_inputs = [np.arange(5), torch.arange(6)]
    inputs = []
    inputs.extend(file_input_names)
    inputs.extend(tensor_inputs)
    expected = [
        ("mrc1", InputFileType.MRC),
        ("mrc2", InputFileType.MRC),
        ("mrc3", InputFileType.MRC),
        ("pdb", InputFileType.PDB),
        ("tensor_4", InputFileType.MEM),
        ("tensor_5", InputFileType.MEM)
    ]

    with patch("builtins.print") as _print:
        sut = _get_file_mgr(tmp_path, inputs, output_plots=False, verbose=verbose)
        assert len(inputs) == len(sut.inputs)
        for i, exp in enumerate(expected):
            got = sut.inputs[i]
            assert got[1] == exp[0]
            assert got[2] == exp[1]

        if verbose:
            assert _print.call_count == 2 * len(file_input_names) + len(tensor_inputs)


def test_parse_inputs_raises_on_invalid_type(tmp_path):
    inputs = [17]
    with raises(ValueError, match="Invalid type"):
        _get_file_mgr(tmp_path, inputs) # type: ignore


def test_parse_inputs_raises_on_missing_file(tmp_path):
    inputs = ["file_that_doesnt_exist.xlsx"]
    with raises(ValueError, match="Input file"):
        _get_file_mgr(tmp_path, inputs, make_input_files=False)


@mark.parametrize("existing_file_count", [0, 3])
def test_get_template_output_filename(tmp_path, existing_file_count: int):
    basename = "myfile"

    if existing_file_count > 0:
        seeds = [(basename, existing_file_count)]
        expected = f"templates_fourier_{basename}_{existing_file_count}.pt"
        expected_power = f"power_spectrum_{basename}_{existing_file_count}.png"
    else:
        seeds = []
        expected = f"templates_fourier_{basename}.pt"
        expected_power = f"power_spectrum_{basename}.png"

    sut = _get_file_mgr(tmp_path, [], True, False, file_seeds=seeds)
    res = sut._get_output_names(basename)

    assert str(res.template_file_name.name) == expected
    assert str(res.power_plot_file.name) == expected_power


@mark.parametrize("include_pdbs", [True, False])
def test_inputs_include_pdb_files(tmp_path, include_pdbs: bool):
    inputs = ["mrc1.mrc", "mrc2.mrc"]
    if include_pdbs:
        inputs.append("file.pdb")
    sut = _get_file_mgr(tmp_path, inputs)
    assert sut.inputs_include_pdb_files() == include_pdbs


@mark.parametrize("do_plots", (True, False))
def test_process_inputs(tmp_path, do_plots):
    # NOTE: we *expect* plotting not to make a difference,
    # since whether plots are actually generated is the
    # responsibility of the callback function creator,
    # not the file mgr.
    mock_plotter = Mock()
    mock_reader = Mock()
    with patch(f"{PKG}.save") as torch_save:
        inputs = ["mrc1.mrc", "mrc2.mrc", "pdb.pdb"]
        expected_out = ["templates_fourier_mrc1.pt",
                        "templates_fourier_mrc2.pt",
                        "templates_fourier_pdb.pt"]
        sut = _get_file_mgr(
            tmp_path,
            inputs,
            do_plots,
            make_input_files=True,
            plotter_fn = mock_plotter,
            template_maker_fn = mock_reader
        )
        sut.process_inputs()
        assert torch_save.call_count == len(inputs)
        assert mock_plotter.call_count == len(inputs)
        assert mock_reader.call_count == len(inputs)
        assert len(sut.output_files) == len(inputs)
        for i, n in enumerate(expected_out):
            actual = Path(sut.output_files[i])
            assert actual.name == n


def test_save_file_list(tmp_path):
    inputs = ["mrc1.mrc"]
    sut = _get_file_mgr(tmp_path, inputs)
    file_list_path = sut._output_base / "template_file_list.npy"
    assert not file_list_path.is_file()
    sut.save_file_list()
    assert file_list_path.is_file()
