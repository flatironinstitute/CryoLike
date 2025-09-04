from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, NamedTuple, Sequence, Tuple, TYPE_CHECKING
from torch import save, Tensor
from numpy import save as np_save
from numpy import ndarray

if TYPE_CHECKING: # pragma: no cover
    from cryolike.stacks import Templates
    
from .file_base import make_dir
from cryolike.util import InputFileType


MRC_EXTENSIONS = ['.mrc', '.mrcs', '.map']
PDB_EXTENSIONS = ['.pdb']

class TemplateOutputFiles(NamedTuple):
    template_file_name: Path
    fourier_images_plot_file: Path
    power_plot_file: Path
    physical_images_plot_file: Path


INPUT_FILE_DESC = Tuple[Path, str, Literal[InputFileType.MRC] | Literal[InputFileType.PDB]]
INPUT_MEM_DESC = Tuple[ndarray | Tensor, str, Literal[InputFileType.MEM]]
TEMPLATE_INPUT_DESC = INPUT_FILE_DESC | INPUT_MEM_DESC


def _get_input_file_type(input: Path):
    ext = input.suffix
    if ext in MRC_EXTENSIONS:
        return InputFileType.MRC
    elif ext in PDB_EXTENSIONS:
        return InputFileType.PDB
    else:
        raise ValueError(f'Unsupported file type: {ext}.')


def _get_file_input_name(input: Path, verbose: bool = False):
    if verbose:
        print(f"Processing {input.absolute()}...")
    input_type = _get_input_file_type(input)
    input_name = input.stem
    if verbose:
        print(f"{input_type.name} name: {input_name}")
    return (input_name, input_type)


def _get_memory_input_name(iteration_count: int, verbose: bool = False):
    if verbose:
        print(f"Processing in-memory tensor [input {iteration_count}]...")
    name = f"tensor_{iteration_count}"
    return (name, InputFileType.MEM)


def _parse_inputs(list_of_inputs: Sequence[str | ndarray | Tensor], verbose: bool = False) -> list[TEMPLATE_INPUT_DESC]:
    processed: list[TEMPLATE_INPUT_DESC] = []
    for i, input in enumerate(list_of_inputs):
        if isinstance(input, str):
            inpath = Path(input)
            if not inpath.is_file():
                raise ValueError(f"Input file {inpath.absolute()} does not exist.")
            name, t = _get_file_input_name(inpath, verbose)
            processed.append((inpath, name, t))
        elif isinstance(input, ndarray) or isinstance(input, Tensor):
            name, t = _get_memory_input_name(i, verbose)
            processed.append((input, name, t))
        else:
            raise ValueError("Invalid type in input list.")
    return processed


class TemplateFileManager():
    _output_base: Path
    _plot_base: Path
    inputs: list[TEMPLATE_INPUT_DESC]
    output_files: list[str]
    template_maker_fn: Callable[[TEMPLATE_INPUT_DESC], Templates]
    plotter_fn: Callable[[Templates, TemplateOutputFiles], None]

    def __init__(self,
        folder_output: str,
        output_plots: bool,
        list_of_inputs: Sequence[str | ndarray | Tensor],
        template_maker_fn: Callable[[TEMPLATE_INPUT_DESC], Templates],
        plotter_fn: Callable[[Templates, TemplateOutputFiles], None],
        verbose: bool = False
    ):
        self.inputs = _parse_inputs(list_of_inputs, verbose)
        self.output_files = []
        self._output_base = Path(folder_output)
        self._plot_base = self._output_base / 'plots'
        self.template_maker_fn = template_maker_fn
        self.plotter_fn = plotter_fn
        self._make_tree(output_plots)


    def _make_tree(self, output_plots: bool):
        branch = 'plots' if output_plots else ''
        make_dir(self._output_base, branch)


    def _get_output_names(self, name: str):
        signame = name
        basename = f"templates_fourier_{signame}.pt"
        template_fn = self._output_base / basename
        same_name_count = 0
        while (template_fn.exists()):
            same_name_count += 1
            signame = f"{name}_{same_name_count}"
            basename = f"templates_fourier_{signame}.pt"
            template_fn = self._output_base / basename

        return TemplateOutputFiles(
            template_file_name = template_fn,
            fourier_images_plot_file = self._plot_base / f"templates_fourier_{signame}.png",
            power_plot_file = self._plot_base / f"power_spectrum_{signame}.png",
            physical_images_plot_file = self._plot_base / f"templates_phys_{signame}.png"
        )
        

    def inputs_include_pdb_files(self) -> bool:
        return any([x[2] == InputFileType.PDB for x in self.inputs])


    def process_inputs(self):
        for input in self.inputs:
            out_names = self._get_output_names(input[1])
            tp = self.template_maker_fn(input)
            self.plotter_fn(tp, out_names)
            save(tp.images_fourier, out_names.template_file_name)
            self.output_files.append(str(out_names.template_file_name))


    def save_file_list(self):
        output_file = self._output_base / "template_file_list.npy"
        np_save(output_file, self.output_files)
