from __future__ import annotations

from typing import Literal, NamedTuple, TYPE_CHECKING
from pathlib import Path
from numpy import unique, where, int64

if TYPE_CHECKING: # pragma: no cover
    from cryolike.metadata import ImageDescriptor, LensDescriptorBuffer
    from torch import Tensor

from torch import save
from .file_base import make_dir, get_input_filename, check_files_exist
from cryolike.metadata import save_combined_params, LensDescriptor
from cryolike.util import IntArrayType


def ensure_input_files_exist(files: list[str]):
    (all_exist, misses) = check_files_exist(files)
    if not all_exist:
        errs = "\n\t".join(misses)
        raise ValueError(f'Files not found:\n{errs}')


def get_filenames_and_indices(
    lens_desc: LensDescriptor,
    folder_mrc: str = ''
) -> list[tuple[str, IntArrayType, IntArrayType]]:
    """Extract selected images from the files described in the metadata
    and associate them with the defocus and phase-shift values.

    Once the metadata file has been read, we have to do some matching.
    The metadata 'files' field describes the MRC files which we'll need
    to read to get actual images. The metadata 'idxs' field lists the index,
    within the corresponding MRC file, of the image described by that row of
    the source data file (Cryosparc or Starfile).

    Example:
      * files = ['file1.mrc', 'file1.mrc', 'file2.mrc', 'file1.mrc']
      * idxs = [1, 3, 1, 2]

    This would indicate that row 0 of the source data file describes the
    2nd image of the stack in file1.mrc; row 1 of the source data file
    describes the 4th image; row 2 of the source data describes the 2nd
    image of the stack in file2.mrc, etc.

    The job of this function is to convert this list full of duplicates into
    a single record per MRC file, so that we can load the appropriate images
    from the stack in the MRC file and also pull in the appropriate slice of
    the per-image data of the LensDescriptor.

    So in the example above, we would want to return:
      * ('file1.mrc', [1, 3, 2], [0, 1, 3]),
      * ('file2.mrc', [1], [2])

    since
      - rows 0, 1, and 3 of the source file refer to file1.mrc (specifically
        the images at indexes 1, 3, and 2 of its image stack); and
      - row 2 of the source file refers to the image at index 1 of file2.mrc

    During this process, this function also skips over any MRC files that
    can't be found on the disk.

    Args:
        lens_desc (LensDescriptor): LensDescriptor object with the 'files'
            and 'idxs' fields set to non-None (thus ndarray of strings and
            of ints, respectively). The 'files' field's every entry is the MRC file
            containing the image described by that row of the source
            (Cryosparc or Starfile) file. (This is likely to have many repeats.)
            The 'idxs' field has the index value, within the MRC file in 'files',
            described by that row of the source file. This will always be a scalar
            integer, as every row describes one image.
        folder_mrc (str): The root of the folder where the MRC files
            described in 'files' are located. If non-empty, we will discard any
            path information from the source file.

    Returns:
        list[tuple[str, IntArrayType, IntArrayType]]: A list of tuples, one for each
            MRC image file described in the 'files' list. The first part of the
            tuple is the file path to the MRC file; the second part is the indices
            (within the MRC file) of the images to extract from that file; and the
            third part is the corresponding rows from the LensDescriptor records.
    """
    assert lens_desc.files is not None
    assert lens_desc.idxs is not None
    path_mrc = Path(folder_mrc)
    assert path_mrc.is_dir()

    # note: np.unique with return_inverse returns:
    #   - a list of the unique elements, and
    #   - an "index list", same length as the original list, where each position is the
    #     index of the value in the unique list that goes in that position of the original.
    #
    # So if OL = ['a', 'b', 'a', 'c']:
    #    UL, IL = np.unique(OL, return_inverse = True)
    #    UL = ['a', 'b', 'c']  # unique items in OL
    #    IL = [0, 1, 0, 2]     # (UL[0] = 'a', UL[1] = 'b', UL[0] = 'a', UL[2] = 'c')
    # [UL[IL[i]] for i in range(len(IL))] always reconstructs the OL.
    unique_files, indices_files = unique(lens_desc.files, return_inverse=True)
    mrc_files_with_img_indices = []

    for i, mrcs_file in enumerate(unique_files):
        ## mrcs_file is bytes literal, so we need to decode it to a string
        if isinstance(mrcs_file, bytes):
            mrcs_file = mrcs_file.decode('utf-8')
        if mrcs_file.startswith('>'):
            mrcs_file = mrcs_file[1:]
        mrcs_file = Path(str(mrcs_file).strip('\''))
        mrc_file_path = path_mrc / mrcs_file.name

        if not mrc_file_path.is_file():
            print("File %s does not exist, skipping..." % str(mrc_file_path))
            continue
        # we are now sure that mrc_file_path points to an existing mrc file.

        # Now we need to index into the defocusU/v/Angle and phaseShift values
        # corresponding to these images. Look up the (non-deduplicated file list)
        # entries that correspond to the file we're currently processing:
        original_file_list_indices = where(indices_files == i)[0]
        # Now look up the selected image/metadata indices from the cs_idxs field
        # for those entries:
        selected_img_indices = (lens_desc.idxs[original_file_list_indices]).squeeze()
        selected_img_indices = selected_img_indices.astype(int64)
        entry = (str(mrc_file_path), selected_img_indices, original_file_list_indices)
        mrc_files_with_img_indices.append(entry)
    if len(mrc_files_with_img_indices) == 0:
        raise ValueError("None of the MRC files in the indexed file were found. Check your pathing.")
    return mrc_files_with_img_indices


class OutputFilenames(NamedTuple):
    phys_stack: Path
    fourier_stack: Path
    params_filename: Path


class CryosparcJobPath():
    file_cs: Path
    folder_type: Literal["restacked"] | Literal["downsample"]
    mrc_folder: Path

    def __init__(self, folder_cryosparc: str, job_number: int):
        cryo_dir = Path(folder_cryosparc)
        folder_job = cryo_dir / f"J{job_number}"
        if not folder_job.is_dir():
            raise ValueError("Error: folder not found: ", str(folder_job))
        self.file_cs = folder_job / f"J{job_number}_passthrough_particles.cs"

        self.mrc_folder = folder_job / "restack"
        if self.mrc_folder.is_dir():
            self.folder_type = "restacked"
            return
        self.mrc_folder = folder_job / "downsample"
        if self.mrc_folder.is_dir():
            self.folder_type = 'downsample'
            return
        raise ValueError(f"Error: directory not found: {str(folder_job)} must contain either /restack or /downsample child directory.")
        

    def get_mrc_filename(self, i_file: int):
        mrc_path = self.mrc_folder / f"batch_{i_file:06}_{self.folder_type}.mrc"
        return mrc_path


class ParticleConversionFileManager():
    _out_base: Path
    _out_plots: Path
    _out_fft: Path
    _out_phys: Path

    def __init__(self, folder_output: str):
        self._out_base = Path(folder_output)
        self._out_plots = self._out_base / "plots"
        self._out_fft = self._out_base / "fft"
        self._out_phys = self._out_base / "phys"
        self._make_tree()


    def _make_tree(self):
        for x in ['plots', 'fft', 'phys']:
            make_dir(self._out_base, x)


    def get_output_filenames(self, i_stack: int) -> OutputFilenames:
        return OutputFilenames(
            phys_stack=get_input_filename(self._out_phys, i_stack, 'phys'),
            fourier_stack=get_input_filename(self._out_fft, i_stack, 'fourier'),
            params_filename=get_input_filename(self._out_fft, i_stack, 'params')
        )
    

    def get_plot_filenames(self, i_stack: int):
        count_suffix = f"stack_{i_stack:06}.png"
        fn_phys = self._out_plots / f"particles_phys_{count_suffix}"
        fn_four = self._out_plots / f"particles_fourier_{count_suffix}"
        fn_power = self._out_plots / f"power_spectrum_{count_suffix}"

        return (str(fn_phys), str(fn_four), str(fn_power))


    def read_job_dir(self, folder_cryosparc: str, job_number: int = 0):
        job_path = CryosparcJobPath(folder_cryosparc, job_number)
        i_file = 0
        mrc_paths: list[str] = []
        while True:
            mrc_path = job_path.get_mrc_filename(i_file)
            if not mrc_path.is_file():
                print(f"Looked for file {str(mrc_path)} but it does not exist, breaking the loop and continuing to the next step...")
                break
            mrc_paths.append(str(mrc_path))
            i_file += 1

        return (str(job_path.file_cs), mrc_paths)


    def write_batch(self,
        stack_cnt: int,
        phys_batch: Tensor,
        fourier_batch: Tensor,
        img_desc: ImageDescriptor,
        lens_batch: LensDescriptorBuffer,
        overwrite: bool,
        overall_batch_start: int | None = None
    ):
        actual_batch_size = phys_batch.shape[0]
        print(f"Stacking {actual_batch_size} images")
        output_fns = self.get_output_filenames(stack_cnt)
        save(phys_batch, output_fns.phys_stack)
        save(fourier_batch, output_fns.fourier_stack)

        save_combined_params(
            str(output_fns.params_filename),
            img_desc,
            lens_batch,
            n_imgs_this_stack=actual_batch_size,
            overall_batch_start=overall_batch_start,
            overwrite=overwrite
        )
