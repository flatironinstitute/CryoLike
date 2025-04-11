from collections import deque
import numpy as np
from os import path
import torch
from typing import Literal, NamedTuple

from .particle_stacks_pathing import JobPaths, OutputFolders
from .particle_stacks_buffers import ImgBuffer

from cryolike.plot import plot_images, plot_power_spectrum
from cryolike.stacks import Images
from cryolike.util import FloatArrayType, IntArrayType, TargetType, project_descriptor, ensure_positive
from cryolike.metadata import (
    ImageDescriptor,
    LensDescriptor,
    save_combined_params,
    LensDescriptorBuffer
)


def _ensure_files_exist(files: list[str]):
    for f in files:
        if not path.exists(f):
            raise ValueError("Error: file not found: ", f)


def _do_skip_exist(skip_exist: bool, image_fourier_file: str, image_param_file: str, im: Images, stack_count: int) -> bool:
    raise NotImplementedError
    if not skip_exist:
        return False
    if not path.exists(image_fourier_file) or not path.exists(image_param_file):
        return False
    n_images = np.load(image_param_file, allow_pickle = True)['n_images']
    if n_images == im.n_images:
        print("Skipping stack %d" % stack_count)
        return True
    return False


def _get_filenames_and_indices(
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
    unique_files, indices_files = np.unique(lens_desc.files, return_inverse=True)

    mrc_files_with_img_indices = []

    for i, mrcs_file in enumerate(unique_files):
        ## mrcs_file is bytes literal, so we need to decode it to a string
        if isinstance(mrcs_file, bytes):
            mrcs_file = mrcs_file.decode('utf-8')
        if mrcs_file.startswith('>'):
            mrcs_file = mrcs_file[1:]
        mrcs_file = str(mrcs_file).strip('\'')
        if folder_mrc == '':
            mrc_file_path = mrcs_file
        else:
            basename = path.basename(mrcs_file)
            mrc_file_path = path.join(folder_mrc, basename)
        if not path.exists(mrc_file_path):
            print("File %s does not exist, skipping..." % mrc_file_path)
            continue
        # we are now sure that mrc_file_path points to an existing mrc file.

        # Now we need to index into the defocusU/v/Angle and phaseShift values
        # corresponding to these images. Look up the (non-deduplicated file list)
        # entries that correspond to the file we're currently processing:
        original_file_list_indices = np.where(indices_files == i)[0]
        # Now look up the selected image/metadata indices from the cs_idxs field
        # for those entries:
        selected_img_indices = (lens_desc.idxs[original_file_list_indices]).squeeze()
        entry = (mrc_file_path, selected_img_indices, original_file_list_indices)
        mrc_files_with_img_indices.append(entry)
    if len(mrc_files_with_img_indices) == 0:
        raise ValueError("None of the MRC files in the indexed file were found. Check your pathing.")
    return mrc_files_with_img_indices


# This was done at the beginning of each conversion function, but the result
# wasn't used anywhere.
def _collect_image_tag_list(folder_output: str):
    tag_file = path.join(folder_output, "image_file_tag_list.npy")
    if not path.exists(tag_file):
        return []
    image_file_tag_list = np.load(tag_file, allow_pickle = True)
    return image_file_tag_list.tolist()



class StarfileInput(NamedTuple):
    particle_file: str
    star_file: str
    defocus_is_degree: bool
    phase_shift_is_degree: bool


class Indexed(NamedTuple):
    mrc_file: str
    selected_img_indices: IntArrayType
    selected_lensdesc_indices: IntArrayType


class SequentialCryosparc(NamedTuple):
    mrc_file: str


DataSource = tuple[Literal["starfile"], StarfileInput] | \
             tuple[Literal["indexed_cryosparc"], Indexed] | \
             tuple[Literal["indexed_starfile"], Indexed] | \
             tuple[Literal["sequential_cryosparc"], SequentialCryosparc]


# TODO: Verbosity flag
class ParticleStackConverter():
    """Object that manages converting images in Starfile or Cryosparc format to
    the internal format used by this package.

    Attributes:
        inputs_buffer (deque[DataSource]): Internal deque storing partially
            processed data sources
        img_desc (ImageDescriptor): ImageDescriptor expected to validly describe
            all images to be converted
        lens_desc (LensDescriptor): LensDescriptor that describes the experimental
            apparatus for all images to be converted. For Starfile sources, this
            will be reset with every new source file.
        images_buffer (ImgBuffer): Buffer of images processed from mrc files.
            Used in restacking.
        lens_desc_buffer (LensDescriptorBuffer): Buffer of per-image lens descriptor
            properties (defocus and phase shift). Used in restacking.
        _must_flush_buffer (bool): Internal tracking. If set, will completely
            empty the image buffer before processing a new file (the default
            behavior for starfile sources).
        _stack_start_file (int):  Internal tracking. This represents the overall
            image number at which we began outputting the current batch. Used
            for sequential cryosparc outputs.
        i_stacks (int): Total number of stacks output by the converter
        _stack_absolute_index (int): Internal tracking. Used for Starfile data
            sources, which may be split into multiple files, to record the range
            of images in the source file which are output in the current stack.
            So if we have 150 images in the input Starfile and are outputting batch
            sizes of 100, we will emit two files; the first will have an absolute
            index of 0 and the second will have an absolute index of 100.
        device (torch.device): Device to use for converting MRC image files.
            Defaults to CPU.
        out_dirs (OutputFolders): Object that maintains directory structure for
            output files
        max_stacks (int): If set to a value greater than 0, the converter will
            stop processing once this many stacks have been emitted
        pixel_size (FloatArrayType | None): Pixel size describing the physical
            images being processed. For indexed Cryosparc files, this data may
            be present in the source file. If so, the configured value must
            match the one in the source file, or an error will be generated.
        downsample_factor (int): If set, downsample by this factor
        downsample_type (Literal['mean'] | Literal['max']): The type of downsampling to use in physical space
        skip_exist (bool): If set, this will cause the converter to attempt to skip files that appear
            to have already been processed.
        output_plots (bool): If True, we will emit plots of the processed images
        max_imgs_to_plot (int): Sets the maximum number of images to plot; has
            no effect if output_plots is False.
    """
    inputs_buffer: deque[DataSource]
    # can be particle file/star file pairs; files-indices pairs; or mrc filenames.
    # When outputting, can use unified lens descriptor (with buffering) or new one
    img_desc: ImageDescriptor
    lens_desc: LensDescriptor
    images_buffer: ImgBuffer
    lens_desc_buffer: LensDescriptorBuffer
    _must_flush_buffer: bool
    _stack_start_file: int
    i_stacks: int
    _stack_absolute_index: int
    device: torch.device

    out_dirs: OutputFolders
    max_stacks: int
    pixel_size: FloatArrayType | None
    downsample_factor: int
    downsample_type: Literal['mean'] | Literal['max']
    skip_exist: bool
    overwrite : bool
    output_plots: bool
    max_imgs_to_plot: int


    def __init__(self,
        image_descriptor: str | ImageDescriptor,
        folder_output: str = '',
        n_stacks_max: int = -1,
        pixel_size: float | FloatArrayType | None = None,
        downsample_factor: int = 1,
        downsample_type: Literal['mean'] | Literal['max'] = 'mean',
        skip_exist: bool = False,
        overwrite: bool = False,
        flag_plots: bool = True,
        device: str | torch.device = 'cpu'
    ):
        """Constructor for object that handles particle stack conversion.

        Args:
            image_descriptor (str | ImageDescriptor): Object detailing the
                grids that describe the input images. If a string, it should
                be the path to a file on the filesystem with an image
                descriptor.
            folder_output (str, optional): Base for the output directory structure,
                which will be created if it does not exist. Defaults to the current
                directory.
            n_stacks_max (int, optional): If set to a positive number, the converter
                will stop converting image files after this many stacks have been
                emitted. Defaults to -1 (no limit).
            pixel_size (float | FloatArrayType | None, optional): The size, in Angstrom,
                of the pixels in the images to convert. Defaults to None, which is only
                appropriate if the actual pixel size is present in the descriptor file
                (as it may be for indexed Cryosparc files). If processing a Cryosparc
                file which has an internal image size set, an error will be generated
                if the value passed here conflicts with the value from the file.
            downsample_factor (int, optional): If set to a positive integer,
                downsampling will be done to that level during image processing.
                Defaults to 1 (no downsampling).
            skip_exist (bool, optional): Not inmplemented. Defaults to False.
            overwrite (bool, optional): Whether to overwrite exiting files. Defaults to False.
            flag_plots (bool, optional): Whether to emit plots of a few images
                of each processed batch. Defaults to True.
            device (str | torch.device, optional): Device on which to do image conversion
                from MRC files. Defaults to 'cpu'.
        """
        self.inputs_buffer = deque()
        self.img_desc = ImageDescriptor.ensure(image_descriptor)
        self.images_buffer = ImgBuffer()
        self.lens_desc_buffer = LensDescriptorBuffer(LensDescriptor())

        self.out_dirs = OutputFolders(folder_output)
        self.max_stacks = n_stacks_max
        # TODO: standardize options for pixel size handling
        if np.isscalar(pixel_size):
            ps = float(pixel_size)          # type: ignore
            ensure_positive(ps, "pixel size")
            self.pixel_size = project_descriptor(ps, "pixel size", 2, TargetType.FLOAT)
            self.img_desc.update_pixel_size(self.pixel_size)
            print(f"Pixel size manually set to {self.pixel_size}")
        else:
            self.pixel_size = pixel_size    # type: ignore
        self.downsample_factor = downsample_factor
        self.downsample_type = downsample_type
        if skip_exist:
            raise NotImplementedError("Skip-exist is not yet implemented.")
        self.skip_exist = skip_exist
        self.overwrite = overwrite
        self.output_plots = flag_plots
        self.max_imgs_to_plot = 16
        self.i_stacks = 0
        self._stack_start_file = 0
        self._stack_absolute_index = 0
        self._must_flush_buffer = False
        self.device = torch.device(device)


    def _can_load_cryosparc(self):
        if len(self.inputs_buffer) == 0: return True
        for x in self.inputs_buffer:
            if x[0] != "starfile":
                return False
        return True


    def _confirm_pixel_size(self, ignore_manual_pixel_size: bool):
        if self.lens_desc.ref_pixel_size is not None:
            if self.pixel_size is None:
                self.pixel_size = self.lens_desc.ref_pixel_size
                # TODO: cs_pixel_size will be ONE-D. Need to upscale.
            elif np.allclose(self.pixel_size, self.lens_desc.ref_pixel_size):
                pass
            else: # a pixel size was manually set, and does not match the one in the file
                msg = f"Manually set pixel size ({self.pixel_size}) does not match record pixel size ({self.lens_desc.ref_pixel_size})."
                if not ignore_manual_pixel_size:
                    raise ValueError(f"{msg}\nAborting unless override is set.")
                else:
                    print(f"{msg}\nUsing file-specified size over manually set size.")
                    self.pixel_size = self.lens_desc.ref_pixel_size
                    self.img_desc.update_pixel_size(self.pixel_size)
        if self.pixel_size is None:
            raise ValueError("Pixel size was never set.")


    def prepare_star_files(self,
        particle_file_list: list[str],
        star_file_list: list[str],
        defocus_angle_is_degree: bool = True,
        phase_shift_is_degree: bool = True
    ):
        """Preprocesses image and starfiles so they are ready for conversion.

        Args:
            particle_file_list (list[str]): List of filesystem paths pointing to MRC
                files containing image records
            star_file_list (list[str]): List of filesystem paths pointing to Starfile
                descriptors for the image records
            defocus_angle_is_degree (bool, optional): If True, the defocus angle values in
                the starfiles are presumed to be in degrees and will be converted to radians.
                Defaults to True.
            phase_shift_is_degree (bool, optional): If True, the phase shift values in the
                starfiles are presumed to be in degrees and will be converted to radians.
                Defaults to True.
        """
        _ensure_files_exist(particle_file_list)
        _ensure_files_exist(star_file_list)
        for p_file, s_file in zip(particle_file_list, star_file_list):
            input = StarfileInput(p_file, s_file, defocus_angle_is_degree, phase_shift_is_degree)
            self.inputs_buffer.append(("starfile", input))


    def prepare_indexed_star_file(self,
        star_file: str,
        folder_mrc: str = '',
        ignore_manual_pixel_size: bool = False
    ):
        """Preprocesses an indexed Starfile file for conversion.

        Args:
            starfile (str): Path to Starfile to process
            folder_mrc (str, optional): Folder where MRC files can be found. If left as an empty
                string (''), assume that the source Starfile contains a correct relative path to
                the MRC files. Defaults to ''.
            ignore_manual_pixel_size (bool, optional): If True, will attempt to resolve conflicts
                between the pxiel size in the Starfile and one that's manually input by the caller.
                This may not be a good idea. Defaults to False.
        """
        if not self._can_load_cryosparc():
            raise ValueError("Refusing to batch additional indexed Starfile with a non-empty buffer.")
        self.lens_desc = LensDescriptor.from_indexed_starfile(star_file, True)
        self._confirm_pixel_size(ignore_manual_pixel_size)
        self.lens_desc_buffer.update_parent(self.lens_desc)
        files_idxs = _get_filenames_and_indices(self.lens_desc, folder_mrc)
        for mrc_file, img_idxs, row_idxs in files_idxs:
            input = Indexed(mrc_file=mrc_file, selected_img_indices=img_idxs, selected_lensdesc_indices=row_idxs)
            self.inputs_buffer.append(("indexed_starfile", input))


    def prepare_indexed_cryosparc(self, file_cs: str, folder_cryosparc: str, ignore_manual_pixel_size: bool = False):
        """Preprocesses an indexed Cryosparc file (which contains descriptors of multiple image files).

        Args:
            file_cs (str): Cryosparc file to process
            folder_cryosparc (str): Folder where the MRC files described in the Cryosparc file
                can be located
            ignore_manual_pixel_size (bool, optional): Indexed Cryosparc files may contain a pixel
                size for the images they describe. However, the user can also pass a pixel size
                manually when setting up a ParticleStackConverter. If this parameter is False
                (the default), then the converter will throw a fatal error if these two values
                conflict. If this parameter is True, the converter emits a warning about the
                situation, and proceeds using the value from the source Cryosparc file. This
                parameter has no effect if the Cyrosparc file does not have pixel size information.
        """
        if not self._can_load_cryosparc():
            raise ValueError("Refusing to batch additional Cryosparc files with a non-empty buffer.")
        self.lens_desc = LensDescriptor.from_cryosparc_file(file_cs, get_fs_data=True)
        self._confirm_pixel_size(ignore_manual_pixel_size)
        self.lens_desc_buffer.update_parent(self.lens_desc)
        files_with_indices = _get_filenames_and_indices(self.lens_desc, folder_cryosparc)
        for mrc_file, img_idxs, row_idxs in files_with_indices:
            input = Indexed(mrc_file=mrc_file, selected_img_indices=img_idxs, selected_lensdesc_indices=row_idxs)
            self.inputs_buffer.append(("indexed_cryosparc", input))


    def prepare_sequential_cryosparc(self, folder_cryosparc: str, job_number: int = 0):
        """Preprocesses a set of sequential files with a Cryosparc descriptor. The expected
        directory structure is as follows. Assume:
            - the parent folder is FOLDER
            - the job number is 15
        Then we expect the following directories to exist:
            - FOLDER/J15/J15_passthrough_particles.cs
            - An MRC file folder. This should be one of:
                - FOLDER/J15/restack
                - FOLDER/J15/downsample
              If both exist, the `restack` folder will be used.
            - MRC files in this folder should follow the naming convention:
                - batch_0_restacked.mrc, batch_1_restacked.mrc, ... for `restack` or
                - batch_000000_downsample.mrc, batch_000001_downsample.mrc, ... for `downsample`
        MRC files will be processed sequentially until the first missing number in the sequence.

        Args:
            folder_cryosparc (str): Path to directory on the filesystem where files
                are located
            job_number (int, optional): The number of the job, used to build out the
                expected directory structure for the source MRC files. Defaults to 0.
        """
        if not self._can_load_cryosparc():
            raise ValueError("Refusing to batch additional Cryosparc files with a non-empty buffer.")
        job_paths = JobPaths(folder_cryosparc, job_number)
        self.lens_desc = LensDescriptor.from_cryosparc_file(job_paths.file_cs)
        self.lens_desc_buffer.update_parent(self.lens_desc)
        i_file = 0
        while True:
            mrc_path = job_paths.get_mrc_filename(i_file)
            if mrc_path is None:
                break
            i_file += 1
            self.inputs_buffer.append(("sequential_cryosparc", SequentialCryosparc(mrc_path)))

    
    def _transform_and_normalize_images(self, im: Images):
        if self.downsample_factor > 1:
            im.downsample_images_phys(self.downsample_factor, self.downsample_type)
        print(f"Physical images shape: {im.images_phys.shape}")
        im.center_physical_image_signal()
        im.transform_to_fourier(polar_grid=self.img_desc.polar_grid, precision=self.img_desc.precision)
        im.normalize_images_fourier(ord=2, use_max=False)
        print(f"Fourier images shape: {im.images_fourier.shape}")


    def convert_stacks(self, batch_size: int = 1024, never_combine_input_files: bool = False):
        """After preprocessing is complete, this function actually does the image conversion
        and outputs regular-sized batches. For Starfile inputs, each input file will result
        in one or more output stacks; for Cryosparc files, image inputs will be buffered
        and restacked, with only the final stack being smaller than the requested batch size.

        If desired, the Starfile behavior (ensure each source file gets its own stack or stacks)
        can be emulated for Cryosparc files.

        Args:
            batch_size (int, optional): Target stack size. Defaults to 1024.

            never_combine_input_files (bool, optional): If set, Cryosparc source files will
                be restacked in the same way as Starfile sources, i.e. one source file will
                generate one or more output stacks, but no output stack will contain images
                from multiple source files. Defaults to False.

        """
        if len(self.inputs_buffer) == 0:
            print(f"Warning: you must prepare input files before running convert_stacks.")
            return

        while(self._load_next_input()):
            target_buffer_size = batch_size
            using_overall_counter = False
            if never_combine_input_files or self._must_flush_buffer:
                target_buffer_size = 1
                # When we are not combining files, we may still split them.
                # In that case, we want to write the start/end index from the
                # split file into the output, resetting with every file.
                using_overall_counter = True
                self._stack_absolute_index = 0
            while self.images_buffer.stack_size >= target_buffer_size:
                self._write_batch(batch_size, using_overall_counter)
                self.i_stacks += 1
                if self.max_stacks > 0 and self.i_stacks >= self.max_stacks:
                    # Abandon unprocessed files and anything left in the buffer
                    return
        # handle remainder
        if self.images_buffer.stack_size > 0:
            # overall counter is only used when doing a complete buffer flush,
            # in which case this section of code should never be executed
            self._write_batch(batch_size, False)
            self.i_stacks += 1


    def _load_next_input(self):
        if len(self.inputs_buffer) == 0:
            return False
        row = self.inputs_buffer.popleft()
        if self.pixel_size is None:
            raise ValueError("Pixel size was never set. This shouldn't happen.")
        if row[0] == 'starfile':
            self._load_starfile(row[1])
        if row[0] == 'indexed_cryosparc':
            self._load_indexed(row[1])
        if row[0] == 'indexed_starfile':
            self._load_indexed(row[1])
        if row[0] == 'sequential_cryosparc':
            self._load_sequential_cryosparc(row[1])
        return True

    
    def _load_starfile(self, r: StarfileInput):
        lens_desc = LensDescriptor.from_starfile(r.star_file, r.defocus_is_degree, r.phase_shift_is_degree)
        self.lens_desc_buffer.update_parent(lens_desc)
        im = Images.from_mrc(r.particle_file, self.pixel_size, self.device)
        self._transform_and_normalize_images(im)

        self.images_buffer.append_imgs(im)
        self.lens_desc_buffer.enqueue(lens_desc.batch_whole())
        self._must_flush_buffer = True


    def _load_indexed(self, r: Indexed):
        if r.selected_img_indices is None:
            print(f"Warning: empty image selection for {r.mrc_file}, skipping {r.mrc_file}.")
            return
        # if len(r.selected_img_indices) < 1:
        #     print(f"Warning: empty image selection for {r.mrc_file}, skipping {r.mrc_file}.")
        #     return
        im = Images.from_mrc(r.mrc_file, self.pixel_size, self.device)
        im.select_images(r.selected_img_indices)
        self._transform_and_normalize_images(im)

        self.images_buffer.append_imgs(im)
        self.lens_desc_buffer.enqueue(self.lens_desc.get_selections(r.selected_lensdesc_indices))
        self._must_flush_buffer = False
    

    def _load_sequential_cryosparc(self, r: SequentialCryosparc):
        im = Images.from_mrc(r.mrc_file, self.pixel_size, self.device)
        self._transform_and_normalize_images(im)
        slice_start = self._stack_start_file
        slice_end = slice_start + im.n_images
        self._stack_start_file = slice_end

        self.images_buffer.append_imgs(im)
        self.lens_desc_buffer.enqueue(self.lens_desc.get_slice(slice_start, slice_end))
        self._must_flush_buffer = False


    def _write_batch(self, batch_size: int, use_overall_counter: bool = False):
        lens_batch = self.lens_desc_buffer.pop_batch(batch_size)
        (phys_batch, fourier_batch) = self.images_buffer.pop_imgs(batch_size)
        if lens_batch.stack_size != len(phys_batch):
            raise ValueError(f"Length mismatch between images ({len(phys_batch)}) and descriptors ({lens_batch.stack_size}).")

        # The actual batch size may be smaller than target if we're clearing a remainder
        actual_batch_size = phys_batch.shape[0]
        print(f"Stacking {actual_batch_size} images")
        output_fns = self.out_dirs.get_output_filenames(self.i_stacks)

        if self.output_plots:
            self._plot_imgs(phys_batch, fourier_batch)
        torch.save(phys_batch, output_fns.phys_stack)
        torch.save(fourier_batch, output_fns.fourier_stack)

        save_combined_params(
            output_fns.params_filename,
            self.img_desc,
            lens_batch,
            n_imgs_this_stack=actual_batch_size,
            overall_batch_start=self._stack_absolute_index if use_overall_counter else None,
            overwrite=self.overwrite
        )
        if use_overall_counter:
            self._stack_absolute_index += actual_batch_size


    def _plot_imgs(self, phys_img: torch.Tensor, fourier_img: torch.Tensor):
        n_plots = min(self.max_imgs_to_plot, phys_img.shape[0])
        plot_dir = self.out_dirs.folder_output_plots
        count_suffix = f"{self.i_stacks:06}.png"
        fn_phys = path.join(plot_dir, f"particles_phys_stack_{count_suffix}")
        fn_four = path.join(plot_dir, f"particles_fourier_stack_{count_suffix}")
        fn_power = path.join(plot_dir, f"power_spectrum_stack_{count_suffix}")
        pol_grid = self.img_desc.polar_grid
        c_grid = self.img_desc.cartesian_grid

        plot_images(phys_img, grid=c_grid, n_plots=n_plots, filename=fn_phys)
        plot_images(fourier_img, grid=pol_grid, n_plots=n_plots, filename=fn_four)
        power_spec_tuple = (fourier_img, pol_grid, c_grid.box_size)
        plot_power_spectrum(power_spec_tuple, filename_plot=fn_power)


