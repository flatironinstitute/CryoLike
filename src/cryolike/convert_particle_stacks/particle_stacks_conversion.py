from os import path
import torch
from typing import TypeVar
from numpy import ndarray
import numpy as np

from .particle_stacks_pathing import JobPaths, OutputFolders
from .particle_stacks_metadata import _Metadata
from .particle_stacks_buffers import ImgBuffer, _MetadataBuffer

from cryolike.grids import CartesianGrid2D, PolarGrid, PhysicalImages
from cryolike.microscopy import read_star_file, ensure_parameters, print_parameters, ParsedParameters
from cryolike.plot import plot_images, plot_power_spectrum
from cryolike.stacks import Images
from cryolike.util import FloatArrayType, Precision, TargetType, project_descriptor, ensure_positive


T2 = TypeVar("T2", bound=float | FloatArrayType | ndarray | None)
def _copy(d: T2) -> T2:
    if isinstance(d, ndarray):
        return d.copy()
    return d


def _ensure_files_exist(files: list[str]):
    for f in files:
        if not path.exists(f):
            raise ValueError("Error: file not found: ", f)


def _make_polar_grid_from_params(params: ParsedParameters) -> PolarGrid:
    polar_grid = PolarGrid(
        radius_max = params.radius_max,
        dist_radii = params.dist_radii,
        n_inplanes = params.n_inplanes,
        uniform = True
    )
    return polar_grid


def _do_skip_exist(skip_exist: bool, image_fourier_file: str, image_param_file: str, im: Images, stack_count: int) -> bool:
    if not skip_exist:
        return False
    if not path.exists(image_fourier_file) or not path.exists(image_param_file):
        return False
    n_images = np.load(image_param_file, allow_pickle = True)['n_images']
    if n_images == im.n_images:
        print("Skipping stack %d" % stack_count)
        return True
    return False


def _do_image_normalization(im: Images, polar_grid: PolarGrid,
                            precision: Precision,
                            downsample_physical: int = 1,
                            use_cuda: bool = True):
    assert im.images_phys is not None
    if downsample_physical > 1:
        im.downsample_images_phys(downsample_physical)
    print("Physical images shape: ", im.images_phys.shape)
    im.center_physical_image_signal()
    im.transform_to_fourier(polar_grid=polar_grid, precision=precision, use_cuda=use_cuda)
    im.normalize_images_fourier(ord=2, use_max=False)
    assert im.images_fourier is not None
    print("Fourier images shape: ", im.images_fourier.shape)


def _get_unbuffered_batch(i_batch: int, batch_size: int, im: Images) -> tuple[int, int, Images]:
    batch_start = int(i_batch * batch_size)
    batch_end = min(batch_start + batch_size, im.n_images)
    assert im.images_phys is not None
    im_batch = Images(
        phys_images_data = PhysicalImages(
            images_phys = im.images_phys[batch_start:batch_end],
            pixel_size = im.phys_grid.pixel_size
        )
    )
    return (batch_start, batch_end, im_batch)


def _plot_images(
        do_plots: bool,
        im: Images,
        dirs: OutputFolders,
        i_stack: int,
        phys_img: torch.Tensor | None = None,
        fourier_img: torch.Tensor | None = None
    ):
    if not do_plots:
        return
    _phys_img = phys_img if phys_img is not None else im.images_phys
    _fourier_imgs = fourier_img if fourier_img is not None else im.images_fourier
    assert _phys_img is not None
    assert _fourier_imgs is not None
    _plot_images_inner(dirs, _phys_img, im.phys_grid, _fourier_imgs, im.polar_grid, i_stack)


def _plot_images_inner(
        dirs: OutputFolders,
        phys_img: torch.Tensor,
        phys_grid: CartesianGrid2D,
        fourier_img: torch.Tensor,
        polar_grid: PolarGrid,
        i_stack: int
    ):
    n_plots = min(16, phys_img.shape[0])

    fn_plots_phys = path.join(dirs.folder_output_plots, f"particles_phys_stack_{i_stack:06}.png")
    plot_images(phys_img, grid=phys_grid, n_plots=n_plots, filename=fn_plots_phys)
    fn_plots_fourier = path.join(dirs.folder_output_plots, f"particles_fourier_stack_{i_stack:06}.png")
    plot_images(fourier_img, grid=polar_grid, n_plots=n_plots, filename=fn_plots_fourier)
    fn_plots_power = path.join(dirs.folder_output_plots, f"power_spectrum_stack_{i_stack:06}.png")
    plot_power_spectrum((fourier_img, polar_grid, phys_grid.box_size), filename_plot=fn_plots_power)


# This was done at the beginning of each conversion function, but the result
# wasn't used anywhere.
def _collect_image_tag_list(folder_output: str):
    tag_file = path.join(folder_output, "image_file_tag_list.npy")
    if not path.exists(tag_file):
        return []
    image_file_tag_list = np.load(tag_file, allow_pickle = True)
    return image_file_tag_list.tolist()



def convert_particle_stacks_from_star_files(
    params_input: str | ParsedParameters,
    particle_file_list: list[str] = [],
    star_file_list: list[str] = [],
    folder_output: str = '',
    batch_size: int = 1024,
    pixel_size: float | FloatArrayType | None = None,
    defocus_angle_is_degree: bool = True,
    phase_shift_is_degree: bool = True,
    skip_exist: bool = False,
    flag_plots: bool = True,
    use_cuda: bool = True
):
    """Transcode a set of particle files, with metadata described in starfile format,
    to consistent batches in a specified output folder.
    
    Each source image file will be centered and normalized, and persisted in both
    physical/Cartesian and Fourier representations.

    Args:
        params_input (str | ParsedParameters): Parameters of the intended output. Mainly
            used to describe the polar grid for the Fourier-space representation and
            define the precision of our output.
        particle_file_list (list[str], optional): List of particle files to process. Defaults to [].
        star_file_list (list[str], optional): List of star files to process. Defaults to [].
            Should be of the same length and in the same order as the particle files.
        folder_output (str, optional): Folder in which to output transcoding results. Defaults to '',
            i.e. outputting in the current working directory.
        batch_size (int, optional): Maximum number of images per output file. Defaults to 1024.
        pixel_size (float | FloatArrayType | None, optional): Size of each pixel, in Angstroms,
            as a square side-length or Numpy array of (x, y). If unset (the default), it will
            be read from the MRC particle files.
        defocus_angle_is_degree (bool, optional): Whether the defocus angle in the metadata
            is in degrees (as opposed to radians). Defaults to True (for star files).
        phase_shift_is_degree (bool, optional): Whether the phase shift angle in the metadata
            is in degrees (as opposed to radians). Defaults to True (for star files).
        skip_exist (bool, optional): If True, we will skip processing on files that appear to
            have already been processed. This is currently not implemented, to avoid
            inadvertently dropping data. Defaults to False.
        flag_plots (bool, optional): Whether to plot images and power spectrum along with
            the transcoding results. Defaults to True.
    """
    if pixel_size is not None:
        ensure_positive(pixel_size, "pixel size")
        pixel_size = project_descriptor(pixel_size, "pixel size", 2, TargetType.FLOAT)
        print("pixel_size:", pixel_size)

    params = ensure_parameters(params_input)
    print_parameters(params)

    polar_grid = _make_polar_grid_from_params(params)
    _output_dirs = OutputFolders(folder_output)
    i_stack = 0

    _ensure_files_exist(particle_file_list)
    _ensure_files_exist(star_file_list)

    for particle_file, star_file in zip(particle_file_list, star_file_list):
        meta = _Metadata.from_star_file(star_file, defocus_is_degree=defocus_angle_is_degree, phase_shift_is_degree=phase_shift_is_degree)
        im = Images.from_mrc(particle_file, pixel_size=pixel_size)

        # NOTE: Not implemented at present due to the challenges of ensuring nothing is missed.
        # if _do_skip_exist(skip_exist, image_fourier_file, image_param_file, im, i_stack):
        #     i_stack += 1
        #     continue

        # NOTE: A prior version branched on whether there would be more than one batch; this
        # is probably not actually useful as the content was otherwise the same.
        # However, originally the skip_exist check was not present if there were multiple batches.
        n_batches = int(np.ceil(im.n_images / batch_size))
        for i_batch in range(n_batches):
            (batch_start, batch_end, im_batch) = _get_unbuffered_batch(i_batch, batch_size, im)
            output_filenames = _output_dirs.get_output_filenames(i_stack)

            _do_image_normalization(im_batch, polar_grid, params.precision, use_cuda)
            
            _plot_images(flag_plots, im_batch, _output_dirs, i_stack)
            torch.save(im_batch.images_phys, output_filenames.phys_stack)
            torch.save(im_batch.images_fourier, output_filenames.fourier_stack)

            batched_meta = meta.take_range(batch_start, batch_end)
            batched_meta.save_params_star(output_filenames.params_filename, im_batch.n_images, im_batch, batch_start, batch_end)
            i_stack += 1


def _get_filenames_and_image_selection_indices(
    metadata: _Metadata,
    folder_cryosparc: str
) -> list[tuple[str, np.ndarray]]:
    """Extract selected images from the files described in the metadata
    and associate them with the defocus and phase-shift values.

    Once the metadata file has been read, we have to do some matching.
    The metadata 'cs_files' field describes the MRC files which we'll need
    to read to get actual images. The metadata 'cs_idxs' field lists, for each
    image file, the indices into the per-image descriptor arrays (defocusU,
    defocusV, defocusAngle, phaseShift) that describe the images in that file.

    This function iterates over a unique list of MRC files from the Cryosparc
    description. We skip any files that can't be found. For the ones that are
    found, we return the actual verified path to the MRC file, as well as the
    indices into the source file data that will let us find the images we
    care about and the image descriptors for them.

    Args:
        metadata (_Metadata): Metadata populated from the main Cryosparc
            file. This is assumed to have the cs_idxs and cs_files fields set.
        folder_cryosparc (str): The root of the folder where the MRC files
            described in the Cryosparc file are stored.

    Returns:
        list[tuple[str, np.ndarray]]: A list of tuples, one for each existing
            MRC image file described in the cs_files field. The first part of the
            tuple is the file path to the MRC file; the second part is the complete
            set of indices (into the individual images and the defocus fields of the
            metadata) which identify the images/metadata values taken from the file.
    """
    assert metadata.cs_idxs is not None
    assert metadata.cs_files is not None

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
    unique_files, indices_files = np.unique(metadata.cs_files, return_inverse=True)

    mrc_files_with_img_indices = []

    for i, file in enumerate(unique_files):
        file = file.decode('utf-8')
        if file[0] == '>':
            file = file[1:]
        mrc_file_path = path.join(folder_cryosparc, file)
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
        selected_img_indices = metadata.cs_idxs[original_file_list_indices]
        entry = (mrc_file_path, selected_img_indices)
        mrc_files_with_img_indices.append(entry)
    return mrc_files_with_img_indices


def _make_Images_from_mrc_file(
    mrc_file_path: str,
    selected_imgs: ndarray,
    _pixel_size: FloatArrayType
):
    im = Images.from_mrc(mrc_file_path, pixel_size = _pixel_size)
    im.select_images(selected_imgs)

    return im


def _drain_buffer(
    batch_size: int,
    i_restack: int,
    metadata_buffer: _MetadataBuffer,
    img_buffer: ImgBuffer,
    _output_dirs: OutputFolders,
    im: Images,
    flag_plots: bool
):
    batch_meta = metadata_buffer.pop_batch(batch_size)
    (img_phys_batch, img_fourier_batch) = img_buffer.pop_imgs(batch_size)
    
    this_batch_size = img_phys_batch.shape[0]
    print(f"Stacking {this_batch_size} images")
    output_filenames = _output_dirs.get_output_filenames(i_restack)

    _plot_images(flag_plots, im, _output_dirs, i_restack, phys_img=img_phys_batch, fourier_img=img_fourier_batch)
    torch.save(img_phys_batch, output_filenames.phys_stack)
    torch.save(img_fourier_batch, output_filenames.fourier_stack)

    # NOTE: we are assuming that the physical grid is consistent between image files.
    # This is probably true, but should perhaps be checked
    batch_meta.save_params(output_filenames.params_filename, this_batch_size, im)
    

def convert_particle_stacks_from_cryosparc(
    params_input: str | ParsedParameters,
    file_cs: str,
    folder_cryosparc: str = '',
    folder_output: str = '',
    batch_size: int = 1024,
    n_stacks_max: int = -1,
    pixel_size: float | FloatArrayType | None = None,
    downsample_physical: int = 1,
    skip_exist: bool = False,
    flag_plots: bool = True,
    use_cuda: bool = True
):
    """Transcodes a set of MRC files, with a cryosparc metadata file, into internal
    representation, with optional downsampling.

    Each source image file will be centered and normalized, and persisted in
    both physical/Cartesian and Fourier representations.

    The cryospark metadata file is expected to contain fields identifying the MRC
    files to be processed (blob/path and blob/idx). Transcoded data will be batched
    into outputs of a specified maximum image count.

    Args:
        params_input (str | ParsedParameters): Parameters of the intended
            output. Mainly used to describe the polar grid for the Fourier
            representation and define output precision.
        file_cs (str, optional): Path to cryosparc metadata file.
        folder_cryosparc (str, optional): Parent folder of the cryospark
            output jobs. Assumes directory structure is
            'folder_cryosparc/J_/' where _ is the job number, with no
            padding. Defaults to the current directory.
        folder_output (str, optional): Directory to use for outputting
            transcoded image files and metadata. Defaults to the current
            direcotry.
        batch_size (int, optional): Maximum number of images to include
            in each output file. Defaults to 1024.
        n_stacks_max (int, optional): Maximum number of stacks to emit
            before aborting the transcoding process. If -1 (the default),
            the entire source data will be processed.
        pixel_size (float | FloatArrayType | None): Side length of each pixel,
            in Angstroms. If a scalar, this will be treated as the side length
            of square pixels. If a Numpy array, the first two elements will
            be taken as the x and y dimensions. If None (the default), we will
            attempt to read the value from the cryosparc file (blob/psize_A field).
        downsample_physical (int, optional): Downsampling factor to
            apply. Downsampling is skipped if the factor is 1 or below.
            Defaults to 1 (no downsampling).
        skip_exist (bool, optional): If True, we will skip processing
            of files that appear to have already been processed.
            This is currently not implemented.
        flag_plots (bool, optional): If True (the default), the function
            will output images and power spectrum along with the
            transcoding results.
    """
    params = ensure_parameters(params_input)
    print_parameters(params)

    polar_grid = _make_polar_grid_from_params(params)
    _output_dirs = OutputFolders(folder_output)

    metadata = _Metadata.from_cryospark_file(file_cs, get_fs_data=True)
    assert metadata.cs_files is not None
    assert metadata.cs_idxs is not None

    if pixel_size is None:
        pixel_size = metadata.cs_pixel_size
        if pixel_size is None:
            raise ValueError('Pixel size was not set and was not present in the cryosparc file.')
    _pixel_size = pixel_size if isinstance(pixel_size, ndarray) else np.array([pixel_size, pixel_size], dtype = float)

    n_images = metadata.defocusU.shape[0] 
    print("n_images:", n_images)

    files_with_indices = _get_filenames_and_image_selection_indices(metadata, folder_cryosparc)

    i_restack = 0
    img_buffer = ImgBuffer()
    metadata_buffer = _MetadataBuffer(metadata)

    _last_good_im = None
    for entry in files_with_indices:
        (mrc_filename, selected_indices) = entry
        im = _make_Images_from_mrc_file(mrc_filename, selected_indices, _pixel_size)
        metadata_buffer.append_batch(
            metadata.defocusU[selected_indices],
            metadata.defocusV[selected_indices],
            metadata.defocusAngle[selected_indices],
            metadata.phaseShift[selected_indices]
        )
        _last_good_im = im
        
        _do_image_normalization(im, polar_grid, params.precision, downsample_physical, use_cuda)
        img_buffer.append_imgs(im.images_phys, im.images_fourier)

        # Currently not implemented -- unclear if it would go here
        # if _do_skip_exist(skip_exist, image_fourier_file, image_param_file, im, i_restack):
        #     i_restack += 1
        #     continue

        while img_buffer.stack_size >= batch_size:
            _drain_buffer(
                batch_size,
                i_restack,
                metadata_buffer,
                img_buffer,
                _output_dirs,
                im,
                flag_plots
            )

            i_restack += 1
            if n_stacks_max > 0 and i_restack >= n_stacks_max:
                # Abandon unprocessed files and anything remaining in the buffer
                return
    # Catch any remainder
    if _last_good_im is not None:
        _drain_buffer(
            batch_size,
            i_restack,
            metadata_buffer,
            img_buffer,
            _output_dirs,
            _last_good_im,
            flag_plots
        )


def convert_particle_stacks_from_cryosparc_restack(
    params_input: str | ParsedParameters,
    folder_cryosparc: str = '',
    job_number: int = 0,
    folder_output: str = '',
    batch_size: int = 1024,
    n_stacks_max: int = -1,
    pixel_size: float = -1.,
    downsample_physical: int = 1,
    skip_exist: bool = False,
    flag_plots: bool = True,
    use_cuda: bool = True
):
    """Transcodes a set of (previously restacked) MRC files into internal
    representation, with optional downsampling.

    Each source image file will be centered and normalized, and persisted in
    both physical/Cartesian and Fourier representations.
    
    This function assumes metadata is stored in a cryosparc format,
    and the previously restacked images are stored in a single
    job directory as sequentially numbered files whose names match
    the pattern 'batch_000000_restacked.mrc' (for non-downsampled files)
    or 'batch_000000_downsample.mrc' (for previously downsampled files).

    Args:
        params_input (str | ParsedParameters): Parameters of the intended
            output. Mainly used to describe the polar grid for the Fourier
            representation and define output precision.
        folder_cryosparc (str, optional): Parent folder of the cryospark
            output jobs. Assumes directory structure is
            'folder_cryosparc/J_/' where _ is the job number, with no
            padding. Defaults to the current directory.
        job_number (int, optional): Job number of the cryosparc output.
            Used to form the job output directory. The cryosparc metadata
            file describing the job is assumed to follow the format
            'Jx_passthrough_particles.cs', where x is the job number
            (unpadded).
        folder_output (str, optional): Directory to use for outputting
            transcoded image files and metadata. Defaults to the current
            direcotry.
        batch_size (int, optional): Maximum number of images to include
            in each output file. Defaults to 1024.
        n_stacks_max (int, optional): Maximum number of stacks to emit
            before aborting the transcoding process. If -1 (the default),
            the entire source data will be processed.
        pixel_size (float): Side length of each pixel, in Angstroms.
            Pixels are assumed square.
        downsample_physical (int, optional): Downsampling factor to
            apply. Downsampling is skipped if the factor is 1 or below.
            Defaults to 1 (no downsampling).
        skip_exist (bool, optional): If True, we will skip processing
            of files that appear to have already been processed.
            This is currently not implemented.
        flag_plots (bool, optional): If True (the default), the function
            will output images and power spectrum along with the
            transcoding results.
    """
    params = ensure_parameters(params_input)
    print_parameters(params)

    if pixel_size <= 0:
        raise ValueError("Error: pixel size must be positive.")
    _pixel_size: FloatArrayType = np.array([pixel_size, pixel_size], dtype = float)

    polar_grid = _make_polar_grid_from_params(params)
    _output_dirs = OutputFolders(folder_output)
    _job_paths = JobPaths(folder_cryosparc, job_number)

    metadata = _Metadata.from_cryospark_file(_job_paths.file_cs)

    n_images = metadata.defocusU.shape[0] 
    print("n_images:", n_images)

    stack_start_file = 0
    stack_end_file = 0
    i_restack = 0
    i_file = 0
    img_buffer = ImgBuffer()
    metadata_buffer = _MetadataBuffer(metadata)

    im = None
    while True:
        mrc_path = _job_paths.get_mrc_filename(i_file)
        if mrc_path is None:
            break
        im = Images.from_mrc(mrc_path, pixel_size = _pixel_size)

        i_file += 1
        _do_image_normalization(im, polar_grid, params.precision, downsample_physical, use_cuda)
        
        stack_end_file = stack_start_file + im.n_images
        img_buffer.append_imgs(im.images_phys, im.images_fourier)
        metadata_buffer.append_batch(
            metadata.defocusU[stack_start_file:stack_end_file],
            metadata.defocusV[stack_start_file:stack_end_file],
            metadata.defocusAngle[stack_start_file:stack_end_file],
            metadata.phaseShift[stack_start_file:stack_end_file],
        )
        stack_start_file = stack_end_file
        
        while img_buffer.stack_size >= batch_size:
            
            _drain_buffer(
                batch_size,
                i_restack,
                metadata_buffer,
                img_buffer,
                _output_dirs,
                im,
                flag_plots
            )
            i_restack += 1
            
            if n_stacks_max > 0 and i_restack >= n_stacks_max:
                return

    # Handle remainder
    if im is not None:
        _drain_buffer(
            batch_size,
            i_restack,
            metadata_buffer,
            img_buffer,
            _output_dirs,
            im,
            flag_plots
        )
