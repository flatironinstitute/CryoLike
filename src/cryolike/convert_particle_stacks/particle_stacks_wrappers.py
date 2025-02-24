from typing import Literal

from cryolike.util import FloatArrayType
from cryolike.metadata import ImageDescriptor
from .particle_stacks_converter import ParticleStackConverter

# TODO QUERY: should we allow a pixel_size separate from the parameters file?
def convert_particle_stacks_from_star_files(
    params_input: str | ImageDescriptor,
    particle_file_list: list[str],
    star_file_list: list[str],
    folder_output: str = '',
    batch_size: int = 1024,
    pixel_size: float | FloatArrayType | None = None,
    defocus_angle_is_degree: bool = True,
    phase_shift_is_degree: bool = True,
    downsample_factor: int = 1,
    downsample_type: Literal['mean'] | Literal['max'] = 'mean',
    skip_exist: bool = False,
    flag_plots: bool = True,
    overwrite: bool = False
):
    """Transcode a set of particle files, with metadata described in starfile format,
    to consistent batches in a specified output folder.
    
    Each source image file will be centered and normalized, and persisted in both
    physical/Cartesian and Fourier representations.

    Args:
        params_input (str | ImageDescriptor): Parameters of the intended output. Mainly
            used to describe the polar grid for the Fourier-space representation and
            define the precision of our output.
        particle_file_list (list[str]): List of particle files to process. Defaults to [].
        star_file_list (list[str]): List of star files to process. Defaults to [].
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
        overwrite (bool, optional): Whether to overwrite existing stacks. Defaults to False.
    """
    converter = ParticleStackConverter(
        image_descriptor=params_input,
        folder_output=folder_output,
        n_stacks_max=-1,
        pixel_size=pixel_size,
        downsample_factor=downsample_factor,
        downsample_type=downsample_type,
        flag_plots=flag_plots
    )
    converter.prepare_star_files(
        particle_file_list=particle_file_list,
        star_file_list=star_file_list,
        defocus_angle_is_degree=defocus_angle_is_degree,
        phase_shift_is_degree=phase_shift_is_degree
    )
    converter.convert_stacks(batch_size=batch_size, overwrite=overwrite)


def convert_particle_stacks_from_indexed_star_files(
    params_input: str | ImageDescriptor,
    star_file: str,
    folder_mrc: str = '',
    folder_output: str = '',
    batch_size: int = 1024,
    downsample_factor: int = 1,
    downsample_type: Literal['mean'] | Literal['max'] = 'mean',
    pixel_size: float | FloatArrayType | None = None,
    flag_plots: bool = True
):
    """Transcode a set of particle files, with metadata described in starfile format,
    to consistent batches in a specified output folder.
    
    Each source image file will be centered and normalized, and persisted in both
    physical/Cartesian and Fourier representations.

    Args:
        params_input (str | ImageDescriptor): Parameters of the intended output. Mainly
            used to describe the polar grid for the Fourier-space representation and
            define the precision of our output.
        star_file_list (str): List of star files to process. Defaults to [].
            Should be of the same length and in the same order as the particle files.
        folder_mrc (str): Folder containing the MRC files. If set to '', use relative path stated in the star file.
        folder_output (str, optional): Folder in which to output transcoding results. Defaults to '',
            i.e. outputting in the current working directory.
        batch_size (int, optional): Maximum number of images per output file. Defaults to 1024.
        pixel_size (float | FloatArrayType | None, optional): Size of each pixel, in Angstroms,
            as a square side-length or Numpy array of (x, y). If unset (the default), it will
            be read from the MRC particle files.
        flag_plots (bool, optional): Whether to plot images and power spectrum along with
            the transcoding results. Defaults to True.
    """
    converter = ParticleStackConverter(
        image_descriptor=params_input,
        folder_output=folder_output,
        n_stacks_max=-1,
        pixel_size=pixel_size,
        downsample_factor=downsample_factor,
        downsample_type=downsample_type,
        flag_plots=flag_plots
    )
    converter.prepare_indexed_star_file(
        star_file=star_file,
        folder_mrc=folder_mrc,
    )
    converter.convert_stacks(batch_size=batch_size)


# TODO QUERY: Do we need a separate pixel size from the one in the image descriptor
def convert_particle_stacks_from_cryosparc(
    params_input: str | ImageDescriptor,
    file_cs: str,
    folder_cryosparc: str = '',
    folder_output: str = '',
    batch_size: int = 1024,
    n_stacks_max: int = -1,
    pixel_size: float | FloatArrayType | None = None,
    downsample_factor: int = 1,
    downsample_type: Literal['mean'] | Literal['max'] = 'mean',
    skip_exist: bool = False,
    flag_plots: bool = True
):
    """Transcodes a set of MRC files, with a cryosparc metadata file, into internal
    representation, with optional downsampling.

    Each source image file will be centered and normalized, and persisted in
    both physical/Cartesian and Fourier representations.

    The cryospark metadata file is expected to contain fields identifying the MRC
    files to be processed (blob/path and blob/idx). Transcoded data will be batched
    into outputs of a specified maximum image count.

    Args:
        params_input (str | ImageDescriptor): Parameters of the intended
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
        downsample_factor (int, optional): Downsampling factor to
            apply. Downsampling is skipped if the factor is 1 or below.
            Defaults to 1 (no downsampling).
        downsample_type (Literal['mean'] | Literal['max']): The type of downsampling to use in physical space
        skip_exist (bool, optional): If True, we will skip processing
            of files that appear to have already been processed.
            This is currently not implemented.
        flag_plots (bool, optional): If True (the default), the function
            will output images and power spectrum along with the
            transcoding results.
    """
    converter = ParticleStackConverter(
        image_descriptor=params_input,
        folder_output=folder_output,
        n_stacks_max=n_stacks_max,
        pixel_size=pixel_size,
        downsample_factor=downsample_factor,
        downsample_type=downsample_type,
        flag_plots=flag_plots
    )
    converter.prepare_indexed_cryosparc(file_cs=file_cs, folder_cryosparc=folder_cryosparc)
    converter.convert_stacks(batch_size=batch_size)


def convert_particle_stacks_from_cryosparc_restack(
    params_input: str | ImageDescriptor,
    folder_cryosparc: str = '',
    job_number: int = 0,
    folder_output: str = '',
    batch_size: int = 1024,
    n_stacks_max: int = -1,
    pixel_size: float = -1.,
    downsample_factor: int = 1,
    downsample_type: Literal['mean'] | Literal['max'] = 'mean',
    skip_exist: bool = False,
    flag_plots: bool = True
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
        params_input (str | ImageDescriptor): Parameters of the intended
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
        downsample_factor (int, optional): Downsampling factor to
            apply. Downsampling is skipped if the factor is 1 or below.
            Defaults to 1 (no downsampling).
        skip_exist (bool, optional): If True, we will skip processing
            of files that appear to have already been processed.
            This is currently not implemented.
        flag_plots (bool, optional): If True (the default), the function
            will output images and power spectrum along with the
            transcoding results.
    """
    converter = ParticleStackConverter(
        image_descriptor=params_input,
        folder_output=folder_output,
        n_stacks_max=n_stacks_max,
        pixel_size=pixel_size,
        downsample_factor=downsample_factor,
        downsample_type=downsample_type,
        flag_plots=flag_plots
    )
    converter.prepare_sequential_cryosparc(folder_cryosparc, job_number)
    converter.convert_stacks(batch_size=batch_size)
