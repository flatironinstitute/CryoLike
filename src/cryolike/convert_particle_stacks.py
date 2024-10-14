import os
from typing import Literal, NamedTuple, TypeVar, cast
from cryolike.cartesian_grid import CartesianGrid2D
from cryolike.util.enums import Precision
from cryolike.util.reformatting import TargetType, project_descriptor
from cryolike.util.typechecks import ensure_positive
from matplotlib import pyplot as plt
import numpy as np
import torch

from cryolike.polar_grid import PolarGrid
from cryolike.util.types import ComplexArrayType, FloatArrayType
from cryolike.image import Images, PhysicalImages
from cryolike.star_file import read_star_file
from cryolike.plot import plot_images, plot_power_spectrum
from cryolike.parameters import ensure_parameters, print_parameters, ParsedParameters


T = TypeVar("T", bound=FloatArrayType | ComplexArrayType | torch.Tensor)
def _pop_batch(u: T, batch_size: int) -> tuple[T, T]:
    head = cast(T, u[:batch_size])
    tail = cast(T, u[batch_size:])
    return (head, tail)


T2 = TypeVar("T2", bound=float | FloatArrayType | np.ndarray | None)
def _copy(d: T2) -> T2:
    if isinstance(d, np.ndarray):
        return d.copy()
    return d


def _ensure_np(d: float | FloatArrayType) -> FloatArrayType:
    if isinstance(d, np.ndarray):
        return d
    return np.array(d)


def _batchify(d: T2, start: int, end: int) -> T2:
    if isinstance(d, np.ndarray):
        return cast(T2, d[start:end])
    return d


def _ensure_files_exist(files: list[str]):
    for f in files:
        if not os.path.exists(f):
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
    if not os.path.exists(image_fourier_file) or not os.path.exists(image_param_file):
        return False
    n_images = np.load(image_param_file, allow_pickle = True)['n_images']
    if n_images == im.n_images:
        print("Skipping stack %d" % stack_count)
        return True
    return False


def _do_image_normalization(im: Images, polar_grid: PolarGrid, precision: Precision, downsample_physical: int = 1):
    assert im.images_phys is not None
    if downsample_physical > 1:
        im.downsample_images_phys(downsample_physical)
    print("Physical images shape: ", im.images_phys.shape)
    im.center_physical_image_signal()
    im.transform_to_fourier(polar_grid=polar_grid, precision=precision, use_cuda=True)
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


class OutputFilenames(NamedTuple):
    phys_stack: str
    fourier_stack: str
    params_filename: str


class OutputFolders():
    folder_output_plots: str
    folder_output_particles_fft: str
    folder_output_particles_phys: str

    def __init__(self, folder_output: str):
        self.folder_output_plots = os.path.join(folder_output, 'plots')
        os.makedirs(self.folder_output_plots, exist_ok=True)
        self.folder_output_particles_fft = os.path.join(folder_output, "fft")
        os.makedirs(self.folder_output_particles_fft, exist_ok=True)
        self.folder_output_particles_phys = os.path.join(folder_output, "phys")
        os.makedirs(self.folder_output_particles_phys, exist_ok=True)


    def get_output_filenames(self, i_stack: int) -> OutputFilenames:
        phys_stack = os.path.join(self.folder_output_particles_phys, f"particles_phys_stack_{i_stack:06}.pt")
        fourier_base = os.path.join(self.folder_output_particles_fft, f"particles_fourier_stack_{i_stack:06}")
        fourier_stack = fourier_base + ".pt"
        params_fn = fourier_base + ".npz"
        return OutputFilenames(phys_stack, fourier_stack, params_fn)


class JobPaths():
    file_cs: str
    folder_type: Literal["restacks"] | Literal["downsample"]
    restacks_folder: str
    downsample_folder: str


    def __init__(self, folder_cryosparc: str, job_number: int):
        folder_job = os.path.join(folder_cryosparc, "J%d" % job_number)
        if not os.path.exists(folder_job):
            raise ValueError("Error: folder not found: ", folder_job)
        self.restacks_folder = os.path.join(folder_job, "restack")
        self.downsample_folder = os.path.join(folder_job, "downsample")
        if not os.path.exists(self.restacks_folder) and not os.path.exists(self.downsample_folder):
            raise ValueError("Error: folder not found: ", self.restacks_folder, " and ", self.downsample_folder)
        self.folder_type = "restacks" if os.path.exists(self.restacks_folder) else "downsample"
        self.file_cs = os.path.join(folder_job, "J%d_passthrough_particles.cs" % job_number)
        

    def get_mrc_filename(self, i_file: int):
        # TODO: Might want to look for them in both padded and unpadded versions
        if self.folder_type == "restacks":
            mrc_path = os.path.join(self.restacks_folder, f"batch_{i_file:06}_restacked.mrc")
        elif self.folder_type == "downsample":
            mrc_path = os.path.join(self.downsample_folder, f"batch_{i_file:06}_downsample.mrc")
        else:
            raise NotImplementedError("Impossible value for paths folder type.")
        if not os.path.exists(mrc_path):
            print(f"File {mrc_path} does not exist, breaking...")
            return None
        return mrc_path


class _Metadata():
    defocusU: FloatArrayType
    defocusV: FloatArrayType
    defocusAngle: FloatArrayType
    phaseShift: FloatArrayType
    sphericalAberration: float | FloatArrayType
    voltage: float | FloatArrayType
    amplitudeContrast: float | FloatArrayType
    defocus_is_degree: bool = True
    phase_shift_is_degree: bool = True
    ctfBfactor: float | FloatArrayType | None = None
    ctfScalefactor: float | FloatArrayType | None = None
    cs_files: np.ndarray | None = None
    cs_idxs: np.ndarray | None = None
    cs_pixel_size: np.ndarray | None = None


    def __init__(
        self,
        defocusU: float | FloatArrayType,
        defocusV: float | FloatArrayType,
        defocusAngle: float | FloatArrayType,
        sphericalAberration: float | FloatArrayType,
        voltage: float | FloatArrayType,
        amplitudeContrast: float | FloatArrayType,
        phaseShift: float | FloatArrayType,
        defocus_is_degree: bool = True,
        phase_shift_is_degree: bool = True,
        ctfBfactor: float | FloatArrayType | None = None,
        ctfScalefactor: float | FloatArrayType | None = None,
        cs_files: np.ndarray | None = None,
        cs_idxs: np.ndarray | None = None,
        cs_pixel_size: FloatArrayType | None = None
    ):
        self.defocusU = _ensure_np(defocusU)
        self.defocusV = _ensure_np(defocusV)
        self.defocusAngle = _ensure_np(defocusAngle)
        self.sphericalAberration = sphericalAberration
        self.voltage = voltage
        self.amplitudeContrast = amplitudeContrast
        self.phaseShift = _ensure_np(phaseShift)
        self.defocus_is_degree = defocus_is_degree
        self.phase_shift_is_degree = phase_shift_is_degree
        self.ctfBfactor = ctfBfactor
        self.ctfScalefactor = ctfScalefactor
        self.cs_files = cs_files
        self.cs_idxs = cs_idxs
        self.cs_pixel_size =cs_pixel_size


    @classmethod
    def from_star_file(cls, star_file: str, defocus_is_degree: bool, phase_shift_is_degree: bool):
        dataBlock, _ = read_star_file(star_file)
        for param in dataBlock.keys():
            y = dataBlock[param]
            y_unique = np.unique(y)
            if len(y_unique) == 1:
                dataBlock[param] = y_unique[0]

        defocusU = dataBlock["DefocusU"]
        defocusV = dataBlock["DefocusV"]
        defocusAngle = dataBlock["DefocusAngle"]
        sphericalAberration = dataBlock["SphericalAberration"]
        voltage = dataBlock["Voltage"]
        try:
            amplitudeContrast = dataBlock["AmplitudeContrast"]
        except:
            print("Warning: CTF amplitude contrast not found, using default value 0.1")
            amplitudeContrast = 0.1
        try:
            phaseShift = dataBlock["PhaseShift"]
        except:
            print("Warning: CTF phase shift not found, using default value 0.0")
            phaseShift = np.zeros_like(defocusU)
        try:
            ctfBfactor = dataBlock["CtfBfactor"]
        except:
            print("Warning: CTF B-factor not found, using default value 0.0")
            ctfBfactor = 0.0
        try:
            ctfScalefactor = dataBlock["CtfScalefactor"]
        except:
            print("Warning: CTF scale factor not found, using default value 1.0")
            ctfScalefactor = 1.0
        
        metadata = cls(
            defocusU,
            defocusV,
            defocusAngle,
            sphericalAberration,
            voltage,
            amplitudeContrast,
            phaseShift,
            defocus_is_degree=defocus_is_degree,
            phase_shift_is_degree=phase_shift_is_degree,
            ctfBfactor=ctfBfactor,
            ctfScalefactor=ctfScalefactor
        )
        return metadata



    @classmethod
    def from_cryospark_file(cls, file: str, get_fs_data: bool = False):
        if not os.path.exists(file):
            raise ValueError("Error: file not found: ", file)
        # prior versions have also considered looking at:
        # - image_shape = cs_data['blob/shape']
        # - ctfBfactor = cs_data['ctf/bfactor_A2']
        # - ctfScalefactor = cs_data['ctf/scale']
        # - uid = cs_data['uid']

        cs_data = np.load(file)
        defocusU = cs_data['ctf/df1_A']
        defocusV = cs_data['ctf/df2_A']
        defocusAng = cs_data['ctf/df_angle_rad']
        sphericalAberration = cs_data['ctf/cs_mm'][0]
        voltage = cs_data['ctf/accel_kv'][0]
        amplitudeContrast = cs_data['ctf/amp_contrast'][0]
        phaseShift = cs_data['ctf/phase_shift_rad']

        if get_fs_data:
            files = cs_data['blob/path']
            idxs = cs_data['blob/idx']
            try:
                pixel_size = cs_data['blob/psize_A'][0]
            except:
                pixel_size = None
        else:
            files = None
            idxs = None
            pixel_size = None

        metadata = cls(
            defocusU,
            defocusV,
            defocusAng,
            sphericalAberration,
            voltage,
            amplitudeContrast,
            phaseShift,
            defocus_is_degree=False,
            phase_shift_is_degree=False,
            cs_files = files,
            cs_idxs = idxs,
            cs_pixel_size = pixel_size
        )
        return metadata


    def take_range(self, batch_start: int, batch_end: int):
        return _Metadata(
            defocusU = _batchify(self.defocusU, batch_start, batch_end),
            defocusV = _batchify(self.defocusV, batch_start, batch_end),
            defocusAngle = _batchify(self.defocusAngle, batch_start, batch_end),
            sphericalAberration = _batchify(self.sphericalAberration, batch_start, batch_end),
            voltage = _batchify(self.voltage, batch_start, batch_end),
            amplitudeContrast = _batchify(self.amplitudeContrast, batch_start, batch_end),
            phaseShift = _batchify(self.phaseShift, batch_start, batch_end),
            defocus_is_degree = self.defocus_is_degree,
            phase_shift_is_degree = self.phase_shift_is_degree,
            ctfBfactor = _batchify(self.ctfBfactor, batch_start, batch_end),
            ctfScalefactor = _batchify(self.ctfScalefactor, batch_start, batch_end),
            cs_files = _batchify(self.cs_files, batch_start, batch_end),
            cs_idxs = _batchify(self.cs_files, batch_start, batch_end),
            cs_pixel_size = _batchify(self.cs_pixel_size, batch_start, batch_end)
        )


    def save_params_star(self, filename: str, n_images: int, im: Images, stack_start: int | None, stack_end: int | None):
        if(self.ctfBfactor is None or self.ctfScalefactor is None):
            raise ValueError("The save_params_star function requires that ctf B-factor and scale factor are set.")
        _start = 0 if stack_start is None else stack_start
        _end = im.n_images if stack_end is None else stack_end
        if (_start >= _end):
            raise ValueError("Request to save metadata parameters for a stack of non-positive length.")
        np.savez_compressed(
            filename,
            n_images = n_images,
            stack_start = _start,
            stack_end = _end,
            defocusU = self.defocusU,
            defocusV = self.defocusV,
            defocusAng = self.defocusAngle,
            sphericalAberration = self.sphericalAberration,
            voltage = self.voltage,
            amplitudeContrast = self.amplitudeContrast,
            ctfBfactor = self.ctfBfactor,
            ctfScalefactor = self.ctfScalefactor,
            phaseShift = self.phaseShift,
            box_size = im.box_size,
            n_pixels = im.phys_grid.n_pixels,
            pixel_size = im.phys_grid.pixel_size,
            defocus_angle_is_degree = self.defocus_is_degree,
            phase_shift_is_degree = self.phase_shift_is_degree
        )


    def save_params(self, filename: str, n_images: int, im: Images):
        np.savez_compressed(
            filename,
            n_images = n_images,
            defocusU = self.defocusU,
            defocusV = self.defocusV,
            defocusAng = self.defocusAngle,
            sphericalAberration = self.sphericalAberration,
            voltage = self.voltage,
            amplitudeContrast = self.amplitudeContrast,
            phaseShift = self.phaseShift,
            box_size = im.box_size,
            n_pixels = im.phys_grid.n_pixels,
            pixel_size = im.phys_grid.pixel_size,
            defocus_angle_is_degree = self.defocus_is_degree,
            phase_shift_is_degree = self.phase_shift_is_degree
        )


class _MetadataBuffer(_Metadata):
    defocusU: FloatArrayType
    defocusV: FloatArrayType
    defocusAngle: FloatArrayType
    phaseShift: FloatArrayType
    sphericalAberration: float
    voltage: float
    amplitudeContrast: float
    stack_size: int


    def __init__(self, meta: _Metadata):
        self.stack_size = 0
        self.defocusU = np.array([])
        self.defocusV = np.array([])
        self.defocusAngle = np.array([])
        self.phaseShift = np.array([])

        _abb = float(meta.sphericalAberration[0].item()) if isinstance(meta.sphericalAberration, np.ndarray) else float(meta.sphericalAberration)
        _volt = float(meta.voltage[0].item()) if isinstance(meta.voltage, np.ndarray) else float(meta.voltage)
        _ampl = float(meta.amplitudeContrast[0].item()) if isinstance(meta.amplitudeContrast, np.ndarray) else float(meta.amplitudeContrast)
        self.sphericalAberration = _abb
        self.voltage = _volt
        self.amplitudeContrast = _ampl
        self.defocus_is_degree = meta.defocus_is_degree
        self.phase_shift_is_degree = meta.phase_shift_is_degree
        self.ctfBfactor = meta.ctfBfactor
        self.ctfScalefactor = meta.ctfScalefactor


    def _ensure_size_consistency(self):
        if (self.defocusU.shape[0] != self.stack_size or
            self.defocusV.shape[0] != self.stack_size or
            self.defocusAngle.shape[0] != self.stack_size or
            self.phaseShift.shape[0] != self.stack_size
        ):
            raise ValueError("Inconsistent size in metadata buffers.")


    def make_copy(self, defU: FloatArrayType, defV: FloatArrayType, defA: FloatArrayType, phase: FloatArrayType):
        copy = _MetadataBuffer(self)
        copy.defocusU = defU
        copy.defocusV = defV
        copy.defocusAngle = defA
        copy.phaseShift = phase
        return copy


    def append_batch(self, defocusU: FloatArrayType, defocusV: FloatArrayType, defocusAng: FloatArrayType, phaseShift: FloatArrayType):
        if self.stack_size == 0:
            self.defocusU = defocusU
            self.defocusV = defocusV
            self.defocusAngle = defocusAng
            self.phaseShift = phaseShift
        else:
            self.defocusU = np.concatenate((self.defocusU, defocusU), axis = 0)
            self.defocusV = np.concatenate((self.defocusV, defocusV), axis = 0)
            self.defocusAngle = np.concatenate((self.defocusAngle, defocusAng), axis = 0)
            self.phaseShift = np.concatenate((self.phaseShift, phaseShift), axis = 0)
        self.sack_size = self.defocusU.shape[0]
        self._ensure_size_consistency()


    def pop_batch(self, batch_size: int):
        _b = min(batch_size, self.stack_size)
        (head_defocusU, tail_defocusU) = _pop_batch(self.defocusU, _b)
        (head_defocusV, tail_defocusV) = _pop_batch(self.defocusV, _b)
        (head_defocusAngle, tail_defocusAngle) = _pop_batch(self.defocusAngle, _b)
        (head_phaseShift, tail_phaseShift) = _pop_batch(self.phaseShift, _b)
        batch_meta = self.make_copy(head_defocusU, head_defocusV, head_defocusAngle, head_phaseShift)
        self.defocusU = tail_defocusU
        self.defocusV = tail_defocusV
        self.defocusAngle = tail_defocusAngle
        self.phaseShift = tail_phaseShift
        self.stack_size = self.defocusU.shape[0]
        self._ensure_size_consistency()
        batch_meta._ensure_size_consistency()
        return batch_meta


class ImgBuffer():
    images_phys: torch.Tensor
    images_fourier: torch.Tensor
    stack_size: int

    def __init__(self, ):
        self.images_phys = torch.tensor([])
        self.images_fourier = torch.tensor([])
        self.stack_size = 0


    def append_imgs(self, phys: torch.Tensor | None, fourier: torch.Tensor | None):
        if (phys is None or fourier is None):
            return
        if self.stack_size == 0:
            self.images_phys = phys
            self.images_fourier = fourier
        else:
            self.images_phys = torch.concatenate((self.images_phys, phys), dim = 0)
            self.images_fourier = torch.concatenate((self.images_fourier, fourier), dim = 0)
        self.stack_size = self.images_phys.shape[0]
        if (self.images_fourier.shape[0] != self.stack_size):
            raise ValueError("Physical and Fourier image buffers have differing lengths.")


    def pop_imgs(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        _b = min(batch_size, self.stack_size)
        assert self.images_phys is not None
        assert self.images_fourier is not None
        (phys_head, phys_tail) = _pop_batch(self.images_phys, _b)
        (fourier_head, fourier_tail) = _pop_batch(self.images_fourier, _b)
        self.images_phys = phys_tail
        self.images_fourier = fourier_tail
        return (phys_head, fourier_head)


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
    fn_plots_phys = os.path.join(dirs.folder_output_plots, f"particles_phys_stack{i_stack:06}.png")
    plot_images(phys_img, phys_grid=phys_grid, n_plots=n_plots, filename=fn_plots_phys)
    fn_plots_fourier = os.path.join(dirs.folder_output_plots, f"particles_fourier_stack{i_stack:06}.png")
    plot_images(fourier_img, polar_grid=polar_grid, n_plots=n_plots, filename=fn_plots_fourier)
    fn_plots_power = os.path.join(dirs.folder_output_plots, f"power_spectrum_stack{i_stack:06}.png")
    plot_power_spectrum(images_fourier=fourier_img, polar_grid=polar_grid, box_size=phys_grid.box_size, filename_plot=fn_plots_power)


# This was done at the beginning of each conversion function, but the result
# wasn't used anywhere.
def _collect_image_tag_list(folder_output: str):
    tag_file = os.path.join(folder_output, "image_file_tag_list.npy")
    if not os.path.exists(tag_file):
        return []
    image_file_tag_list = np.load(tag_file, allow_pickle = True)
    return image_file_tag_list.tolist()



def convert_particle_stacks(
    params_input: str | ParsedParameters,
    particle_file_list: list[str] = [],
    star_file_list: list[str] = [],
    folder_output: str = '',
    batch_size: int = 1024,
    pixel_size: float | FloatArrayType | None = None,
    defocus_angle_is_degree: bool = True,
    phase_shift_is_degree: bool = True,
    skip_exist: bool = False,
    flag_plots: bool = True
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

            _do_image_normalization(im_batch, polar_grid, params.precision)
            
            _plot_images(flag_plots, im_batch, _output_dirs, i_stack)
            torch.save(im_batch.images_phys, output_filenames.phys_stack)
            torch.save(im_batch.images_fourier, output_filenames.fourier_stack)

            batched_meta = meta.take_range(batch_start, batch_end)
            batched_meta.save_params_star(output_filenames.params_filename, im_batch.n_images, im_batch, batch_start, batch_end)
            i_stack += 1


def _process_indexed_cryosparc_file(
    unique_files: np.ndarray,
    indices_files: np.ndarray,
    i_file: int,
    folder_cryosparc: str,
    metadata: _Metadata,
    metadata_buffer: _MetadataBuffer,
    _pixel_size: FloatArrayType
) -> Images | None:
    assert metadata.cs_idxs is not None

    file = unique_files[i_file]
    file = file.decode('utf-8')
    if file[0] == '>':
        file = file[1:]
    mrc_file_path = os.path.join(folder_cryosparc, file)
    if not os.path.exists(mrc_file_path):
        print("File %s does not exist, skipping..." % mrc_file_path)
        return None
    
    indices_stack = np.where(indices_files == i_file)[0]
    idxs_stack = metadata.cs_idxs[indices_stack]
    
    im = Images.from_mrc(mrc_file_path, pixel_size = _pixel_size)
    im.select_images(idxs_stack)
    
    metadata_buffer.append_batch(
        metadata.defocusU[indices_stack],
        metadata.defocusV[indices_stack],
        metadata.defocusAngle[indices_stack],
        metadata.phaseShift[indices_stack]
    )

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
    _pixel_size = pixel_size if isinstance(pixel_size, np.ndarray) else np.array([pixel_size, pixel_size], dtype = float)

    n_images = metadata.defocusU.shape[0] 
    print("n_images:", n_images)

    unique_files, indices_files = np.unique(metadata.cs_files, return_inverse = True)
    n_unique_files = len(unique_files)

    i_restack = 0
    img_buffer = ImgBuffer()
    metadata_buffer = _MetadataBuffer(metadata)

    _last_good_im = None
    for i_file in range(n_unique_files):
        im = _process_indexed_cryosparc_file(unique_files, indices_files,
            i_file, folder_cryosparc,
            metadata, metadata_buffer,
            _pixel_size
        )
        if im is None:
            continue
        else:
            _last_good_im = im
        
        _do_image_normalization(im, polar_grid, params.precision, downsample_physical)
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
    pixel_size: float | None = None,
    downsample_physical: int = 1,
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

    polar_grid = _make_polar_grid_from_params(params)
    _output_dirs = OutputFolders(folder_output)
    _job_paths = JobPaths(folder_cryosparc, job_number)

    metadata = _Metadata.from_cryospark_file(_job_paths.file_cs)

    if pixel_size is None:
        raise ValueError("Error: pixel size not found")
    _pixel_size: FloatArrayType = np.array([pixel_size, pixel_size], dtype = float)

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
        _do_image_normalization(im, polar_grid, params.precision, downsample_physical)
        
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
