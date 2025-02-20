from os import path
from typing import TypeVar, cast
from numpy import ndarray
import numpy as np

from cryolike.util import FloatArrayType
from cryolike.stacks import Images
from cryolike.microscopy import read_star_file


def _ensure_np(d: float | FloatArrayType) -> FloatArrayType:
    if isinstance(d, ndarray):
        return d
    return np.array(d)


T2 = TypeVar("T2", bound=float | FloatArrayType | ndarray | None)
def _batchify(d: T2, start: int, end: int) -> T2:
    if isinstance(d, ndarray):
        return cast(T2, d[start:end])
    return d


class _Metadata():
    defocusU: FloatArrayType
    defocusV: FloatArrayType
    defocusAngle: FloatArrayType
    phaseShift: FloatArrayType
    sphericalAberration: float | FloatArrayType
    voltage: float | FloatArrayType
    amplitudeContrast: float | FloatArrayType
    defocus_is_degree: bool
    phase_shift_is_degree: bool
    ctfBfactor: float | FloatArrayType | None
    ctfScalefactor: float | FloatArrayType | None
    cs_files: ndarray | None
    cs_idxs: ndarray | None
    cs_pixel_size: ndarray | None


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
        cs_files: ndarray | None = None,
        cs_idxs: ndarray | None = None,
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
        self.cs_pixel_size = cs_pixel_size


    @classmethod
    def from_star_file(cls, star_file: str, defocus_is_degree: bool = True, phase_shift_is_degree: bool = True):
        dataBlock, _ = read_star_file(star_file)
        for param in dataBlock.keys():
            y = dataBlock[param]
            y_unique = np.unique(y)
            if len(y_unique) == 1 and param != 'DefocusAngle':
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
        # Not super necessary to test this--np.load will throw FileNotFound
        if not path.exists(file):
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
        # Q: shouldn't we check if n_images matches what we'd expect from stack_start and stack_end?
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

